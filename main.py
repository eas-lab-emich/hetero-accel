import logging
import traceback
import os.path
import pickle
import numpy as np
import pandas as pd
import yaml
from copy import deepcopy
from types import SimpleNamespace
from tabulate import tabulate
from src import dataset_dirs
from src.workload import MultiDNNWorkload
from src.utils import env_cfg, handle_model_subapps
from src.args import OperationMode
from src.net_wrapper import TorchNetworkWrapper
from src.compression.compressor import PruningQuantizationCompressor
from src.dataset import load_data
from src.accelerator_cfg import AcceleratorProfile
from src.optimization.optimizer import AcceleratorOptimizer
from src.baseline import run_baseline
from src.sota import run_sota
from src.partition import run_partition_comparison
from src.other_heuristics import run_genetic_algorithm, run_random_search

BASELINE_PRECISION = 8


logger = logging.getLogger(__name__)


def main():
    """Main executing function, supporting the execution of either
       our optimization, or others for comparisons
    """
    logging.basicConfig(level=logging.INFO)
    args = env_cfg()
    args.logdir = logging.getLogger().logdir
    # save arguments as pkl, for reproducibility
    with open(os.path.join(args.logdir, 'args.pkl'), 'wb') as f:
        pickle.dump(vars(args), f)

    # initialize the workload
    workload = setup_workload(args)
    # create a LUT of quantization profiles for each DNN-precision pairing
    dnn_accuracy_lut, compressors = quant_exploration(args, workload)

    if args.operation_mode == OperationMode.Ours:
        # perform a DSE to define the sub-accelerator architectures
        accelerator_exploration(args, workload, dnn_accuracy_lut)
        # # prune the DNNs to produce the final scheduler
        # schedule, metrics = pruned_schedule(args, workload, dnn_accuracy_lut,
        #                                     compressors, optimizer, optimizer.state)

    # evaluate a given baseline accelerator architecture
    elif args.operation_mode == OperationMode.Baseline:
        run_baseline(args, workload, dnn_accuracy_lut)

    # execute the optimizations in the state-of-the-art
    elif args.operation_mode == OperationMode.SOTA:
        run_sota(args, workload, dnn_accuracy_lut)

    # compare our technique against partition-aware scheduling
    elif args.operation_mode == OperationMode.Partition:
        run_partition_comparison(args, workload, dnn_accuracy_lut)

    # compare against a genetic algorithm
    elif args.operation_mode == OperationMode.Genetic:
        run_genetic_algorithm(args, workload, dnn_accuracy_lut)

    # compare against a random-search approach
    elif args.operation_mode == OperationMode.RandomSearch:
        run_random_search(args, workload, dnn_accuracy_lut)


def setup_workload(args):
    """Initialize the multi-DNN workload
    """
    dnn_args = deepcopy(args)
    dnns = {}
    datasets = {}
    print_frequency = {}

    # load workload file
    with open(args.workload_cfg_file, 'r') as stream:
        workloads = yaml.safe_load(stream)
    multi_dnn_workload = workloads['workloads'][args.workload_idx]

    # setup each DNN separately
    for idx, workload_dict in enumerate(multi_dnn_workload):

        # gather workload arguments
        for name, value in workload_dict.items():
            setattr(dnn_args, name, value)
        # specific handling of batch size
        if 'batch_size' not in workload_dict:
            dnn_args.batch_size = args.batch_size

        # configure and save DNN wrapper
        net_wrapper = TorchNetworkWrapper.from_args(dnn_args)
        dnns[net_wrapper.arch] = net_wrapper
        print_frequency[net_wrapper.arch] = dnn_args.batch_print_frequency

        if dnn_args.dataset not in datasets:
            # configure dataset
            data_loaders = load_data(
                dnn_args.dataset,
                dataset_dirs[dnn_args.dataset],
                net_wrapper.arch,
                dnn_args.batch_size,
                args.workers,
                args.validation_split,
                args.effective_train_size,
                args.effective_valid_size,
                args.effective_test_size,
                args.evaluate_model_mode,
                True, #self.args.verbose
                args.cpu
            )
            datasets[dnn_args.dataset] = data_loaders

        # manually execute the summary, if not already done
        if getattr(net_wrapper, 'summary', None) is None:
            net_wrapper.run_summary(datasets[dnn_args.dataset][2]) # use the test dataset for exploring network geometry

        # execute sub-applications
        if handle_model_subapps(net_wrapper, data_loaders, args):
            exit(0)

    return MultiDNNWorkload(dnns, datasets, print_frequency)


def init_compressor(args, workload, arch, net_wrapper):
    compression_args = SimpleNamespace(logdir=args.logdir,
                                       pruning_high=args.pruning_high,
                                       pruning_low=args.pruning_low,
                                       quant_high=args.quant_high,
                                       quant_low=args.quant_low,
                                       layer_type_whitelist=args.layer_type_whitelist,
                                       pruning_group_type=args.pruning_group_type,
                                       accelerator_cfg=AcceleratorProfile(args.accelerator_arch_type),
                                       # DNN args for inheritance from TorchNetworkWrapper
                                       optimizer_type=args.optimizer_type,
                                       profile_model=False,
                                       gpus=args.gpus,
                                       cpu=args.cpu,
                                       print_frequency=workload.print_frequency[arch],
                                       verbose=args.model_verbose)
    return PruningQuantizationCompressor(compression_args,
                                         workload.datasets[net_wrapper.dataset],
                                         net_wrapper.model)


def quant_exploration(args, workload):
    """Exploration of possible quantization profiles
    """
    skip_exploration = False
    if getattr(args, 'dnn_accuracy_lut_file', None) is not None and \
       os.path.exists(args.dnn_accuracy_lut_file):
        skip_exploration = True
        preloaded_dnn_accuracy_lut = pd.read_csv(args.dnn_accuracy_lut_file)
        assert all(arch in preloaded_dnn_accuracy_lut['Network'].unique() for arch in workload.dnns), \
            f"All DNNs {list(workload.dnns.keys())} must be included in the preloaded accuracy LUT: {args.dnn_accuracy_lut_file}"
        logger.info(f'=> Skipping exhaustive exploration: loaded LUT from {args.dnn_accuracy_lut_file}')
        
    # LUT structure
    columns = ['Network', 'QuantBits', 
               'Accuracy', 'AccuracySubMetric1', 'AccuracySubMetric2', 'AccuracySubMetric3',
               'Sparsity', 'Size', 'Valid']

    accuracy_columns = [column for column in columns if 'accuracy' in column.lower()]
    max_accuracy_metrics_recorded = len(accuracy_columns)

    # initialize LUT
    if skip_exploration:
       df = preloaded_dnn_accuracy_lut
    else:
        df = pd.DataFrame(columns=columns)

    compressors = {}
    for arch, net_wrapper in workload.dnns.items():

        # check if there is at least one accuracy constraint set
        assert any(
            getattr(args, f'{metric}_constraint', None) is not None for metric in net_wrapper.accuracy_metrics
        ), f"No accuracy constraint was set! Define at least one of the following in {args.yaml_cfg_file}: " \
           f"{' | '.join([f'{metric}_constraint'] for metric in net_wrapper.accuracy_metrics)}"

        if not skip_exploration:
            # initialize compressor
            compressor = init_compressor(args, workload, arch, net_wrapper)
            compressors[arch] = compressor
            logger.info(f'=> Beginning exhaustive exploration for {arch}')
            compressor.quantize(BASELINE_PRECISION)

            # compute accuracy statistics
            accuracy_stats = compressor.validate() if args.use_validation_set else compressor.test()
            accuracy_stats = list(accuracy_stats)
            accuracy_stats.extend(
                max(0, max_accuracy_metrics_recorded - len(accuracy_stats)) * [0.0]
            )
            assert len(accuracy_stats) >= max_accuracy_metrics_recorded

            # compute the rest and group together
            # model_stats, _ = compressor.compute_model_statistics()
            og_stats = {
                'accuracy': accuracy_stats[0],
                'sparsity': 0, 'size': 0,
                **{
                    metric: accuracy_stats[i] for i, metric in enumerate(net_wrapper.accuracy_metrics)
                }
            }

            # save the statistics to the LUT
            df.loc[len(df.index)] = ([arch, BASELINE_PRECISION,
                                      *accuracy_stats[:max_accuracy_metrics_recorded],
                                      0, 0, 1])

        else:
            og_stats = df.loc[(df['Network'] == arch) & (df['QuantBits'] == BASELINE_PRECISION)].iloc[0].to_dict()
            og_stats.pop('Network')
            og_stats.pop('Unnamed: 0', None)
            for metric, column in zip(net_wrapper.accuracy_metrics, accuracy_columns):
                og_stats[metric] = og_stats[column]

        og_stats_logstr = ', '.join([f'{metric.capitalize()}={value:.2f}' if metric != 'size' else
                                    f'{metric.capitalize()}={value:.2e}'
                                    for metric, value in og_stats.items()])
        logger.info(f'{arch}: Original statistics; Precision of {BASELINE_PRECISION}: {og_stats_logstr}')

        # iterate over quantization bits
        quant_bits_options = np.arange(args.quant_low, args.quant_high + 1, args.quant_incr)
        # if 16 not in quant_bits_options:
        #     quant_bits_options = np.append(quant_bits_options, 16)

        for quant_bits in quant_bits_options:

            if not skip_exploration:
                logger.info(f'{arch}: Testing quantization of {quant_bits} bits')
                # reset the previous state of the network
                compressor.reset()
                # execute the compression profile
                compressor.quantize(quant_bits)
                # evaluate for accuracy and network statistics
                accuracy_stats = compressor.validate() if args.use_validation_set else compressor.test(use_quant=True)
                accuracy_stats = list(accuracy_stats)
                accuracy_stats.extend(
                    max(0, max_accuracy_metrics_recorded - len(accuracy_stats)) * [0.0]
                )
                assert len(accuracy_stats) >= max_accuracy_metrics_recorded

                # model_stats, _ = compressor.compute_model_statistics()
                stats = {
                    'accuracy': accuracy_stats[0],
                    'sparsity': 0, 'size': 0,
                    **{
                        metric: accuracy_stats[i] for i, metric in enumerate(net_wrapper.accuracy_metrics)
                    }
                }

                # save the statistics to the LUT, except of the 'valid' flag
                df.loc[len(df.index)] = ([arch, quant_bits,
                                          *accuracy_stats[:max_accuracy_metrics_recorded],
                                          0, 0, 0])

            else:
                stats = df.loc[(df['Network'] == arch) & (df['QuantBits'] == quant_bits)].iloc[0].to_dict()
                stats.pop('Network')
                stats.pop('Unnamed: 0', None)
                for metric, column in zip(net_wrapper.accuracy_metrics, accuracy_columns):
                    stats[metric] = stats[column]

            stats_logstr = ', '.join([f'{metric.capitalize()}={value:.2f}' if metric != 'size' else
                                      f'{metric.capitalize()}={value:.2e}'
                                      for metric, value in stats.items()])
            logger.info(f'{arch}: Compressed statistics: {stats_logstr}')

            # binary flag whether at least one of the accuracy constraints are satisfied
            valid = 0
            if all(
                getattr(args, f'{metric}_constraint', None) is None or
                stats[metric] >= og_stats[metric] - getattr(args, f'{metric}_constraint')
                for metric in net_wrapper.accuracy_metrics
            ):
                valid = 1

            # save the binary flag
            df.loc[(df['Network'] == arch) & (df['QuantBits'] == quant_bits), 'Valid'] = valid
            logger.info(f"Is quantization valid? -> {bool(valid)}")

    # check if any valid solutions were found
    assert df['Valid'].sum() > 1, "No valid solutions were found, consider changing the compression settings or " \
                                  "loosen the accuracy constraints"

    # save LUT to .csv file
    df.to_csv(os.path.join(args.logdir, 'lut.csv'))

    return df, compressors


def accelerator_exploration(args, workload, accuracy_lut):
    """Exploration to design/discover the sub-accelerator architectures
    """
    precision_options = sorted(set(
        accuracy_lut.loc[accuracy_lut['Valid'] == 1]['QuantBits']
    ))
    # remove 16 and 32 bits from the options
    try:
        precision_options.remove(32)
    except ValueError:
        pass
    try:
        precision_options.remove(16)
    except ValueError:
        pass

    # check if there is at least one valid option for each DNN
    if any(
        accuracy_lut.loc[
            (accuracy_lut['Network'] == arch) & 
            (accuracy_lut['QuantBits'] != 32) & 
            (accuracy_lut['QuantBits'] != 16)
        ]['Valid'].sum() == 0
        for arch in workload.dnns
    ):
        which_dnns = [
            arch for arch in workload.dnns
            if accuracy_lut.loc[
                (accuracy_lut['Network'] == arch) & 
                (accuracy_lut['QuantBits'] != 32) & 
                (accuracy_lut['QuantBits'] != 16)
            ]['Valid'].sum() == 0
        ]
        raise ValueError("The following DNNs cannot be used, as all quantization options "
                         f"violate the accuracy constraint: {', '.join(which_dnns)}")

    accel_cfg = AcceleratorProfile(args.accelerator_arch_type)
    accel_cfg.design_space['precision'] = precision_options

    logger.debug(f"Examining design space: {accel_cfg.design_space}")

    # initialize and run optimizer
    optimizer = AcceleratorOptimizer(args=args,
                                     num_accelerators=len(precision_options),
                                     accelerator_cfg=accel_cfg,
                                     workload=workload,
                                     accuracy_lut=accuracy_lut,
                                     hw_constraints=SimpleNamespace(deadline=args.deadline_constraint,
                                                                    area=args.area_constraint),
                                     logdir=args.logdir
                                     )

    if not args.skip_exploration:
        optimizer.run()

    logger.info("*------------------*")
    comment = ''
    if args.skip_exploration and getattr(args, 'load_state_from', None) is not None:
        comment = f' (loaded from: {args.load_state_from}) ' 
    logger.info(f"Final heterogeneous accelerator{comment}:")
    for state in optimizer.best_state:
        logger.info(f'\t{state}')

    # get the scheduling evaluation from the best accelerator state
    optimizer.set_state(optimizer.best_state)
    logger.info(f"Final scheduling:")
    optimizer.energy(initial=False, save_best=False)

    logger.info("*------------------*")
    optimizer.close()


def pruned_schedule(args, workload, dnn_accuracy_lut, compressors, optimizer, hetero_accel):
    """Prune the DNNs according to the final heterogeneous accelerator architecture
    """
    def lut_entry(arch, precision):
        return dnn_accuracy_lut.loc[
                (dnn_accuracy_lut['Network'] == arch) &
                (dnn_accuracy_lut['QuantBits'] == precision)
            ]

    def is_valid(arch, accuracy_stats):
        og_entry = lut_entry(arch, BASELINE_PRECISION)
        og_stats = tuple(og_entry[metric].iloc[0]
                         for metric in dnn_accuracy_lut.columns
                         if 'accuracy' in metric.lower())
        return all(
                getattr(args, f'{metric}_constraint', None) is None or
                accuracy_stats[idx] >= og_stats[idx] - getattr(args, f'{metric}_constraint')
                for idx, metric in enumerate(compressor.accuracy_metrics)
            )
    
    def compare_dense_pruned_results(key, pruned_mappings, prefix=""):
        df = pd.DataFrame(columns=['Unpruned', 'Pruned'], index=['Energy', 'Latency', 'EDP'])
        df.loc['Energy', 'Unpruned'] = optimizer.energy_dict[key]
        df.loc['Latency', 'Unpruned'] = optimizer.latency_dict[key]
        df.loc['EDP', 'Unpruned'] = optimizer.energy_dict[key] * optimizer.latency_dict[key]
        df.loc['Energy', 'Pruned'] = pruned_mappings.energy[key]
        df.loc['Latency', 'Pruned'] = pruned_mappings.latency[key]
        df.loc['EDP', 'Pruned'] = pruned_mappings.energy[key] * pruned_mappings.latency[key]
        t = tabulate(df, headers='keys', tablefmt='psql', floatfmt=".3e")
        logger.info(f"{prefix}Unpruned vs Pruned results for {key[0]} -> {key[1]}:\n{t}")

    def get_scheduling_results(schedule, mapping, hetero_accel, prefix="", verbose=False):
        total_area = sum([
            mapping.area[single_accel] for single_accel in hetero_accel
        ])
        total_energy = sum([
            mapping.energy[(entry.tag, entry.bin)] for entry in schedule.entries
        ])
        total_latency = max([
            sum([
                mapping.latency[(entry.tag, entry.bin)] for entry in entries
            ]) for bin, entries in schedule.as_dict(main_key='bin').items()
        ])
        metrics = SimpleNamespace(energy=total_energy,
                                  latency=total_latency,
                                  edp=total_energy * total_latency,
                                  area=total_area)

        if verbose:
            prefix = f"[{prefix}] " if prefix != "" else ""
            logger.info(f"=> {prefix}schedule metrics:")
            for metric, value in vars(metrics).items():
                logger.info(f"\t{metric.capitalize()} = {value:.3e}")

        return metrics

    # load measurements
    skip_mapping = False
    if getattr(args, 'load_pruned_mappings_from', None) is not None and \
       os.path.exists(args.load_pruned_mappings_from):
        skip_mapping = True
        with open(args.load_pruned_mappings_from, 'rb') as f:
            pruned_mappings = pickle.load(f)
        logger.info(f"=> Loaded pruned mappings from {args.load_pruned_mappings_from}")

    else:
        # create dictionaries to save measurements
        edp_dict = {key: optimizer.energy_dict[key] * optimizer.latency_dict[key]
                    for key in optimizer.energy_dict}
        pruned_mappings = SimpleNamespace(energy=deepcopy(optimizer.energy_dict),
                                          latency=deepcopy(optimizer.latency_dict),
                                          edp=edp_dict,
                                          area=deepcopy(optimizer.area_dict))

    # re-initialize compressors if they were not loaded
    if len(compressors) == 0:
        for arch, net_wrapper in workload.dnns.items():
            compressors[arch] = init_compressor(args, workload, arch, net_wrapper)

    # evaluate the previous schedule with the unpruned mappings
    unpruned_schedule = deepcopy(optimizer.latest_schedule)
    unpruned_schedule_metrics = SimpleNamespace(energy=optimizer.latest_energy,
                                                latency=optimizer.latest_latency,
                                                edp=optimizer.latest_energy * optimizer.latest_latency,
                                                area=optimizer.latest_area)

    logger.info("=> Beginning accelerator-aware pruning")

    # iterate over each accelerator
    for accelerator in hetero_accel:
        if skip_mapping:
            continue

        logger.info(f"\tEvaluating accelerator: {accelerator}")
        # iterate over each DNN
        for arch, compressor in compressors.items():
            logger.info(f"\t\tCompressing DNN: {arch}")

            # check the accuracy of the quantized DNN
            entry = lut_entry(arch, accelerator.precision)
            accuracy_satisfied = entry['Valid'].iloc[0]
            if not accuracy_satisfied:
                logger.info(f"\t\tSkipping pruning of quantized DNN: accuracy already violated")
                # Invalid scheduling mappings are marked with negative weight (latency)
                pruned_mappings.energy[(arch, accelerator)] = -1
                pruned_mappings.latency[(arch, accelerator)] = -1
                pruned_mappings.edp[(arch, accelerator)] = -1
                continue
 
            # # the accuracy metric is the first on returned from the accuracy meter
            # # that has a valid given constraint
            # accuracy_metric = next(metric for metric in compressor.accuracy_metrics
            #                        if getattr(args, metric + '_constraint', None) is not None)

            # check compliance with accuracy threshold, in case of mismatches from previous run
            accuracy_stats = tuple(entry[column].iloc[0]
                                   for column in entry.columns
                                   if 'accuracy' in column.lower())
            if accuracy_satisfied and not is_valid(arch, accuracy_stats):
                logger.warning(f"A solution was accepted but surpasses the given constraint")

            # increamentally prune the DNN until an accuracy violation
            prune_ratio = 0.0
            while accuracy_satisfied:
                prune_ratio += 0.05

                # execute quantization and pruning with new ratio
                compressor.reset()
                compressor.prune_and_quantize(prune_ratio, accelerator.precision)

                # evaluate for accuracy and network statistics
                accuracy_stats = compressor.validate() if args.use_validation_set else compressor.test()
                # check constraint again
                accuracy_satisfied = is_valid(arch, accuracy_stats)

            # continue from lastly pruned model that satisfied the accuracy constraint
            prune_ratio = max(0.0, prune_ratio - 0.05)

            # if no pruning can be handled, then skip the evaluation, as the metrics from the
            #  exact DNN-accelerator mapping are already stored
            if prune_ratio == 0.0 and (arch, accelerator) in pruned_mappings.energy:
                logger.info(f"\t\tSkipping pruning of quantizated DNN {arch}: accuracy would be violated")
                continue

            # execute pruning and quantization
            compressor.reset()
            compressor.prune_and_quantize(prune_ratio, accelerator.precision)

            # evaluate the final pruned/quantized accuracy and sparsity statistics
            accuracy_stats = compressor.validate() if args.use_validation_set else compressor.test()
            total_stats, layer_stats = compressor.compute_model_statistics()
            logger.info(f"\t\tCompleted pruning/quantization for {arch}: ratio={prune_ratio}, bits={accelerator.precision}")
            logger.info("\t\tResults: " + ', '.join([
                f'{metric}={accuracy_stats[i]:.2f}'
                for i, metric in enumerate(compressor.accuracy_metrics)
            ]))
            logger.info(f"\t\tResults: sparsity={total_stats['sparsity']*100:.2f}%")

            # evaluate the DNN within timeloop
            energy = latency = 0
            optimizer.timeloop_wrapper.adjust_architecture(accelerator)
            for problem_name in optimizer.timeloop_problems_per_dnn[arch]:
                layer_name = optimizer.timeloop_problem_to_layer_name[arch][problem_name]

                # check extreme sparsity cases
                filters = optimizer.timeloop_wrapper.workloads[problem_name].config['instance']['M']
                channels = optimizer.timeloop_wrapper.workloads[problem_name].config['instance']['C']
                if (filters * (1 - layer_stats[layer_name]['sparsity_filters']) <= 1 and filters > 1) or \
                   (channels * (1 - layer_stats[layer_name]['sparsity_channels']) <= 1 and channels > 1):
                    logger.warning(f"\t\t{arch} layer {layer_name}: Attempting to leave less than 1 filter/channel "
                                   f"unpruned. Considering this layer completely pruned.")
                    continue

                # adjust for removed filters/channels
                optimizer.timeloop_wrapper.adjust_problem_dimension(
                    problem_name, 'M', adjust_by=(1 - layer_stats[layer_name]['sparsity_filters'])
                )
                optimizer.timeloop_wrapper.adjust_problem_dimension(
                    problem_name, 'C', adjust_by=(1 - layer_stats[layer_name]['sparsity_channels'])
                )
                # run timeloop and get results
                optimizer.timeloop_wrapper.run(problem_name)
                results = optimizer.timeloop_wrapper.get_results()
                optimizer.timeloop_wrapper.cleanup()
                energy += results.energy
                latency += results.cycles

            # save DNN results
            pruned_mappings.energy[(arch, accelerator)] = energy
            pruned_mappings.latency[(arch, accelerator)] = latency
            pruned_mappings.edp[(arch, accelerator)] = energy * latency

        if accelerator not in pruned_mappings.area:
            pruned_mappings.area[accelerator] = getattr(results, 'area', 0.0)

    logger.info(f"{'Skipped' if skip_mapping else 'Completed'} pruned mapping evaluation")
    savefile = os.path.join(optimizer.logdir, 'pruned_mappings.pkl')
    with open(savefile, 'wb') as f:
        pickle.dump(pruned_mappings, f)
    logger.info(f"Saved pruned mappings in {savefile}")

    # execute and evaluate the final schedule with the pruned mappings
    schedule = optimizer.scheduler.run(items=list(compressors.keys()),
                                       bins=hetero_accel,
                                       weight_dict=pruned_mappings.energy,
                                       cost_dict=pruned_mappings.latency,
                                       solver_type=args.solver_type)
    pruned_schedule_metrics = get_scheduling_results(schedule, pruned_mappings, hetero_accel,
                                                     verbose=True, prefix=f"Pruned mappings, final")
    schedule_str = '\n\t'.join([f'{entry.tag} -> {entry.bin}' for entry in schedule.entries])
    logger.info(f"Scheduler results:\n\t{schedule_str}")

    # check if the pruned schedule is better than the unpruned one
    if pruned_schedule_metrics.edp >= unpruned_schedule_metrics.edp:
        # check if the previous schedule (but with pruned DNNs) is better than the final one
        unpruned_schedule_updated_metrics = get_scheduling_results(unpruned_schedule, pruned_mappings, hetero_accel,
                                                                verbose=True, prefix="Pruned mappings, previous")
        if unpruned_schedule_updated_metrics.edp >= unpruned_schedule_metrics.edp:
            # scheduling has failed: neither pruning nor another schedule improved the EDP    
            for entry in unpruned_schedule.entries:
                compare_dense_pruned_results((entry.tag, entry.bin), pruned_mappings)
            raise ValueError("Pruning did not improve the EDP scheduling efficiency")
        else:
            logger.warning("Pruning did not improve the EDP scheduling efficiency, but the previous schedule is better")
            logger.info("=> Using the previous schedule with pruned DNNs")
            best_schedule = unpruned_schedule
            final_metrics = unpruned_schedule_updated_metrics
    else:
        best_schedule = schedule
        final_metrics = pruned_schedule_metrics

    return best_schedule, final_metrics


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n-- KeyboardInterrupt --")
    except Exception as e:
        if logger is not None:
            # We catch unhandled exceptions here in order to log them to the log file
            # However, using the logger as-is to do that means we get the trace twice in stdout - once from the
            # logging operation and once from re-raising the exception. So we remove the stdout logging handler
            # before logging the exception
            handlers_bak = logger.handlers
            logger.handlers = [h for h in logger.handlers if type(h) != logging.StreamHandler]
            logger.error(traceback.format_exc())
            logger.handlers = handlers_bak
        raise
    finally:
        if logger is not None and hasattr(logging.getLogger(), 'log_filename'):
            logger.info('')
            logger.info('Log file for this run: ' + os.path.realpath(logging.getLogger().log_filename))
            exit()


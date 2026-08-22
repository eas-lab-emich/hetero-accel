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
from src.mapping.api import ConvolutionProblem
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

def extract_problems(dnn_summary) -> list[ConvolutionProblem]:
    eligible_layer_types = ['conv2d', 'linear']
    layers = []
    for layer in dnn_summary.values():
        if layer.layer_type.lower() in eligible_layer_types:
            layers.append(ConvolutionProblem(
                input_channels=layer.dimensions['C'],
                output_channels=layer.dimensions['K'],
                input_width=layer.dimensions['Xi'],
                input_height=layer.dimensions['Yi'],
                padding_width=layer.dimensions['Wpad'],
                padding_height=layer.dimensions['Hpad'],
                stride_width=layer.dimensions['Wstr'],
                stride_height=layer.dimensions['Hstr'],
                batch_size=layer.dimensions['N'],
                kernel_width=layer.dimensions['S'],
                kernel_height=layer.dimensions['R']
            ))
    return layers

def main():
    """Main executing function, supporting the execution of either
       our optimization, or others for comparisons
    """
    logging.basicConfig(level=logging.DEBUG)
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
                True,  # self.args.verbose
                args.cpu
            )
            datasets[dnn_args.dataset] = data_loaders

        # manually execute the summary, if not already done
        if getattr(net_wrapper, 'summary', None) is None:
            net_wrapper.run_summary(
                datasets[dnn_args.dataset][2])  # use the test dataset for exploring network geometry

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
    workload = {
        dnn_name: extract_problems(dnn.summary) for dnn_name, dnn in workload.dnns.items()
    }

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

    # # get the scheduling evaluation from the best accelerator state
    # optimizer.set_state(optimizer.best_state)
    # logger.info(f"Final scheduling:")
    # optimizer.energy(initial=False)

    logger.info("*------------------*")
    optimizer.close()


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

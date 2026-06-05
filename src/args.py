import argparse
from enum import Enum
from src.accelerator_cfg import AcceleratorType
from src.compression.pruning import PruningGroupType
from src.optimization.scheduling import SchedulerType, SolverType


def app_args(parser):
    """Arguments for running the main application"""
    parser.add_argument("--mode", "-m", dest='operation_mode',
                        type=operation_mode_arg, default='ours',
                        help='Define the operation mode. Possible option are: '
                              '\'ours\' (default, execute our SA-based optimization), '
                              '\'baseline\' (execute a baseline calculation), ')

    parser.add_argument('--out-dir', '-o', dest='output_dir', default='logs',
                         help='Path to dump logs and checkpoints')
    parser.add_argument('--name', '-n', help='Experiment name')
    parser.add_argument('--deterministic', '--det', action='store_true',
                        help='Set for deterministic execution. Preferably, set the experiment --seed as well')
    parser.add_argument('--seed', dest='global_seed', type=int,
                        help='Global seed for deterministic execution')
    parser.add_argument('--yaml-cfg-file', required=True,
                        help='YAML file containing the experiment description')
    parser.add_argument('--workload-cfg-file', required=True,
                        help='YAML file describing the multi-DNN workloads (required)')
    parser.add_argument('--dnn-accuracy-lut-file', metavar='PATH',
                        help='Path where the DNN-accuracy LUT is saved. Should contain '
                             'statistics for each DNN, per quantization profile')
    parser.add_argument('--scheduler-type', type=scheduler_type_arg, default='ours',
                        help='Select scheduler type. Default is our scheduler.')
    parser.add_argument('--solver-type', type=solver_type_arg, default='mthggreedy',
                        help='Select solver type for the scheduling algorithm. Default is MTHG-Greedy')
    parser.add_argument('--load-state-from', metavar='PATH',
                        help='Load the state of the SA optimization from the given file')
    parser.add_argument('--load-pruned-mappings-from', metavar='PATH',
                        help='Pickle file to load the pruned DNN-to-accelerator mappings')

    dnn_op_mode = parser.add_argument_group("DNN execution mode arguments")
    dnn_op_mode_exc = dnn_op_mode.add_mutually_exclusive_group()
    dnn_op_mode_exc.add_argument('--evaluate-model', dest='evaluate_model_mode', action='store_true',
                                 help='Evaluate DNN model on test set')
    dnn_op_mode_exc.add_argument('--train-model', dest='train_model_mode', action='store_true',
                                 help='Train DNN model on train set')
    dnn_op_mode_exc.add_argument('--test-timeloop-accelergy', dest='test_timeloop_accelergy_mode', action='store_true',
                                 help='Execute an example to test if timeloop+accelergy works well')
    dnn_op_mode_exc.add_argument('--model-summary', dest='model_summary_mode', type=model_summary_type_arg, default='',
                                 help='Specify the type of summary of the given DNNs')
    dnn_op_mode_exc.add_argument('--test-pruning-quantization', dest='test_pruning_quant_mode', action='store_true',
                                 help='Execute a test of the effect of pruning on energy consumption')
    
    optimizer_args = parser.add_argument_group('Optimizer arguments')
    optimizer_args.add_argument('--optimizer-type', '--ot', type=optimizer_type_arg, default='adam',
                                help='Choose optimizer type')
    return parser


def workload_args(parser):
    """Arguments for configuring the multi-DNN workload"""
    workload_args = parser.add_argument_group("DNN related arguments")
    workload_args.add_argument('--workload', dest='workload_idx', default='A', choices=['A', 'B', 'C'],
                               help='')
    workload_args.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                               help='Number of data loading workers (default: 4)')
    workload_args.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                               help='Mini-batch size (default: 256)')
    workload_args.add_argument('--load-serialized', dest='load_serialized', action='store_true', default=False,
                               help='Load a model without DataParallel wrapping it')
    workload_args.add_argument('--gpus', metavar='DEV_ID', nargs='*',
                               help='List of GPU device IDs to be used. Can be used as space- or comma-separated. '
                                    'Default is to use all available devices.')
    workload_args.add_argument('--cpu', action='store_true', default=False,
                               help='Use CPU only. \n'
                                    'Flag not set => uses GPUs according to the --gpus flag value.'
                                    'Flag set => overrides the --gpus flag')
    workload_args.add_argument('--validation-split', '--valid-size', '--vs', dest='validation_split', type=float, default=0.1,
                               help='Portion of training dataset to set aside for validation')
    workload_args.add_argument('--effective-train-size', '--etrs', type=float, default=1.,
                               help='Portion of training dataset to be used in each epoch. '
                                    'NOTE: If --validation-split is set, then the value of this argument is applied '
                                    'AFTER the train-validation split according to that argument')
    workload_args.add_argument('--effective-valid-size', '--evs', type=float, default=1.,
                               help='Portion of validation dataset to be used in each epoch. '
                                    'NOTE: If --validation-split is set, then the value of this argument is applied '
                                    'AFTER the train-validation split according to that argument')
    workload_args.add_argument('--effective-test-size', '--etes', type=float, default=1.,
                               help='Portion of test dataset to be used in each epoch')
    workload_args.add_argument('--model-verbose', '--mv', action='store_true', 
                               help='Log various messages for the application model')
    return parser


def compression_args(parser):
    """Arguments for pruning/quantization exploration
    """
    compression_args = parser.add_argument_group("Compression related arguments")
    compression_args.add_argument('--pruning-group-type', type=pruning_group_type_arg, default='columns',
                                  help='Pruning group type. Default is column pruning')
    compression_args.add_argument('--use-validation-set', action='store_true',
                                  help='Whether to use validation set during inference. If not specified, test set is used')
    compression_args.add_argument('--pruning-high', type=float, default=0.95,
                                  help='Highest bound for pruning')
    compression_args.add_argument('--pruning-low', type=float, default=0.0,
                                  help='Lowest bound for pruning')
    compression_args.add_argument('--pruning-increment', '--pruning-incr', dest='pruning_incr', type=float, default=0.05,
                                  help='Increment step for pruning exploration')
    compression_args.add_argument('--quant-high', type=int, default=8,
                                  help='Highest bound for weight quantization')
    compression_args.add_argument('--quant-low', type=int, default=2,
                                  help='Lowest bound for weight quantization')
    compression_args.add_argument('--quantization-increment', '--quant-incr', dest='quant_incr', type=int, default=2,
                                  help='Increment step for quantization exploration')
    # Constraints
    compression_args.add_argument('--top1-constraint', type=float, default=1,
                                  help='Top1 accuracy loss constraint for image classification (default is 1\%)')
    compression_args.add_argument('--top5-constraint', type=float,
                                  help='Top5 accuracy loss constraint for image classification')
    compression_args.add_argument('--loss-constraint', type=float,
                                  help='Objective loss constraint for image classification')
    compression_args.add_argument('--miou-constraint', type=float, dest='mIOU_constraint', default=1,
                                  help='Mean intersect-over-union constraint for segmentation')
    compression_args.add_argument('--pixel-acc-constraint', type=float, dest='AvgPixelAcc_constraint',
                                  help='Average pixel accuracy constraint for segmentation')
    compression_args.add_argument('--map-50t95-constraint', type=float, dest='mAP[IoU=0.50:0.95]_constraint', default=1,
                                  help='Mean average precision ([IoU=0.50:0.95]) constraint for detection')
    compression_args.add_argument('--map-50-constraint', type=float, dest='mAP[IOU=0.5]_constraint',
                                  help='Mean average precision ([IoU=0.5]) constraint for detection')
    compression_args.add_argument('----map-75-constraint', type=float, dest='mAP[IOU=0.75]_constraint',
                                  help='Mean average precision ([IoU=0.75]) constraint for detection')
    return parser


def accel_args(parser):
    """Arguments related to the hardware DNN accelerator
    """
    accel_args = parser.add_argument_group("Accelerator-related arguments")
    accel_args.add_argument('--accelerator-type', dest='accelerator_arch_type',
                            type=accelerator_type_arg, default='eyeriss',
                            help='Type of accelerator architecture. Default is Eyeriss-like')
    accel_args.add_argument('--deadline-constraint', type=float,
                            help='Deadline constraint for scheduling')
    accel_args.add_argument('--area-constraint', type=float, default=0.1,
                            help='Area constraint for heterogeneous accelerator. Specify '
                                 'as a percentage compared to the baseline. Default is 0.1, '
                                 'which means at most 90% of the baseline\'s area')
    return parser


def simanneal_args(parser):
    """Arguments related to Simulated Annealing
    """
    simanneal_args = parser.add_argument_group("Simulated Annealing-related arguments")
    # setting defaults according to https://github.com/perrygeo/simanneal#readme
    simanneal_args.add_argument('--simanneal-tmax', dest='simanneal_Tmax', type=float, default=25000.0,
                                help='Maximum temperature for simulated annealing (default is 25000)')
    simanneal_args.add_argument('--simanneal-tmin', dest='simanneal_Tmin', type=float, default=2.5,
                                help='Minimum temperature for simulated annealing (default is 2.5')
    simanneal_args.add_argument('--simanneal-steps', dest='simanneal_steps', type=int, default=50000,
                                help='Number of steps (iterations) for simulated annealing (default is 50000')
    simanneal_args.add_argument('--simanneal-updates', dest='simanneal_updates', type=int, default=100,
                                help='Number of updates for simulated annealing (default is 100')
    simanneal_args.add_argument('--simanneal-auto-schedule', dest='simanneal_auto_schedule', action='store_true',
                                help='Set to produce automatic scheduling of simulated annealing parameters. '
                                     'Overrides the other simulated annealing-related arguments')
    simanneal_args.add_argument('--simanneal-state-delta', dest='simanneal_state_delta', type=int, default=0.5,
                                help='Percentage [0, 1] that the state of the SA will be altered by')
    simanneal_args.add_argument('--simanneal-optimization-metric', type=metric_type_arg, default='edp',
                                help='Select the optimization metric to minimize during simmulated annealing')
    parser.add_argument('--skip-exploration', action='store_true',
                        help='Skip the simmulated annealing entirely')
    return parser


def baseline_args(parser):
    """Arguments related to the baseline accelerator. These can be used to evaluate
       any single accelerator as well
    """
    baseline_args = parser.add_argument_group('Baseline-related arguments')
    baseline_args.add_argument('--baseline-num-accelerators', type=int, default=4,
                               help='Define the number of sub-accelerators of the baseline '
                                    'accelerator. Default is 4')
    baseline_args.add_argument('--baseline-homogeneous', action='store_true',
                               help='Set to use a homogeneous accelerator baseline')
    # NOTE: These arguments have the same name ('baseline_' + arg) as the attributes of the accelerator cfg.
    #       If another type of accelerator is added, add more arguments to correspond to its new attributes.
    #       To do this, follow the same recipe: arguments have nargs='+', to be translated into lists
    baseline_args.add_argument('--baseline-pe-array-x', type=int, nargs='+', default=[],
                               help='List the PE row size per-accelerator in the heterogeneous '
                                    'case or a single value for the homogeneous one.')
    baseline_args.add_argument('--baseline-pe-array-y', type=int, nargs='+', default=[],
                               help='List the PE column size per-accelerator in the heterogeneous '
                                    'case or a single value for the homogeneous one.')
    baseline_args.add_argument('--baseline-precision', type=int, nargs='+', default=[],
                               help='List the precision per-accelerator in the heterogeneous '
                                    'case or a single value for the homogeneous one.')
    baseline_args.add_argument('--baseline-sram-size', type=int, nargs='+', default=[],
                               help='List the SRAM size per-accelerator in the heterogeneous '
                                    'case or a single value for the homogeneous one.')
    baseline_args.add_argument('--baseline-ifmap-spad-size', type=int, nargs='+', default=[],
                               help='List the size of the IFM scratchpad per-accelerator in the '
                                    'heterogeneous case or a single value for the homogeneous one.')
    baseline_args.add_argument('--baseline-weights-spad-size', type=int, nargs='+', default=[],
                               help='List the size of the weights scratchpad per-accelerator in the '
                                    'heterogeneous case or a single value for the homogeneous one.')
    baseline_args.add_argument('--baseline-psum-spad-size', type=int, nargs='+', default=[],
                               help='List the size of the partial sum scratchpad per-accelerator in the '
                                    'heterogeneous case or a single value for the homogeneous one.')
    return parser


def sota_args(parser):
    """State-of-the-art evaluation-related arguments
    """
    sota_args = parser.add_argument_group('SOTA-related arguments')
    sota_args.add_argument('--sota-load-baseline-results', metavar='PATH',
                           help='File where the evaluation results of a baseline accelerator')
    return parser


def partition_args(parser):
    """Arguments related to the partition-aware compared technique
    """
    partition_args = parser.add_argument_group('Partition-related arguments')
    partition_args.add_argument('--partition-results-path', metavar='PATH',
                                help='Directory where partitioning results from CNN-Parted are located')
    partition_args.add_argument('--partition-optimization-metric', dest='partition_optim_metric_type',
                                type=metric_type_arg, default='edp',
                                help='Select the optimization metric for the partitioning design space')
    partition_args.add_argument('--partition-optimization-iterations', dest='partition_optim_iterations',
                                type=int, default=10,
                                help='Number of iterations for the partition-aware optimization')
    partition_args.add_argument('--partition-selection-probability', type=float, default=0.6,
                                help='Probability within [0, 1] that a partitioning instance from previous '
                                     'iterations will be kept')
    partition_args.add_argument('--partition-use-constrained', action='store_true',
                                help='Set to use constrained partitions, based on the accuracy constraints')
    return parser


def check_args(args):
    """Check for logical errors in argument parsing
    """
    assert args.effective_train_size is None or 0 <= args.effective_train_size <= 1
    assert args.effective_valid_size is None or 0 <= args.effective_valid_size <= 1
    assert args.effective_train_size is None or 0 <= args.effective_train_size <= 1
    assert args.validation_split is None or 0 <= args.validation_split <= 1
    assert 0 <= args.pruning_low <= args.pruning_high <= 1
    assert 2 <= args.quant_low <= args.quant_high <= 8
    assert args.simanneal_state_delta is None or 0 <= args.simanneal_state_delta <= 1
    assert args.partition_selection_probability is None or 0 <= args.partition_selection_probability <= 1



### Enumerators for argument parsing ###

class OperationMode(Enum):
    Ours = 1
    Baseline = 2
    SOTA = 3
    Partition = 4
    GA = 5
    RandomSearch = 6

def operation_mode_arg(argstr):
    str_to_operation_mode_map = {entry.name.lower(): entry
                                 for entry in OperationMode}
    try:
        return str_to_operation_mode_map[argstr.lower()]
    except KeyError:
        raise argparse.ArgumentTypeError(f"--operation-mode argument must be one of the following: "
                                         f"{str_to_operation_mode_map.keys()}. Invalid argument {argstr}")


class OptimizerType(Enum):
    SGD = 1
    Adam = 2

def optimizer_type_arg(argstr):
    str_to_optimizer_type_map = {'sgd': OptimizerType.SGD,
                                 'adam': OptimizerType.Adam}
    try:
        return str_to_optimizer_type_map[argstr.lower()]
    except KeyError:
        raise argparse.ArgumentTypeError(f"--optimizer-type argument must be one of the following: "
                                         f"{str_to_optimizer_type_map.keys()}. Invalid argument {argstr}")


def pruning_group_type_arg(argstr):
    try:
        pruning_group_type_dict = {str(entry.name).lower(): entry for entry in PruningGroupType}
        return pruning_group_type_dict[argstr.lower()]
    except KeyError:
        raise argparse.ArgumentTypeError(f"Invalid argument {argstr} for --pruning-group-type argument")


def accelerator_type_arg(argstr):
    str_to_accelerator_type_map = {str(entry.name).lower(): entry for entry in AcceleratorType}
    try:
        return str_to_accelerator_type_map[argstr.lower()]
    except KeyError:
        raise argparse.ArgumentTypeError(f"--accelerator-type argument must be one of the following: "
                                         f"{str_to_accelerator_type_map.keys()}. Invalid argument {argstr}")


class ModelSummaryType(Enum):
    Compute = 1
    Sparsity = 2
    Dummy = 3

def model_summary_type_arg(argstr):
    str_to_summary_type_map = {'compute': ModelSummaryType.Compute,
                               'sparsity': ModelSummaryType.Sparsity,
                               '': ModelSummaryType.Dummy}
    try:
        return str_to_summary_type_map[argstr.lower()]
    except KeyError:
        raise argparse.ArgumentTypeError(f"--model-summary argument must be one of the following: "
                                         f"{str_to_summary_type_map.keys()}. Invalid argument {argstr}")


def scheduler_type_arg(argstr):
    str_to_scheduler_type_map = {'ours': SchedulerType.Ours,
                                 'random': SchedulerType.Random,
                                 'multiknapsack': SchedulerType.MultiKnapsack,
                                 'sota': SchedulerType.SOTA,
                                 'partitionaware': SchedulerType.PartitionAware}
    try:
        return str_to_scheduler_type_map[argstr.lower()]
    except KeyError:
        raise argparse.ArgumentTypeError(f"--scheduler-type argument must be one of the following: "
                                         f"{str_to_scheduler_type_map.keys()}. Invalid argument {argstr}")


def solver_type_arg(argstr):
    str_to_solver_type_map = {str(entry.name).lower(): entry for entry in SolverType}
    try:
        return str_to_solver_type_map[argstr.lower()]
    except KeyError:
        raise argparse.ArgumentTypeError(f"--solver-type argument must be one of the following: "
                                         f"{str_to_solver_type_map.keys()}. Invalid argument {argstr}")


class MetricType(Enum):
    Energy = 1
    Latency = 2
    EDP = 3
    EDP_Artificial = 4
    Area = 5


def metric_type_arg(argstr):
    str_to_metric_type_map = {'energy': MetricType.Energy,
                              'latency': MetricType.Latency,
                              'edp': MetricType.EDP,
                              'edp_artificial': MetricType.EDP_Artificial,
                              'area': MetricType.Area}

    try:
        return str_to_metric_type_map[argstr.lower()]
    except KeyError:
        raise argparse.ArgumentTypeError(f"--metric-type argument must be one of the following: "
                                         f"{str_to_metric_type_map.keys()}. Invalid argument {argstr}")

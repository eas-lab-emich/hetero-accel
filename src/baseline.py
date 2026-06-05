import logging
import os.path
from collections import OrderedDict
from src.accelerator_cfg import AcceleratorProfile
from src.optimization.optimizer import AcceleratorOptimizer
from src.args import MetricType
from src.optimization.scheduling import Scheduler

logger = logging.getLogger(__name__)


def run_baseline(args, workload, accuracy_lut):
    """Implementation of a baseline homogeneous accelerator
    """
    accel_cfg = AcceleratorProfile(args.accelerator_arch_type)

    # create the baseline multi-accelerator
    bl_evaluator = BaselineEvaluator(args, accel_cfg, workload, accuracy_lut)
    logger.info("Baseline accelerator:")
    for accelerator in bl_evaluator.state:
        logger.info(f"\t{accelerator}")

    # get the results of the evaluation (automatically saved)
    bl_evaluator.evaluate()


class BaselineEvaluator(AcceleratorOptimizer):
    """Class to create and evaluate a multi-accelerator architecture which
       serves as our baseline (inheritance mostly for the timeloop wrapper)
    """
    def __init__(self,
                 args,
                 accelerator_cfg,
                 workload,
                 accuracy_lut
        ):
        self.accelerator_cfg = accelerator_cfg
        self.workload = workload
        self.accuracy_lut = accuracy_lut
        self.num_accelerators = args.baseline_num_accelerators
        self.logdir = args.logdir
        self.metric = MetricType.EDP
        self.energy_dict = OrderedDict()
        self.latency_dict = OrderedDict()
        self.edp_dict = OrderedDict()
        self.area_dict = OrderedDict()
        self.latest_energy = self.latest_latency = self.latest_edp = self.latest_area = None
        self.latest_schedule = None

        # initialize accelerator
        self.init_accelerator(args)        
        if getattr(args, 'load_state_from', None) is not None and \
           os.path.exists(args.load_state_from):
            self.load_state(args.load_state_from)
        # initialize timeloop
        self.init_timeloop(args.layer_type_whitelist,
                           timeloop_workdir=os.path.join(self.logdir, 'timeloop_baseline'))
        # initialize scheduler
        self.scheduler = Scheduler(args.scheduler_type)
        self.solver_type = args.solver_type

    def init_accelerator(self, args):
        """Create the multi-accelerator architecture
        """
        self.state = []
        for accelerator_idx in range(args.baseline_num_accelerators):

            values = []
            # get the value for each attribute of the accelerator
            for field in self.accelerator_cfg.state._fields:
                # if the attribute is not given as an argument, keep default
                if getattr(args, 'baseline_' + field, None) is None:
                    value = getattr(self.accelerator_cfg, field)
                else:
                    # if the default value is overriden by an argument
                    value = getattr(args, 'baseline_' + field)
                    try:
                        value = value[0] if args.baseline_homogeneous else value[accelerator_idx]
                    except (IndexError, TypeError):
                        raise ValueError(f'Wrong number of arguments for --baseline-{field} ({value}) and a '
                                         f'{"homogeneous" if args.baseline_homogeneous else "heterogeneous"} '
                                         f'baseline accelerator of {args.baseline_num_accelerators} accelerators. '
                                         'NOTE: In case of a non-homogeneous baseline accelerator, '
                                         'provide as many arguments as the number of accelerators. '
                                         'A homogenous baseline would use the first provided argument.')

                values.append(value)

            # create the accelerator instance
            self.state.append(self.accelerator_cfg.state(*values))

    def evaluate(self):
        super().energy(save_best=False)

    def save_state(self, **kwargs):
        """Save the baseline measurements
        """
        savefile = os.path.join(self.logdir, 'baseline.results.pkl')
        super().save_state(save_state_to=savefile)

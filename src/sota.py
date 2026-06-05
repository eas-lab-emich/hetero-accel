import logging
import os.path
import pickle
from shutil import copy
from src.accelerator_cfg import AcceleratorProfile
from src.optimization.optimizer import AcceleratorOptimizer
from src.args import MetricType
from src.optimization.scheduling import Scheduler, SchedulerType

logger = logging.getLogger(__name__)


def run_sota(args, workload, accuracy_lut):
    """Implementation of the state-of-the-art technique described in:
       https://ieeexplore.ieee.org/document/9789220
    """
    accel_cfg = AcceleratorProfile(args.accelerator_arch_type)
    assert 'pe_array_x' in accel_cfg.state._fields and \
           'pe_array_y' in accel_cfg.state._fields, "The accelerator must have " \
            "the 'pe_array_x' and 'pe_array_y' attributes"

    # create the multi-accelerator
    sota_evaluator = SOTAEvaluator(args, accel_cfg, workload, accuracy_lut)
    sota_evaluator.save_state()
    logger.info("Accelerator architecture:")
    for accelerator in sota_evaluator.state:
        logger.info(f"\t{accelerator}")

    # get the results of the evaluation
    sota_evaluator.evaluate()


class SOTAEvaluator(AcceleratorOptimizer):
    """Class to create and evaluate a multi-accelerator architecture
       according to the state-of-the-art technique described in:
       https://ieeexplore.ieee.org/document/9789220
    """
    def __init__(self,
                 args,
                 accelerator_cfg,
                 workload,
                 accuracy_lut
        ):
        assert all(
            hasattr(accelerator_cfg.state, attribute)
            for attribute in ['pe_array_x', 'pe_array_y', 'precision']
        ), "The accelerator must have the following attributes: 'pe_array_x'," \
           "'pe_array_y' and 'precision'"
        self.accelerator_cfg = accelerator_cfg
        self.workload = workload
        self.accuracy_lut = accuracy_lut
        self.state = None
        self.solver_type = args.solver_type
        self.metric = MetricType.EDP
        self.scheduler = None
        self.logdir = args.logdir

        self.precision_options = sorted(set(
            accuracy_lut.loc[accuracy_lut['Valid'] == 1]['QuantBits']
        ), reverse=True)
        # remove 16 and 32 bits from the options
        try:
            self.precision_options.remove(32)
        except ValueError:
            pass
        try:
            self.precision_options.remove(16)
        except ValueError:
            pass
        self.num_accelerators = len(self.precision_options) 

        # initialize timeloop
        self.init_timeloop(args.layer_type_whitelist,
                           timeloop_workdir=os.path.join(self.logdir, 'timeloop_sota'))
        # load baseline results
        self.load_baseline_results(args.sota_load_baseline_results)
        # load state if provided
        if getattr(args, 'load_state_from', None) is not None and \
           os.path.exists(args.load_state_from):
            # load the initial state of the accelerator
            self.load_state(args.load_state_from)

        # discover the accelerator architecture
        self.create_accelerator(getattr(args, 'area_constraint', None))
        # restore the desired scheduler
        self.scheduler = Scheduler(args.scheduler_type)
        self.solver_type = None

    def load_baseline_results(self, load_from):
        """Load the evaluation results of the baseline accelerator
        """
        assert load_from is not None and os.path.exists(load_from), \
            f"SOTA evaluations require the use of evaluation results of the baseline accelerator ({load_from})"
        with open(load_from, 'rb') as f:
            bl_state_dict = pickle.load(f)
        copy(load_from, self.logdir)
        logger.info(f"Loaded baseline evaluation results from checkpoint ({load_from})")
        logger.debug(f"Checkpoint contents:\n{bl_state_dict}")

        # load the baseline mapping evaluations
        try:
            self.energy_dict = bl_state_dict['energy']
            self.latency_dict = bl_state_dict['latency']
            self.edp_dict = bl_state_dict['edp']
            self.area_dict = bl_state_dict['area']
        except KeyError:
            raise KeyError("The baseline state_dict should include all evaluation metrics")
        # load the baseline architecture/state as a single accelerator 
        assert len(bl_state_dict['state']) == 1, "A single baseline accelerator is needed as a baseline"
        self.baseline_state = bl_state_dict['state'][0]
        # load the baseline area of the single accelerator
        self.baseline_area = bl_state_dict['latest_area']
        assert self.baseline_area is not None or self.baseline_state != 0.0
        logger.info(f"Baseline area: {self.baseline_area:.3e}")

    def create_accelerator(self, area_constraint):
        """Create the multi-accelerator architecture, based on a given
           area constraint, by iteratively performing area evaluations to
           find the optimal number of PEs for the systolic array
        """
        assert (area_constraint is not None) and (0 < area_constraint < 1), \
            "An area constraint ([0, 1]) is needed to decide the dimensions of the PE array"
        area_min = self.baseline_area * (1 - area_constraint)
        area_max = self.baseline_area * (1 + area_constraint)
        logger.info(f"Area bounds: [{area_min:.3e}, {area_max:.3e}]")

        # For the area evaluations, we use our Scheduler. This will be reverted later
        self.scheduler = Scheduler(SchedulerType.Ours)

        state = []
        previous_pes = self.baseline_state.pe_array_x * self.baseline_state.pe_array_y
        for accelerator_idx in range(self.num_accelerators):
            # initialize the area and number of PEs
            area = 2 * area_max
            num_pes = previous_pes
            pe_x = 2 * max(self.accelerator_cfg.pe_array_x, self.accelerator_cfg.pe_array_y)

            # make sure that as the precision lowers, the number of PEs increase and
            #  at the same time the area is kept within the margins
            # NOTE: We assume a 'rectangular' PE array (i.e., pe_x == pe_y)
            
            # TODO: Here we assume that as the precision lowers, the number of PEs increase.
            #       But we change some things in Timeloop and the memory buffers. So, the area
            #       of an accelerator is not lower than the area of the acceleartor with higher precision,
            #       with equal number of PEs. We need to change this, but for now we remove the constraint
            #       on the number of PEs.
            # while not (num_pes >= previous_pes and area_min <= area <= area_max):

            while not (area_min <= area <= area_max):
                # decrease the number of PEs
                pe_x -= 1
                assert pe_x > 0, "Invalid PE elements. Relax the area " \
                                 "constraint to find a valid solution"

                # create the accelerator instance, with specific precision and PE elements
                # the rest of the attributes are the same as the baseline
                values = []
                for field in self.accelerator_cfg.state._fields:
                    if field in ('pe_array_x', 'pe_array_y'):
                        value = pe_x
                    elif field == 'precision':
                        value = self.precision_options[accelerator_idx]
                    else:
                        # value = getattr(self.accelerator_cfg, field)
                        value = getattr(self.baseline_state, field)
                    values.append(value)
                accelerator = self.accelerator_cfg.state(*values)
                num_pes = accelerator.pe_array_x * accelerator.pe_array_y
                self.state = [accelerator]

                # evaluate to get the area
                self.evaluate()
                area = self.latest_area

            # save accelerator instance
            state.append(accelerator)
            logger.info(f"=> Accepting accelerator: {accelerator}")
            # update area and number of PEs for next accelerator
            previous_pes = num_pes

        self.state = state

    def evaluate(self):
        super().energy()

    def save_state(self, **kwargs):
        """Save the baseline measurements
        """
        savefile = os.path.join(self.logdir, 'sota.results.pkl')
        super().save_state(save_state_to=savefile)

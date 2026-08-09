import itertools
import logging
import random
import os.path
import math
import pickle
import uuid
from concurrent.futures import as_completed
from types import SimpleNamespace
from collections import OrderedDict
from time import time
from shutil import copy
from simanneal import Annealer

from src.evaluation_result import EvaluationResult
from src.logging.subaccelerator_params_logger import SubacceleratorParamsLogger
from src.logging.accelerator_metric_logger import AcceleratorMetricLogger
from src.mapping.api import AcceleratorConfiguration, MappingRequest
from src.mapping.impl.async_timeloop import AsyncTimeloopMapper
from src.mapping.impl.distributed_timeloop import DistributedTimeloopMapper

from src.optimization.evaluation import SchedulePenalizer
from src.optimization.scheduling import SolverType, Scheduler
from src.utils import get_contents_table

__all__ = ['DesignSpace', 'AcceleratorOptimizer']

logger = logging.getLogger(__name__)


class DesignSpace(SimpleNamespace):
    """Wrapper for the design space of possible accelerator architectures
    """

    def __init__(self, accelerator_state_class, **kwargs):
        super().__init__(**kwargs)
        self._fields = list(self.__dict__.keys())
        self.accelerator_state_class = accelerator_state_class
        for key, value in kwargs.items():
            assert key in accelerator_state_class._fields, f'{key}'
            assert isinstance(value, (list, tuple)) and len(value) > 0

    # def __setattr__(self, key, val):
    #     """Emulating the functionality of a namedtuple"""
    #     raise AttributeError('Cannot set new values for DesignSpace')

    def sample(self, override_dict=None):
        """Get a random sample from the design space. A semi-random sample
           can be obtained be setting specific values to the override dict
        """
        override_dict = {} if override_dict is None else override_dict
        values = {
            field: random.choice(getattr(self, field))
            if field not in override_dict else override_dict[field]
            for field in self._fields
        }
        # return super().__class__(**values)
        return self.accelerator_state_class(**values)

    def extract(self, *args, **kwargs):
        """Extract a specific solution from the design space
        """
        assert len(args) == 0 or len(kwargs) == 0, "Only one type of input is supported"
        if len(args) > 0:
            values_to_get = args[0] if isinstance(args[0], list) and len(args) == 1 else args
        elif len(kwargs) > 0:
            values_to_get = [kwargs[field] for field in self._fields]
        else:
            raise ValueError("No inputs were given")

        for value, field in zip(values_to_get, self._fields):
            assert value in getattr(self, field), f"Invalid value {value} for {field}"

        return self.accelerator_state_class(*values_to_get)


class AcceleratorOptimizer(Annealer):
    """
    Implementation of the annealing optimizer for heterogeneous accelerators.


    """

    def __init__(self,
                 args,
                 num_accelerators,
                 accelerator_cfg,
                 workload,
                 accuracy_lut,
                 hw_constraints,
                 logdir
                 ):
        self.latest_penalty_details = None
        self.num_accelerators = num_accelerators
        self.accelerator_cfg = accelerator_cfg
        self.workload = workload
        self.accuracy_lut = accuracy_lut
        self.hw_constraints = hw_constraints
        self.energy_dict = OrderedDict()
        self.latency_dict = OrderedDict()
        self.edp_dict = OrderedDict()
        self.area_dict = OrderedDict()
        self.step = 0
        self.state = None
        self.evaluated_state = None
        self.latest_energy = self.latest_penalty = self.latest_latency = self.latest_edp = self.latest_area = self.latest_evaluation_result = None
        self.solver_type = SolverType.MTHGGreedyRegret
        self.logdir = logdir
        self.accelerator_metric_logger = AcceleratorMetricLogger(self.logdir)
        self.subaccelerator_params_logger = SubacceleratorParamsLogger(self.logdir)
        self.design_space = DesignSpace(accelerator_cfg.state,
                                        **accelerator_cfg.design_space)

        # initialize timeloop
        # self.accelerator_mapper = AsyncTimeloopMapper(os.path.join(self.logdir, 'mapper_workspace'))
        self.accelerator_mapper = DistributedTimeloopMapper()
        self.accelerator_mapper.start()
        # initialize scheduler
        self.scheduler = Scheduler(args.scheduler_type)
        self.schedule_penalizer = SchedulePenalizer(self.accuracy_lut)

        initial_state = self.get_initial_state()
        super().__init__(initial_state, getattr(args, 'simanneal_load_state', None))
        assert self.state == initial_state

        # load previous evaluations
        self.loaded_state = None
        if getattr(args, 'load_state_from', None) is not None and \
                os.path.exists(args.load_state_from):
            # later, use the loaded state as the first move of the annealing procedure
            self.loaded_state = self.load_state(args.load_state_from)
        self.best_state = self.copy_state(self.loaded_state)

        # get baseline measurements
        initial_metric = self.energy(initial=True)
        self.initial_metric = initial_metric
        self.initial_energy = self.latest_energy
        self.initial_latency = self.latest_latency
        self.initial_edp = self.latest_edp
        self.initial_area = self.latest_area
        logger.info("Initial results -> "
                    f"Energy={self.initial_energy:.3e}, "
                    f"Latency={self.initial_latency:.3e}, "
                    f"EDP={self.initial_edp:.3e}, "
                    f"Area={self.initial_area:.3e}")

        # setup scheduling parameters during annealing
        self.copy_strategy = 'deepcopy'
        self.state_delta = args.simanneal_state_delta
        self.state_delta = 1 if self.state_delta is None else self.state_delta
        if args is None or \
                getattr(args, 'simanneal_auto_schedule', False) or \
                any(getattr(args, arg, None) is None
                    for arg in ['simanneal_Tmax', 'simanneal_Tmin', 'simanneal_steps', 'simanneal_updates']):
            # automatic annealing schedule
            self.set_schedule(self.auto(minutes=10))
        else:
            # user-defined annealing schedule
            self.Tmax = args.simanneal_Tmax
            self.Tmin = args.simanneal_Tmin
            self.steps = args.simanneal_steps
            self.updates = args.simanneal_steps

    def close(self):
        self.accelerator_metric_logger.close()
        self.subaccelerator_params_logger.close()
        self.accelerator_mapper.stop()

    def get_initial_state(self):
        """Configure the initial state of the optimizer, w.r.t. the
           selected heterogeneity of he accelerator
        """
        initial_state = []
        # build a heterogeneous accelerator, with specific precision for each accelerator
        for accelerator_idx in range(self.num_accelerators):
            precision = self.accelerator_cfg.design_space['precision'][accelerator_idx]
            values = [
                precision if 'precision' in field.lower() else getattr(self.accelerator_cfg, field)
                for field in self.design_space._fields
            ]
            initial_state.append(self.design_space.extract(*values))

        logger.info("=> Initial state:")
        for state in initial_state:
            logger.info(f"\t{state}")
        return initial_state

    def load_state(self, load_from, save_state_to=None):
        """Load the state and its results from a given file
        """
        with open(load_from, 'rb') as f:
            state_dict = pickle.load(f)
        logger.info(f"Loaded initial state from checkpoint ({load_from})")
        logger.info(f"Checkpoint contents:\n{get_contents_table(state_dict)}\n")
        save_state_to = save_state_to or os.path.join(self.logdir, 'state.sa.pkl')
        copy(load_from, save_state_to)

        self.energy_dict.update(state_dict.get('energy', {}))
        self.latency_dict.update(state_dict.get('latency', {}))
        self.edp_dict.update(state_dict.get('edp', {}))
        self.area_dict.update(state_dict.get('area', {}))
        if getattr(self, 'hw_constraints', None) is None:
            self.hw_constraints = state_dict.get('constraints', None)
        self.latest_schedule = state_dict.get('schedule', None)
        return state_dict.get('state', None)

    def set_state(self, state):
        """Set a given state
        """
        self.state = state
        self.latest_energy = self.latest_latency = self.latest_edp = self.latest_area = None
        self.latest_schedule = None

    def save_state(self, save_state_to=None, state_to_save=None):
        """Save the state and results from fitness calculation
        """
        state_dict = {'energy': self.energy_dict,
                      'latency': self.latency_dict,
                      'edp': self.edp_dict,
                      'area': self.area_dict,
                      'schedule': self.latest_schedule,
                      'state': self.state if state_to_save is None else state_to_save,
                      'constraints': getattr(self, 'hw_constraints', None),
                      'latest_energy': self.latest_energy,
                      'latest_latency': self.latest_latency,
                      'latest_edp': self.latest_edp,
                      'latest_area': self.latest_area}

        save_state_to = save_state_to or os.path.join(self.logdir, 'state.sa.pkl')
        with open(save_state_to, 'wb') as f:
            pickle.dump(state_dict, f)
        logger.info(f"Saved state in: {save_state_to}")

    def run(self):
        """Run Simulated Annealing
        """
        self.anneal()
        # save the best state
        self.save_state(os.path.join(self.logdir, 'best_state.sa.pkl'))

    def update(self, step, T, E, acceptance, improvement):
        """Internal update for the status of the simulated annealing
        """

        def time_string(seconds):
            """Returns time in seconds as a string formatted HHHH:MM:SS."""
            s = int(round(seconds))  # round to nearest second
            h, s = divmod(s, 3600)  # get hours and remainder
            m, s = divmod(s, 60)  # split remainder into minutes and seconds
            return '%4i:%02i:%02i' % (h, m, s)

        if step != 0:
            elapsed = time() - self.start
            remain = (self.steps - step) * (elapsed / step)
            logger.info(f"Update --> temperature={T:8.3e}, energy_metric={E:8.3e}, "
                        f"accept={acceptance:6.2%}, improvement={improvement:6.2%},"
                        f"time_elapsed={time_string(elapsed)}, time_remaining={time_string(remain)}")

        evaluation_result = self.latest_evaluation_result if self.latest_evaluation_result else EvaluationResult.UNKNOWN

        self.accelerator_metric_logger.log(
            iteration=self.step,
            is_improved=improvement,
            sim_temperature=T,
            energy=self.latest_energy,
            latency=self.latest_latency,
            edp=self.latest_edp,
            penalty=self.latest_penalty,
            area=self.latest_area,
            scheduled=self.latest_schedule,
            penalty_details=self.latest_penalty_details,
            evaluation_result=evaluation_result
        )
        state = self.evaluated_state if self.evaluated_state else self.state
        for accl in state:
            self.subaccelerator_params_logger.log(
                iteration=self.step,
                is_improved=improvement,
                pe_array_x=accl.pe_array_x,
                pe_array_y=accl.pe_array_y,
                precision=accl.precision,
                sram_size=accl.sram_size,
                ifmap_spad_size=accl.ifmap_spad_size,
                weights_spad_size=accl.weights_spad_size,
                psum_spad_size=accl.psum_spad_size,
                evaluation_result=evaluation_result
            )

    def move(self):
        """Alter the current state
        """
        self.step += 1
        # If this is the first evaluation, use the loaded state as the first move of the annealing procedure
        if self.step == 1 and getattr(self, 'loaded_state', None) is not None:
            new_state = self.loaded_state
        # Otherwise, generate a semi-random accelerator with static precision
        # NOTE: This works for accelerators with the attribute 'precision'
        else:
            new_state = self.state

            # change only a number of features from the previous state, according to delta
            fields_to_change = random.choices(self.design_space._fields,
                                              k=math.ceil(self.state_delta * len(self.design_space._fields)))

            # make sure the new state is different
            while new_state == self.state:
                new_state = []
                for accelerator_idx in range(self.num_accelerators):
                    # generate a random architecture from the design space
                    new_accelerator = self.design_space.sample(
                        override_dict={'precision': self.accelerator_cfg.design_space['precision'][accelerator_idx]}
                    )
                    # set the accelerator values as a combination from the new and old ones (previous state)
                    values = [
                        getattr(new_accelerator, field) if field in fields_to_change
                        else getattr(self.state[accelerator_idx], field)
                        for field in self.design_space._fields
                    ]
                    new_accelerator = self.design_space.extract(*values)
                    new_state.append(new_accelerator)

        self.state = self.evaluated_state = new_state
        logger.info(f"=> Move #{int(self.step)} taken. New state:")
        for state in new_state:
            logger.info(f"\t{state}")

    def energy(self, initial=False, save_best=True):
        """Wrapper function for estimating the SA energy metric
        """
        start = time()
        logger.info(f"=> Beginning {'initial ' if initial else ''}state evaluation")
        self.latest_evaluation_result = self._evaluation()
        logger.info(f"Completed state evaluation in {time() - start:.3e}s")

        # save the results
        self.save_state()
        if save_best:
            # save the best state
            self.save_state(save_state_to=os.path.join(self.logdir, 'best_state.sa.pkl'),
                            state_to_save=self.best_state)

        if self.latest_schedule is not None:
            logger.info(f"Evaluation results:\n"
                        f"\tEnergy={self.latest_energy:.3e}\n"
                        f"\tLatency={self.latest_latency:.3e}\n"
                        f"\tEDP={self.latest_edp:.3e}\n"
                        f"\tArea={self.latest_area:.3e}")
        elif initial:
            raise ValueError("Initial metric calculation cannot be invalid")

        logger.info("*--------------*")

        if self.latest_edp is None:
            self.latest_penalty = math.inf
            self.latest_penalty_details = None
            return self.latest_penalty

        self.latest_penalty_details = self.schedule_penalizer.penalize(self.latest_schedule)
        self.latest_penalty = self.latest_penalty_details.total_penalty
        return self.latest_edp + self.latest_penalty

    def _evaluation(self) -> EvaluationResult:
        """Evaluate the fitness of the current state
           Returns a boolean variable, indicating a successful/unsuccessful evaluation
        """

        def violated_deadline(schedule):
            deadline = getattr(getattr(self, 'hw_constraints', None), 'deadline', None)
            # the constraint is not violated if a deadline is not given
            return deadline is not None and \
                any(end_timestamp >= deadline for end_timestamp in schedule.end_timestamp.values())

        def violated_area_constraint(area):
            return not (
                    getattr(self, 'initial_area', None) is None or
                    getattr(self, 'hw_constraints', None) is None or
                    getattr(self.hw_constraints, 'area', None) is None or
                    area < self.initial_area * (1 + self.hw_constraints.area)
            )

        def violated_accuracy_constraint(arch, precision):
            try:
                return self.accuracy_lut.loc[
                    (self.accuracy_lut['Network'] == arch) &
                    (self.accuracy_lut['QuantBits'] == precision)
                    ]['Valid'].iloc[0] == 0
            except IndexError:
                return True

        # metrics to be accumulated
        energy_dict = {}
        latency_dict = {}
        edp_dict = {}

        deferred_mappings = {}
        for accelerator, (dnn_name, layers) in itertools.product(self.state, self.workload.items()):
            logger.info(f"\t\tQueuing evaluation on aceelerator={accelerator}, dnn={dnn_name}")

            # check if this evaluation was executed before
            if (dnn_name, accelerator) in self.energy_dict:
                # NOTE: This is not as accurate as accumulate layer-wise EDP results,
                #       but it is a good approximation for not re-running the simulation
                if (dnn_name, accelerator) not in self.edp_dict:
                    self.edp_dict[(dnn_name, accelerator)] = self.energy_dict[(dnn_name, accelerator)] * \
                                                             self.latency_dict[
                                                                 (dnn_name, accelerator)]
                logger.info(f"\t\tSkipping evaluation: already estimated")
                continue

            # check accuracy constraint
            if violated_accuracy_constraint(dnn_name, accelerator.precision):
                logger.info(f"\t\tSkipping evaluation: accuracy violation")
                # Invalid scheduling mappings are marked with negative weight (latency)
                self.energy_dict[(dnn_name, accelerator)] = -1
                self.latency_dict[(dnn_name, accelerator)] = -1
                self.edp_dict[(dnn_name, accelerator)] = -1
                continue

            energy_dict[(dnn_name, accelerator)] = 0
            latency_dict[(dnn_name, accelerator)] = 0
            edp_dict[(dnn_name, accelerator)] = 0
            # iterate over each timeloop problem (layer) of the DNN
            for layer in layers:
                accelerator_config = AcceleratorConfiguration(
                    pe_array_x=accelerator.pe_array_x,
                    pe_array_y=accelerator.pe_array_y,
                    precision=accelerator.precision,
                    sram_size=accelerator.sram_size,
                    ifmap_spad_size=accelerator.ifmap_spad_size,
                    weights_spad_size=accelerator.weights_spad_size,
                    psum_spad_size=accelerator.psum_spad_size
                )
                request = MappingRequest(
                    id=uuid.uuid4(),
                    accelerator_config=accelerator_config,
                    problem=layer
                )
                deferred_mappings[self.accelerator_mapper.map(request)] = (dnn_name, accelerator)

        for deferred in as_completed(deferred_mappings):
            metric_key = deferred_mappings[deferred]
            try:
                results = deferred.result()
                energy_dict[metric_key] += results.energy
                latency_dict[metric_key] += results.cycles
                edp_dict[metric_key] += results.edp

                # store the accelerator area from the results of the last mapping
                # all layers with the same accelerator should give the same area
                _, accelerator = metric_key
                if accelerator not in self.area_dict:
                    self.area_dict[accelerator] = getattr(results, 'area', None)
                    logger.debug(f"\tSet accelerator area: {self.area_dict[accelerator]}")
            except FileNotFoundError: # TODO add specific exception
                self.latest_schedule = self.latest_energy = self.latest_latency = None
                logger.error(f"Invalid timeloop/accelergy simulation for {metric_key}")
                return EvaluationResult.INVALID_SIMULATION # TODO retry logic? Add cancel method to mapper?

        for dnn_name, accelerator in energy_dict.keys():
            logger.debug(f"\t\tEvaluation results for {dnn_name} on {accelerator}:\n"
                         f"\t\t\tEnergy={energy_dict[(dnn_name, accelerator)]:.3e}\n"
                         f"\t\t\tLatency={latency_dict[(dnn_name, accelerator)]:.3e}\n"
                         f"\t\t\tEDP={edp_dict[(dnn_name, accelerator)]:.3e}")

        # update stored metrics with executed evaluations
        self.energy_dict.update(energy_dict)
        self.latency_dict.update(latency_dict)
        self.edp_dict.update(edp_dict)

        logger.info("Completed mapping evaluation")

        # check the area constraint
        self.latest_area = sum([self.area_dict[accelerator] for accelerator in self.state])
        if violated_area_constraint(self.latest_area):
            self.latest_schedule = self.latest_energy = self.latest_latency = self.latest_edp = None
            logger.info("Violated area constraint")
            return EvaluationResult.AREA_CONSTRAINT

        # perform the scheduling and get a concrete DNN-to-accelerator mapping
        start = time()
        # TODO: Consider the metrics used for weight_dict and cost_dict
        schedule = self.scheduler.run(items=list(self.workload.keys()),
                                      bins=self.state,
                                      cost_dict=self.energy_dict,
                                      weight_dict=self.latency_dict,
                                      solver_type=self.solver_type)
        # save schedule of latest move
        self.latest_schedule = schedule
        logger.debug(f"Schedule created in {time() - start:.3e}s")

        if schedule is None:
            # return in case of invalid schedule
            self.latest_energy = self.latest_latency = self.latest_edp = None
            logger.info(f"Could not find valid schedule")
            return EvaluationResult.SCHEDULE_CONSTRAINT

        # get results for energy and latency based on the final schedule
        self.latest_energy = sum([
            self.energy_dict[(entry.tag, entry.bin)] for entry in schedule.entries
        ])
        self.latest_latency = max([
            sum([
                self.latency_dict[(entry.tag, entry.bin)] for entry in entries
            ]) for bin, entries in schedule.as_dict(main_key='bin').items()
        ])
        self.latest_edp = self.latest_energy * self.latest_latency

        # log the results of the scheduling
        schedule_str = '\n\t'.join([f'{entry.tag} -> {entry.bin}' for entry in schedule.entries])
        logger.info(f"Scheduler results:\n\t{schedule_str}")

        # check deadline constraint
        if violated_deadline(schedule):
            logger.info(f"Violated deadline constraint")
            return EvaluationResult.DEADLINE_CONSTRAINT

        return EvaluationResult.SUCCESS

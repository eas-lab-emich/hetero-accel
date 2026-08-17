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
from src.mapping.impl.distributed_timeloop import DistributedTimeloopMapper

from src.optimization.evaluation import SchedulePenalizer, StepResult
from src.optimization.scheduling import SolverType, Scheduler
from src.utils import get_contents_table

__all__ = ['DesignSpace', 'AcceleratorOptimizer']

logger = logging.getLogger(__name__)


class DesignSpace(SimpleNamespace):
    """Wrapper for the design space of possible accelerator architectures
    """

    def __init__(self, accelerator_state_class, **kwargs):
        super().__init__(**kwargs)
        self._fields = ['pe_array_x',
                        'pe_array_y',
                        'sram_size',
                        'ifmap_spad_size',
                        'weights_spad_size',
                        'psum_spad_size']
        self.accelerator_state_class = accelerator_state_class
        for key, value in kwargs.items():
            assert key in accelerator_state_class._fields, f'{key}'
            assert isinstance(value, (list, tuple)) and len(value) > 0

    def neighborhood_move(self, accelerator):
        fields_to_change = random.sample(self._fields, k=2)
        accel_dict = accelerator._asdict()
        for field_name in fields_to_change:
            param_val = accel_dict[field_name]
            possible_vals = getattr(self, field_name)
            current_val_idx = possible_vals.index(param_val)
            if current_val_idx == 0:
                idx = 1
            elif current_val_idx == len(possible_vals) - 1:
                idx = current_val_idx - 1
            else:
                idx = current_val_idx + random.choice([-1, 1])
            accel_dict[field_name] = possible_vals[idx]

        return self.accelerator_state_class(**accel_dict)


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
        self.under_eval_state : list | None = None
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
        super().__init__(initial_state)
        assert self.state == initial_state

        # get baseline measurements
        self.under_eval_state = self.state
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
           selected heterogeneity of the accelerator
        """
        initial_state = []
        # build a heterogeneous accelerator, with specific precision for each accelerator
        for accelerator_idx in range(self.num_accelerators):
            values = {
                field: getattr(self.accelerator_cfg, field) for field in self.design_space._fields
            }
            values['precision'] = self.accelerator_cfg.design_space['precision'][accelerator_idx]
            initial_state.append(self.design_space.accelerator_state_class(**values))

        logger.info("=> Initial state:")
        for state in initial_state:
            logger.info(f"\t{state}")
        return initial_state


    def set_state(self, state):
        """Set a given state
        """
        self.state = state
        self.latest_energy = self.latest_latency = self.latest_edp = self.latest_area = None
        self.latest_schedule = None


    def run(self):
        """Run Simulated Annealing
        """
        self.anneal()

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

        self.schedule_penalizer.ingest_results(
            StepResult(edp=self.latest_edp, penalty=self.latest_penalty, accepted=acceptance, improved=improvement,
                       schedule=self.latest_schedule, evaluation_result=evaluation_result))

        self.accelerator_metric_logger.log(
            iteration=self.step,
            is_improved=improvement,
            is_accepted=acceptance,
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
        for accl in self.under_eval_state or []:
            self.subaccelerator_params_logger.log(
                iteration=self.step,
                is_improved=improvement,
                is_accepted=acceptance,
                pe_array_x=accl.pe_array_x,
                pe_array_y=accl.pe_array_y,
                precision=accl.precision,
                sram_size=accl.sram_size,
                ifmap_spad_size=accl.ifmap_spad_size,
                weights_spad_size=accl.weights_spad_size,
                psum_spad_size=accl.psum_spad_size,
                evaluation_result=evaluation_result
            )
        self.under_eval_state = None

    def move(self):
        """Alter the current state
        """
        self.step += 1
        # randomly step in the design space for the given accelerator
        self.under_eval_state = self.state = [self.design_space.neighborhood_move(accelerator)
                                 for accelerator in self.state]

        logger.info(f"=> Move #{int(self.step)} taken. New state:")
        for state in self.under_eval_state:
            logger.info(f"\t{state}")

    def energy(self, initial=False):
        """Wrapper function for estimating the SA energy metric
        """
        start = time()
        logger.info(f"=> Beginning {'initial ' if initial else ''}state evaluation")
        self.latest_evaluation_result = self._evaluation()
        logger.info(f"Completed state evaluation in {time() - start:.3e}s")

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
        for accelerator, (dnn_name, layers) in itertools.product(self.under_eval_state, self.workload.items()):
            logger.info(f"\t\tQueuing evaluation on accelerator={accelerator}, dnn={dnn_name}")

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

        deferred_count = len(deferred_mappings.values())
        completed = 0
        for deferred in as_completed(deferred_mappings):
            metric_key = deferred_mappings[deferred]
            try:
                results = deferred.result()
                energy_dict[metric_key] += results.energy
                latency_dict[metric_key] += results.cycles
                edp_dict[metric_key] += results.edp
                completed += 1

                logger.info(
                    f"Received results for (accel,dnn)={metric_key}, id={results.id}, mappings_complete={completed}/{deferred_count}")

                # store the accelerator area from the results of the last mapping
                # all layers with the same accelerator should give the same area
                _, accelerator = metric_key
                if accelerator not in self.area_dict:
                    self.area_dict[accelerator] = getattr(results, 'area', None)
                    logger.info(f"\tSet accelerator area: {self.area_dict[accelerator]}")
            except FileNotFoundError:  # TODO add specific exception
                self.latest_schedule = self.latest_energy = self.latest_latency = None
                logger.error(f"Invalid timeloop/accelergy simulation for {metric_key}")
                return EvaluationResult.INVALID_SIMULATION  # TODO retry logic? Add cancel method to mapper?

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

        self.latest_area = sum([self.area_dict[accelerator] for accelerator in self.under_eval_state])
        # check the area constraint #TODO do we keep this long term or remove? It's a completely different objective...
        # if violated_area_constraint(self.latest_area):
        #     self.latest_schedule = self.latest_energy = self.latest_latency = self.latest_edp = None
        #     logger.info("Violated area constraint")
        #     return EvaluationResult.AREA_CONSTRAINT

        # perform the scheduling and get a concrete DNN-to-accelerator mapping
        start = time()
        # TODO: Consider the metrics used for weight_dict and cost_dict
        schedule = self.scheduler.run(items=list(self.workload.keys()),
                                      bins=self.under_eval_state,
                                      cost_dict=self.energy_dict,
                                      weight_dict=self.latency_dict,
                                      solver_type=self.solver_type)
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

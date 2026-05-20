import logging
import os.path
import random
import re
import subprocess
import numpy as np
from collections import OrderedDict, namedtuple, deque
from enum import Enum
from time import time
# matplotlib imports
import matplotlib as mpl

mpl.rcParams.update(mpl.rcParamsDefault)
from matplotlib import pyplot as plt
from matplotlib import rcParams
from src import project_dir


__all__ = ['SchedulerType', 'ScheduleEntry', 'Schedule',
           'SolverType', 'solver_args_dict'
           'Scheduler']

logger = logging.getLogger(__name__)



class SchedulerType(Enum):
    Ours = 1
    Random = 2
    MultiKnapsack = 3
    SOTA = 4
    PartitionAware = 5


ScheduleEntry = namedtuple('ScheduleEntry',
                           ['start', 'end', 'bin', 'tag'])


class Schedule:
    def __init__(self, bins):
        self.bins = bins
        # the ending timestamp of the last inserted item for each bin
        self.end_timestamp = {bin: 0 for bin in bins}
        self.entries = []
        self.assigned = {}

    def add(self, item, to_bin, duration, start=None):
        """Add entry to the schedule
        """
        assert item not in self.assigned, f"Item {item} is already assigned to bin {self.assigned[item]}"
        if start is None:
            start = self.end_timestamp[to_bin]
        start = max(start, self.end_timestamp[to_bin])

        end = start + duration
        self.end_timestamp[to_bin] = end
        self.entries.append(ScheduleEntry(start=start,
                                          end=end,
                                          bin=to_bin,
                                          tag=item))
        self.assigned[item] = to_bin

    def as_dict(self, main_key='bin'):
        assert main_key in ScheduleEntry._fields, f"Select as main_key one of the followning {ScheduleEntry._fields}"
        keys = sorted({getattr(entry, main_key) for entry in self.entries})
        return OrderedDict([
            (key, [entry for entry in self.entries if getattr(entry, main_key) == key])
            for key in keys
        ])

    def visualize(self, savefile=None):
        """Visualize the current schedule
        """
        # rcParams.update({
        #     # 'text.latex.preamble': r"\usepackage{lmodern}",
        #     'font.size': "7",    
        #     "text.usetex": True,
        #     "font.family": "lmodern",
        #     "font.serif": ["Computer Modern Roman"]
        # })

        # organize schedule in batches: each batch contains one item per bin, and
        # there are as many batches as max number of items in one bin
        bin_dict = self.as_dict('bin')
        batch_entries = []
        batch_index = 0
        last_added = True
        while last_added:
            this_batch = []
            last_added = False
            for bin, bin_entries in bin_dict.items():
                try:
                    entry_for_this_batch = bin_entries[batch_index]
                    this_batch.append(entry_for_this_batch)
                    last_added = True
                except IndexError:
                    this_batch.append(ScheduleEntry(0, 0, bin, tag=''))

            if last_added:
                batch_entries.append(this_batch)
                batch_index += 1

        # build schedule figure
        fig, ax = plt.subplots(figsize=(3.3, 1.8))
        left = np.zeros(len(bin_dict))
        for batch_entry in batch_entries:
            widths = [entry.end - entry.start for entry in batch_entry]
            y = [repr(bin) for bin in bin_dict.keys()]

            bar_container = ax.barh(y=y,
                                    width=widths,
                                    height=0.9,
                                    align='center',
                                    left=left,
                                    joinstyle='round',
                                    capstyle='round',
                                    fill=False,
                                    linewidth=1.0,
                                    edgecolor='black',)
            ax.bar_label(bar_container,
                         labels=[f'{entry.tag}\n{entry.start}->{entry.end}'
                                  for entry in batch_entry],
                         label_type='center',
                         fontsize='x-small',
                         color='black')
            left += widths

        # save schedule figure
        if savefile is None:
            logdir = logging.getLogger().logdir
            savefile = os.path.join(logdir, 'latest_schedule.png')
        plt.savefig(savefile, bbox_inches='tight', pad_inches=0)
        logger.info(f"Schedule visualization was saved in {savefile}")


# TODO: Study the possible options for the solver

class SolverType(Enum):
    Greedy = 1
    GreedyRegret = 2
    MTHGGreedy = 3
    MTHGGreedyRegret = 4
    LocalSearch = 5
    ColumnGenerationGreedy = 6
    ColumnGenerationLimitedDiscrepency = 7
    Random = 8
    LocalSolver = 9
    MixedIntegerLinearCBC = 10
    MixedIntegerLinearCPLEX = 11
    MixedIntegerLineargGurobi = 12
    MixedIntegerLinearKnitro = 13
    ConstraintGecode = 14
    ConstraintCPLEX = 15

solver_args_dict = {
    SolverType.Greedy: '\"greedy -f wij\"',
    SolverType.GreedyRegret: '\"greedy-regret -f wij\"',
    SolverType.MTHGGreedy: '\"mthg -f wij\"',
    SolverType.MTHGGreedyRegret: '\"mthg-regret -f wij\"',
    SolverType.LocalSearch: '\"local-search --threads 4\"',
    SolverType.ColumnGenerationGreedy: '\"column-generation-heuristic-greedy --linear-programming-solver cplex\"',
    SolverType.ColumnGenerationLimitedDiscrepency: '\"column-generation-heuristic-limited-discrepancy-search --linear-programming-solver cplex\"',
    SolverType.Random: 'random',
    SolverType.LocalSolver: 'localsolver',
    SolverType.MixedIntegerLinearCBC: 'milp-cbc',
    SolverType.MixedIntegerLinearCPLEX: 'milp-cplex',
    SolverType.MixedIntegerLineargGurobi: 'milp-gurobi',
    SolverType.MixedIntegerLinearKnitro: 'milp-knitro',
    SolverType.ConstraintGecode: 'constraint-programming-gecode',
    SolverType.ConstraintCPLEX: 'constraint-programming-cplex',
}


class Scheduler:
    """Implementations for the scheduler"""
    def __init__(self, scheduler_type=SchedulerType.Ours):
        self.type = scheduler_type
        self.__run_f = {
            SchedulerType.Ours: self._run_ours,
            SchedulerType.Random: self._run_random_scheduling,
            SchedulerType.MultiKnapsack: self._run_with_identical_bins,
            SchedulerType.SOTA: self._run_sota,
            SchedulerType.PartitionAware: self._run_partition_aware
        }.get(scheduler_type)

    def run(self, *args, **kwargs):
        """Wrapper over the main scheduling function, may be needed
        """
        if self.type == SchedulerType.Ours:
            if 'solver_type' not in kwargs:
                kwargs['solver_type'] = SolverType.MTHGGreedy
        return self.__run_f(*args, **kwargs)

    def _run_ours(self, items, bins, cost_dict, weight_dict,
                  max_capacity=None, solver_type=SolverType.MTHGGreedy, use_value=False):
        """Static scheduling with heterogeneous bins w.r.t. cost/value and weight per item,
           i.e., cost_dict and weight_dict have different values for different bins.
           This is an implementation of the generalized assignment problem. We use the solver
           algorithmic options found in: https://github.com/fontanf/generalizedassignmentsolver
        """
        # NOTE: Other options: Dynamic programming/Branch-and-bound?
        def write_input_file(infile):
            """Create the input file for the solver
            """
            with open(infile, 'w') as f:
                # write the number of bins (agents) and items (tasks)
                f.write(f'{len(bins)} {len(items)}\n')

                # write the cost or value/profit
                for bin in bins:
                    costs = []
                    for item in items:
                        # in the case of an invalid mapping, the cost/value/profit does not matter
                        if weight_dict[(item, bin)] < 0:
                            costs.append(0)
                        # in the case of value/profit
                        elif use_value:
                            costs.append(
                                max([value for key, value in cost_dict.items() if item in key]) - cost_dict[(item, bin)]
                            )
                        # in the case of cost
                        else:
                            costs.append(
                                cost_dict[(item, bin)]
                            )
                    # TODO: Check if casting to int here is a problem. Floats do not work for this solver
                    f.write(' '.join([str(int(cost)) for cost in costs]) + '\n')

                # write the weights
                for bin_idx, bin in enumerate(bins):
                    # negative weights are marked as invalid mappings, and are assigned higher than the
                    # maximum capacity of the bin, to make that mapping impossible
                    weights = [weight_dict[(item, bin)] if weight_dict[(item, bin)] > 0 else capacities[bin_idx] + 1
                               for item in items]
                    f.write(' '.join([str(int(weight)) for weight in weights]) + '\n')

                # write the maximum weight (capacity) of each bin (agent)
                f.write(' '.join([str(int(capacity)) for capacity in capacities]))

        weight_latencies = ", ".join([f"(network={key[0]}, accel={key[1].precision} bits, latency={value})" for key, value in weight_dict.items()])
        schedule = Schedule(bins)

        if max_capacity:
            capacities = [max_capacity for _ in bins]
        else:
            alpha = 1.82
            # alpha = 5.82
            latency_per_bin = [
                sum(weight_dict[(item, bin)] for item in items
                    if weight_dict[(item, bin)] > 0)
                for bin in bins
            ]
            speeds = [1 / latency if latency > 0 else 0
                      for latency in latency_per_bin]
            speed_total = sum(speeds)
            normalized_speeds = [speed / speed_total for speed in speeds]
            latencies_per_item = [
                [weight_dict[(item, bin)] for bin in bins if weight_dict[(item, bin)] > 0]
                for item in items
            ]
            best_case_latency = sum(
                min(latencies) for latencies in latencies_per_item if latencies
            )
            capacities = [int(alpha * n_speed * best_case_latency) for n_speed in normalized_speeds]

        solver_dir = os.path.join(project_dir, 'generalizedassignmentsolver')
        logdir = logging.getLogger().logdir
        resdir = os.path.join(logdir, 'scheduler_solver')
        os.makedirs(resdir, exist_ok=True)
        infile = os.path.join(resdir, 'inputs')
        outfile = os.path.join(resdir, 'output')
        solution_file = os.path.join(resdir, 'solution')
        logfile = os.path.join(resdir, 'log')

        # write cost/profit and weight to file
        write_input_file(infile)
        # construct command for solver
        solver_args = solver_args_dict.get(solver_type)
        command = f"cd {solver_dir} && " \
                  f"./bazel-bin/generalizedassignmentsolver/main -v 3 " \
                  f"-a {solver_args} " \
                  f"-i {infile} -o {outfile} -c {solution_file} " \
                  f"2>&1 | tee {logfile}"
        logger.debug(f"GeneralizedAssignmentSolver command:\n{command}")

        # run command 
        start = time()
        p = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.debug(f"Executed solver command in {time() - start:.3e} with exitcode: {p.returncode}")

        # check if all items are assigned in the solution
        if re.search('Number of items.*[(](\d+)%', p.stdout).group(1) != '100':
            return

        # get assignment via the stdout of the command
        assigns = re.search('Item\s+Agent\n.*?\n(.*)', p.stdout, re.DOTALL).group(1)
        assigns = [re.sub('\s+', ' ', assign.strip()).split(' ') for assign in assigns.split('\n')[:-1]]
        # complete the item-to-bin assignment and define the schedule
        for item_idx, bin_idx in assigns:
            item = items[int(item_idx)]
            bin = bins[int(bin_idx)]
            schedule.add(item, bin, weight_dict[(item, bin)])
        return schedule

    def _run_random_scheduling(self, items, bins, cost_dict, weight_dict, **kwargs):
        """Random assignment of items to bins
        """
        schedule = Schedule(bins)
        for item in items:
            bin_sel = random.choice(bins)
            while (item, bin_sel) not in weight_dict:
                bin_sel = random.choice(bins)
            schedule.add(item, bin_sel, weight_dict[(item, bin_sel)])
        return schedule

    def _run_with_identical_bins(self, *args, **kwargs):
        """Static scheduling with homogeneous bins w.r.t. of values and weights per item
           This is an implementation of the Multiple Knapsack problem
        """
        # TODO: This is a temporary implementation for baseline. We use the same GAP solver as ours,
        #       but expect to have the same cost/weight values for all bins
        return self._run_ours(*args, **kwargs)

    def _run_sota(self, items, bins, cost_dict, weight_dict, **kwargs):
        """Execute the scheduling described in: https://ieeexplore.ieee.org/document/9789220.
           We assume a queue of (randomly) shuffled items that are mapped to bins in the order
           of the queue.
        """
        schedule = Schedule(bins)

        random.shuffle(items)
        queue = deque(items, maxlen=len(items))
        ready_list = {bin: [] for bin in bins}
        response_time = {item: {bin: -1 for bin in bins} for item in items}

        # populate the ready list
        while len(queue) > 0:
            next_item = queue.pop()
            # gather the available bins for that item: only those with positive weight
            #  (i.e., valid mapping) are considered
            available_bins = [bin for bin in bins if (next_item, bin) in weight_dict and
                                                     weight_dict[(next_item, bin)] > 0]

            for available_bin in available_bins:
                # TODO: Here we do design-time scheduling, so the agent (bin) is not
                #       currently executing any tasks. Rather, we assign tasks to its
                #       ready list so they can be executed in order
                bin_workload = 0
                weight_ready_list = sum([weight_dict[(item, available_bin)] for item in ready_list[available_bin]])
                weight_next_item = weight_dict[(next_item, available_bin)]
                assert weight_next_item >= 0

                # the response time is the sum of the current workload of the agent (bin),
                #  the total weight of the items on the bin's ready list and the weight of 
                #  the current item to-be-assigned
                response_time[next_item][available_bin] = bin_workload + weight_ready_list + weight_next_item

            # the item/task is assigned to the bin with the minimum response time
            selected_bin = min({bin: rtime for bin, rtime in response_time[next_item].items() if rtime >= 0},
                               key=response_time[next_item].get)
            # assign the item to the ready list of the bin
            ready_list[selected_bin].append(next_item)
            # add to the schedule
            schedule.add(next_item, selected_bin, weight_dict[(next_item, selected_bin)])

        return schedule

    def _run_partition_aware(self, partitions, bins):
        """Greedy scheduling algorithm for partitioned tasks
        """
        schedule = Schedule(bins)

        # prioritize partitions based on their overall EDP (execution + transfer)
        sorted_partitions = sorted(partitions,
                                   reverse=True,
                                   key=lambda entry: (entry.metrics.overall_latency + entry.metrics.overall_link_latency) *
                                                     (entry.metrics.overall_energy + entry.metrics.overall_link_energy))

        # bookkeeping dictionaries
        track_bin_execution = schedule.end_timestamp
        subpartition_to_be_executed = {partition.tag: 0 for partition in partitions}
        track_partition_execution = {partition.tag: 0.0 for partition in partitions}

        # execute until all partitions are fully assigned to bins
        while not all(subpartition_to_be_executed[partition.tag] == len(partition.assignment) for partition in partitions):
            assigned = False

            # order bins by their execution time up to this point
            ordered_bins = sorted(bins, key=lambda bin: track_bin_execution[bin])
            for selected_bin in ordered_bins:
                if assigned: continue

                # assign the first sub-partition available, in order of partition priority
                for partition in sorted_partitions:
                    if assigned: continue

                    subpartition_index = subpartition_to_be_executed[partition.tag]
                    # check if that sub-partition was selected for the specific bin
                    if subpartition_index < len(partition.assignment) and \
                        partition.assignment[subpartition_index] == selected_bin:

                        # duration is the sum of the transfer latency from the previous subpartition-bin assignment
                        #  and the execution time/latency for the selected one
                        duration = partition.metrics.partition_latency[subpartition_index]
                        if subpartition_index != 0:
                            duration += partition.metrics.partition_link_latency[subpartition_index - 1]

                        # schedule subpartition
                        schedule.add(item=partition.tag + f'_{subpartition_index}',
                                     to_bin=selected_bin,
                                     duration=duration,
                                     # make sure that the subpartition would be executed 
                                     # not before the previous one from the same partition
                                     # has finished
                                     start=track_partition_execution[partition.tag])

                        # update the partition execution time and index
                        track_partition_execution[partition.tag] = track_bin_execution[selected_bin]
                        subpartition_to_be_executed[partition.tag] += 1
                        assigned = True

            # this is an exhaustive type of search, so at least one assignment has to be found
            assert assigned

        return schedule


if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)
    from collections import OrderedDict
    from src.accelerator_cfg import EyerissAcceleratorState

    solver_type = SolverType.MTHGGreedyRegret
    items = ['vgg11', 'resnet18', 'vgg16']
    bins = [EyerissAcceleratorState(pe_array_x=14, pe_array_y=12, precision=6, sram_size=108000, ifmap_spad_size=24,
                                    weights_spad_size=448, psum_spad_size=48),
            EyerissAcceleratorState(pe_array_x=14, pe_array_y=12, precision=7, sram_size=108000, ifmap_spad_size=24,
                                    weights_spad_size=448, psum_spad_size=48),
            EyerissAcceleratorState(pe_array_x=14, pe_array_y=12, precision=8, sram_size=108000, ifmap_spad_size=24,
                                    weights_spad_size=448, psum_spad_size=48)]

    cost_dict = OrderedDict([(('vgg11', EyerissAcceleratorState(pe_array_x=14, pe_array_y=12, precision=6, sram_size=108000, ifmap_spad_size=24, weights_spad_size=448, psum_spad_size=48)), 3718503.03), (('resnet18', EyerissAcceleratorState(pe_array_x=14, pe_array_y=12, precision=6, sram_size=108000, ifmap_spad_size=24, weights_spad_size=448, psum_spad_size=48)), 794431.9399999998), (('vgg16', EyerissAcceleratorState(pe_array_x=14, pe_array_y=12, precision=6, sram_size=108000, ifmap_spad_size=24, weights_spad_size=448, psum_spad_size=48)), 7034727.320000001), (('vgg11', EyerissAcceleratorState(pe_array_x=14, pe_array_y=12, precision=7, sram_size=108000, ifmap_spad_size=24, weights_spad_size=448, psum_spad_size=48)), 3800196.9199999995), (('resnet18', EyerissAcceleratorState(pe_array_x=14, pe_array_y=12, precision=7, sram_size=108000, ifmap_spad_size=24, weights_spad_size=448, psum_spad_size=48)), 811770.2600000001), (('vgg16', EyerissAcceleratorState(pe_array_x=14, pe_array_y=12, precision=7, sram_size=108000, ifmap_spad_size=24, weights_spad_size=448, psum_spad_size=48)), 7191403.129999999), (('vgg11', EyerissAcceleratorState(pe_array_x=14, pe_array_y=12, precision=8, sram_size=108000, ifmap_spad_size=24, weights_spad_size=448, psum_spad_size=48)), 3891126.39), (('resnet18', EyerissAcceleratorState(pe_array_x=14, pe_array_y=12, precision=8, sram_size=108000, ifmap_spad_size=24, weights_spad_size=448, psum_spad_size=48)), 831068.5699999998), (('vgg16', EyerissAcceleratorState(pe_array_x=14, pe_array_y=12, precision=8, sram_size=108000, ifmap_spad_size=24, weights_spad_size=448, psum_spad_size=48)), 7365794.509999999)])

    weight_dict = OrderedDict([(('vgg11', EyerissAcceleratorState(pe_array_x=14, pe_array_y=12, precision=6, sram_size=108000, ifmap_spad_size=24, weights_spad_size=448, psum_spad_size=48)), 3569123328.0), (('resnet18', EyerissAcceleratorState(pe_array_x=14, pe_array_y=12, precision=6, sram_size=108000, ifmap_spad_size=24, weights_spad_size=448, psum_spad_size=48)), 735883264.0), (('vgg16', EyerissAcceleratorState(pe_array_x=14, pe_array_y=12, precision=6, sram_size=108000, ifmap_spad_size=24, weights_spad_size=448, psum_spad_size=48)), 6563856384.0), (('vgg11', EyerissAcceleratorState(pe_array_x=14, pe_array_y=12, precision=7, sram_size=108000, ifmap_spad_size=24, weights_spad_size=448, psum_spad_size=48)), 3569123328.0), (('resnet18', EyerissAcceleratorState(pe_array_x=14, pe_array_y=12, precision=7, sram_size=108000, ifmap_spad_size=24, weights_spad_size=448, psum_spad_size=48)), 735883264.0), (('vgg16', EyerissAcceleratorState(pe_array_x=14, pe_array_y=12, precision=7, sram_size=108000, ifmap_spad_size=24, weights_spad_size=448, psum_spad_size=48)), 6563856384.0), (('vgg11', EyerissAcceleratorState(pe_array_x=14, pe_array_y=12, precision=8, sram_size=108000, ifmap_spad_size=24, weights_spad_size=448, psum_spad_size=48)), 3569123328.0), (('resnet18', EyerissAcceleratorState(pe_array_x=14, pe_array_y=12, precision=8, sram_size=108000, ifmap_spad_size=24, weights_spad_size=448, psum_spad_size=48)), 735883264.0), (('vgg16', EyerissAcceleratorState(pe_array_x=14, pe_array_y=12, precision=8, sram_size=108000, ifmap_spad_size=24, weights_spad_size=448, psum_spad_size=48)), 6563856384.0)])

    logging.getLogger().logdir = project_dir + "/solver_temp"

    logger.info(f"Accelerators: {bins}")
    weights = ", ".join([f"(network={key[0]}, accel={key[1].precision} bits, latency={value})" for key, value in weight_dict.items()])
    logger.info(f"Weights/latencies: {weights}")

    underTest = Scheduler()
    result = underTest.run(bins=bins, items=items, cost_dict=cost_dict,
                           weight_dict=weight_dict, solver_type=solver_type)
    if not result:
        logger.error("No valid assignments found!")
    else:
        assignments = ", ".join([f"(network={entry.tag}, assignedAccel={entry.bin.precision} bits)" for entry in result.entries])
        logger.info(assignments)

import logging
import os.path
import pickle
import random
import re
import pandas as pd
from time import time
from copy import deepcopy
from collections import namedtuple, OrderedDict
from types import SimpleNamespace
from glob import glob
from src.accelerator_cfg import AcceleratorProfile
from src.baseline import BaselineEvaluator
from src.args import MetricType
from src.optimization.scheduling import SchedulerType

logger = logging.getLogger(__name__)


def run_partition_comparison(args, workload, accuracy_lut):
   """Experiment comparing a partition-aware scheduling technique
      for multi-DNN workloads with ours
   """
   accel_cfg = AcceleratorProfile(args.accelerator_arch_type)

   # evaluate our scheduling with a set architecture
   args.sceduler_type = SchedulerType.Ours
   ours = BaselineEvaluator(args, accel_cfg, workload, accuracy_lut)
   ours.evaluate()

   # preprocessing for partition aware scheduling
   # number of valid mappings per NN, such that the accuracy constraint is satisfied
   accelerators_per_network = {arch: len(
                                 accuracy_lut.loc[
                                    (accuracy_lut['Network'] == arch) & 
                                    (accuracy_lut['QuantBits'] != 32) & 
                                    (accuracy_lut['QuantBits'] != 16) & 
                                    (accuracy_lut['Valid'] == 1),
                                 'QuantBits'].unique()
                              ) for arch in workload.dnns.keys()}
   mappings = None
   if args.partition_use_constrained:
      # all mapping evaluations, for evaluating non-partitioned NNs
      mappings = SimpleNamespace(energy=deepcopy(ours.energy_dict),
                                 latency=deepcopy(ours.latency_dict),
                                 edp={key: ours.energy_dict[key] * ours.latency_dict[key]
                                    for key in ours.energy_dict},
                                 area=deepcopy(ours.area_dict))

   # initializing and running partition-aware scheduling technique
   theirs = PartitionEvaluator(args, ours.state, accelerators_per_network, mappings)
   theirs.run_optimization()

   # repeat the evaluation results from our method in the end
   logger.info(f"Ours: evaluation results:\n"
            f"\tEnergy={ours.latest_energy:.3e}\n"
            f"\tLatency={ours.latest_latency:.3e}\n"
            f"\tEDP={ours.latest_energy * ours.latest_latency:.3e}\n"
            f"\tArea={ours.latest_area:.3e}")


PartitionMetrics = namedtuple('PartitionMetrics', ['partition_latency', 'partition_link_latency',
                                                   'partition_energy', 'partition_link_energy',
                                                   'overall_latency', 'overall_energy',
                                                   'maximum_throughput', 'overall_link_latency', 'overall_link_energy'])

PartitionInstance = namedtuple('PartitionInstance', ['tag', 'num_partitions', 'partition_points', 'assignment', 'metrics'])


class PartitionEvaluator:
   """Implementation of a parititioning-aware scheduling optimizer 
   """
   def __init__(self, args, accelerators, accelerators_per_network, mappings=None):
      self.logdir = args.logdir
      self.best_energy = self.best_edp = self.best_latency = None
      self.latest_energy = self.latest_edp = self.latest_latency = None
      self.best_schedule = self.latest_schedule = None
      self.best_partition = self.latest_partition = None
      self.latest_execution = self.latest_transfer = None
      self.constrained = mappings is not None

      self.optimization_metric = args.partition_optim_metric_type
      self.num_iterations = args.partition_optim_iterations
      self.selection_probability = args.partition_selection_probability
      self.num_accelerators = len(accelerators)

      # initialize scheduler
      self.scheduler = Scheduler(SchedulerType.PartitionAware)

      # latency (energy) is given in ms (mJ) instead of ns (uJ)
      self.latency_multiplier = 1e6
      self.energy_multiplier = 1e3

      # preapare partitioning data for scheduling
      unpartitioned_networks = self.load_partition_results(args.partition_results_path,
                                                           accelerators_per_network)
      if self.constrained:
         self.load_unpartitioned_networks(unpartitioned_networks, accelerators, mappings)

      # determine type of optimization
      # TODO: This method is based on sample/evaluate iterations. Find a more sophisticated
      # TODO:  technique that takes into account all partitionings (not only sampled ones)
      self.optimize = self._optimize_search

   def load_partition_results(self, path, accelerator_per_networks):
      """Load the partitioning results
      """
      def isnumber(entry):
         try:
            return float(entry) == entry
         except ValueError:
            return False

      def first_consecutive_numbers(number_list):
         consecutives = []
         for number in number_list:
            if number + 1 in number_list:
               consecutives.append(number)
               continue
            elif len(consecutives) != 0:
               consecutives.append(number)
               return consecutives
            
      def keep_uniques(seq):
         seen = set()
         seen_add = seen.add
         return [x for x in seq if not (x in seen or seen_add(x))]

      # get the used networks
      networks = list(accelerator_per_networks.keys())

      # import and load the csv files for all NNs
      files = [file for file in glob(f'{path}/*/*result_nondom.csv')
               if re.search(f'/(\w+)_result_nondom[.]csv', file).group(1) in networks]
      logger.debug(f"Found {len(files)} csv files: ({' | '.join(files)})")
      # NNs whose corresponding csv files were not found
      missing_networks = [network for network in networks
                          if not any(network in file for file in files)]
      logger.debug(f'Missing networks: {missing_networks}')
      single_accelerator_networks = []
      if self.constrained:
         # NNs that cannot sustain partitioning, due to accuracy constraints
         single_accelerator_networks = [arch for arch, num_accelerators in accelerator_per_networks.items()
                                       if num_accelerators == 1]
         logger.debug(f'Non-partitioned networks: {single_accelerator_networks}')

      dfs = OrderedDict([
         (os.path.basename(file).replace('_result_nondom.csv', ''),
          pd.read_csv(file, header=None, index_col=0))
         for file in files
         if os.path.basename(file).replace('_result_nondom.csv', '') not in missing_networks + single_accelerator_networks
      ])
      assert not any(df.empty for df in dfs.values()), \
         f"Empty result file(s) detected: {' | '.join([file for file in files if pd.read_csv(file).empty])}"

      self.partitions = OrderedDict()
      for arch, df in dfs.items():
         logger.debug(f'Parsing partition data for {arch}')

         # Separate column indices in sections, not counting the index
         # ----------------------------------------------------------#
         # first, the 2nd column shows the number of unique partition points
         num_unique_partition_points_column = 2
         # columns with numbers (i.e., not layer names)
         numeric_columns = [column for column in df.columns
                            if all(isnumber(entry) for entry in df.loc[:, column])]
         # columns with the names of partition points (i.e., layers)
         partition_point_columns = first_consecutive_numbers([column for column in df.columns
                                                              if column not in numeric_columns])
         total_num_partitions = len(partition_point_columns) + 1
         # columns with the latency of each partition
         start_part_latency = num_unique_partition_points_column
         end_part_latency = start_part_latency + total_num_partitions
         partition_latency_columns = list(df.columns[start_part_latency:end_part_latency])
         # columns with the energy consumption of each partition
         start_part_energy = end_part_latency
         end_part_energy = start_part_energy + total_num_partitions
         partition_energy_columns = list(df.columns[start_part_energy:end_part_energy])
         # columns with the latency of the link between partitions
         start_part_link_latency = end_part_energy
         end_part_link_latency = start_part_link_latency + (total_num_partitions - 1) + 1  # extra dummy column with zeros
         partition_link_latency_columns = list(df.columns[start_part_link_latency:end_part_link_latency])
         # columns with the energy consumption of the link between partitions
         start_part_link_energy = end_part_link_latency
         end_part_link_energy = start_part_link_energy + (total_num_partitions - 1)  + 1  # extra dummy column with zeros
         partition_link_energy_columns = list(df.columns[start_part_link_energy:end_part_link_energy])
         # columns with the assignment of partitions to accelerators
         start_assignment = partition_point_columns[-1]
         end_assignment = start_assignment + total_num_partitions
         partition_assignment_columns = list(df.columns[start_assignment:end_assignment])
         # columns with overall metrics
         overall_metric_columns = list(df.columns[end_assignment:-1])
         # check if all columns have been covered
         assert partition_latency_columns + partition_energy_columns + \
                partition_link_latency_columns + partition_link_energy_columns + \
                partition_point_columns + partition_assignment_columns + overall_metric_columns \
                  == list(df.columns[num_unique_partition_points_column:-1]), arch
         assert len(partition_assignment_columns) == len(partition_latency_columns) == len(partition_energy_columns) == \
                len(partition_link_latency_columns) == len(partition_link_energy_columns) == len(partition_point_columns) + 1, arch
         # ----------------------------------------------------------#

         # gather information about each group of partitions (one group per network)
         self.partitions[arch] = []
         for row_index, row in df.iterrows():

            this_num_partition_points = row.loc[num_unique_partition_points_column]
            if this_num_partition_points == 1:
               assert len(df) != 1, f"{arch}: Only a single non-partitioned schedule was found"
               continue

            partition_points = [row.loc[column] for column in partition_point_columns]
            partition_latency = [row.loc[column] for column in partition_latency_columns]
            partition_energy = [row.loc[column] for column in partition_energy_columns]
            partition_link_latency = [row.loc[column] for column in partition_link_latency_columns[1:]]  # removing dummy column
            partition_link_energy = [row.loc[column] for column in partition_link_energy_columns[1:]]

            # overall schedule metrics
            overall_metrics = {metric: row.loc[column] for metric, column in 
                                zip(PartitionMetrics._fields[-len(overall_metric_columns):], overall_metric_columns)}
            
            # special handling for assignments: shift the numbers such that the highest indexed accelerator
            # always participates (it is the one with highest precision)
            assignments = [row.loc[column] for column in partition_assignment_columns]
            if (min(assignments), max(assignments)) != (1, self.num_accelerators):
               shift_by = self.num_accelerators - accelerator_per_networks[arch]
               assignments = [assignment + shift_by for assignment in assignments]

            # save each schedule as a namedtuple
            partition_instance = PartitionInstance(
               tag=f'{arch}_{row_index}',
               num_partitions=this_num_partition_points + 1,
               partition_points=partition_points,
               assignment=assignments,
               metrics=PartitionMetrics(
                  partition_latency=partition_latency,
                  partition_energy=partition_energy,
                  partition_link_latency=partition_link_latency,
                  partition_link_energy=partition_link_energy,
                  **overall_metrics)
            )
            self.partitions[arch].append(partition_instance)
            logger.debug(f'\t{row_index}: {partition_instance}')

      return list(set(missing_networks + single_accelerator_networks)) 

   def load_unpartitioned_networks(self, networks, accelerators, mappings=None):
      """Construct dummy partition-like instance for unpartitioned networks,
         based on already evaluated mappings
      """
      if mappings is None or networks is None or len(networks) == 0:
         return

      # transform from our units, according to the partition data
      # this is reversed later on the aggregate results
      energy_multiplier = 1 / self.energy_multiplier
      latency_multiplier = 1 / self.latency_multiplier

      sorted_accelerators = sorted(accelerators, key=lambda accelerator: accelerator.precision)
      for arch in networks:
         energy = [mappings.energy[(arch, accelerator)] for accelerator in sorted_accelerators]
         latency = [mappings.latency[(arch, accelerator)] for accelerator in sorted_accelerators]
         # assign to corresponding accelerators
         energy, latency, assignment = zip(*[(_energy, _latency, index + 1) 
                                              for index, (_energy, _latency) in enumerate(zip(energy, latency))
                                              if _energy != -1 and latency != -1])

         assert len(energy) == len(latency) == 1, f'{arch}: Found {len(energy)} possible assignments ' \
                                                   'for hypothetically un-partitioned network'

         energy = [item * energy_multiplier for item in energy]
         latency = [item * latency_multiplier for item in latency]

         # include as partition object
         partition_instance = PartitionInstance(
            tag=f'{arch}_0',
            num_partitions=1,
            partition_points=['input'],
            assignment=assignment,
            metrics=PartitionMetrics(
               partition_latency=latency,
               partition_energy=energy,
               partition_link_latency=[0],
               partition_link_energy=[0],
               overall_latency=latency[0],
               overall_energy=energy[0],
               maximum_throughput=0,
               overall_link_latency=0,
               overall_link_energy=0
            )
         )
         self.partitions[arch] = [partition_instance]

   def save_results(self):
      """Save the most recent results
      """
      state_dict = {'best_partition': self.best_partition,
                    'best_schedule': self.best_schedule,
                    'best_energy': self.best_energy,
                    'best_latency': self.best_latency,
                    'best_edp': self.best_edp,}

      savefile = os.path.join(self.logdir, 'partition.results.pkl')
      with open(savefile, 'wb') as f:
         pickle.dump(state_dict, f)
      logger.info(f"Saved partition results in: {savefile}")

   def run_optimization(self):
      """Wrapper for optimization algorithm 
      """
      start = time()
      logger.info(f"=> Beginning partition-aware optimization")
      try:
         is_successful = self.optimize()
      finally:
         self.save_results()
      assert self.best_schedule is not None
      logger.info(f"Completed optimization in {time() - start:.3e}s")

      # log the results of the final scheduling
      logger.info("*--------------*")
      schedule_str = '\n\t'.join([f'{entry.tag} -> {entry.bin}' for entry in self.best_schedule.entries])
      logger.debug(f"Final scheduling:\n\t{schedule_str}")
      logger.info(f"Evaluation results:\n"
                  f"\tEnergy={self.best_energy:.3e}\n"
                  f"\tLatency={self.best_latency:.3e}\n"
                  f"\tEDP={self.best_energy * self.best_latency:.3e}\n")
      logger.info("*--------------*\n")

   def _optimize_search(self):
      """Search-based optimization algorithm to find optimal schedules for partitioned DNNs
      """
      for iteration in range(1, self.num_iterations + 1):
         logger.info("*--------------*")
         logger.info(f"Iteration {iteration}/{self.num_iterations}")
         start = time()

         self.sample_partitions()
         self.evaluate_partitions()

         # select optimization metric         
         latest_metric, best_metric = {
            MetricType.Energy: (self.latest_energy, self.best_energy),
            MetricType.Latency: (self.latest_latency, self.best_latency),
            MetricType.EDP: (self.latest_edp, self.best_edp)
         }.get(self.optimization_metric)
         assert latest_metric is not None, f"Optimization metric {self.optimization_metric} is not valid"

         # check whether to reject the sampled partitions
         if best_metric is None or latest_metric < best_metric:
            self.best_partition = self.latest_partition
            self.best_schedule = self.latest_schedule
            self.best_energy, self.best_latency, self.best_edp = self.latest_energy, self.latest_latency, self.latest_edp

            # save and log the results
            if best_metric is not None:
               logger.info("\033[92m" + f"New {self.optimization_metric.name.lower()} improvement " \
                           f"on iteration {iteration}: {best_metric:.3e} < {latest_metric:.3e}" + "\033[0m")
         else:
            logger.info(f"No improvement on iteration {iteration}: {best_metric:.3e} > {latest_metric:.3e}")

         logger.info(f'Completed iteration {iteration} in {time() - start:.3e}s')

   def sample_partitions(self):
      """Semi-random sampling of one partitioning instance from each network
      """
      self.latest_partition = OrderedDict()
      for arch, partition_instances in self.partitions.items():
         if self.best_partition is None or random.random() > self.selection_probability:
            self.latest_partition[arch] = random.choice(partition_instances)
         else:
            self.latest_partition[arch] = self.best_partition[arch]

      logstr = '\n\t'.join([f'{arch}: {partition_instance}' for arch, partition_instance in self.latest_partition.items()])
      logger.debug(f"New sampled partitions:\n\t{logstr}")

   def evaluate_partitions(self):
      """Run scheduling on the selected partition to gather evaluation metrics
      """
      bins = list(range(1, self.num_accelerators + 1))
      partitions = list(self.latest_partition.values())

      start = time()
      self.latest_schedule = self.scheduler.run(partitions, bins)
      logger.debug(f"Completed schedule in {time() - start:.3e}s")

      if self.latest_schedule is None:
         logger.warning(f'Could not find valid schedule')
         self.latest_energy = self.latest_latency = self.latest_edp = None
         return

      # make sure no errors in sub-partition assignment have occured
      assert all(
         all(
            entries[i].end <= entries[i+1].start
            for i in range(len(entries)-1)
         ) 
         for bin, entries in self.latest_schedule.as_dict('bin').items()
      )

      # gather metrics from the schedule
      self.latest_energy = sum([
         instance.metrics.overall_energy + instance.metrics.overall_link_energy
         for instance in self.latest_partition.values()
      ])
      self.latest_latency = max([
         entries[-1].end
         for bin, entries in self.latest_schedule.as_dict(main_key='bin').items()
      ])
      # apply numerical transformations
      self.latest_latency *= self.latency_multiplier
      self.latest_energy *= self.energy_multiplier
      self.latest_edp = self.latest_energy * self.latest_latency


if __name__ == "__main__":
   args = SimpleNamespace(logdir=None,
                          scheduler_type=None,
                          solver_type=None,
                          partition_optim_metric_type=None,
                          partition_optim_iterations=2,
                          partition_selection_probability=0.8,
                          partition_results_path='data/cnn-parted')
   e = PartitionEvaluator(args, num_accelerators=4)
   e.run_optimization()
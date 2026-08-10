import logging
import os
import re
import shutil
import subprocess
import yaml

from copy import deepcopy
from glob import glob
from time import time
from types import SimpleNamespace
from uuid import UUID

from src import project_dir
from src.mapping.api import AcceleratorConfiguration, ConvolutionProblem, MappingRequest, MappingStats, \
    AcceleratorMapper

__all__ = ['TimeloopWrapper', 'TimeloopTemplate', 'TimeloopProblem', 'TimeloopArch', 'TimeloopMapper']

logger = logging.getLogger(__name__)

TIMELOOP_ACCELERGY_VERSION = 0.4


def force_quotes_on_str(nested_dict, filter_fn=None):
    """Add quotes to each str located within a nested dict
    """

    class quoted(str):
        pass

    def quoted_presenter(dumper, data):
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')

    yaml.add_representer(quoted, quoted_presenter)

    if filter_fn is None:
        filter_fn = lambda entry: isinstance(entry, str)

    def recursion_f(this_dict):
        for key, value in this_dict.items():
            if isinstance(value, dict):
                recursion_f(this_dict[key])
            elif filter_fn(value):
                this_dict[key] = quoted(value)
            elif isinstance(value, (list, tuple)):
                for v in value:
                    assert isinstance(v, dict)
                    recursion_f(v)

    # execute the recursion function
    recursion_f(nested_dict)


class TimeloopWrapper(AcceleratorMapper):
    """Wrapper for Timeloop+Accelergy tool
    """

    def __init__(self, workdir, *, cleanup=True):
        self.template = TimeloopTemplate()
        self.workdir = workdir
        self.cleanup = cleanup
        os.makedirs(self.workdir, exist_ok=True)
        self.mapper = TimeloopMapper(mapper_file=os.path.join(self.workdir, 'mapper.yaml'))

    def map(self, request: MappingRequest) -> MappingStats:
        logger.debug(f"Evaluating MappingRequest with id={request.id}, name={request.name}")

        request_dir = os.path.join(self.workdir, str(request.id))
        os.makedirs(request_dir, exist_ok=True)

        problem_filepath = os.path.join(request_dir, 'problem.yaml')
        problem = TimeloopProblem(request.id, request.problem, problem_filepath)

        arch_dir = os.path.join(request_dir, 'arch')
        os.makedirs(arch_dir, exist_ok=True)
        architecture = TimeloopArch(arch_dir, component_files=self.template.arch_components)
        architecture.adjust(request.accelerator_config)
        architecture.to_yaml()

        constraint_dir = os.path.join(request_dir, 'constraints')
        os.makedirs(constraint_dir, exist_ok=True)

        output_dir = os.path.join(request_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)

        # copy constraint files to the created directory
        for constraint_file in self.template.constraint_files:
            shutil.copy2(constraint_file, constraint_dir)

        logfile = os.path.join(request_dir, 'timeloop-mapper.log')

        command = f'timeloop-mapper ' \
                  f'{architecture.arch_filepath} ' \
                  f'{" ".join(architecture.component_files)} ' \
                  f'{problem_filepath} ' \
                  f'{self.mapper.mapper_filepath} ' \
                  f'{constraint_dir}/*.yaml ' \
                  f'-o {output_dir} 2>&1 | tee {logfile}'
        logger.debug(f'timeloop-mapper command: {command}')
        start = time()
        completed_process = subprocess.run(["bash", "-lc", command], check=True, capture_output=True)
        logger.debug(f"Executed timeloop-mapper command in {time() - start:.3e}s "
                     f"with exitcode: {completed_process.returncode}")
        results = self._get_results(request.id, output_dir)
        if self.cleanup:
            shutil.rmtree(request_dir)
        logger.debug(f"Layer-wise results: "
                          f"id={request.id}, name={request.name}, energy={results.energy:.3e}, latency={results.cycles:.3e}, edp={results.edp:.3e}")
        return results

    @staticmethod
    def _get_results(uuid: UUID, output_dir) -> MappingStats:
        """Get the results of a succesfull run from Timeloop. Note, timeloop provides
           a script that does a more analytical parsing: 
           https://github.com/NVlabs/timeloop/blob/master/scripts/parse_timeloop_output.py#L55
        """

        def _get_area_from_ART(area_file=None):
            """Gather an area measurement from the ART file. Note, the area
               measurements in the file are per unit
            """
            area_summary_file = area_file or os.path.join(output_dir, 'timeloop-mapper.ART.yaml')
            assert os.path.exists(area_summary_file)
            with open(area_summary_file, 'r') as stream:
                area_dict = yaml.safe_load(stream)
            # this is in um^2
            area = 0.0
            for item in area_dict['ART']['tables']:
                num_units = 1
                if re.search('\d+[.]{2}\d+', item['name']):
                    num_units = re.search("[.][.](\d+)[]]", item['name']).group(1)
                area += (int(num_units) + 1) * item['area']
            return area * 1e-6  # in mm^2

        stats_file = os.path.join(output_dir, 'timeloop-mapper.stats.txt')
        with open(stats_file, 'r') as f:
            stats = f.read()

        # NOTE: More results can be extracted here, but not needed for now
        gflops = re.search('GFLOPs .*?: ([\d.]+)', stats).group(1)
        gflops = float(gflops)  # @1GHz
        utilization = re.search('Utilization: ([\d.]+)', stats).group(1)
        utilization = float(utilization)  # non-unit
        cycles = re.search('Cycles: ([\d.]+)', stats).group(1)
        cycles = float(cycles)  # non-unit
        energy = re.search('Energy: ([\d.]+)', stats).group(1)
        energy = float(energy)  # uJ
        edp = re.search('EDP.*?: (.*)', stats).group(1)
        edp = float(edp)  # J * cycle

        # get the area in mm^2 from the stats file
        area = re.search('Area: ([\d.]+)', stats).group(1)
        # if area is 0.0 from the stats file, we override with ART values
        area = _get_area_from_ART() if float(area) <= 0.0 else float(area)

        return MappingStats(id=uuid, gflops=gflops, utilization=utilization,
                            cycles=cycles, energy=energy, edp=edp,
                            area=area)

    def adjust_architecture(self, accelerator: AcceleratorConfiguration):
        """Adjust the architectural parameters of the accelerator
        """
        self.arch.adjust(accelerator)
        self.arch._to_yaml()


# TODO break into two functions
class TimeloopTemplate:
    """Configuration environment for Timeloop files
    """

    def __init__(self):
        eyeriss_timeloop_dir = os.path.join(project_dir, 'timeloop-accelergy-exercises',
                                            'workspace', 'exercises', '2020.ispass', 'timeloop',
                                            '06-mapper-convlayer-eyeriss')
        self.arch_components = glob(os.path.join(eyeriss_timeloop_dir, 'arch', 'components', '*.yaml'))
        self.constraint_files = glob(os.path.join(eyeriss_timeloop_dir, 'constraints', '*.yaml'))


# TODO convert to function
class TimeloopProblem:
    """Utility class to handle and create a Timeloop-related workload
    """

    def __init__(self, id: UUID, problem: ConvolutionProblem, problem_filepath):
        self.problem_filepath = problem_filepath
        self.config = None

        self._config_conv_layer(id, problem)
        self._to_yaml()

    def _to_yaml(self, filepath=None):
        """Create a yaml description of the workload
        """
        if filepath is None:
            assert self.problem_filepath is not None
            filepath = self.problem_filepath
        with open(filepath, 'w') as f:
            f.write(yaml.dump({'problem': self.config}))

    def _config_conv_layer(self, id: UUID, problem):
        """Create the configuration for a Convolutional-type layer
        """
        output_width = int(
            (problem.input_width - problem.kernel_width + 2 * problem.padding_width) / problem.stride_width) + 1
        output_height = int(
            (problem.input_height - problem.kernel_height + 2 * problem.padding_height) / problem.stride_height) + 1
        dimensions = ['C', 'M', 'R', 'S', 'N', 'P', 'Q']

        config = {'shape': {}}
        config['shape']['name'] = str(id)
        config['shape']['dimensions'] = dimensions
        config['shape']['coefficients'] = [
            {
                'name': 'Hdilation',
                'default': 1
            },
            {
                'name': 'Wdilation',
                'default': 1
            },
            {
                'name': 'Hstride',
                'default': problem.stride_height
            },
            {
                'name': 'Wstride',
                'default': problem.stride_width
            }
        ]
        config['shape']['data-spaces'] = [
            {
                'name': 'Weights',
                'projection': [
                    [['M']],
                    [['C']],
                    [['R']],
                    [['S']]
                ]
            },
            {
                'name': 'Inputs',
                'projection': [
                    [['N']],
                    [['C']],
                    [['R', 'Wdilation'], ['Q', 'Wstride']],
                    [['S', 'Hdilation'], ['P', 'Hstride']],
                ]
            },
            {
                'name': 'Outputs',
                'projection': [
                    [['N']],
                    [['M']],
                    [['P']],
                    [['Q']]
                ],
                'read-write': True
            }
        ]
        config['instance'] = {
            'C': problem.input_channels,
            'M': problem.output_channels,
            'R': problem.kernel_width,
            'S': problem.kernel_height,
            'N': problem.batch_size,
            'Q': output_width,
            'P': output_height
        }

        self.config = config


class TimeloopArch:
    """Utility class to handle the architectural parameters of the accelerator
       when using timeloop
    """

    def __init__(self, workdir, component_files):
        self.name = self.__class__.__name__
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
        for component_file in component_files:
            shutil.copy2(component_file, workdir)
        self.component_files = [os.path.join(workdir, os.path.basename(component_file))
                                for component_file in component_files]
        self.arch_filepath = os.path.join(workdir, 'architecture.yaml')

        # initialize dict with parameters
        self.get_default_params()
        self.init_params = deepcopy(self.params)
        self.get_config()
        self._sync_version()
        self.to_yaml()

    def _sync_version(self):
        """Set the correct timeloop+accelergy version to all files
        """
        for component_file in self.component_files:
            with open(component_file, 'r') as stream:
                yaml_dict = yaml.safe_load(stream)

            if 'compound_components' in yaml_dict:
                yaml_dict['compound_components']['version'] = TIMELOOP_ACCELERGY_VERSION
            else:
                yaml_dict['version'] = TIMELOOP_ACCELERGY_VERSION

            with open(component_file, 'w') as f:
                f.write(yaml.dump(yaml_dict))

    def to_yaml(self, filepath=None):
        """Write the configuration of the architecture to a yaml file
        """
        use_default_flow_style = True
        if TIMELOOP_ACCELERGY_VERSION > 0.3:
            # make sure to use quotes on each str item
            try:
                force_quotes_on_str(self.config)
            except AssertionError:
                logger.error(f"Config failed:\n{self.config}")
                with open('temp.yaml', 'w') as f:
                    f.write(yaml.dump({'architecture': self.config},
                                      default_flow_style=use_default_flow_style))
                raise
            use_default_flow_style = False

        if filepath is None:
            assert self.arch_filepath is not None
            filepath = self.arch_filepath

        with open(filepath, 'w') as f:
            f.write(yaml.dump({'architecture': self.config},
                              default_flow_style=use_default_flow_style))

    def adjust_params(self, params):
        """Generic function to override parameter values
        """
        for param_name, value in params.items():
            assert hasattr(self.params, param_name), f'{param_name} is not a valid parameter'
            setattr(self.params, param_name, value)
        # update the configuration with new parameters
        self.get_config()

    def adjust_pe_array(self, pe_x, pe_y):
        """Adjust the dimensions of the PE array
        """
        if self.params.pe_array_x == pe_x and \
                self.params.pe_array_y == pe_y:
            return

        params = {'pe_array_x': pe_x,
                  'pe_array_y': pe_y}
        self.adjust_params(params)

    def adjust_mem_width(self, buffer_name, word_bits, block_size, num_clusters=1):
        """Adjust the memory width of a buffer.

           For each memory unit/buffer, the two following assertions must be satisfied:
           1) width % (word_bits * block_size) == 0
                https://github.com/NVlabs/timeloop/blob/be27768a6466aeae18c52d0221ce778b8b58870c/src/model/buffer.cpp#L285
                This dictates that the width of the buffer has to be perfectly divided
                by word_bits * block_size
           2) specs_.instances.Get() % specs_.cluster_size.Get() == 0
                https://github.com/NVlabs/timeloop/blob/be27768a6466aeae18c52d0221ce778b8b58870c/src/model/buffer.cpp#L1275
                This needs the number of instances of a buffer (i.e., when instantiated
                in a mesh) to be dividable by the cluster size. The formula for the cluster
                size is: cluster_size = width / (word_bits * block_size).   
            The architecture of the memory and the dependecies of its parameters are well illustrated here:
                https://github.com/NVlabs/timeloop/pull/176
            
            We modify the memory width, via the number of clusters, to comply with the
            above assertions, whilst keeping the same block size.
        """
        width = num_clusters * word_bits * block_size
        cluster_size = width / (word_bits * block_size)
        assert self.params.pe_array_x % cluster_size == 0, \
            f"{buffer_name}: Instances ({self.params.pe_array_x}) are not dividable by cluster_size " \
            f"({cluster_size} = {width} / ({word_bits} * {block_size}))"
        return width

    ### Accelerator-specific functions: Eyeriss ###

    def adjust(self, accelerator: AcceleratorConfiguration):
        """Adjust the parameter of the architecture based on the
           given accelerator instance
        """
        # NOTE: VERY important to change the PE array dimensions first,
        #       such that the memory width can adjust to the new dimensions
        #       and the change in precision
        self.adjust_pe_array(accelerator.pe_array_x,
                             accelerator.pe_array_y)
        self.adjust_memories(accelerator.sram_size,
                                      accelerator.ifmap_spad_size,
                                      accelerator.weights_spad_size,
                                      accelerator.psum_spad_size)
        self.adjust_precision(accelerator.precision)

    def adjust_memories(self, sram_size, ifmap_spad_size,
                                 weights_spad_size, psum_spad_size):
        """Adjust each specific memory unit of the Eyeriss-like accelerator
        """
        params = {
            'sram_depth': sram_size,
            'ifmap_spad_depth': ifmap_spad_size,
            'weights_spad_depth': weights_spad_size,
            'psum_spad_depth': psum_spad_size
        }
        self.adjust_params(params)

    def adjust_precision(self, precision):
        """Adjust the data precision of the Eyeriss sarchitecture, including memory and compute units.
           We only change the parameters of the MAC unit and the scratchpads, not the DRAM or SRAM.
        """
        if self.params.mac_datawidth == precision:
            return

        params = {
            # MAC unit
            'mac_datawidth': precision,  # multiplying activations/inpus * weights
            'mac_class': 'fpmac' if precision == 32 else 'intmac',
            # word bits of scratchpads/dummy register file
            'ifmap_spad_word_bits': precision,  # activations
            'weights_spad_word_bits': precision,  # weights
            'psum_spad_word_bits': precision,  # outputs from MAC units
            'regfile_word_bits': precision,  # memory used by MACs
            # width of scratchpads/dummy register file
            'ifmap_spad_width': self.adjust_mem_width('ifmap_spad',
                                                      precision,
                                                      self.init_params.ifmap_spad_block_size,
                                                      getattr(self.params, 'ifmap_spad_cluster_size', 1)),
            # memory bus for ifmap (activations/input features)
            'weights_spad_width': self.adjust_mem_width('weights_spad',
                                                        precision,
                                                        self.init_params.weights_spad_block_size,
                                                        getattr(self.params, 'weights_spad_cluster_size', 1)),
            # memory bus for weights
            'psum_spad_width': self.adjust_mem_width('psum_spad',
                                                     precision,
                                                     self.init_params.psum_spad_block_size,
                                                     getattr(self.params, 'psum_spad_cluster_size', 1)),
            # partial sum/outputs from MAC units bus width
            'regfile_width': self.adjust_mem_width('dummy_regfile',
                                                   precision,
                                                   self.init_params.regfile_block_size,
                                                   getattr(self.params, 'dummy_regfile_cluster_size', 1)),
            # bus width used by register files/used by MACs
        }
        self.adjust_params(params)

    def get_default_params(self):
        """Get the default parameters for all levels of an
           Eyeriss-like architecture
        """
        self.params = SimpleNamespace()
        self.params.pe_array_x = 14
        self.params.pe_array_y = 16

        self.params.technology = '45nm'
        # external DRAM attributes
        self.params.dram_width = 64
        self.params.dram_word_bits = 16
        self.params.dram_block_size = 4
        # global SRAM attributes
        self.params.sram_class = 'smartbuffer_SRAM'
        self.params.sram_depth = 16384
        self.params.sram_width = 64
        self.params.sram_n_banks = 32
        self.params.sram_word_bits = 16
        self.params.sram_block_size = 4
        self.params.sram_read_bandwidth = 16
        self.params.sram_write_bandwidth = 16
        # dummy register file attributes
        self.params.regfile_depth = 16
        self.params.regfile_width = 16
        self.params.regfile_word_bits = 16
        self.params.regfile_block_size = 1
        # class for implementing scratchpads
        self.params.spad_class = 'smartbuffer_RF'
        # attributes for IFM scratchpad
        self.params.ifmap_spad_depth = 12
        self.params.ifmap_spad_width = 16
        self.params.ifmap_spad_word_bits = 16
        self.params.ifmap_spad_block_size = 1
        self.params.ifmap_spad_read_bandwidth = 2
        self.params.ifmap_spad_write_bandwidth = 2
        # attributes for Weights' scratchpad
        self.params.weights_spad_depth = 192
        self.params.weights_spad_width = 16
        self.params.weights_spad_word_bits = 16
        self.params.weights_spad_block_size = 1
        self.params.weights_spad_read_bandwidth = 2
        self.params.weights_spad_write_bandwidth = 2
        # attributes for Partial Sums' scratchpad
        self.params.psum_spad_depth = 16
        self.params.psum_spad_width = 16
        self.params.psum_spad_update_fifo_depth = 2
        self.params.psum_spad_word_bits = 16
        self.params.psum_spad_block_size = 1
        self.params.psum_spad_read_bandwidth = 2
        self.params.psum_spad_write_bandwidth = 2
        # MAC unit attributes
        self.params.mac_class = 'intmac'
        self.params.mac_datawidth = 16

    def get_config(self):
        """Write the architectural description of an Eyeriss-like
           architecture in a dict format
        """
        config = {}
        config['version'] = TIMELOOP_ACCELERGY_VERSION

        level1 = {}
        level1['name'] = 'system'
        level1['local'] = [
            {
                'name': 'DRAM',
                'class': 'DRAM',
                'attributes': {
                    'type': 'LPDDR4',
                    'width': self.params.dram_width,
                    'block-size': self.params.dram_block_size,
                    'word-bits': self.params.dram_word_bits
                }
            }
        ]

        level2 = {}
        level2['name'] = 'eyeriss'
        level2['attributes'] = {
            'technology': self.params.technology
        }
        level2['local'] = [
            {
                'name': 'shared_glb',
                'class': self.params.sram_class,
                'attributes': {
                    'memory_depth': self.params.sram_depth,
                    'memory_width': self.params.sram_width,
                    'n_banks': self.params.sram_n_banks,
                    'block-size': self.params.sram_block_size,
                    'word-bits': self.params.sram_word_bits,
                    'read_bandwidth': self.params.sram_read_bandwidth,
                    'write_bandwidth': self.params.sram_write_bandwidth
                }
            },
            {
                'name': f'DummyBuffer[0..{self.params.pe_array_x - 1}]',
                'class': 'regfile',
                'attributes': {
                    'depth': self.params.regfile_depth,
                    'width': self.params.regfile_width,
                    'word-bits': self.params.regfile_word_bits,
                    'block-size': self.params.regfile_block_size,
                    'meshX': self.params.pe_array_x
                }
            }
        ]

        level3 = {'name': f'PE[0..{(self.params.pe_array_x * self.params.pe_array_y) - 1}]'}
        level3_ifmap = {
            'name': 'ifmap_spad',
            'class': self.params.spad_class,
            'attributes': {
                'memory_depth': self.params.ifmap_spad_depth,
                'memory_width': self.params.ifmap_spad_width,
                'block-size': self.params.ifmap_spad_block_size,
                'word-bits': self.params.ifmap_spad_word_bits,
                'meshX': self.params.pe_array_x,
                'read_bandwidth': self.params.ifmap_spad_read_bandwidth,
                'write_bandwidth': self.params.ifmap_spad_write_bandwidth,
            }
        }
        level3_weights = {
            'name': 'weights_spad',
            'class': self.params.spad_class,
            'attributes': {
                'memory_depth': self.params.weights_spad_depth,
                'memory_width': self.params.weights_spad_width,
                'block-size': self.params.weights_spad_block_size,
                'word-bits': self.params.weights_spad_word_bits,
                'meshX': self.params.pe_array_x,
                'read_bandwidth': self.params.weights_spad_read_bandwidth,
                'write_bandwidth': self.params.weights_spad_write_bandwidth,
            }
        }
        level3_psum = {
            'name': 'psum_spad',
            'class': self.params.spad_class,
            'attributes': {
                'memory_depth': self.params.psum_spad_depth,
                'memory_width': self.params.psum_spad_width,
                'update_fifo_depth': self.params.psum_spad_update_fifo_depth,
                'block-size': self.params.psum_spad_block_size,
                'word-bits': self.params.psum_spad_word_bits,
                'meshX': self.params.pe_array_x,
                'read_bandwidth': self.params.psum_spad_read_bandwidth,
                'write_bandwidth': self.params.psum_spad_write_bandwidth,
            }
        }
        level3_mac = {
            'name': 'mac',
            'class': self.params.mac_class,
            'attributes': {
                'datawidth': self.params.mac_datawidth,
                'meshX': self.params.pe_array_x
            }
        }
        level3['local'] = [level3_ifmap, level3_weights, level3_psum, level3_mac]

        level2['subtree'] = [level3]
        level1['subtree'] = [level2]
        config['subtree'] = [level1]

        self.config = config


# TODO turn into function
class TimeloopMapper:
    """Utility wrapper class fot the mapping optimizer
    """

    def __init__(self, mapper_file):
        self.mapper_filepath = mapper_file
        self._get_params()
        self._get_config()
        self._to_yaml()

    def _get_params(self):
        """Collect the default configuration parameters of the mapper 
        """
        self.params = SimpleNamespace()
        self.params.optimization_metrics = ['edp']
        self.params.live_status = False
        self.params.num_threads = 1
        self.params.timeout = 15000
        self.params.victory_condition = 500
        self.params.algorithm = 'random-pruned'
        self.params.max_permutations_per_if_visit = 16

    def _get_config(self):
        """Write the configuration parameters to a yaml-like dict
        """
        config = {
            'optimization-metrics': self.params.optimization_metrics,
            'live-status': self.params.live_status,
            'num-threads': self.params.num_threads,
            'timeout': self.params.timeout,
            'victory-condition': self.params.victory_condition,
            'algorithm': self.params.algorithm,
            'max-permutations-per-if-visit': self.params.max_permutations_per_if_visit
        }
        self.config = config

    def _to_yaml(self, filepath=None):
        """Write the configuration of the mapper to a yaml file
        """
        if filepath is None:
            assert self.mapper_filepath is not None
            filepath = self.mapper_filepath

        with open(filepath, 'w') as f:
            f.write(yaml.dump({'mapper': self.config}))

# TODO incorporate into a mock of timeloop wrapper
# def timeloop_execution_mock(timeloop_wrapper: TimeloopWrapper, problem_name: str) -> TimeloopStats:
#     energy_base=6.608e+04
#     latency_base=3.303e+07
#     area_base = 12.715
#     energy = uniform(energy_base - 1e4, energy_base + 1e4)
#     latency = ceil(uniform(latency_base - 1e7, latency_base + 1e7))
#     area = uniform(area_base - 1, area_base + 1)
#     return TimeloopStats(gflops=None, utilization=None, energy=energy, cycles=latency,
#                             edp=energy * latency, area=area)

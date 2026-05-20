from math import ceil
from random import uniform

import crimson_magick.cifar_zoo
import yaml
import os
import shutil
import logging
import subprocess
import re
import numpy as np
from time import time
from copy import deepcopy
from glob import glob
from collections import OrderedDict, namedtuple

from crimson_magick.cifar_zoo import Cifar

from src.net_wrapper import TorchNetworkWrapper
from types import SimpleNamespace
from src import eyeriss_timeloop_dir, simba_timeloop_dir, project_dir
from src.accelerator_cfg import AcceleratorType
from src.args import OptimizerType
from src.utils import force_quotes_on_str

__all__ = ['TimeloopStats', 'TimeloopWrapper', 'TimeloopTemplate', 'TimeloopProblem', 'TimeloopArch', 'TimeloopMapper', 'timeloop_execution']

logger = logging.getLogger(__name__)

TIMELOOP_ACCELERGY_VERSION = 0.4

TimeloopStats = namedtuple('TimeloopStats', ['gflops', 'utilization', 'cycles',
                                             'energy', 'edp', 'area'])


class TimeloopWrapper:
    """Wrapper for Timeloop+Accelergy tool
    """

    def __init__(self, accelerator_type, workdir, workloads=None):
        self.template = TimeloopTemplate(accelerator_type)
        self.workdir = workdir
        os.makedirs(self.workdir, exist_ok=True)
        self.workloads = workloads if workloads is not None else OrderedDict()
        self.init_files()
        self.init_arch(accelerator_type)
        self.init_mapper()

    def init_files(self):
        """Initialize files and directories
        """
        workload_dir = os.path.join(self.workdir, 'problem')
        os.makedirs(workload_dir, exist_ok=True)
        self.workload_dir = workload_dir
        constraint_dir = os.path.join(self.workdir, 'constraints')
        os.makedirs(constraint_dir, exist_ok=True)
        self.constraint_dir = constraint_dir
        output_dir = os.path.join(self.workdir, 'output')
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        # copy constraint files to the created directory
        for constraint_file in self.template.constraint_files:
            shutil.copy2(constraint_file, constraint_dir)

    def init_problem(self, problem_name, problem_type, dimensions, problem_filepath=None):
        """Initialize a workload wrapper
        """
        if problem_name in self.workloads:
            return

        if problem_filepath is None:
            problem_filepath = os.path.join(self.workload_dir, f'{problem_name}_{problem_type}.yaml')

        problem = TimeloopProblem(problem_name, problem_type, dimensions, problem_filepath)
        self.workloads[problem_name] = problem

    def init_arch(self, accelerator_type):
        """Initialize the accelerator architecture description
        """
        self.arch = TimeloopArch(accelerator_type=accelerator_type,
                                 workdir=os.path.join(self.workdir, 'arch'),
                                 component_files=self.template.arch_components)

    def init_mapper(self):
        """Initialize the mapper object for the mapping optimization
        """
        self.mapper = TimeloopMapper(mapper_file=os.path.join(self.workdir, 'mapper.yaml'))

    def run(self, problem_name):
        """Execute the Timeloop+Accelergy infrastructure via command-line
        """
        logfile = os.path.join(self.output_dir, problem_name, 'timeloop-mapper.log')
        os.makedirs(f"{self.output_dir}/{problem_name}", exist_ok=True)
        command = f'timeloop-mapper ' \
                  f'{self.arch.arch_filepath} ' \
                  f'{" ".join(self.arch.component_files)} ' \
                  f'{self.workloads[problem_name].problem_filepath} ' \
                  f'{self.mapper.mapper_filepath} ' \
                  f'{self.constraint_dir}/*.yaml ' \
                  f'-o {self.output_dir}/{problem_name} 2>&1 | tee {logfile}'
        logger.debug(f'timeloop-mapper command: {command}')
        start = time()
        p = subprocess.run(["bash", "-lc", command], check=True, capture_output=True)
        logger.debug(f"Executed timeloop-mapper command in {time() - start:.3e}s "
                     f"with exitcode: {p.returncode}")
        return p

    def get_results(self, problem_name) -> TimeloopStats:
        """Get the results of a succesfull run from Timeloop. Note, timeloop provides
           a script that does a more analytical parsing: 
           https://github.com/NVlabs/timeloop/blob/master/scripts/parse_timeloop_output.py#L55
        """

        def _get_area_from_ART(area_file=None):
            """Gather an area measurement from the ART file. Note, the area
               measurements in the file are per unit
            """
            area_summary_file = area_file or os.path.join(self.output_dir, problem_name, 'timeloop-mapper.ART.yaml')
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

        stats_file = os.path.join(self.output_dir, problem_name, 'timeloop-mapper.stats.txt')
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

        return TimeloopStats(gflops, utilization, cycles, energy, edp, area)

    def cleanup(self, problem_name, override_outdir=None):
        """Remove files from the output directory after a run
        """
        if override_outdir is not None:
            outdir = override_outdir
        elif len(glob(f'{self.output_dir}/{problem_name}/*')) > 0:
            outdir = f"{self.output_dir}/{problem_name}"
        else:
            return
        for file in glob(os.path.join(outdir, 'timeloop-mapper*')):
            os.remove(file)

    def adjust_architecture(self, accelerator_instance, adjust_components=False):
        """Adjust the architectural parameters of the accelerator
        """
        self.arch.adjust(accelerator_instance)
        self.arch.to_yaml(do_components=adjust_components)

    def adjust_precision(self, precision):
        """Adjust only the precision of the architecture
        """
        self.arch.adjust_precision(precision)
        self.arch.to_yaml()

    def adjust_problem_dimension(self, problem_name, dimension, value=None, adjust_by=None):
        """Adjust the dimension size for the given workload-problem name
        """
        self.workloads[problem_name].adjust_dimension(dimension, value, adjust_by)
        self.workloads[problem_name].to_yaml()

    def adjust_mapper(self, param_name, param_value):
        self.mapper.adjust_param(param_name, param_value)
        self.mapper.to_yaml()


class TimeloopTemplate:
    """Configuration environment for Timeloop files
    """

    def __init__(self, accelerator_type):
        if accelerator_type == AcceleratorType.Eyeriss:
            self.arch_components = glob(os.path.join(eyeriss_timeloop_dir, 'arch', 'components', '*.yaml'))
            self.constraint_files = glob(os.path.join(eyeriss_timeloop_dir, 'constraints', '*.yaml'))

        elif accelerator_type == AcceleratorType.Simba:
            self.arch_components = [os.path.join(simba_timeloop_dir, 'components.yaml')]
            self.constraint_files = [os.path.join(simba_timeloop_dir, 'architecture_constraints.yaml'),
                                     os.path.join(simba_timeloop_dir, 'mapspace_constraints.yaml')]

        else:
            raise NotImplementedError(f"Accelerator type {accelerator_type} is not supported")


class TimeloopProblem:
    """Utility class to handle and create a Timeloop-related workload
    """

    def __init__(self, name, problem_type, dimensions, problem_filepath):
        self.name = name
        self.dims = dimensions
        self.problem_filepath = problem_filepath
        self.config = None

        assert problem_type in ['Linear', 'Conv2d', 'AvgPool2d', 'MaxPool2d'], \
            f"Layer of type {problem_type} is not supported"
        self.problem_type = problem_type
        self.get_config()
        self.to_yaml()

    def get_config(self):
        """Gather the configuration of the problem/workload in dict format
        """
        if self.problem_type in ['Conv2d', 'Linear']:
            self.config_conv_layer()
        elif 'pool' in self.problem_type.lower():
            self.config_pool_layer()

    def adjust_dimension(self, dimension, value=None, adjust_by=None):
        """Change/Adjust the value of a given workload dimension
        """
        if value is not None:
            self.config['instance'][dimension] = int(max(1, value))
        elif adjust_by is not None:
            self.config['instance'][dimension] = int(max(1, self.config['instance'][dimension] * adjust_by))
        else:
            raise ValueError("To change a workload dimension, specify either the absolute value or relative change")

    def to_yaml(self, filepath=None):
        """Create a yaml description of the workload
        """
        if self.config is None:
            self.get_config()

        if filepath is None:
            assert self.problem_filepath is not None
            filepath = self.problem_filepath
        with open(filepath, 'w') as f:
            f.write(yaml.dump({'problem': self.config}))

    def config_conv_layer(self):
        """Create the configuration for a Convolutional-type layer
        """
        self.dims['Q'] = int((self.dims['Xi'] - self.dims['S'] + 2 * self.dims['Wpad']) / self.dims['Wstr']) + 1
        self.dims['P'] = int((self.dims['Yi'] - self.dims['R'] + 2 * self.dims['Hpad']) / self.dims['Hstr']) + 1
        dimensions = ['C', 'M', 'R', 'S', 'N', 'P', 'Q']
        coefficients = ['Wstride', 'Hstride', 'Wdilation', 'Hdilation']

        config = {}
        config['shape'] = {}
        config['shape']['name'] = self.name
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
                'default': self.dims['Hstr']
            },
            {
                'name': 'Wstride',
                'default': self.dims['Wstr']
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
                    [['R', 'Wdilation'], ['P', 'Wstride']],
                    [['S', 'Hdilation'], ['Q', 'Hstride']],
                ]
            },
            {
                'name': 'Outputs',
                'projection': [
                    [['N']],
                    [['M']],
                    [['Q']],
                    [['P']]
                ],
                'read-write': True
            }
        ]
        config['instance'] = {
            'C': self.dims['C'],
            'M': self.dims['K'],
            'R': self.dims['R'],
            'S': self.dims['S'],
            'N': self.dims['N'],
            'P': self.dims['P'],
            'Q': self.dims['Q']
        }

        self.config = config

    def config_pool_layer(self):
        """Create the configuration for a Pooling layer
        """
        raise NotImplementedError


class TimeloopArch:
    """Utility class to handle the architectural parameters of the accelerator
       when using timeloop
    """

    def __init__(self, accelerator_type, workdir, component_files):
        self.name = self.__class__.__name__
        self.workdir = workdir
        self.type = accelerator_type
        os.makedirs(workdir, exist_ok=True)
        for component_file in component_files:
            shutil.copy2(component_file, workdir)
        self.component_files = [os.path.join(workdir, os.path.basename(component_file))
                                for component_file in component_files]
        self.arch_filepath = os.path.join(workdir, 'architecture.yaml')

        # functions for specific accelerator types
        if accelerator_type == AcceleratorType.Eyeriss:
            self.get_default_params = self._get_default_params_eyeriss
            self.get_config = self._get_config_eyeriss
            self.adjust = self._adjust_eyeriss
            self.adjust_precision = self._adjust_precision_eyeriss

        elif accelerator_type == AcceleratorType.Simba:
            self.get_default_params = self._get_default_params_simba
            self.get_config = self._get_config_simba
            self.adjust = self._adjust_simba
            self.adjust_precision = self._adjust_precision_simba

        else:
            raise NotImplementedError(f"Accelerator type {accelerator_type} is not supported")

        # initialize dict with parameters
        self.get_default_params()
        self.init_params = deepcopy(self.params)
        self.get_config()
        self._sync_version()
        self.to_yaml(do_components=accelerator_type == AcceleratorType.Simba)

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

    def to_yaml(self, filepath=None, do_components=False):
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

            # reload the component files with minor changes (quoting string values in yaml)
            # if do_components:
            #     # use quotes only on 'nm' and 'ns' strings
            #     filter_fn = None if self.type != AcceleratorType.Simba else \
            #                 lambda entry: isinstance(entry, str) and ('ns' in entry or 'nm' in entry)
            #
            #     for component_file in self.component_files:
            #         with open(component_file, 'r') as stream:
            #             comp_cfg = yaml.safe_load(stream)
            #         force_quotes_on_str(comp_cfg, filter_fn=filter_fn)
            #         with open(component_file, 'w') as f:
            #             f.write(yaml.dump(comp_cfg, default_flow_style=False))

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

    def _adjust_eyeriss(self, accelerator_instance):
        """Adjust the parameter of the architecture based on the
           given accelerator instance
        """
        # NOTE: VERY important to change the PE array dimensions first,
        #       such that the memory width can adjust to the new dimensions
        #       and the change in precision
        self.adjust_pe_array(accelerator_instance.pe_array_x,
                             accelerator_instance.pe_array_y)
        self._adjust_memories_eyeriss(accelerator_instance.sram_size,
                                      accelerator_instance.ifmap_spad_size,
                                      accelerator_instance.weights_spad_size,
                                      accelerator_instance.psum_spad_size)
        self._adjust_precision_eyeriss(accelerator_instance.precision)

    def _adjust_memories_eyeriss(self, sram_size, ifmap_spad_size,
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

    def _adjust_precision_eyeriss(self, precision):
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

    def _get_default_params_eyeriss(self):
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

    def _get_config_eyeriss(self):
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

    ### Accelerator-specific functions: Simba ###

    def _adjust_simba(self, accelerator_instance):
        """Adjust the parameters of the Simba-like architecture based on the
           given accelerator instance
        """
        # NOTE: VERY important to change the PE array dimensions first,
        #       such that the memory width can adjust to the new dimensions
        #       and the change in precision
        self.adjust_pe_array(accelerator_instance.pe_array_x,
                             accelerator_instance.pe_array_y)
        self._adjust_memories_simba(accelerator_instance.sram_size,
                                    accelerator_instance.input_buffer_size,
                                    accelerator_instance.weight_buffer_size,
                                    accelerator_instance.accum_buffer_size)
        self._adjust_precision_simba(accelerator_instance.precision)

    def _adjust_memories_simba(self, sram_size, input_buffer_size, weight_buffer_size, accum_buffer_size):
        """Adjust the size of the Simba-related memory units
        """
        params = {
            'sram_depth': sram_size,
            'input_buffer_depth': input_buffer_size,
            'weight_buffer_depth': weight_buffer_size,
            'accum_buffer_depth': accum_buffer_size,
        }
        # NOTE: We also adjust the cluster-size of the weight register to be
        #       perfectly dividable by the meshX paremeter
        if self.params.weight_reg_cluster_size % (self.params.pe_array_x * self.params.pe_array_y) != 0:
            params['weight_reg_cluster_size'] = self.params.pe_array_x * self.params.pe_array_y

        self.adjust_params(params)

    def _adjust_precision_simba(self, precision):
        """Adjust the precision of a Simba-like architecture
        """
        if precision == self.params.mac_precision:
            return

        params = {
            'mac_precision': precision,
            'system_precision': precision,
            'dram_word_bits': precision,
            'sram_word_bits': precision,
            'input_buffer_word_bits': precision,
            'weight_buffer_word_bits': precision,
            'accum_buffer_precision': precision,
            'weight_reg_word_bits': precision,
            # adjust the width of the memories
            'dram_width': self.adjust_mem_width('dram',
                                                precision,
                                                self.params.dram_block_size,
                                                getattr(self.params, 'dram_cluster_size', 1)),
            'sram_width': self.adjust_mem_width('sram',
                                                precision,
                                                self.params.sram_block_size,
                                                getattr(self.params, 'sram_cluster_size', 1)),
            'input_buffer_width': self.adjust_mem_width('input_buffer',
                                                        precision,
                                                        self.params.input_buffer_block_size,
                                                        getattr(self.params, 'input_buffer_cluster_size', 1)),
            # Weight and accumulatio buffers do not have a 'width' parameter
            # 'weight_reg_sth': # TODO: not sure what to do here
        }
        self.adjust_params(params)

    def _get_default_params_simba(self):
        """Get the default parameters for all levels of a Simba-like architecture
        """
        self.params = SimpleNamespace()
        # system parameters
        self.params.system_precision = 8
        self.params.technology = '45nm'
        # DRAM
        self.params.dram_width = 64
        self.params.dram_block_size = 8
        self.params.dram_word_bits = 8
        # SRAM
        self.params.sram_depth = 2048
        self.params.sram_width = 256
        self.params.sram_word_bits = 8
        self.params.sram_block_size = 32
        self.params.sram_n_banks = 4
        self.params.sram_n_ports = 2
        # PE array
        self.params.pe_array_x = 4
        self.params.pe_array_y = 4
        # Internal buffers (inputs, weights, partial sums)
        self.params.input_buffer_depth = 128
        self.params.input_buffer_width = 64
        self.params.input_buffer_word_bits = 8
        self.params.input_buffer_block_size = 8
        self.params.weight_buffer_depth = 4096
        self.params.weight_buffer_word_bits = 8
        self.params.weight_buffer_block_size = 8
        self.params.weight_buffer_n_banks = 8
        self.params.accum_buffer_depth = 128
        self.params.accum_buffer_precision = 24
        # weight registers
        self.params.weight_reg_depth = 1
        self.params.weight_reg_word_bits = 8
        self.params.weight_reg_cluster_size = 64
        self.params.weight_reg_num_ports = 2
        # MAC
        self.params.mac_subclass = 'lmac'
        self.params.mac_precision = 8

    def _get_config_simba(self):
        """Write the architectural description of a Simba-like architecture in a dict format
        """
        config = {}
        config['version'] = TIMELOOP_ACCELERGY_VERSION

        level1 = {}
        level1['name'] = 'system'
        level1['attributes'] = {
            'datawidth': self.params.system_precision,
            'word-bits': self.params.system_precision,
            'technology': self.params.technology,
        }
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
        level2['name'] = 'simba'  # 'ws' in template file
        level2['local'] = [
            {
                'name': 'GlobalBuffer',
                'class': 'storage',
                'subclass': 'smartbuffer_SRAM',
                'attributes': {
                    'memory_depth': self.params.sram_depth,
                    'memory_width': self.params.sram_width,
                    'word-bits': self.params.sram_word_bits,
                    'block-size': self.params.sram_block_size,
                    'n_banks': self.params.sram_n_banks,
                    'nports': self.params.sram_n_ports,
                    'meshX': 1  # Could this be removed?
                }
            }
        ]

        level3 = {}
        level3['name'] = f'PE[0..{self.params.pe_array_x * self.params.pe_array_y - 1}]'
        level3['local'] = [
            {
                'name': 'PEInputBuffer',
                'class': 'storage',
                'subclass': 'smartbuffer_RF',
                'attributes': {
                    'memory_depth': self.params.input_buffer_depth,
                    'memory_width': self.params.input_buffer_width,
                    'word-bits': self.params.input_buffer_word_bits,
                    'block-size': self.params.input_buffer_block_size,
                    'meshX': self.params.pe_array_x * self.params.pe_array_y
                }
            },
            {
                'name': f'PEWeightBuffer[0..{self.params.pe_array_x - 1}]',
                'class': 'storage',
                'subclass': 'smartbuffer_RF',
                'attributes': {
                    'memory_depth': self.params.weight_buffer_depth,
                    'word-bits': self.params.weight_buffer_word_bits,
                    'meshX': self.params.pe_array_x * self.params.pe_array_y,
                    'block-size': self.params.weight_buffer_block_size,
                    'n_banks': self.params.weight_buffer_n_banks
                }
            },
            {
                'name': f'PEAccuBuffer[0..{self.params.pe_array_x - 1}]',
                'class': 'storage',
                'subclass': 'smartbuffer_RF',
                'attributes': {
                    'memory_depth': self.params.accum_buffer_depth,
                    'word-bits': self.params.accum_buffer_precision,
                    'datawidth': self.params.accum_buffer_precision,
                    'meshX': self.params.pe_array_x * self.params.pe_array_y,
                }
            },
            {
                'name': f'PEWeightRegs[0..{self.params.pe_array_x * self.params.pe_array_y - 1}]',
                'class': 'storage',
                'subclass': 'reg_storage',
                'attributes': {
                    'memory_depth': self.params.weight_reg_depth,
                    'word-bits': self.params.weight_reg_word_bits,
                    'cluster-size': self.params.weight_reg_cluster_size,
                    'num-ports': self.params.weight_reg_num_ports,
                    'meshX': self.params.pe_array_x * self.params.pe_array_y,
                }
            },
            {
                'name': f'LMAC[0..{self.params.pe_array_x * self.params.pe_array_y - 1}]',
                'class': 'compute',
                'subclass': self.params.mac_subclass,
                'attributes': {
                    'datawidth': self.params.mac_precision,
                    'meshX': self.params.pe_array_x * self.params.pe_array_y
                }
            }
        ]

        level2['subtree'] = [level3]
        level1['subtree'] = [level2]
        config['subtree'] = [level1]

        self.config = config


class TimeloopMapper:
    """Utility wrapper class fot the mapping optimizer
    """

    def __init__(self, mapper_file):
        self.mapper_filepath = mapper_file
        self.get_params()
        self.get_config()
        self.to_yaml()

    def get_params(self):
        """Collect the default configuration parameters of the mapper 
        """
        self.params = SimpleNamespace()
        self.params.optimization_metrics = ['edp']
        self.params.live_status = False
        self.params.num_threads = 8
        self.params.timeout = 15000
        self.params.victory_condition = 500
        self.params.algorithm = 'random-pruned'
        self.params.max_permutations_per_if_visit = 16

    def get_config(self):
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

    def to_yaml(self, filepath=None):
        """Write the configuration of the mapper to a yaml file
        """
        if filepath is None:
            assert self.mapper_filepath is not None
            filepath = self.mapper_filepath

        with open(filepath, 'w') as f:
            f.write(yaml.dump({'mapper': self.config}))

    def adjust_param(self, param_name, value):
        """Generic function to override a parameter value
        """
        assert hasattr(self.params, param_name), f'{param_name} is not a valid parameter'
        setattr(self.params, param_name, value)
        # update the configuration with new parameters
        self.get_config()

def timeloop_execution(timeloop_wrapper: TimeloopWrapper, problem_name: str) -> TimeloopStats:
    logger.debug(f"\t\t\tEvaluating layer/problem: {problem_name}")
    timeloop_wrapper.run(problem_name)
    results = timeloop_wrapper.get_results(problem_name)
    logger.debug(f"\t\t\tLayer-wise results: "
                 f"energy={results.energy:.3e}, latency={results.cycles:.3e}, edp={results.edp:.3e}")
    timeloop_wrapper.cleanup(problem_name)
    return results

def timeloop_execution_mock(timeloop_wrapper: TimeloopWrapper, problem_name: str) -> TimeloopStats:
    energy_base=6.608e+04
    latency_base=3.303e+07
    area_base = 12.715
    energy = uniform(energy_base - 1e4, energy_base + 1e4)
    latency = ceil(uniform(latency_base - 1e7, latency_base + 1e7))
    area = uniform(area_base - 1, area_base + 1)
    return TimeloopStats(gflops=None, utilization=None, energy=energy, cycles=latency,
                            edp=energy * latency, area=area)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    accel_type = AcceleratorType.Eyeriss
    DO_EXPLORATION = False
    # prob_name = 'resnet18__layer0_conv1'
    # prob_name = 'vgg13__layer0_features.0'

    tw = TimeloopWrapper(accel_type, project_dir + '/test_tl')

    # prob_fp = os.path.join(tw.workload_dir, prob_name + '.yaml')
    # have the file already in the test_tl/yamls/ directory
    # shutil.copyfile(project_dir + f'/test_problems/{prob_name}.yaml', prob_fp)

    # tw.workloads[prob_name] = SimpleNamespace()
    # tw.workloads[prob_name].problem_filepath = prob_fp

    model_config = SimpleNamespace(arch='resnet50', dataset='cifar100', batch_size=1, gpus=0, cpu=False,
                                   load_serialized=False, pretrained=True, resumed_checkpoint_path=None, optimizer_type=
                                   OptimizerType.Adam, print_frequency=100, verbose=True)
    prob_name = "conv_test"
    net_wrapper = TorchNetworkWrapper(model_config)
    dataset = crimson_magick.cifar_zoo.get_test_loader(Cifar.CIFAR100)
    net_wrapper.run_summary(dataset)
    to_serialize = net_wrapper.summary['model.conv1']
    tw.init_problem(prob_name, "Conv2d", to_serialize.dimensions)


    from src.accelerator_cfg import AcceleratorProfile

    accel_cfg = AcceleratorProfile(accel_type)

    accel = accel_cfg.state(pe_array_x=accel_cfg.pe_array_x,
                            pe_array_y=accel_cfg.pe_array_y,
                            precision=4,
                            sram_size=accel_cfg.sram_size,
                            ifmap_spad_size=accel_cfg.ifmap_spad_size,
                            weights_spad_size=accel_cfg.weights_spad_size,
                            psum_spad_size=accel_cfg.psum_spad_size)


    logger.info(f'Accelerator:{accel}')
    tw.adjust_architecture(accel, adjust_components=True)
    tw.cleanup(prob_name)
    p = tw.run(prob_name)
    results = tw.get_results(prob_name)
    print(results._asdict())


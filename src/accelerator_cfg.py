from enum import Enum
from collections import namedtuple


class AcceleratorType(Enum):
    Eyeriss = 1
    Simba = 2
    DianNao = 3
    TPU = 4


EyerissAcceleratorState = namedtuple("EyerissAcceleratorState",
                                     ['pe_array_x', 'pe_array_y',
                                      'precision', 'sram_size',
                                      'ifmap_spad_size', 'weights_spad_size', 'psum_spad_size'])


SimbaAcceleratorState = namedtuple("SimbaAcceleratorState",
                                   ['pe_array_x', 'pe_array_y',
                                    'precision', 'sram_size',
                                    'input_buffer_size', 'weight_buffer_size', 'accum_buffer_size'])


class AcceleratorProfile:
    """General characteristics from the given accelerator.

        Attributes that must be defined:
            type: definition of the specific accelerator (AcceleratorType)
            state: the characteristics changed during a DSE. Precision must be included
            design_space: a dict with the same keys as state, which contains
                          the possible options for each field in state
            pe_array_x: a scalar with the length (x) dimension of the innermost PE array
            pe_array_y: same as pe_array_x, for the PE array width (y) dimension 
    """
    def __init__(self, accelerator_type):
        self.type = accelerator_type

        if accelerator_type == AcceleratorType.Eyeriss:
            self.state = EyerissAcceleratorState

            # accelerator parameters, according to https://ieeexplore.ieee.org/document/7738524
            self.technology = 65                                # in nm (LP 1P9M)
            self.chip_size = [4.0, 4.0]                         # in mm
            self.core_area = [3.5, 3.5]                         # in mm
            self.gate_count = 1176000                           # 2-input NAND
            self.core_supply_voltage = [0.82, 1.17]             # in V
            self.io_supply_voltage = 1.8                        # in V
            self.core_clock_rate = 250 * 1e6                    # in Hz (100 - 250 MHz allowed)
            self.link_clock_rate = 90 * 1e6                     # in Hz
            self.peak_throughput = [16.8, 42.0]                 # in GMACs

            # useful for exploration
            self.pe_array_x = 14
            self.pe_array_y = 12
            self.num_pes = self.pe_array_x * self.pe_array_y
            self.precision_weights = 16                         # in bits, fixed-point
            self.precision_activations = 16                     # in bits, fixed-point
            self.sram_size = 108000                             # in bytes
            self.ifmap_spad_size = 24                           # in bytes (12b x 16b)
            self.weights_spad_size = 448                        # in bytes (224b x 16b)
            self.psum_spad_size = 48                            # in bytes (24b x 16b)
            self.dataflow = 'row_stationary'
            self.num_banks = 25
            self.sram_per_bank = 4000                           # in bytes (512b x 64b)
            self.dram_precision = 64                            # in bits

            # design space parameters
            pe_array_x_options = pe_array_y_options = [8, 10, 12, 14, 16, 20, 25]
            ifmap_spad_size_options = [8, 12, 16, 24, 32, 40, 48, 64]
            weights_spad_size_options = [256, 320, 384, 448, 512, 576, 640]
            psum_spad_size_options = [16, 24, 32, 40, 48, 64, 80, 96]
            sram_size_options = [45000, 60000, 80000, 108000, 120000, 140000, 180000]
            precision_options = [8, 16, 32]

            # prepare dictionary with design space parameters, according to the self.state class
            self.design_space = {
                'pe_array_x': pe_array_x_options,
                'pe_array_y': pe_array_y_options,
                'precision': precision_options,
                'sram_size': sram_size_options,
                'ifmap_spad_size': ifmap_spad_size_options,
                'weights_spad_size': weights_spad_size_options,
                'psum_spad_size': psum_spad_size_options
            }

        elif accelerator_type == AcceleratorType.Simba:
            self.state = SimbaAcceleratorState

            # accelerator parameters, according to https://dl.acm.org/doi/10.1145/3352460.3358302
            self.num_chiplets = 36
            self.package_size = [47.5, 47.5]                    # in mm
            self.core_voltage = [0.52, 1.1]                     # in V, [min, max]
            self.pe_clock_frequency = [0.48, 1.8]               # in GHz, [min, max]
            self.nop_interc_bandwidth = 100                     # in GB/s/chiplet
            self.nop_interc_latency = 20                        # in ns/hop
            self.nop_interc_energy = [0.82, 1.75]               # in pJ/bit, approximate range
            self.technology = 16                                # in nm (FinFET)
            self.chiplet_voltage = [0.42, 1.2]                  # in V
            self.chiplet_pe_clock_frequency = [0.16, 2.0]       # in GHz
            self.routers_per_global_pe = 3
            self.chiplet_noc_interc_bandwidth = 68              # in GB/s/PE
            self.chiplet_noc_interc_latency = 10                # in ns/hop
            
            # useful for exploration
            self.pe_array_x = 4
            self.pe_array_y = 4
            self.num_pes = self.pe_array_x * self.pe_array_y
            self.precision_weights = 8                          # in bits, fixed-point
            self.precision_accumulator = 24                     # in bits, fixed-point
            self.dataflow = 'weight_stationary'
            self.dram_precision = 8                             # in bits
            self.sram_size = 64000                              # in bytes
            self.input_buffer_size = 8000                       # in bytes
            self.weight_buffer_size = 32000                     # in bytes
            self.accum_buffer_size = 3000                       # in bytes
            self.vector_mac_width = 8
            self.num_vector_macs = 8

            # design space parameters
            pe_array_x_options = pe_array_y_options = [2, 4, 6, 8, 10, 12, 16, 20]
            precision_options = [8, 16, 32]
            sram_size_options = [24000, 32000, 40000, 48000, 56000, 64000, 80000, 96000]
            input_buffer_size_options = [800, 2000, 4000, 8000, 12000, 20000, 30000, 50000]
            weight_buffer_size_options = [4000, 8000, 16000, 24000, 32000, 48000, 64000, 80000]
            accum_buffer_size_options = [500, 1000, 3000, 5000, 8000, 12000]
            # prepare dictionary with design space parameters, according to the self.state class
            self.design_space = {
                'pe_array_x': pe_array_x_options,
                'pe_array_y': pe_array_y_options,
                'precision': precision_options,
                'sram_size': sram_size_options,
                'input_buffer_size': input_buffer_size_options,
                'weight_buffer_size': weight_buffer_size_options,
                'accum_buffer_size': accum_buffer_size_options,
            }

        else:
            raise NotImplementedError(f"Accelerator type {accelerator_type} is not supported")

    def __repr__(self) -> str:
        return f'{self.type}->{self.precision_weights}b'

    def __str__(self) -> str:
        return f'{self.type}->{self.precision_weights}b'


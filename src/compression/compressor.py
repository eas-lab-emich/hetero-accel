import os
import torch
import logging
import numpy as np
from types import SimpleNamespace
from copy import deepcopy
from src import project_dir
from src.compression.quantization import Quantizer
from src.net_wrapper import TorchNetworkWrapper
from src.utils import compute_model_statistics
from src.accelerator_cfg import AcceleratorProfile
from src.compression.pruning import Pruner
from src.worker.timeloop import TimeloopWrapper


logger = logging.getLogger(__name__)


class PruningQuantizationCompressor(TorchNetworkWrapper):
    """Class to handle all the compression-related actions
    """
    def __init__(self, args, data_loaders, model=None):
        super().__init__(args, model)

        self.original_model = deepcopy(self.model)
        self.train_loader, self.valid_loader, self.test_loader = data_loaders
        self.layers_to_compress = [name for name, module in self.model.named_modules()
                                   if isinstance(module, args.layer_type_whitelist)]
        # logger.debug(f"{self.model.arch}: Layers to compress: {self.layers_to_compress}")

        # pruner and quantizer for compression
        self.pruner = Pruner(self.pruning_group_type, self.layers_to_compress,
                             eridanus_window_w=self.accelerator_cfg.pe_array_x,
                             eridanus_window_h=self.accelerator_cfg.pe_array_y)
        self.quantizer = Quantizer()
        self.timeloop_wrapper = None

    @classmethod
    def from_args(cls, args, data_loaders, model=None):
        compression_args = SimpleNamespace(logdir=args.logdir,
                                           pruning_high=args.pruning_high,
                                           pruning_low=args.pruning_low,
                                           quant_high=args.quant_high,
                                           quant_low=args.quant_low,
                                           layer_type_whitelist=(torch.nn.Conv2d,),
                                           pruning_group_type=args.pruning_group_type,
                                           accelerator_cfg=AcceleratorProfile(args.accelerator_arch_type),
                                           # DNN args for inheritance from TorchNetworkWrapper
                                           gpus=args.gpus,
                                           cpu=args.cpu,
                                           verbose=args.model_verbose)
        return cls(compression_args, data_loaders, model)

    def init_timeloop_wrapper(self):
        # timeloop wrapper to execute mapping searches and energy/area estimation
        tl_workdir = os.path.join(self.logdir, f'timeloop_compression_{self.model.arch}')
        self.timeloop_wrapper = TimeloopWrapper(self.accelerator_cfg.type, tl_workdir)

    def reset(self):
        del self.model
        self.model = deepcopy(self.original_model)

    def quantize(self, q_bits):
        self.prune_and_quantize(None, q_bits)

    def prune(self, pruning_ratio):
        self.prune_and_quantize(pruning_ratio, None)

    def prune_and_quantize(self, pruning_ratio=None, q_bits=None):
        if pruning_ratio is not None and pruning_ratio != 0.0:
            assert self.pruning_low <= pruning_ratio <= self.pruning_high
            self.pruner.prune(self.model, pruning_ratio)
        if q_bits is not None:
            # NOTE: Assuming no accuracy degradation INT8, so quantization is skipped
            if q_bits > max(self.quant_high, 8):
                return

            # WTF are you doing validating app config inside the quantize method?
            # assert self.quant_low <= q_bits <= self.quant_high
            self.model = self.quantizer.quantize(self.model, q_bits, self.test_loader.dataset)


    def translate_pruning_action(self, pruning_action):
        pruning_action = pruning_action * (self.pruning_high - self.pruning_low) + self.pruning_low
        return np.round(pruning_action, 2).astype(float)

    def translate_quant_action(self, quant_action):
        quant_action = quant_action * (self.quant_high - self.quant_low) + self.quant_low
        return int(np.round(quant_action, 0))

    def compute_model_statistics(self):
        return compute_model_statistics(self.model, self.layers_to_compress)

    def compute_accelerator_statistics(self, init=False):
        """Calculate the hardware-related efficiency metrics of the accelerator
           using Timeloop/Accelergy
        """
        if self.timeloop_wrapper is None:
            self.init_timeloop_wrapper()

        total_area = total_latency = total_power = total_energy = 0
        _, layer_stats = self.compute_model_statistics()
        # itearate over all layers
        layer_idx = 0
        for name, module in self.model.named_modules():
            if name not in self.layers_to_compress:
                continue

            # if specified, create the Timeloop workload files for each layer
            if init:
                problem_filepath = os.path.join(self.timeloop_wrapper.workload_dir, f'layer{layer_idx}_{name}.yaml')
                self.timeloop_wrapper.init_problem(name,
                                                   self.summary[name].layer_type,
                                                   self.summary[name].dimensions,
                                                   problem_filepath)

            # modifying the problem file to adjust for pruning
            self.timeloop_wrapper.adjust_problem_dimension(name, 'M',
                                                           adjust_by=(1 - layer_stats[name]['sparsity_filters']))
            self.timeloop_wrapper.adjust_problem_dimension(name, 'C',
                                                           adjust_by=(1 - layer_stats[name]['sparsity_channels']))
            # modifying the architecture to adjust for quantization
            bits = getattr(getattr(module, 'quant_metadata', None), 'bits', 32)
            self.timeloop_wrapper.adjust_precision(bits)

            # execute timeloop
            self.timeloop_wrapper.run(name)
            # gather results from simulation
            results = self.timeloop_wrapper.get_results(project_dir)
            logger.debug(f'Timeloop results for layer {layer_idx} ({name}): '
                         f'GFLOPS={results.gflops}, Utilization={results.utilization}, Cycles={results.cycles}, '
                         f'Energy={results.energy}, EDP={results.edp}, Area={results.area}')

            total_latency += results.cycles
            total_energy += results.energy

            layer_idx += 1
            
        total_area = results.area
        return total_area, total_latency, total_power, total_energy

    def train(self, epochs):
        return super().train(epochs, self.train_loader)

    def validate(self):
        return super().validate(self.valid_loader)

    def test(self, use_quant=False):
        return super().test(self.test_loader, use_quant)


import gc
import logging
from copy import deepcopy
from types import SimpleNamespace

import torch

from src.accelerator_cfg import AcceleratorProfile
from src.compression.quantization import quantize
from src.net_wrapper import TorchNetworkWrapper
from src.utils import compute_model_statistics

logger = logging.getLogger(__name__)


class Compressor(TorchNetworkWrapper):
    """Class to handle all the compression-related actions
    """

    def __init__(self, args, data_loaders, model=None):
        super().__init__(args,  deepcopy(model))
        self.train_loader, self.valid_loader, self.test_loader = data_loaders
        self.layers_to_compress = [name for name, module in self.model.named_modules()
                                   if isinstance(module, args.layer_type_whitelist)]

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


    def quantize(self, q_bits):
        if q_bits is not None:
            # NOTE: Assuming no accuracy degradation INT8, so quantization is skipped
            if q_bits > max(self.quant_high, 8):
                return
            self.model = quantize(self.model, q_bits, self.test_loader.dataset)

    def compute_model_statistics(self):
        return compute_model_statistics(self.model, self.layers_to_compress)

    def train(self, epochs):
        return super().train(epochs, self.train_loader)

    def validate(self):
        return super().validate(self.valid_loader)

    def test(self, use_quant=False):
        return super().test(self.test_loader, use_quant)

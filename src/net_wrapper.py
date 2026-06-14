import torch
import logging
import re
from types import SimpleNamespace

from crimson_magick.cifar_zoo import Cifar, Arch, load_model

from src import pretrained_checkpoint_paths, dataset_dirs
from src.utils import weight_init, load_checkpoint, model_summary, save_checkpoint
from src.meter import *
from src.train_test import validate
from src.models import create_model, DNNType
from src.args import OptimizerType


logger = logging.getLogger(__name__)


class TorchNetworkWrapper:
    """DNN wrapper with training/testing functionality
    """
    def __init__(self, args, model=None):
        self.print_frequency = None
        self.verbose = None
        for name, value in vars(args).items():
            setattr(self, name, value)

        self.config_compute_device()
        self.model = model
        if model is None:
            self.init_model()
        self.run_summary()

        # loss function and accuracy meter
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.accuracy_meter = ImageClassificationMeter()

        # optimizer
        if args.optimizer_type == OptimizerType.Adam:
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=0.01, weight_decay=1e-4)
        elif args.optimizer_type == OptimizerType.SGD:
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=0.01, weight_decay=1e-4)

    @classmethod
    def from_args(cls, args):
        args = SimpleNamespace(arch=args.arch,
                               dataset=args.dataset,
                               batch_size=args.batch_size,
                               gpus=args.gpus,
                               cpu=args.cpu,
                               load_serialized=args.load_serialized,
                               pretrained=args.pretrained,
                               resumed_checkpoint_path=args.resumed_checkpoint_path,
                               optimizer_type=args.optimizer_type,
                               print_frequency=args.batch_print_frequency,
                               verbose=args.model_verbose,
                               logdir=args.logdir,
                               )
        return cls(args)

    def config_compute_device(self):
        if self.cpu or not torch.cuda.is_available():
            # Set GPU index to -1 if using CPU
            self.device = 'cpu'
            self.gpus = -1
        else:
            self.device = 'cuda'
            self.gpus = [self.gpus] if not isinstance(self.gpus, list) else self.gpus
            if self.gpus is not None:
                if len(self.gpus) == 1 and re.search('[ ,]', str(self.gpus[0])):
                    self.gpus = [int(device_idx.strip()) for device_idx in re.split('[ ,]', self.gpus[0])]
                else:
                    self.gpus = [int(device_idx) for device_idx in self.gpus]

                available_gpus = torch.cuda.device_count()
                for device_idx in self.gpus:
                    if device_idx >= available_gpus:
                        raise ValueError(f'ERROR: GPU device ID {device_idx} requested, but only {available_gpus} devices available')
                # Set default device in case the first one on the list != 0
                torch.cuda.set_device(self.gpus[0])

    def init_model(self):
        """Initialize the DNN architecture with the option of pretrained weights
        """
        assert not (self.pretrained and self.resumed_checkpoint_path is not None), "Only pretrained models are needed! " \
            "Specify either the '--pretrained' or '--resumed-checkpoint-path' arguments"
        if 'cifar' in self.dataset:
            arch: Arch = Arch.MOBILENETV1 if self.arch == "mobilenet" else Arch[self.arch.upper()]
            cifar: Cifar = Cifar[self.dataset.upper()]
            self.model = load_model(arch, cifar, self.device)
            for name, module in self.model.named_modules():
                module.full_name = name
            return

        if self.pretrained and self.arch + '_' + self.dataset in pretrained_checkpoint_paths:
            self.pretrained = False
            self.resumed_checkpoint_path = pretrained_checkpoint_paths[self.arch + '_' + self.dataset]

        self.model = create_model(self.arch,
                                  self.dataset,
                                  self.batch_size,
                                  self.pretrained,
                                  parallel=not self.load_serialized,
                                  device_ids=self.gpus,)
                                  #verbose=self.verbose)

        if not self.pretrained:
            self.model.apply(weight_init)

        if self.resumed_checkpoint_path is not None:
            self.model, _ = load_checkpoint(
                self.model,
                self.resumed_checkpoint_path,
                model_device=self.device,
                to_cpu=self.device == 'cpu',
                #verbose=self.verbose
            )

    def run_summary(self, data_loader=None):
        """Create a summary for the DNN model
        """
        if data_loader is None and getattr(self.model, 'input_shape', None) is None:
            self.summary = self.num_layers = None
        else:
            dummy_input = next(iter(data_loader))[0] if data_loader else None
            if dummy_input is not None:
                dummy_input = dummy_input.to(self.device)
            self.summary = model_summary(self.model, dummy_input)
            self.num_layers = len(self.summary)


    def log_model(self, test_loader, tb_logger):
        """Record the graph of the model and its various statistics using TensorBoard
        """
        # save the model graph using Tensorboard
        inputs, labels = next(iter(test_loader))
        tb_logger.add_graph(self.model, inputs)

        # histograms of model parameters and their gradients
        for name, param in self.model.named_parameters():
            tb_logger.add_histogram(name, param, 0)
            if param.grad is not None:
                tb_logger.add_histogram(name + '.grad', param.grad, 0)

    @property
    def accuracy_metrics(self):
        """Gather accuracy metrics
        """
        return self.accuracy_meter.metrics 

    def validate(self, valid_loader):
        """Run inference on the validation set
        """
        logger.debug(f"Running inference on validation set")
        self.accuracy_meter.reset()
        return self._run_inference(valid_loader)

    def test(self, test_loader, use_quant=False):
        """Run inference on the test set
        """
        logger.debug(f"Running inference on test set")
        self.accuracy_meter.reset()
        return self._run_inference(test_loader, use_quant)

    def _run_inference(self, data_loader, use_quant=False):
        """Wrapper function for running inerence with a give data loader
        """
        accuracy_metrics = validate(valid_loader=data_loader,
                                    model=self.model,
                                    criterion=self.criterion,
                                    accuracy_meter=self.accuracy_meter,
                                    epoch=0,
                                    verbose=self.verbose,
                                    print_frequency=self.print_frequency,
                                    use_quant=use_quant,
                                    device=self.device)
        return accuracy_metrics

    def save_model(self, name=None, episode=None, is_best=False, verbose=True):
        """Save the current version of the model
        """
        save_checkpoint(arch=self.model.arch, model=self.model,
                        epoch=episode, is_best=is_best, verbose=verbose,
                        name=name, savedir=self.logdir)


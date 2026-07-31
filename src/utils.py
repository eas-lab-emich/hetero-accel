"""Utility functions"""
import logging
import logging.config
import argparse
import yaml
import os
import sys
import re
import numpy as np
import pandas as pd
import torch
import random
import subprocess as sp
from numbers import Number
from enum import Enum
from datetime import datetime
from errno import ENOENT
from numbers import Number
from copy import deepcopy
from collections import OrderedDict
from types import SimpleNamespace
from glob import glob
from tabulate import tabulate
from time import time
from src import project_dir, eyeriss_timeloop_dir
from src.args import app_args, workload_args, compression_args, accel_args, \
                     simanneal_args, check_args, baseline_args, sota_args, partition_args
from src.args import ModelSummaryType


__all__ = [
    'env_cfg', 'logging_cfg', 'cfg_from_yaml', 'set_deterministic',
    'load_checkpoint', 'save_checkpoint', 'weight_init', 'transform_model',
    'log_training_progress', 'perfect_divisors', 'get_contents_table',
    'force_quotes_on_str',
    'get_sparsity', 'compute_model_statistics', 'get_dummy_input', 'model_summary',
    'handle_model_subapps',
    'iou', 'iou_wh', 'nms'
]

logger = logging.getLogger(__name__)


def env_cfg():
    """Configure the environment to run the optimization"""
    parser = argparse.ArgumentParser("Hetero-Accel")
    parser = app_args(parser)
    parser = workload_args(parser)
    parser = compression_args(parser)
    parser = accel_args(parser)
    parser = simanneal_args(parser)
    parser = baseline_args(parser)
    parser = sota_args(parser)
    parser = partition_args(parser)
    # parse command arguments 
    args = parser.parse_args()
    # NOTE: adding the layer whitelist here, maybe somewhere else would be a better place
    args.layer_type_whitelist = (torch.nn.Conv2d, torch.nn.Linear)

    # overwrite if a yaml configuration file is given
    assert args.yaml_cfg_file is not None, "Specify a yaml file to configure the experiment"
    cfg_from_yaml(args, args.yaml_cfg_file)

    # check if there were errors in the argument parsing
    check_args(args)

    # configure logging 
    logging_cfg(args)

    # setup deterministic execution
    if args.deterministic:
        set_deterministic(args.global_seed)

    # fix deepcopy recursion problem by increasing the limit, if necessary
    if sys.getrecursionlimit() < 10000:
        sys.setrecursionlimit(10000)

    return args


def cfg_from_yaml(args, cfg_yaml_file):
    """Configure environment based on arguments from a yaml file
    """
    def replace_arg(name, value):
        # special handling for Enum type of arguments
        if isinstance(getattr(args, name, None), Enum) and value is not None:
            assert isinstance(value, str)
            value = next(entry.value for entry in getattr(args, name).__class__
                         if entry.name.lower() == value)
            value = getattr(args, name).__class__(value)
        # special handling for lists (nargs='+/*/?')
        elif isinstance(getattr(args, name, None), list) and value is not None:
            value = [value] if not isinstance(value, list) else value
        setattr(args, name, value)

    # read configuration file
    with open(cfg_yaml_file, 'r') as stream:
        yaml_dict = yaml.safe_load(stream)

    # inspect all arguments
    for name, value in yaml_dict.items():
        # we assume a two-level nested dictionary
        if isinstance(value, dict):
            # usually this branch gets executed
            for _name, _value in value.items():
                replace_arg(_name, _value)
        else:
            # this is rarely executed
            replace_arg(_name, _value)


def logging_cfg(args):
    """Configure logging for entire framework"""
    if not os.path.exists(os.path.join(project_dir, 'logs')):
        os.makedirs(os.path.join(project_dir, 'logs'))

    # set the name of the log file and directory
    timestr = datetime.utcnow().strftime("%Y.%m.%d-%H.%M.%S.%f")[:-3]
    exp_full_name = timestr if args.name is None else args.name + '___' + timestr
    logdir = os.path.join(project_dir, 'logs', exp_full_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # use the logging config file
    log_filename = os.path.join(logdir, exp_full_name + '.log')
    logging.config.fileConfig(
        os.path.join(project_dir, 'logging.conf'),
        disable_existing_loggers=False,
        defaults={
            'main_log_filename': f'{project_dir}/logs/out.log',
            'all_log_filename': log_filename,
        }
    )

    # disable logging for specific modules
    for module_name in ['absl', 'matplotlib', 'PIL']:
        logging.getLogger(module_name).setLevel(logging.WARNING)

    # save the experiment logging directory and file as root logger attributes
    logging.getLogger().logdir = logdir
    logging.getLogger().log_filename = log_filename

    # first messages logging the command executed for this experiment
    logger.info('Log file for this run: ' + os.path.realpath(log_filename))
    logger.debug("Command line: {}".format(" ".join(sys.argv)))
    arguments = {argument: getattr(args, argument) for argument in dir(args)
                 if not callable(getattr(args, argument)) and not argument.startswith('__')}
    logger.debug(f"Arguments: {arguments}")

    # Create a symbollic link to the last log file created (for easier access)
    try:
        os.unlink("latest_log_file")
    except FileNotFoundError:
        pass
    try:
        os.unlink("latest_log_dir")
    except FileNotFoundError:
        pass
    try:
        os.symlink(logdir, "latest_log_dir")
        os.symlink(log_filename, "latest_log_file")
    except OSError:
        logger.debug("Failed to create symlinks to latest logs")


def set_deterministic(seed):
    """Try to configure the system for reproducible results.
       Seed the PRNG for the CPU, Cuda, numpy and Python
    """
    logger.debug('Deterministic configuration was invoked')
    if seed is None:
        seed = 123
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_checkpoint(model, chkpt_path=None, model_device='cuda', to_cpu=False, strict=False, verbose=True):
    """Load a pytorch training checkpoint.
    """
    def _load_optimizer():
        """Initialize optimizer with model parameters and load src_state_dict"""
        try:
            cls, src_state_dict = checkpoint['optimizer_type'], checkpoint['optimizer_state_dict']
            # Initialize the dest_optimizer with a dummy learning rate,
            # this is required to support SGD.__init__()
            dest_optimizer = cls(model.parameters(), lr=1)
            dest_optimizer.load_state_dict(src_state_dict)
            logger.info('Optimizer of type {type} was loaded from checkpoint'.format(
                type=type(dest_optimizer)))
            optimizer_param_groups = dest_optimizer.state_dict()['param_groups']
            logger.debug('Optimizer Args: {}'.format(
                dict((k, v) for k, v in optimizer_param_groups[0].items()
                     if k != 'params')))
            return dest_optimizer
        except KeyError:
            # Older checkpoints do support optimizer loading: They either had an 'optimizer' field
            # (different name) which was not used during the load, or they didn't even checkpoint
            # the optimizer.
            if verbose:
                logger.debug('Optimizer could not be loaded from checkpoint.')
            return None

    chkpt_file = os.path.expanduser(chkpt_path)
    if not os.path.isfile(chkpt_file):
        raise IOError(ENOENT, 'Could not find a checkpoint file at', chkpt_file)

    if verbose:
        logger.info(f"=> loading checkpoint {chkpt_file}")
    checkpoint = torch.load(chkpt_file, map_location=lambda storage, loc: storage)

    # log the contents of the checkpoint
    if verbose:
        logger.info('=> Checkpoint contents: \n{}\n'.format(get_contents_table(checkpoint)))
        if 'extras' in checkpoint:
            logger.info("=> Checkpoint['extras'] contents:\n{}\n".format(get_contents_table(checkpoint['extras'])))

    # load parameters from checkpoint
    if 'state_dict' not in checkpoint:
        # workaround for loading checkpoint from cifar10_100_playground
        if 'net' not in checkpoint:
            raise ValueError("Checkpoint must contain the model parameters under the key 'state_dict'")
        else:
            state_dict = deepcopy(checkpoint.get('net'))
    else:
        state_dict = deepcopy(checkpoint.get('state_dict'))

    new_state_dict = OrderedDict()
    for name, param in state_dict.items():
        new_name = name
        # in case of the checkpoint param names having the DataParallel 'module' prefix
        if to_cpu:
            new_name = re.sub('^module[.]', '', new_name)
        # in case the current model has a 'wrapped_module' but the checkpoint does not
        if re.sub('(weight|bias)', r'wrapped_module.\1', new_name) in model.state_dict():
            new_name = re.sub('(weight|bias)', r'wrapped_module.\1', new_name)
        # in case the current model does not contain a 'wrapped_module', load the parameter without the wrapper
        if 'wrapped_module' in new_name and new_name.replace('.wrapped_module', '') in model.state_dict():
            new_name = new_name.replace('.wrapped_module', '')

        new_state_dict[new_name] = param

    # save the newly formed state_dict
    checkpoint['state_dict'] = new_state_dict

    # check for inconsistencies in the new parameters
    anomalous_keys = model.load_state_dict(checkpoint['state_dict'], strict)
    if anomalous_keys:
        missing_keys, unexpected_keys = anomalous_keys
        if verbose:
            logger.debug(f'Missing keys: {missing_keys}')
            logger.debug(f'Unexpected keys: {unexpected_keys}')

        if unexpected_keys:
            if verbose:
                logger.warning(f"Warning: the loaded checkpoint ({chkpt_file}) contains {len(unexpected_keys)} "
                                  f"unexpected state keys")

            # Some masks may have been loaded and characterized as unexpected keys (not recongized in the
            #  initialized model). In that case, register them as buffers to the correct module
            for key in unexpected_keys:
                if 'mask' in key:
                    mod_name = key.replace('.mask', '')
                    if to_cpu:
                        mod_name = re.sub('^module[.]', '', mod_name)
                    module = dict(model.named_modules()).get(mod_name)
                    module.register_buffer('mask', new_state_dict[key])

        if missing_keys:
            raise ValueError(f"The loaded checkpoint ({chkpt_file}) is missing {len(missing_keys)} state keys")

    # load pruning/quantization metadata
    if 'quant_metadata' in checkpoint:
        for module_name, module in model.named_modules():
            if module_name in checkpoint['quant_metadata']:
                setattr(module, 'quant_metadata', checkpoint['quant_metadata'][module_name])
    if 'pruning_metadata' in checkpoint:
        for module_name, module in model.named_modules():
            if module_name in checkpoint['pruning_metadata']:
                setattr(module, 'pruning_metadata', checkpoint['pruning_metadata'][module_name])

    # set all modules to device
    if model_device is not None:
        model.to(model_device)

    optimizer = _load_optimizer()
    return model, optimizer


def save_checkpoint(arch, model, epoch=0, optimizer=None, extras=None, is_best=False, 
                    name=None, savedir='.', verbose=True):
    """Save a pytorch training checkpoint
    Args:
        arch: name of the network architecture/topology
        model: a pytorch model
        epoch: current epoch number
        optimizer: the optimizer used in the training session
        extras: optional dict with additional user-defined data to be saved in the checkpoint.
            Will be saved under the key 'extras'
        is_best: If true, will save the checkpoint with the suffix 'best'
        name: the name of the checkpoint file
        savedir: directory in which to save the checkpoint
    """
    if not os.path.isdir(savedir):
        raise IOError(ENOENT, 'Checkpoint directory does not exist at', os.path.abspath(savedir))

    if extras is not None and not isinstance(extras, dict):
        raise TypeError('extras must be either a dict or None')

    # if not specified, set a name based on the epoch
    if name is None:
        epoch_str = str(epoch).zfill(4) if epoch is not None else str(0).zfill(4)
        name = f'best' if is_best else 'checkpoint_' + str(epoch).zfill(4)

    filename = name + '.pth.tar'
    fullpath = os.path.join(savedir, filename)
    model_filename = 'fullmodel__' + name + '.pth'
    model_fullpath = os.path.join(savedir, model_filename)

    # checkpoint is a dictionary containing different parameters, mostly the state dict
    checkpoint = {'epoch': epoch, 'state_dict': model.state_dict(), 'arch': arch}

    # save pruning/quantization metadata
    checkpoint['quant_metadata'] = {module_name: module.quant_metadata 
                                    for module_name, module in model.named_modules()
                                    if hasattr(module, 'quant_metadata')}
    checkpoint['pruning_metadata'] = {module_name: module.pruning_metadata 
                                      for module_name, module in model.named_modules()
                                      if hasattr(module, 'pruning_metadata')}

    try:
        checkpoint['is_parallel'] = model.is_parallel
        checkpoint['dataset'] = model.dataset
    except AttributeError:
        pass

    if extras is not None:
        checkpoint['extras'] = extras
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        checkpoint['optimizer_type'] = type(optimizer)

    torch.save(checkpoint, fullpath)
    try:
        torch.save(model, model_fullpath)
    except TypeError:
        # some pickling error when saving full model
        pass

    if verbose:
        logger.info(f"Saving checkpoint to: {fullpath}")


def weight_init(module):
    if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
        torch.nn.init.xavier_normal_(module.weight)


def transform_model(model, replacement_factory, replace_by_name=False):
    """Transform a model by replacing specific modules with new ones
    :param replace_by_name [bool]: if true, the replacement_factory contains
            a layer-name-to-replacement-module type. Otherwise, the mapping
            will be done by module type
    """
    def has_children(module):
        try:
            next(module.children())
            return True
        except StopIteration:
            return False

    def _replace_modules(container, prefix=''):
        for name, module in container.named_children():
            full_name = prefix + name

            if module in processed_modules:
                raise Exception("Has not come up before. Investigate")

            # get the replacement function for this module
            if replace_by_name:
                replace_fn = replacement_factory.get(full_name)
            else:
                replace_fn = replacement_factory.get(type(module))

            if replace_fn is not None:
                new_module = replace_fn(module, full_name)
                setattr(container, name, new_module)
                processed_modules[module] = new_module
                logger.debug(f"Replaced {full_name} with {new_module}")

            if has_children(module):
                _replace_modules(module, full_name + '.')

    processed_modules = OrderedDict()
    _replace_modules(model)


def log_training_progress(stats_dict, epoch, completed, total):
    """Log statistics about the training/inference procedure
    """
    if epoch > -1:
        log = 'Epoch: [{}][{:5d}/{:5d}]    '.format(epoch, completed, int(total))
    else:
        log = 'Test: [{:5d}/{:5d}]    '.format(completed, int(total))
    for name, val in stats_dict.items():
        if isinstance(val, int):
            log = log + '{name} {val:.3}    '.format(name=name, val=val)
        else:
            log = log + '{name} {val:.6f}    '.format(name=name, val=val)

    logger.info(log)


def perfect_divisors(numbers, include_one=False):
    """Get the perfect divisors of the given number/iterable
    """
    def _perfect_divisors(number):
        for i in range(1 if include_one else 2, int(number / 2) + 1):
            if number % i == 0:
                yield i
        yield number

    divisors = []
    if isinstance(numbers, Number):
        numbers = [numbers]
    for number in numbers:
        if number == 0:
            continue
        number = int(number)
        divisors.extend(
            [n for n in _perfect_divisors(number)]
        )
    return divisors


def get_contents_table(content_dict):
    """Get a tabulated format from the content dictionary
    """
    def inspect_val(val):
        if isinstance(val, (Number, str)):
            return val
        elif isinstance(val, type):
            return val.__name__
        return None

    contents = [[k, type(content_dict[k]).__name__, inspect_val(content_dict[k])]
                 for k in content_dict.keys()]
    contents = sorted(contents, key=lambda entry: entry[0])
    return tabulate(contents, headers=["Key", "Type", "Value"], tablefmt="psql")


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


def get_sparsity(param):
    """Calculate the sparsity across diverse dimentions
    """
    with torch.no_grad():
        all_weights = param.numel()
        nonzero_weights = param.ne(0).sum().item()
        # columns
        param_2d = param.view(-1, np.prod(param.shape[1:]))
        all_columns = param_2d.shape[0]
        nonzero_columns = torch.norm(param_2d, p=1, dim=1).ne(0).sum().item()
        # rows
        all_rows = param_2d.shape[1]
        nonzero_rows = torch.norm(param_2d, p=1, dim=0).ne(0).sum().item()
        # channels
        if param.dim() != 4:
            all_channels = nonzero_channels = sparsity_channels = 0.0
        else:
            channel_view = param.transpose(0, 1).contiguous()
            channels_norm = torch.norm(channel_view.view(-1, np.prod(channel_view.shape[1:])), p=1, dim=1)
            all_channels = param.shape[1]
            nonzero_channels = channels_norm.ne(0).sum().item()
            sparsity_channels = 1 - (nonzero_channels / all_channels)
        # filters
        if param.dim() != 4:
            all_filters = nonzero_filters = sparsity_filters = 0.0
        else:
            filters_norm = torch.norm(param.view(-1, np.prod(param.shape[1:])), p=1, dim=1)
            all_filters = param.shape[0]
            nonzero_filters = filters_norm.ne(0).sum().item()
            sparsity_filters = 1 - (nonzero_filters / all_filters)

    return {'all_weights': all_weights, 'nonzero_weights': nonzero_weights, 'weights': 1 - (nonzero_weights / all_weights),
            'all_columns': all_columns, 'nonzero_columns': nonzero_columns, 'columns': 1 - (nonzero_columns / all_columns),
            'all_rows': all_rows, 'nonzero_rows': nonzero_rows, 'rows': 1 - (nonzero_rows / all_rows),
            'all_channels': all_channels, 'nonzero_channels': nonzero_channels, 'channels': sparsity_channels,
            'all_filters': all_filters, 'nonzero_filters': nonzero_filters, 'filters': sparsity_filters}


def compute_model_statistics(model, layers_to_compress=[]):
    """Calculate statistics for each layer of a given model 
    """
    metrics = ['size', 'sparsity',
               'all_weights', 'nonzero_weights', 'sparsity_weights',
               'all_columns', 'nonzero_columns', 'sparsity_columns',
               'all_rows', 'nonzero_rows', 'sparsity_rows',
               'all_channels', 'nonzero_channels', 'sparsity_channels',
               'all_filters', 'nonzero_filters', 'sparsity_filters']
    total_stats = {metric: 0.0 for metric in metrics}
    per_layer_stats = {}

    for module_name, module in model.named_modules():
        if module_name not in layers_to_compress:
            continue


        # match the names of each sparsity-related metric
        sparsity_stats = get_sparsity(module.weight)
        per_layer_stats[module_name] = {metric: sparsity_stats[metric]
                                            if 'all_' in metric or 'nonzero_' in metric
                                            else sparsity_stats[metric.replace('sparsity_', '')]
                                        for metric in metrics
                                            if metric in sparsity_stats or 
                                            metric.replace('sparsity_', '') in sparsity_stats
                                        }
        per_layer_stats[module_name]['sparsity'] = per_layer_stats[module_name]['sparsity_weights']
        bits = getattr(getattr(module, 'quant_metadata', None), 'bits', 32)
        per_layer_stats[module_name]['size'] = (bits/8) * sparsity_stats['nonzero_weights']

        for metric in metrics:
            total_stats[metric] += per_layer_stats[module_name][metric]

    total_stats['sparsity'] = total_stats['sparsity_weights'] = 1 - (total_stats['nonzero_weights'] / total_stats['all_weights'])
    total_stats['sparsity_columns'] = 1 - (total_stats['nonzero_columns'] / total_stats['all_columns'])
    total_stats['sparsity_rows'] = 1 - (total_stats['nonzero_rows'] / total_stats['all_rows'])
    total_stats['sparsity_channels'] = 1 - (total_stats['nonzero_channels'] / total_stats['all_channels'])
    total_stats['sparsity_filters'] = 1 - (total_stats['nonzero_filters'] / total_stats['all_filters'])
    return total_stats, per_layer_stats


def get_dummy_input(device=None, input_shape=None):
    """Generate a representative dummy (random) input.
    Args:
        device (str or torch.device): Device on which to create the input
        input_shape (tuple): Tuple of integers representing the input shape. Can also be a tuple of tuples, allowing
          arbitrarily complex collections of tensors.
    """
    def create_single(shape):
        t = torch.randn(shape)
        if device:
            t = t.to(device)
        return t

    def create_recurse(shape):
        if all(isinstance(x, int) for x in shape):
            return create_single(shape)
        return tuple(create_recurse(s) for s in shape)

    return create_recurse(input_shape)


def model_summary(model, dummy_input=None)  -> OrderedDict[str, SimpleNamespace]:
    """Record statistics for input/output dimensions of each layer of a given model
    """
    def register_hook(module):
        def stats_hook(module, input, output):
            # access info via attribute
            info = SimpleNamespace()
            dimensions = OrderedDict()

            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                # input and output tensor sizes
                try:
                    ifm = input[0].size()
                except AttributeError:
                    ifm = input[0][0].size()
                if isinstance(output, dict) and 'out' in output:
                    output = output['out']
                ofm = output.size()
 
            if isinstance(module, torch.nn.Conv2d):
                # input shape of (batch_size, nfm, fm_h, fm_w)
                # naming conventions taken from DiGamma: https://arxiv.org/abs/2201.11220
                dimensions['K'] = ofm[-3]
                dimensions['Yo'] = ofm[-2]
                dimensions['Xo'] = ofm[-1]
                dimensions['C'] = ifm[-3]
                dimensions['Yi'] = ifm[-2]
                dimensions['Xi'] = ifm[-1]
                dimensions['N'] = ifm[0]
                dimensions['R'] = module.kernel_size[0]
                dimensions['S'] = module.kernel_size[1]
                dimensions['Hpad'] = module.padding[0]
                dimensions['Wpad'] = module.padding[1]
                dimensions['Hstr'] = module.stride[0]
                dimensions['Wstr'] = module.stride[1]
                dimensions['groups'] = module.groups
                info.type_int = 0
                info.weights_volume = np.prod(module.weight.shape)
                info.macs = dimensions['K'] * dimensions['Yo'] * dimensions['Xo'] * \
                            dimensions['C'] / module.groups * dimensions['R'] * dimensions['S']
                info.bits = getattr(getattr(module, 'quant_metadata', None), 'bits', 32)
                info.size = info.bits * info.weights_volume

            elif isinstance(module, torch.nn.Linear):
                # input shape of (batch_size, n_neurons)
                dimensions['K'] = module.out_features
                dimensions['C'] = module.in_features
                dimensions['Yo'] = dimensions['Xo'] = 1
                dimensions['Yi'] = dimensions['Xi'] = 1
                dimensions['R'] = dimensions['S'] = 1
                dimensions['N'] = ifm[0]
                dimensions['Hpad'] = dimensions['Wpad'] = 0
                dimensions['Hstr'] = dimensions['Wstr'] = 1
                dimensions['groups'] = 1
                info.type_int = 1
                info.weights_volume = info.macs = np.prod(module.weight.shape)
                info.bits = getattr(getattr(module, 'quant_metadata', None), 'bits', 32)
                info.size = info.bits * info.weights_volume

            # TODO: Fix the hard-coded override for any dimension being 0 
            # Fixing error that appears on fasterrcnn_mobilenet_v3_large_320_fpn:
            #   the 69th layer is a FC layer, but it comes up with an empty input.
            #   This results in the 'N' dimension being 0, which induced problems
            #   with timeloop. So we are hardcoding the 'N' dimension to 1.
            if any(dim == 0 for dim in dimensions.values()):
                new_dimensions = OrderedDict()
                for dim, dim_value in dimensions.items():
                    new_dimensions[dim] = dim_value if dim_value != 0 else 1
                dimensions = new_dimensions

            #TODO: Include descriptions of pooling layers

            # store all data, accessible by layer name
            info.layer_type = module.__class__.__name__
            info.dimensions = dimensions
            summary[module.full_name] = info

        handle = module.register_forward_hook(stats_hook)
        hook_handles.append(handle)

    summary = OrderedDict()
    hook_handles = []

    # apply the hooks and record the statistics
    model.eval()
    model.apply(register_hook)

    # execute a forward pass
    if dummy_input is None:
        assert getattr(model, 'input_shape', None) is not None, "To complete the summary, either provide " \
                                                                "a pre-configured tensor-shape or attribute " \
                                                                "it to the DNN model"
        dummy_input = get_dummy_input(model.device, model.input_shape)
    model(dummy_input)

    # remove the hooks
    for handle in hook_handles:
        handle.remove()

    return summary


def handle_model_subapps(net_wrapper, data_loaders, args):
    """Used to handle different modes of operation associated with the DNN model
    """
    do_exit = False
    train_loader, valid_loader, test_loader = data_loaders

    # perform inference on the registered DNN model
    if args.evaluate_model_mode:
        net_wrapper.model_verbose = net_wrapper.verbose = True
        # net_wrapper.print_frequency = args.batch_print_frequency
        logger.info(f"Evaluating {net_wrapper.model.arch} model\n{net_wrapper.model}")
        accuracy_metrics = net_wrapper.test(test_loader)
        logger.info(f"Accuracy metrics: {accuracy_metrics}")
        do_exit = True

    # perform training on DNN model
    elif args.train_model_mode:
        net_wrapper.verbose = True
        # net_wrapper.train(args.train_epochs, train_loader)
        net_wrapper.train(1500, train_loader)
        net_wrapper.save_model(verbose=True)
        do_exit = True

    elif args.model_summary_mode:
        do_exit = args.model_summary_mode != ModelSummaryType.Dummy

        if args.model_summary_mode == ModelSummaryType.Compute:
            df = pd.DataFrame(columns=['Name', 'Shape', 'IFM', 'OFM', 'MACs', 'Bits', 'Size'])
            # gather tensor statistics from the DNN
            summary = model_summary(net_wrapper.model)
            size = macs = 0
            for name, param in net_wrapper.model.named_parameters():
                if param.dim() in (2, 4) and 'weight' in name:
                    mname = name.replace('.weight', '')
                    size += summary[mname].size
                    macs += summary[mname].macs
                    df.loc[len(df.index)] = ([name,
                        '(' + ', '.join([str(size) for size in param.size()]) + ')',
                        (summary[mname].dimensions['C'], summary[mname].dimensions['Yi'], summary[mname].dimensions['Xi']),
                        (summary[mname].dimensions['K'], summary[mname].dimensions['Yo'], summary[mname].dimensions['Xo']),
                        int(summary[mname].macs), summary[mname].bits, summary[mname].size
                        ])
            df.loc[len(df.index)] = (['Total:', '-', '-', '-', f'{int(macs):,}', '-', f'{size:,}'])

        elif args.model_summary_mode == ModelSummaryType.Sparsity:
            df = pd.DataFrame(columns=['Name', 'Volume',
                                       'Cols', 'Rows', 'Channels', 'Filters', 'Weights',
                                       'Min', 'Max', 'AbsMean'])
            params = pruned_params = 0
            for name, param in net_wrapper.model.named_parameters():
                if param.dim() in (2, 4) and 'weight' in name:
                    params += param.numel()
                    pruned_params += param.eq(0).sum().item()
                    sparsity = get_sparsity(param)
                    df.loc[len(df.index)] = ([name, np.prod(param.size()),
                                              100 * sparsity['columns'], 100 * sparsity['rows'],
                                              100 * sparsity['channels'], 100 * sparsity['channels'],
                                              100 * sparsity['weights'],
                                              param.min().item(), param.max().item(),
                                              param.abs().mean().item()])
            df.loc[len(df.index)] = (['Total sparsity:'] + 5*['-'] + [f'{pruned_params/params:,}'] + 3*['-'])
 
        t = tabulate(df, headers='keys', tablefmt='psql', floatfmt="2.3f")
        logger.info(f"\n{t}")

    elif args.test_pruning_quant_mode:
        do_exit = True

        # to avoid circular import
        from src.compression.compressor import PruningQuantizationCompressor
        compressor = PruningQuantizationCompressor.from_args(args, data_loaders, net_wrapper.model)
        logger.info('Initialized compressor for testing the energy efficiency of pruning and quantization')

        pruning_ratios = [0.4, 0.7]
        quant_bits = [8, 4]
        compressor.model.eval()

        for i, pruning_ratio in enumerate(pruning_ratios):
            for j, quant_bit in enumerate(quant_bits):
                compressor.reset()

                compressor.prune_and_quantize(pruning_ratio, quant_bit) 
                logger.info(f"Testing with {pruning_ratio*100:.2f}% pruning ratio and {quant_bit} quantization bits")
                stats, _ = compressor.compute_model_statistics()
                area, latency, power, energy = compressor.compute_accelerator_statistics(init=i+j==0)
                logger.info(f"\tSparsity={stats['sparsity']:.3f} - Size={stats['size']:.3e} - Area={area:.3e} - "
                            f"Latency={latency:.3e} - Power={power:.3e} - Energy={energy:.3e}")

    elif args.test_timeloop_accelergy_mode:
        # test accelergy
        accelergy_dir = os.path.join(os.path.dirname(eyeriss_timeloop_dir), 'accelergy')
        inputs_dir = os.path.join(accelergy_dir, '04_eyeriss_like', 'input')
        positional_args = f'{inputs_dir}/*.yaml {inputs_dir}/components/*.yaml'
        # correct accelergy version on files in the exercise
        for arch_file in glob(f'{inputs_dir}/*.yaml') + glob(f'{inputs_dir}/components/*.yaml'):
            command = rf"sed -i 's_\(version:\).*_\1 0.3_' {arch_file}"
            logger.debug(f'Executing sed command:\n{command}')
            p = sp.run(command, shell=True, check=True, capture_output=True, text=True)
            logger.debug(f'Command status: {p.returncode}')

        command = f'accelergy {positional_args} '\
                  f'--outdir {net_wrapper.logdir} ' \
                  f'--output_files energy_estimation ERT_summary ART_summary flattened_arch '\
                  f'--oprefix {net_wrapper.model.arch}__ '\
                  f'--verbose 1 --precision 3'
        logger.debug(f'Accelergy command: {command}')
        start = time()
        p = sp.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.debug(f'Stdout: {p.stdout}')
        logger.debug(f'Stderr: {p.stderr}')
        logger.info(f'Executed accelergy command (exitcode: {p.returncode}) in {time() - start:.3e}s')

        # test timeloop model
        inputs_dir = os.path.join(eyeriss_timeloop_dir, '04-model-conv1d+oc-3levelspatial')
        command = f'timeloop-model '\
                  f'{inputs_dir}/arch/*.yaml '\
                  f'{inputs_dir}/map/conv1d+oc+ic-3levelspatial-cp-ws.map.yaml '\
                  f'{inputs_dir}/prob/*.yaml'
        start = time()
        p = sp.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.debug(f'Stdout: {p.stdout}')
        logger.debug(f'Stderr: {p.stderr}')
        logger.info(f'Executed timeloop-model command (exitcode: {p.returncode}) in {time() - start:.3e}s')

        # test timeloop mapper
        inputs_dir = os.path.join(os.path.dirname(eyeriss_timeloop_dir), 'timeloop+accelergy')
        command = f'timeloop-mapper '\
                  f'{inputs_dir}/arch/eyeriss_like-int16.yaml '\
                  f'{inputs_dir}/arch/components/*.yaml '\
                  f'{inputs_dir}/prob/prob.yaml '\
                  f'{inputs_dir}/mapper/mapper.yaml '\
                  f'{inputs_dir}/constraints/*.yaml '
        start = time()
        p = sp.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.debug(f'Stdout: {p.stdout}')
        logger.debug(f'Stderr: {p.stderr}')
        logger.info(f'Executed timeloop-mapper command (exitcode: {p.returncode}) in {time() - start:.3e}s')

    return do_exit


def iou(bboxes1, bboxes2):
    """ calculate iou between each bbox in `bboxes1` with each bbox in `bboxes2`"""
    px, py, pw, ph = bboxes1[...,:4].reshape(-1, 4).split(1, -1)
    lx, ly, lw, lh = bboxes2[...,:4].reshape(-1, 4).split(1, -1)
    px1, py1, px2, py2 = px - 0.5 * pw, py - 0.5 * ph, px + 0.5 * pw, py + 0.5 * ph
    lx1, ly1, lx2, ly2 = lx - 0.5 * lw, ly - 0.5 * lh, lx + 0.5 * lw, ly + 0.5 * lh
    zero = torch.tensor(0.0, dtype=px1.dtype, device=px1.device)
    dx = torch.max(torch.min(px2, lx2.T) - torch.max(px1, lx1.T), zero)
    dy = torch.max(torch.min(py2, ly2.T) - torch.max(py1, ly1.T), zero)
    intersections = dx * dy
    pa = (px2 - px1) * (py2 - py1) # area
    la = (lx2 - lx1) * (ly2 - ly1) # area
    unions = (pa + la.T) - intersections
    ious = (intersections/unions).reshape(*bboxes1.shape[:-1], *bboxes2.shape[:-1])
    
    return ious


def iou_wh(bboxes1, bboxes2):
    """ calculate iou between each bbox in `bboxes1` with each bbox in `bboxes2`
    The bboxes should be defined by their width and height and are centered around (0,0)
    """ 
    w1 = bboxes1[..., 0].view(-1)
    h1 = bboxes1[..., 1].view(-1)
    w2 = bboxes2[..., 0].view(-1)
    h2 = bboxes2[..., 1].view(-1)

    intersections = torch.min(w1[:,None],w2[None,:]) * torch.min(h1[:,None],h2[None,:])
    unions = (w1 * h1)[:,None] + (w2 * h2)[None,:] - intersections
    ious = (intersections / unions).reshape(*bboxes1.shape[:-1], *bboxes2.shape[:-1])

    return ious


def nms(filtered_tensor, threshold):
    result = []
    for x in filtered_tensor:
        # Sort coordinates by descending confidence
        scores, order = x[:, 4].sort(0, descending=True)
        x = x[order]
        ious = iou(x,x) # get ious between each bbox in x

        # Filter based on iou
        keep = (ious > threshold).long().triu(1).sum(0, keepdim=True).t().expand_as(x) == 0

        result.append(x[keep].view(-1, 6).contiguous())
    return result

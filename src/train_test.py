import logging
import time
from contextlib import nullcontext

import torch
import torchnet.meter as tnt
from collections import OrderedDict

from brevitas.export.inference import quant_inference_mode

from src.utils import log_training_progress

logger = logging.getLogger(__name__)

def validate(valid_loader, model, criterion, accuracy_meter, epoch, verbose, print_frequency, use_quant=False, device="cpu"):
    if verbose:
        logger.info(f'--- validate (epoch={epoch})-----------')

    def _log_validation_progress():
        stats_dict = OrderedDict()
        for accuracy_metric in accuracy_meter.metrics:
            stats_dict[accuracy_metric] = accuracy_meter.value(metric=accuracy_metric)
        stats_dict['Loss'] = losses['Objective Loss'].mean
        stats_dict['Time'] = batch_time.mean
        log_training_progress(stats_dict, epoch, steps_completed, total_steps)

    """Execute the validation/test loop."""
    losses = {'Objective Loss': tnt.AverageValueMeter()}

    batch_time = tnt.AverageValueMeter()
    total_samples = len(valid_loader.sampler)
    batch_size = valid_loader.batch_size
    total_steps = total_samples / batch_size
    if verbose:
        logger.info(f'{total_samples} samples ({batch_size} per mini-batch)')

    # Switch to evaluation mode
    model.eval()
    model.to(device)

    end = time.time()
    quant_cm = quant_inference_mode(model) if use_quant else nullcontext()
    with torch.no_grad(), quant_cm:
        # if use_quant:
        #     quant_top1 = validate_quant(valid_loader, model, stable=True)
        #     print("QUANT_TOP1: " + quant_top1)
        for validation_step, (inputs, target) in enumerate(valid_loader):

            # cast to device
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(device)
            if isinstance(target, torch.Tensor):
                target = target.to(device)

            # compute output from model
            output = model(inputs)

            # compute loss
            loss = criterion(output, target)
            # measure accuracy and record loss
            losses['Objective Loss'].add(loss.item())
            accuracy_meter.add(output, target)

            # measure elapsed time
            batch_time.add(time.time() - end)
            end = time.time()

            steps_completed = validation_step + 1
            if verbose and steps_completed % print_frequency == 0:
                _log_validation_progress()

    if verbose:
        logstr = '   '.join([
            f'{accuracy_metric.capitalize()}: {accuracy_meter.value(metric=accuracy_metric):.3f}'
            for accuracy_metric in accuracy_meter.metrics
        ])
        logger.info(f"==> {logstr}   Loss: {losses['Objective Loss'].mean:.3f}\n")

    return *accuracy_meter.value(metric='all'), losses['Objective Loss'].mean

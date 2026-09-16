import logging
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from brevitas.export.inference import quant_inference_mode
from brevitas.graph.quantize import preprocess_for_quantize
from brevitas_examples.imagenet_classification.ptq.ptq_common import apply_bias_correction
from brevitas_examples.imagenet_classification.ptq.ptq_common import apply_gptq
from brevitas_examples.imagenet_classification.ptq.ptq_common import calibrate
from brevitas_examples.imagenet_classification.ptq.ptq_common import quantize_model
from brevitas_examples.imagenet_classification.utils import SEED
from torchvision.datasets import CIFAR100

from src.datasets.imagenet_dataset import ImagenetDataset

GRAPH_EQ_ITERATIONS = 20

logger = logging.getLogger(__name__)


def _get_quant_config(model, dataset):
    config = {}
    if type(dataset) is ImagenetDataset:
        config['percentile'] = 99.95
        config['target_backend'] = 'layerwise'
        config['merge_bn'] = True
        config['relu6_to_relu'] = True
    elif hasattr(model, 'arch_name') and model.arch_name == 'mobilenet':
        config['percentile'] = 99.95
        config['gptq'] = True
        config['target_backend'] = 'layerwise'
        config['merge_bn'] = False
        config['relu6_to_relu'] = True
    elif hasattr(model, 'arch_name') and model.arch_name == 'mobilenetv2':
        config['percentile'] = 99.95
        config['gptq'] = True
        config['target_backend'] = 'layerwise'
        config['merge_bn'] = True
        config['relu6_to_relu'] = False
    elif type(dataset) is CIFAR100:
        config['percentile'] = 99.95
        config['gptq'] = True
        config['target_backend'] = 'fx'
        config['merge_bn'] = True
        config['relu6_to_relu'] = True
    else:
        config['percentile'] = 99.999
        config['gptq'] = True
        config['target_backend'] = 'fx'
        config['merge_bn'] = True,
        config['relu6_to_relu'] = True

    if hasattr(model, "arch") and 'inception' in model.arch.lower():
        config['size'] = 299
    else:
        config['size'] = 224

    return config


def _generate_ref_input(device, dtype, input_shape):
    return torch.ones(1, 3, input_shape, input_shape, device=device, dtype=dtype)


def _generate_calib_loader(
        dataset,
        batch_size: int = 64,
        num_workers: int = 8,
        n_calib: int = 2048
):
    subset = torch.utils.data.Subset(dataset, list(range(min(n_calib, len(dataset)))))
    loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return loader


def quantize(model, q_bits, dataset):
    q_bits = int(q_bits)
    dtype = next(model.parameters()).dtype
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Get model-specific configurations about input shapes and normalization

    calib_loader = _generate_calib_loader(dataset)
    quant_config = _get_quant_config(model, dataset)

    cudnn.benchmark = False
    model.eval()

    # Preprocess the model for quantization
    preprocessed_model = preprocess_for_quantize(
        model,
        trace_model=True,
        equalize_iters=GRAPH_EQ_ITERATIONS,
        equalize_merge_bias=True,
        merge_bn=quant_config['merge_bn'],
        channel_splitting_ratio=0.0,
        relu6_to_relu=quant_config['relu6_to_relu'],
        channel_splitting_split_input=False)
    del model
    model = preprocessed_model

    device = next(iter(model.parameters())).device

    act_quant_percentile = quant_config['percentile']

    # Define the quantized model
    quant_model = quantize_model(
        model,
        dtype=dtype,
        device=device,
        backend=quant_config['target_backend'],
        scale_factor_type='float_scale',
        bias_bit_width=32,  # TODO should this match other parameters?
        weight_bit_width=q_bits,
        weight_narrow_range=False,
        weight_param_method='stats',
        weight_quant_granularity='per_tensor',
        act_quant_granularity='per_tensor',
        weight_quant_type='sym',
        layerwise_first_last_bit_width=8,
        act_bit_width=q_bits,
        act_param_method='stats',
        act_quant_percentile=act_quant_percentile,
        act_quant_type='sym',
        quant_format='int',
        layerwise_first_last_mantissa_bit_width=4,
        layerwise_first_last_exponent_bit_width=3,
        weight_mantissa_bit_width=4,
        weight_exponent_bit_width=3,
        act_mantissa_bit_width=4,
        act_exponent_bit_width=3,
        act_scale_computation_type='static',
        uint_sym_act_for_unsigned_values=True)

    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    images, _ = next(iter(calib_loader))
    images = images.to(device=device, dtype=dtype)
    with torch.no_grad():
        model(images)

    # Calibrate the quant_model on the calibration dataloader
    print("Starting activation calibration:")
    calibrate(calib_loader, quant_model)

    if quant_config.get('gptq'):
        print("Performing GPTQ:")
        apply_gptq(
            calib_loader,
            quant_model,
            act_order=False,
            create_weight_orig=True,
            use_quant_activations=False,
            max_accumulator_bit_width=None,
            max_accumulator_tile_size=None)

    print("Applying bias correction:")
    apply_bias_correction(calib_loader, quant_model)

    # Validate the quant_model on the validation dataloader
    # print("Starting validation:")
    with torch.no_grad(), quant_inference_mode(quant_model):
        param = next(iter(quant_model.parameters()))
        device, dtype = param.device, param.dtype
        ref_input = _generate_ref_input(device, dtype, quant_config['size'])
        quant_model(ref_input)
        compiled_model = torch.compile(quant_model, fullgraph=True, disable=True)
        del calib_loader
        return compiled_model

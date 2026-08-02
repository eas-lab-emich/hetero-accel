import gc
import os
from types import SimpleNamespace

from src.args import OptimizerType
import logging

from src.mapping.api.accelerator import AcceleratorConfiguration
from src.mapping.api.mapping_request import MappingRequest
from src.mapping.api.problem import ConvolutionProblem
from src.mapping.impl.timeloop import TimeloopWrapper
from src import project_dir, hetero_dataset_dir

logger = logging.getLogger(__name__)


def timeloop_test():
    logging.basicConfig(level=logging.DEBUG)

    import crimson_magick.cifar_zoo

    model_config = SimpleNamespace(arch='resnet50', dataset='cifar100', batch_size=1, gpus=0, cpu=False,
                                   load_serialized=False, pretrained=True, resumed_checkpoint_path=None, optimizer_type=
                                   OptimizerType.Adam, print_frequency=100, verbose=True)
    from src.net_wrapper import TorchNetworkWrapper
    net_wrapper = TorchNetworkWrapper(model_config)
    from crimson_magick.cifar_zoo import Cifar
    dataset = crimson_magick.cifar_zoo.get_test_loader(Cifar.CIFAR100, data_dir=hetero_dataset_dir)
    net_wrapper.run_summary(dataset)
    to_serialize = net_wrapper.summary['model.layer4.1.conv1']
    problem = ConvolutionProblem(
        input_channels=to_serialize.dimensions['C'],
        output_channels=to_serialize.dimensions['K'],
        input_width=to_serialize.dimensions['Xi'],
        input_height=to_serialize.dimensions['Yi'],
        padding_width=to_serialize.dimensions['Wpad'],
        padding_height=to_serialize.dimensions['Hpad'],
        stride_width=to_serialize.dimensions['Wstr'],
        stride_height=to_serialize.dimensions['Hstr'],
        batch_size=to_serialize.dimensions['N'],
        kernel_width=to_serialize.dimensions['S'],
        kernel_height=to_serialize.dimensions['R'],
    )

    from src.accelerator_cfg import AcceleratorProfile
    from src.accelerator_cfg import AcceleratorType
    accel_type = AcceleratorType.Eyeriss
    accel_cfg = AcceleratorProfile(accel_type)

    accelerator_config = AcceleratorConfiguration(
        pe_array_x=accel_cfg.pe_array_x,
        pe_array_y=accel_cfg.pe_array_y,
        precision=4,
        sram_size=accel_cfg.sram_size,
        ifmap_spad_size=accel_cfg.ifmap_spad_size,
        weights_spad_size=accel_cfg.weights_spad_size,
        psum_spad_size=accel_cfg.psum_spad_size
    )

    import uuid
    request = MappingRequest(
        id=uuid.uuid4(),
        accelerator_config=accelerator_config,
        problem=problem
    )

    logger.info(f'Accelerator:{accelerator_config}')
    tw = TimeloopWrapper(project_dir + '/test_tl_2', cleanup=False)
    results = tw.map(request)
    print(results._asdict())
    del net_wrapper.model
    del net_wrapper
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    exit(0)


if __name__ == "__main__":
    timeloop_test()

import gc
import uuid
from types import SimpleNamespace

from crimson_magick import cifar_zoo
from crimson_magick.cifar_zoo import Cifar

from src.accelerator_cfg import AcceleratorProfile, AcceleratorType
from src.args import OptimizerType
import logging

from src import hetero_dataset_dir
from src.mapping.api import ConvolutionProblem, AcceleratorConfiguration, MappingRequest
from src.mapping.impl.distributed_timeloop import DistributedTimeloopMapper
from src.net_wrapper import TorchNetworkWrapper

logger = logging.getLogger(__name__)


def timeloop_test():
    logging.basicConfig(level=logging.DEBUG)


    model_config = SimpleNamespace(arch='resnet50', dataset='cifar100', batch_size=1, gpus=0, cpu=False,
                                   load_serialized=False, pretrained=True, resumed_checkpoint_path=None, optimizer_type=
                                   OptimizerType.Adam, print_frequency=100, verbose=True)
    net_wrapper = TorchNetworkWrapper(model_config)
    dataset = cifar_zoo.get_test_loader(Cifar.CIFAR100, data_dir=hetero_dataset_dir)
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

    request = MappingRequest(
        id=uuid.uuid4(),
        accelerator_config=accelerator_config,
        problem=problem
    )

    logger.info(f'Accelerator:{accelerator_config}')
    mapper = DistributedTimeloopMapper()
    try:
        mapper.start()
        results = mapper.map(request).result(500)
        print(results)
    except Exception as e:
        print(e)
    finally:
        del net_wrapper.model
        del net_wrapper
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        mapper.stop()
        exit(0)


if __name__ == "__main__":
    timeloop_test()

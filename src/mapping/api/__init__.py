__all__ = ['MappingStats', 'AcceleratorConfiguration', 'ConvolutionProblem', 'MappingRequest', 'AcceleratorMapper',
           'AsyncAcceleratorMapper', 'async_mapper']

import logging
import os

from src.mapping.api.mapper import MappingStats, AcceleratorConfiguration, ConvolutionProblem, MappingRequest, \
    AcceleratorMapper, AsyncAcceleratorMapper
from src.mapping.impl.async_timeloop import AsyncTimeloopMapper
from src.mapping.impl.cached_async_mapper import can_use_cache, CachedAsyncMapperFacade
from src.mapping.impl.distributed_timeloop import DistributedTimeloopMapper
from src.mapping.impl.mq_config import MQError

logger = logging.getLogger(__name__)


def async_mapper(logdir) -> AsyncAcceleratorMapper:
    try:
        mapper_impl = DistributedTimeloopMapper()
    except MQError as mq_error:
        logger.warning("Unable to instantiate DistributedTimeloopMapper, using local compute only", exc_info=mq_error)
        mapper_impl = AsyncTimeloopMapper(os.path.join(logdir, 'mapper_workspace'))
    return CachedAsyncMapperFacade(mapper_impl) if can_use_cache() else mapper_impl

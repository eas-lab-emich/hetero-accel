import logging
import multiprocessing
from concurrent.futures import Future
from concurrent.futures.thread import ThreadPoolExecutor

from src.mapping.api import MappingStats, MappingRequest
from src.mapping.api.mapper import AsyncAcceleratorMapper
from src.mapping.impl.timeloop import TimeloopWrapper

logger = logging.getLogger(__name__)


class AsyncTimeloopMapper(AsyncAcceleratorMapper):

    def __init__(self, workdir, *, cleanup=True):
        max_workers = multiprocessing.cpu_count()
        logger.info("Initializing AsyncTimeloopMapper with max_workers=%s", max_workers)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.timeloop_mapper = TimeloopWrapper(workdir, cleanup=cleanup)

    def map(self, request: MappingRequest) -> Future[MappingStats]:
        logger.debug("Submitting MappingRequest, request=%s", request)
        return self.executor.submit(self.timeloop_mapper.map, request)

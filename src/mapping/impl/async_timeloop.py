import logging
import os
from concurrent.futures import Future
from concurrent.futures.thread import ThreadPoolExecutor

from src.mapping.api import MappingStats, MappingRequest
from src.mapping.api.mapper import AsyncAcceleratorMapper
from src.mapping.impl.timeloop import TimeloopWrapper
from src.mapping.impl.timeloop_threads import get_tl_thread_count

logger = logging.getLogger(__name__)


class AsyncTimeloopMapper(AsyncAcceleratorMapper):

    def __init__(self, workdir, *, cleanup=True):
        logger.info("Initializing AsyncTimeloopMapper with max_workers=%s", self.get_parallelism())
        self.executor = ThreadPoolExecutor(max_workers=self.get_parallelism())
        self.timeloop_mapper = TimeloopWrapper(workdir, cleanup=cleanup)

    def map(self, request: MappingRequest) -> Future[MappingStats]:
        logger.debug("Submitting MappingRequest, request=%s", request)
        return self.executor.submit(self.timeloop_mapper.map, request)

    def get_parallelism(self):
        return (os.cpu_count() or 1) // get_tl_thread_count() or 1

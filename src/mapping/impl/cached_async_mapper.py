import logging
import os
import pickle
import time
from concurrent.futures import Future
from pathlib import Path
from threading import Lock, Thread
from typing import Dict

from torch._C import TupleType

from src.mapping.api import AsyncAcceleratorMapper, MappingRequest, MappingStats, ConvolutionProblem, \
    AcceleratorConfiguration

CACHE_FILE_NAME = "mapper_cache.pkl"

logger = logging.getLogger(__name__)


def get_cache_dir():
    return os.environ.get("HETERO_CACHE_DIR")


def can_use_cache():
    return True if get_cache_dir() else False


class CachedAsyncMapperFacade(AsyncAcceleratorMapper):

    def __init__(self, wrapped):
        cache_dir = get_cache_dir()
        if cache_dir is None:
            raise RuntimeError("Cannot initialize CachedAsyncMapper without a cache directory!")
        self.cache_path = Path(os.path.join(cache_dir, CACHE_FILE_NAME))
        self.cache: Dict[tuple[AcceleratorConfiguration, ConvolutionProblem], MappingStats] = {}
        self.wrapped = wrapped
        self.cache_lock = Lock()
        self.flush_pending = False
        self.flush_thread = None
        self.shutdown = False

    def map(self, request: MappingRequest) -> Future[MappingStats]:
        with self.cache_lock:
            cached_stats = self.cache.get(self._cache_key(request))
            if cached_stats:
                result = Future()
                result.set_result(cached_stats.model_copy(
                    update={'id': request.id}
                ))
                return result

        source_result = self.wrapped.map(request)
        result = Future()

        def update_cache(completed: Future[MappingStats]):
            try:
                mapping_stats = completed.result()
                with self.cache_lock:
                    self.cache[self._cache_key(request)] = mapping_stats
                    self.flush_pending = True
                result.set_result(mapping_stats)
            except BaseException as e:
                result.set_exception(e)

        source_result.add_done_callback(update_cache)
        return result

    def start(self):
        if self.cache_path.exists():
            with open(self.cache_path, "rb") as f:
                raw_dict = pickle.load(f)
            for key, result in raw_dict.items():
                model_stats = MappingStats(**dict(zip(MappingStats.model_fields, result)))
                self.cache[key] = model_stats

        self.flush_thread = Thread(target=self._scheduled_flush, daemon=False)
        self.flush_thread.start()
        return self.wrapped.start()

    def stop(self):
        self.wrapped.stop()
        with self.cache_lock:
            self.shutdown = True
        self.flush_thread.join(timeout=5)
        return

    def _scheduled_flush(self):
        shutdown = self.shutdown
        while not shutdown:
            with self.cache_lock:
                shutdown = self.shutdown
                if self.flush_pending:
                    self._flush()
                    self.flush_pending = False
            time.sleep(.5)

    def _flush(self):
        to_flush = {}
        for key, result in self.cache.items():
            to_flush[key] = tuple(result.model_dump().values())
        with open(self.cache_path, "wb") as f:
            pickle.dump(to_flush, f)

    @staticmethod
    def _cache_key(request: MappingRequest):
        return tuple(request.accelerator_config.model_dump().values()), tuple(request.problem.model_dump().values())

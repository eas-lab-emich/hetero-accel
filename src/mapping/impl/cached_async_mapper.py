import logging
import os
import pickle
from concurrent.futures import Future
from pathlib import Path
from threading import Thread, Event
from typing import Dict

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
        if not cache_dir:
            raise RuntimeError("Cannot initialize CachedAsyncMapper without a cache directory!")
        self.cache_path = Path(os.path.join(cache_dir, CACHE_FILE_NAME))
        self.cache: Dict[tuple[AcceleratorConfiguration, ConvolutionProblem], MappingStats] = {}
        self.wrapped = wrapped
        self.flush_pending = False
        self.flush_thread = None
        self.shutdown = False
        self.wake = Event()

    def map(self, request: MappingRequest) -> Future[MappingStats]:
        key = self._cache_key(request)
        cached_stats = self.cache.get(key)
        if cached_stats is not None:
            result = Future()
            result.set_result(cached_stats.model_copy(update={'id': request.id}))
            return result

        source_result = self.wrapped.map(request)
        result = Future()

        def update_cache(completed):
            try:
                mapping_stats = completed.result()
                self.cache[key] = mapping_stats  # atomic
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
        try:
            self.wrapped.stop()
        finally:
            self.shutdown = True
            self.wake.set()
            self.flush_thread.join()

    def _scheduled_flush(self):
        while True:
            self.wake.wait(timeout=60)
            self.wake.clear()
            if self.flush_pending:
                self.flush_pending = False
                try:
                    self._flush()
                except Exception:
                    logger.exception("Failed to flush cache, retrying in 60 seconds")
                    self.flush_pending = True
            if self.shutdown:
                return

    def _flush(self):
        snapshot = dict(self.cache)
        to_flush = {k: tuple(v.model_dump().values()) for k, v in snapshot.items()}
        tmp_path = self.cache_path.with_suffix(".tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump(to_flush, f)
        os.replace(tmp_path, self.cache_path)

    @staticmethod
    def _cache_key(request: MappingRequest):
        return tuple(request.accelerator_config.model_dump().values()), tuple(request.problem.model_dump().values())

import logging
import threading
import time
from concurrent.futures import Future
from uuid import UUID

import pika

from src.mapping.api import MappingStats, MappingRequest
from src.mapping.api.mapper import AsyncAcceleratorMapper
from src.mapping.impl.mq_config import get_mq_config

logger = logging.getLogger(__name__)


class DistributedTimeloopMapper(AsyncAcceleratorMapper):
    REQUEST_QUEUE = "mapping.requests"
    RESULTS_QUEUE = "mapping.results"

    def __init__(self):
        username, password, host = get_mq_config()
        self._parameters = pika.ConnectionParameters(
            host=host,
            credentials=pika.PlainCredentials(username, password),
            heartbeat=0,
            blocked_connection_timeout=30,
        )

        self._request_connection = pika.BlockingConnection(self._parameters)
        self._request_channel = self._request_connection.channel()
        self._request_channel.confirm_delivery()
        self._request_channel.queue_declare(queue=self.REQUEST_QUEUE, durable=True)
        self._request_channel.queue_purge(queue=self.REQUEST_QUEUE)

        # Confirmed publishes pump this connection's I/O loop before returning, so
        # the lock has to span the whole call, not just the frame write. Plain Lock is
        # safe here because no user callback is registered on this connection.
        self._publish_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending_mappings: dict[UUID, Future[MappingStats]] = {}
        self._stopped = False

        self._results_connection = None
        self._results_channel = None
        self._consumer_thread = None
        self._result_timeout = 1200.0
        self._deadlines: dict[UUID, float] = {}
        self._reaper_thread = None
        self._reaper_stop = threading.Event()
        self._reaper_interval = 1.0

    def _publish(self, request: MappingRequest) -> None:
        body = request.model_dump_json().encode()
        properties = pika.BasicProperties(
            content_type="application/json",
            delivery_mode=2,
        )
        with self._publish_lock:
            self._request_channel.basic_publish(
                exchange="",
                routing_key=self.REQUEST_QUEUE,
                body=body,
                properties=properties,
            )

    def map(self, request: MappingRequest) -> Future[MappingStats]:
        logger.debug("Submitting MappingRequest, request=%s", request)
        pending_mapping: Future[MappingStats] = Future()
        with self._pending_lock:
            if self._stopped:
                raise RuntimeError("DistributedTimeloopMapper has been stopped")
            self._pending_mappings[request.id] = pending_mapping
            self._deadlines[request.id] = time.monotonic()
        try:
            self._publish(request)
        except Exception:
            with self._pending_lock:
                self._pending_mappings.pop(request.id, None)
                self._deadlines.pop(request.id, None)
            raise
        return pending_mapping

    def start(self):
        if self._consumer_thread is not None:
            return

        username, password, host = get_mq_config()  # FIXME don't love keeping username and password on the stack
        credentials = pika.PlainCredentials(
            username,
            password
        )
        self._results_connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=host,
                credentials=credentials,
                heartbeat=0
            )
        )
        self._results_channel = self._results_connection.channel()
        self._results_channel.queue_declare(
            queue=self.RESULTS_QUEUE,
            durable=True
        )
        self._results_channel.queue_purge(queue=self.RESULTS_QUEUE)
        self._results_channel.basic_qos(prefetch_count=1000)

        self._consumer_thread = threading.Thread(
            target=self._start_consuming,
            name="rabbitmq-consumer",
            daemon=False
        )
        self._consumer_thread.start()
        self._reaper_thread = threading.Thread(
            target=self._reap_expired,
            name="result-reaper",
            daemon=True,
        )
        self._reaper_thread.start()

    def _start_consuming(self) -> None:
        try:
            self._results_channel.basic_consume(
                queue=self.RESULTS_QUEUE,
                auto_ack=False,
                on_message_callback=self._handle_result,
            )
            self._results_channel.start_consuming()
        except Exception as exc:
            logger.exception("Results consumer failed")
            self._fail_pending(exc)
        finally:
            self._results_connection.close()

    def _handle_result(self, ch, method, properties, body) -> None:
        try:
            result = MappingStats.model_validate_json(body)
            with self._pending_lock:
                future = self._pending_mappings.pop(result.id, None)
                self._deadlines.pop(result.id, None)
            if future is None:
                logger.warning("No future found for id=%s, result=%s", result.id, result)
            elif future.cancelled():
                logger.warning("Future for id=%s was cancelled; discarding result", result.id)
            else:
                future.set_result(result)
        except Exception:
            # An exception escaping into pika's dispatch loop kills this thread and
            # leaves every outstanding future hanging.
            logger.exception("Failed to handle result delivery")
        finally:
            ch.basic_ack(method.delivery_tag)

    def _fail_pending(self, exc: BaseException) -> None:
        with self._pending_lock:
            pending = list(self._pending_mappings.values())
            self._pending_mappings.clear()
        for future in pending:
            if not future.done():
                future.set_exception(exc)

    def stop(self) -> None:
        self._stopped = True

        if self._consumer_thread is not None and self._consumer_thread.is_alive():
            self._results_connection.add_callback_threadsafe(
                self._results_channel.stop_consuming
            )
            self._consumer_thread.join(timeout=5)
            if self._consumer_thread.is_alive():
                logger.warning("Consumer thread still alive after join timeout")
        self._consumer_thread = None
        if self._reaper_thread is not None:
            self._reaper_thread.join(timeout=self._reaper_interval * 2)
            if self._reaper_thread.is_alive():
                logger.warning("Reaper thread still alive after join timeout")
            self._reaper_thread = None
        self._fail_pending(RuntimeError("DistributedTimeloopMapper stopped"))

        if self._results_connection is not None and self._results_connection.is_open:
            self._results_connection.close()
        if self._request_connection.is_open:
            self._request_connection.close()

        self._results_channel = None
        self._results_connection = None

    def _reap_expired(self) -> None:
        while not self._reaper_stop.wait(1.0):
            now = time.monotonic()
            with self._pending_lock:
                expired = [rid for rid, t in self._deadlines.items()
                           if now - t > self._result_timeout]
                for rid in expired:
                    self._deadlines.pop(rid, None)
                    future = self._pending_mappings.pop(rid, None)
                    if future is not None and not future.done():
                        future.set_exception(
                            TimeoutError(f"No result for id={rid}"))

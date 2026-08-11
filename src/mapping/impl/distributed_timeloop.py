import logging
import threading
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
        credentials = pika.PlainCredentials(
            username,
            password
        )

        self._request_connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=host,
                credentials=credentials,
                heartbeat=0
            )
        )

        self._request_channel = self._request_connection.channel()
        self._request_channel.confirm_delivery()
        self._request_channel.queue_declare(
            queue=self.REQUEST_QUEUE,
            durable=True
        )
        self._request_channel.queue_purge(queue=self.REQUEST_QUEUE)

        self._results_connection = None
        self._results_channel = None

        self._consumer_thread = None
        self._pending_lock = threading.Lock()
        self._pending_mappings: dict[UUID, Future[MappingStats]] = {}

    def _publish(self, request: MappingRequest) -> None:
        self._request_channel.basic_publish(
            exchange="",
            routing_key=self.REQUEST_QUEUE,
            body=request.model_dump_json().encode(),
            properties=pika.BasicProperties(
                content_type="application/json",
                delivery_mode=2
            )
        )

    def map(self, request: MappingRequest) -> Future[MappingStats]:
        logger.debug("Submitting MappingRequest, request=%s", request)
        pending_mapping = Future()
        with self._pending_lock:
            self._pending_mappings[request.id] = pending_mapping
        try:
            self._publish(request)
        except Exception:
            with self._pending_lock:
                self._pending_mappings.pop(request.id, None)
            raise
        return pending_mapping

    def start(self):
        if self._consumer_thread is not None:
            return

        username, password, host = get_mq_config() # FIXME don't love keeping username and password on the stack
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

        self._consumer_thread = threading.Thread(
            target=self._start_consuming,
            name="rabbitmq-consumer",
            daemon=False
        )
        self._consumer_thread.start()

    def _start_consuming(self):
        self._results_channel.basic_consume(
            queue=self.RESULTS_QUEUE,
            auto_ack=False,
            on_message_callback=self._handle_result,
        )
        self._results_channel.start_consuming()

        self._results_connection.close()

    def _handle_result(self, ch, method, properties, body):
        result = MappingStats.model_validate_json(body)
        with self._pending_lock:
            future = self._pending_mappings.pop(result.id, None)
        if future is None:
            logger.warning('No future found for id=%s, result=%s', result.id, result)
        else:
            # logger.info('Received result for id=%s, result=%s', result.id, result)
            future.set_result(result)
        ch.basic_ack(method.delivery_tag)

    def stop(self):
        if self._consumer_thread is not None:
            self._results_connection.add_callback_threadsafe(
                self._results_channel.stop_consuming
            )
            self._consumer_thread.join()

        self._request_connection.close()

        self._consumer_thread = None
        self._results_channel = None
        self._results_connection = None

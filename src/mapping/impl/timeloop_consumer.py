import logging
import os
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import dataclass

import pika
from pika.channel import Channel

from src import project_dir
from src.mapping.api import MappingRequest
from src.mapping.impl.async_timeloop import AsyncTimeloopMapper
from src.mapping.impl.mq_config import get_mq_config

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

REQUEST_QUEUE = "mapping.requests"
RESULTS_QUEUE = "mapping.results"

work_dir = os.path.join(project_dir, "mapper_workspace")
async_mapper = AsyncTimeloopMapper(work_dir)


@dataclass
class PendingMapping:
    request: MappingRequest
    delivery_tag: int


pending: dict[Future, PendingMapping] = {}


def main():
    username, password, host = get_mq_config()
    credentials = pika.PlainCredentials(username, password, host)

    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=host,
            credentials=credentials,
        )
    )

    channel: Channel = connection.channel()

    channel.queue_declare(queue=REQUEST_QUEUE, durable=True)
    channel.queue_declare(queue=RESULTS_QUEUE, durable=True)

    max_parallel = os.cpu_count() or 1

    try:
        while True:
            while len(pending) < max_parallel:
                method, properties, body = channel.basic_get(
                    queue=REQUEST_QUEUE,
                    auto_ack=False,
                )

                if method is None:
                    break

                request = MappingRequest.model_validate_json(body)

                logger.info("Submitting request %s", request.id)

                future = async_mapper.map(request)

                pending[future] = PendingMapping(
                    request=request,
                    delivery_tag=method.delivery_tag,
                )


            if not pending:
                connection.sleep(0.1)
                continue

            done, _ = wait(
                pending.keys(),
                timeout=0.1,
                return_when=FIRST_COMPLETED,
            )

            for future in done:
                pending_mapping = pending.pop(future)

                try:
                    result = future.result()

                    channel.basic_publish(
                        exchange="",
                        routing_key=RESULTS_QUEUE,
                        body=result.model_dump_json().encode(),
                        properties=pika.BasicProperties(
                            content_type="application/json",
                            delivery_mode=2
                        )
                    )

                    channel.basic_ack(
                        delivery_tag=pending_mapping.delivery_tag
                    )

                except Exception:
                    logger.exception(
                        "Mapping failed for %s",
                        pending_mapping.request.id,
                    )

                    channel.basic_nack(
                        delivery_tag=pending_mapping.delivery_tag,
                        requeue=False, # send to DLQ instead?
                    )

    finally:
        connection.close()


if __name__ == "__main__":
    main()
"""
Exactly-once processing coordinator.

Combines consumer, producer, and state manager for end-to-end exactly-once semantics.
"""

import logging
from typing import Callable, Dict, Any
from .consumer import ExactlyOnceConsumer, ConsumerConfig
from .producer import HighThroughputProducer, ProducerConfig
from .state_manager import TransactionalStateManager

logger = logging.getLogger(__name__)


class ExactlyOnceProcessor:
    """
    End-to-end exactly-once processor.

    Coordinates consumer, producer, and state manager to achieve
    exactly-once processing guarantees across the entire pipeline.

    Example:
        >>> processor = ExactlyOnceProcessor(
        ...     consumer_config=ConsumerConfig(topics=["input"]),
        ...     producer_config=ProducerConfig(),
        ...     postgres_dsn="postgresql://localhost/db"
        ... )
        >>>
        >>> async def process(msg, producer, state):
        ...     # Check idempotency
        ...     if state.check_idempotency(msg['key']):
        ...         return True
        ...
        ...     # Process and produce output
        ...     async with state.transaction(msg) as tx:
        ...         result = transform(msg['value'])
        ...         await tx.publish_event("output", result)
        ...         await tx.save_to_db("results", result)
        ...
        ...     return True
        >>>
        >>> processor.start(process)
    """

    def __init__(
        self,
        consumer_config: ConsumerConfig,
        producer_config: ProducerConfig,
        postgres_dsn: str,
        redis_url: str = "redis://localhost:6379"
    ):
        """
        Initialize exactly-once processor.

        Args:
            consumer_config: Consumer configuration
            producer_config: Producer configuration
            postgres_dsn: PostgreSQL connection string
            redis_url: Redis URL
        """
        self.consumer = ExactlyOnceConsumer(consumer_config)

        self.producer = HighThroughputProducer(producer_config)

        self.state_manager = TransactionalStateManager(
            postgres_dsn=postgres_dsn,
            redis_url=redis_url,
            kafka_bootstrap=producer_config.bootstrap_servers
        )

        logger.info("Exactly-once processor initialized")

    def start(
        self,
        process_func: Callable[[Dict[str, Any], HighThroughputProducer, TransactionalStateManager], bool],
        max_messages: int = None
    ):
        """
        Start exactly-once processing loop.

        Args:
            process_func: Processing function(msg, producer, state_mgr) -> bool
            max_messages: Maximum messages to process (None = infinite)
        """

        def wrapped_process(msg: Dict[str, Any]) -> bool:
            """Wrapper that provides producer and state manager."""
            return process_func(msg, self.producer, self.state_manager)

        self.consumer.consume(
            process_func=wrapped_process,
            max_messages=max_messages
        )

    def close(self):
        """Close all components."""
        logger.info("Closing exactly-once processor...")
        self.consumer.close()
        self.producer.close()
        self.state_manager.close()
        logger.info("Exactly-once processor closed")

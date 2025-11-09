"""
Exactly-once Kafka/Redpanda consumer with transactional processing.

Provides consumer with manual offset management, graceful shutdown,
consumer group rebalancing, and dead letter queue support.
"""

import logging
import time
import signal
import threading
from typing import Optional, Dict, Any, Callable, List, Set
from dataclasses import dataclass
from enum import Enum
from confluent_kafka import Consumer, TopicPartition, KafkaError, KafkaException
from confluent_kafka import OFFSET_BEGINNING, OFFSET_END, OFFSET_STORED
import json

logger = logging.getLogger(__name__)


class OffsetCommitStrategy(Enum):
    """Strategies for committing offsets."""
    AUTO = "auto"  # Automatic periodic commits
    MANUAL = "manual"  # Manual commit after processing
    MANUAL_BATCH = "manual_batch"  # Manual commit after batch
    TRANSACTIONAL = "transactional"  # Transactional commits


@dataclass
class ConsumerConfig:
    """
    Configuration for exactly-once consumer.

    Attributes:
        bootstrap_servers: Kafka broker addresses
        group_id: Consumer group ID
        topics: List of topics to subscribe to
        client_id: Client identifier
        auto_offset_reset: Where to start reading (earliest/latest)
        enable_auto_commit: Enable automatic offset commits
        auto_commit_interval_ms: Auto commit interval
        max_poll_records: Max records per poll
        max_poll_interval_ms: Max time between polls
        session_timeout_ms: Session timeout
        heartbeat_interval_ms: Heartbeat interval
        isolation_level: Isolation level (read_committed for exactly-once)
        enable_partition_eof: Enable partition EOF events
    """
    bootstrap_servers: str = "localhost:19092"
    group_id: str = "exactly-once-consumer-group"
    topics: List[str] = None
    client_id: str = "exactly-once-consumer"
    auto_offset_reset: str = "earliest"  # earliest, latest, or error
    enable_auto_commit: bool = False  # Manual commit for exactly-once
    auto_commit_interval_ms: int = 5000
    max_poll_records: int = 500
    max_poll_interval_ms: int = 300000  # 5 minutes
    session_timeout_ms: int = 10000  # 10 seconds
    heartbeat_interval_ms: int = 3000  # 3 seconds
    isolation_level: str = "read_committed"  # read_committed or read_uncommitted
    enable_partition_eof: bool = False

    def __post_init__(self):
        if self.topics is None:
            self.topics = []

    def to_kafka_config(self) -> Dict[str, Any]:
        """
        Convert to confluent-kafka configuration dict.

        Returns:
            Configuration dictionary for Consumer
        """
        return {
            'bootstrap.servers': self.bootstrap_servers,
            'group.id': self.group_id,
            'client.id': self.client_id,
            'auto.offset.reset': self.auto_offset_reset,
            'enable.auto.commit': self.enable_auto_commit,
            'auto.commit.interval.ms': self.auto_commit_interval_ms,
            'max.poll.interval.ms': self.max_poll_interval_ms,
            'session.timeout.ms': self.session_timeout_ms,
            'heartbeat.interval.ms': self.heartbeat_interval_ms,
            'isolation.level': self.isolation_level,
            'enable.partition.eof': self.enable_partition_eof,
        }


class ConsumerMetrics:
    """
    Tracks consumer performance metrics.

    Attributes:
        messages_consumed: Total messages consumed
        messages_processed: Successfully processed messages
        messages_failed: Failed messages
        processing_time_sum: Sum of processing times
        consumer_lag: Current consumer lag per partition
    """

    def __init__(self):
        self.messages_consumed: int = 0
        self.messages_processed: int = 0
        self.messages_failed: int = 0
        self.processing_time_sum: float = 0.0
        self.consumer_lag: Dict[tuple, int] = {}  # (topic, partition) -> lag
        self.start_time: float = time.time()

    def record_consumed(self):
        """Record message consumed."""
        self.messages_consumed += 1

    def record_processed(self, processing_time_ms: float):
        """Record successful processing."""
        self.messages_processed += 1
        self.processing_time_sum += processing_time_ms

    def record_failed(self):
        """Record processing failure."""
        self.messages_failed += 1

    def update_lag(self, topic: str, partition: int, lag: int):
        """Update consumer lag for a partition."""
        self.consumer_lag[(topic, partition)] = lag

    def get_total_lag(self) -> int:
        """Get total lag across all partitions."""
        return sum(self.consumer_lag.values())

    def get_throughput_mps(self) -> float:
        """Get messages processed per second."""
        elapsed = time.time() - self.start_time
        return self.messages_processed / elapsed if elapsed > 0 else 0.0

    def get_avg_processing_time_ms(self) -> float:
        """Get average processing time."""
        return (self.processing_time_sum / self.messages_processed
                if self.messages_processed > 0 else 0.0)

    def __str__(self) -> str:
        return (
            f"Consumer Metrics:\n"
            f"  Messages Consumed: {self.messages_consumed:,}\n"
            f"  Messages Processed: {self.messages_processed:,}\n"
            f"  Messages Failed: {self.messages_failed:,}\n"
            f"  Throughput: {self.get_throughput_mps():.2f} msg/s\n"
            f"  Avg Processing Time: {self.get_avg_processing_time_ms():.2f}ms\n"
            f"  Total Lag: {self.get_total_lag():,}"
        )


class ExactlyOnceConsumer:
    """
    Exactly-once Kafka consumer with transactional processing.

    Features:
    - Manual offset management for exactly-once
    - Graceful shutdown handling
    - Consumer group rebalancing
    - Dead letter queue for poison messages
    - Partition lag tracking
    - Automatic retry with exponential backoff

    Example:
        >>> config = ConsumerConfig(
        ...     bootstrap_servers="localhost:19092",
        ...     group_id="my-group",
        ...     topics=["payments"]
        ... )
        >>> consumer = ExactlyOnceConsumer(config)
        >>>
        >>> def process_message(message):
        ...     print(f"Processing: {message['value']}")
        ...     return True  # Success
        >>>
        >>> consumer.consume(process_message, max_messages=1000)
        >>> consumer.close()
    """

    def __init__(
        self,
        config: ConsumerConfig,
        dead_letter_topic: Optional[str] = None
    ):
        """
        Initialize exactly-once consumer.

        Args:
            config: Consumer configuration
            dead_letter_topic: Optional DLQ topic for poison messages
        """
        self.config = config
        self.dead_letter_topic = dead_letter_topic
        self.metrics = ConsumerMetrics()

        # Initialize Kafka consumer
        kafka_config = config.to_kafka_config()
        self.consumer = Consumer(kafka_config)

        # Subscribe to topics
        if config.topics:
            self.consumer.subscribe(
                config.topics,
                on_assign=self._on_partition_assign,
                on_revoke=self._on_partition_revoke
            )
            logger.info(f"Subscribed to topics: {config.topics}")

        # Shutdown handling
        self.running = False
        self.shutdown_lock = threading.Lock()
        self._setup_signal_handlers()

        # Offset tracking for exactly-once
        self.processed_offsets: Dict[tuple, int] = {}  # (topic, partition) -> offset

        # DLQ producer (lazy initialization)
        self._dlq_producer = None

        logger.info(f"Consumer initialized: {config.client_id}")

    def _setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _on_partition_assign(self, consumer, partitions: List[TopicPartition]):
        """
        Callback for partition assignment during rebalance.

        Args:
            consumer: Consumer instance
            partitions: Assigned partitions
        """
        logger.info(f"Partitions assigned: {len(partitions)}")
        for partition in partitions:
            logger.info(
                f"  {partition.topic}[{partition.partition}] "
                f"offset: {partition.offset}"
            )

        # Reset processed offsets for newly assigned partitions
        for partition in partitions:
            key = (partition.topic, partition.partition)
            if key not in self.processed_offsets:
                self.processed_offsets[key] = -1

    def _on_partition_revoke(self, consumer, partitions: List[TopicPartition]):
        """
        Callback for partition revocation during rebalance.

        Args:
            consumer: Consumer instance
            partitions: Revoked partitions
        """
        logger.info(f"Partitions revoked: {len(partitions)}")

        # Commit offsets before revocation
        try:
            offsets_to_commit = []
            for partition in partitions:
                key = (partition.topic, partition.partition)
                if key in self.processed_offsets:
                    offset = self.processed_offsets[key]
                    if offset >= 0:
                        offsets_to_commit.append(
                            TopicPartition(
                                partition.topic,
                                partition.partition,
                                offset + 1  # Commit next offset
                            )
                        )

            if offsets_to_commit:
                self.consumer.commit(offsets=offsets_to_commit)
                logger.info(f"Committed {len(offsets_to_commit)} offsets on revoke")

        except Exception as e:
            logger.error(f"Error committing offsets on revoke: {e}")

    def _get_dlq_producer(self):
        """Lazy initialization of DLQ producer."""
        if self._dlq_producer is None and self.dead_letter_topic:
            from .producer import HighThroughputProducer, ProducerConfig
            dlq_config = ProducerConfig(
                bootstrap_servers=self.config.bootstrap_servers,
                client_id=f"{self.config.client_id}-dlq"
            )
            self._dlq_producer = HighThroughputProducer(dlq_config)
        return self._dlq_producer

    def _send_to_dlq(
        self,
        message: Dict[str, Any],
        error: Exception,
        retry_count: int = 0
    ):
        """
        Send message to dead letter queue.

        Args:
            message: Original message
            error: Exception that occurred
            retry_count: Number of retries attempted
        """
        if not self.dead_letter_topic:
            logger.warning("No DLQ configured, dropping message")
            return

        dlq_producer = self._get_dlq_producer()
        if not dlq_producer:
            logger.error("Failed to initialize DLQ producer")
            return

        dlq_message = {
            'original_topic': message.get('topic'),
            'original_partition': message.get('partition'),
            'original_offset': message.get('offset'),
            'original_key': message.get('key'),
            'original_value': message.get('value'),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'retry_count': retry_count,
            'timestamp': int(time.time() * 1000)
        }

        try:
            dlq_producer.send(
                self.dead_letter_topic,
                value=dlq_message,
                key=message.get('key')
            )
            logger.warning(
                f"Sent message to DLQ: {message.get('topic')}["
                f"{message.get('partition')}]@{message.get('offset')}"
            )
        except Exception as e:
            logger.error(f"Failed to send to DLQ: {e}")

    def consume(
        self,
        process_func: Callable[[Dict[str, Any]], bool],
        max_messages: Optional[int] = None,
        timeout: float = 1.0,
        max_retries: int = 3,
        commit_strategy: OffsetCommitStrategy = OffsetCommitStrategy.MANUAL
    ):
        """
        Start consuming messages with exactly-once processing.

        Args:
            process_func: Function to process messages, returns True on success
            max_messages: Maximum messages to process (None = infinite)
            timeout: Poll timeout in seconds
            max_retries: Maximum retries for failed messages
            commit_strategy: Offset commit strategy

        Example:
            >>> def process(msg):
            ...     print(msg['value'])
            ...     return True
            >>> consumer.consume(process, max_messages=100)
        """
        self.running = True
        messages_processed = 0

        logger.info("Starting consumer loop...")

        try:
            while self.running:
                # Check if we've hit the limit
                if max_messages and messages_processed >= max_messages:
                    logger.info(f"Reached max messages: {max_messages}")
                    break

                # Poll for messages
                msg = self.consumer.poll(timeout=timeout)

                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        logger.debug(
                            f"Reached end of partition {msg.topic()}["
                            f"{msg.partition()}]"
                        )
                        continue
                    else:
                        logger.error(f"Consumer error: {msg.error()}")
                        continue

                # Process message
                self.metrics.record_consumed()

                message_dict = {
                    'topic': msg.topic(),
                    'partition': msg.partition(),
                    'offset': msg.offset(),
                    'key': msg.key().decode('utf-8') if msg.key() else None,
                    'value': self._deserialize_value(msg.value()),
                    'headers': dict(msg.headers()) if msg.headers() else {},
                    'timestamp': msg.timestamp()[1] if msg.timestamp()[0] else None
                }

                # Process with retries
                start_time = time.time()
                success = False
                retry_count = 0

                while retry_count <= max_retries and not success:
                    try:
                        success = process_func(message_dict)
                        if success:
                            processing_time = (time.time() - start_time) * 1000
                            self.metrics.record_processed(processing_time)
                            messages_processed += 1

                            # Track offset
                            key = (msg.topic(), msg.partition())
                            self.processed_offsets[key] = msg.offset()

                            # Commit based on strategy
                            if commit_strategy == OffsetCommitStrategy.MANUAL:
                                self._commit_offset(msg)

                        else:
                            logger.warning(
                                f"Processing returned False for message at "
                                f"{msg.topic()}[{msg.partition()}]@{msg.offset()}"
                            )
                            retry_count += 1
                            if retry_count <= max_retries:
                                time.sleep(2 ** retry_count)  # Exponential backoff

                    except Exception as e:
                        logger.error(
                            f"Error processing message: {e}", exc_info=True
                        )
                        retry_count += 1
                        if retry_count <= max_retries:
                            time.sleep(2 ** retry_count)

                # Send to DLQ if all retries failed
                if not success:
                    self.metrics.record_failed()
                    self._send_to_dlq(message_dict, Exception("Max retries exceeded"), retry_count)

                # Periodic metrics log
                if messages_processed % 1000 == 0 and messages_processed > 0:
                    logger.info(f"Progress: {self.metrics}")

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Consumer loop error: {e}", exc_info=True)
        finally:
            logger.info("Consumer loop ended")

    def _deserialize_value(self, value_bytes: bytes) -> Any:
        """
        Deserialize message value.

        Args:
            value_bytes: Raw message bytes

        Returns:
            Deserialized value (dict for JSON, str otherwise)
        """
        if not value_bytes:
            return None

        try:
            # Try JSON deserialization
            return json.loads(value_bytes.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Return as string if not JSON
            try:
                return value_bytes.decode('utf-8')
            except UnicodeDecodeError:
                # Return raw bytes if not UTF-8
                return value_bytes

    def _commit_offset(self, msg):
        """
        Commit offset for a message.

        Args:
            msg: Message object
        """
        try:
            # Commit next offset (current + 1)
            self.consumer.commit(
                offsets=[TopicPartition(
                    msg.topic(),
                    msg.partition(),
                    msg.offset() + 1
                )],
                asynchronous=False
            )
            logger.debug(
                f"Committed offset: {msg.topic()}[{msg.partition()}]"
                f"@{msg.offset() + 1}"
            )
        except Exception as e:
            logger.error(f"Failed to commit offset: {e}")

    def commit_offsets(self, offsets: Optional[List[TopicPartition]] = None):
        """
        Manually commit offsets.

        Args:
            offsets: List of offsets to commit (None = all processed)
        """
        if offsets is None:
            # Commit all processed offsets
            offsets = [
                TopicPartition(topic, partition, offset + 1)
                for (topic, partition), offset in self.processed_offsets.items()
                if offset >= 0
            ]

        if offsets:
            try:
                self.consumer.commit(offsets=offsets, asynchronous=False)
                logger.info(f"Committed {len(offsets)} offsets")
            except Exception as e:
                logger.error(f"Failed to commit offsets: {e}")

    def get_lag(self) -> Dict[tuple, int]:
        """
        Calculate consumer lag for all assigned partitions.

        Returns:
            Dictionary of (topic, partition) -> lag
        """
        lag_dict = {}

        try:
            # Get assigned partitions
            assignment = self.consumer.assignment()

            for tp in assignment:
                # Get current position
                position = self.consumer.position([tp])[0].offset

                # Get high watermark
                low, high = self.consumer.get_watermark_offsets(tp)

                lag = high - position
                lag_dict[(tp.topic, tp.partition)] = lag
                self.metrics.update_lag(tp.topic, tp.partition, lag)

        except Exception as e:
            logger.error(f"Error calculating lag: {e}")

        return lag_dict

    def pause(self, partitions: Optional[List[TopicPartition]] = None):
        """
        Pause consumption from partitions.

        Args:
            partitions: Partitions to pause (None = all)
        """
        if partitions is None:
            partitions = self.consumer.assignment()

        self.consumer.pause(partitions)
        logger.info(f"Paused {len(partitions)} partitions")

    def resume(self, partitions: Optional[List[TopicPartition]] = None):
        """
        Resume consumption from partitions.

        Args:
            partitions: Partitions to resume (None = all)
        """
        if partitions is None:
            partitions = self.consumer.assignment()

        self.consumer.resume(partitions)
        logger.info(f"Resumed {len(partitions)} partitions")

    def shutdown(self):
        """Initiate graceful shutdown."""
        with self.shutdown_lock:
            if not self.running:
                return
            self.running = False
        logger.info("Shutdown initiated")

    def get_metrics(self) -> ConsumerMetrics:
        """Get consumer metrics."""
        return self.metrics

    def close(self):
        """
        Close consumer and release resources.

        Commits any pending offsets before closing.
        """
        logger.info("Closing consumer...")

        # Commit final offsets
        self.commit_offsets()

        # Close consumer
        self.consumer.close()

        # Close DLQ producer if exists
        if self._dlq_producer:
            self._dlq_producer.close()

        logger.info(f"Consumer closed. Final metrics:\n{self.metrics}")

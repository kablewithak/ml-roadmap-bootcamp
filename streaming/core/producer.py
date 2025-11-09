"""
High-throughput Kafka/Redpanda producer with exactly-once semantics.

Provides idempotent producer with optimized batching, compression,
and async send capabilities targeting 100k+ messages/sec.
"""

import logging
import time
import asyncio
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field
from enum import Enum
from confluent_kafka import Producer, KafkaError, KafkaException
import json
import io
import avro.schema
import avro.io

logger = logging.getLogger(__name__)


class CompressionType(Enum):
    """Compression algorithms supported by Kafka."""
    NONE = "none"
    GZIP = "gzip"
    SNAPPY = "snappy"
    LZ4 = "lz4"
    ZSTD = "zstd"  # Best compression ratio for high-throughput


@dataclass
class ProducerConfig:
    """
    Configuration for high-throughput producer.

    Optimized for 100k+ msg/sec with exactly-once semantics.

    Attributes:
        bootstrap_servers: Kafka broker addresses
        client_id: Client identifier
        compression_type: Compression algorithm
        linger_ms: Time to wait before sending batch (latency vs throughput)
        batch_size: Maximum batch size in bytes
        buffer_memory: Total memory for buffering
        max_in_flight: Max unacknowledged requests (1 for strict ordering)
        acks: Acknowledgment level ('all' for durability)
        retries: Number of retries
        enable_idempotence: Enable idempotent producer
        transactional_id: Transaction ID for exactly-once (optional)
    """
    bootstrap_servers: str = "localhost:19092"
    client_id: str = "high-throughput-producer"
    compression_type: CompressionType = CompressionType.ZSTD
    linger_ms: int = 10  # Wait 10ms to batch messages
    batch_size: int = 1000000  # 1MB batches
    buffer_memory: int = 67108864  # 64MB buffer
    max_in_flight: int = 5  # Pipeline requests for throughput
    acks: str = "all"  # Wait for all replicas
    retries: int = 10  # Retry on transient errors
    enable_idempotence: bool = True
    transactional_id: Optional[str] = None
    max_request_size: int = 1048576  # 1MB max message size
    request_timeout_ms: int = 30000
    retry_backoff_ms: int = 100

    def to_kafka_config(self) -> Dict[str, Any]:
        """
        Convert to confluent-kafka configuration dict.

        Returns:
            Configuration dictionary for Producer
        """
        config = {
            'bootstrap.servers': self.bootstrap_servers,
            'client.id': self.client_id,
            'compression.type': self.compression_type.value,
            'linger.ms': self.linger_ms,
            'batch.size': self.batch_size,
            'buffer.memory': self.buffer_memory,
            'max.in.flight.requests.per.connection': self.max_in_flight,
            'acks': self.acks,
            'retries': self.retries,
            'enable.idempotence': self.enable_idempotence,
            'max.request.size': self.max_request_size,
            'request.timeout.ms': self.request_timeout_ms,
            'retry.backoff.ms': self.retry_backoff_ms,
        }

        if self.transactional_id:
            config['transactional.id'] = self.transactional_id

        return config


class ProducerMetrics:
    """
    Tracks producer performance metrics.

    Attributes:
        messages_sent: Total messages sent successfully
        messages_failed: Total messages that failed
        bytes_sent: Total bytes sent
        send_latency_sum: Sum of send latencies for averaging
        send_count: Count for latency averaging
    """

    def __init__(self):
        self.messages_sent: int = 0
        self.messages_failed: int = 0
        self.bytes_sent: int = 0
        self.send_latency_sum: float = 0.0
        self.send_count: int = 0
        self.start_time: float = time.time()

    def record_success(self, message_size: int, latency_ms: float):
        """Record successful send."""
        self.messages_sent += 1
        self.bytes_sent += message_size
        self.send_latency_sum += latency_ms
        self.send_count += 1

    def record_failure(self):
        """Record failed send."""
        self.messages_failed += 1

    def get_throughput_mps(self) -> float:
        """Get messages per second."""
        elapsed = time.time() - self.start_time
        return self.messages_sent / elapsed if elapsed > 0 else 0.0

    def get_throughput_mbps(self) -> float:
        """Get megabytes per second."""
        elapsed = time.time() - self.start_time
        return (self.bytes_sent / (1024 * 1024)) / elapsed if elapsed > 0 else 0.0

    def get_avg_latency_ms(self) -> float:
        """Get average send latency in milliseconds."""
        return self.send_latency_sum / self.send_count if self.send_count > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"Producer Metrics:\n"
            f"  Messages Sent: {self.messages_sent:,}\n"
            f"  Messages Failed: {self.messages_failed:,}\n"
            f"  Throughput: {self.get_throughput_mps():.2f} msg/s "
            f"({self.get_throughput_mbps():.2f} MB/s)\n"
            f"  Avg Latency: {self.get_avg_latency_ms():.2f}ms"
        )


class HighThroughputProducer:
    """
    High-throughput Kafka producer with exactly-once semantics.

    Features:
    - Idempotent producer (deduplication)
    - Optimized batching and compression
    - Async send with callbacks
    - Automatic retry with exponential backoff
    - Transaction support for exactly-once
    - Avro serialization with schema registry

    Example:
        >>> config = ProducerConfig(bootstrap_servers="localhost:19092")
        >>> producer = HighThroughputProducer(config)
        >>> await producer.send("payments", key="user123", value={"amount": 100})
        >>> producer.flush()
        >>> producer.close()
    """

    def __init__(
        self,
        config: ProducerConfig,
        error_callback: Optional[Callable[[Exception], None]] = None
    ):
        """
        Initialize high-throughput producer.

        Args:
            config: Producer configuration
            error_callback: Optional callback for error handling
        """
        self.config = config
        self.error_callback = error_callback
        self.metrics = ProducerMetrics()

        # Initialize Kafka producer
        kafka_config = config.to_kafka_config()
        self.producer = Producer(kafka_config)

        # Transaction support
        self.in_transaction = False
        if config.transactional_id:
            self.producer.init_transactions()
            logger.info(f"Initialized transactional producer: {config.transactional_id}")
        else:
            logger.info("Initialized idempotent producer")

        # Avro serialization support
        self.avro_writers: Dict[str, avro.io.DatumWriter] = {}

        logger.info(f"Producer initialized: {config.client_id}")

    def _delivery_callback(
        self,
        err: Optional[KafkaError],
        msg,
        start_time: float,
        user_callback: Optional[Callable] = None
    ):
        """
        Internal callback for message delivery.

        Args:
            err: Error if delivery failed
            msg: Message object
            start_time: Time when send was initiated
            user_callback: Optional user-provided callback
        """
        latency_ms = (time.time() - start_time) * 1000

        if err is not None:
            logger.error(f"Message delivery failed: {err}")
            self.metrics.record_failure()

            if self.error_callback:
                self.error_callback(KafkaException(err))

            if user_callback:
                user_callback(err, None)
        else:
            message_size = len(msg.value()) if msg.value() else 0
            self.metrics.record_success(message_size, latency_ms)

            if user_callback:
                user_callback(None, {
                    'topic': msg.topic(),
                    'partition': msg.partition(),
                    'offset': msg.offset(),
                    'latency_ms': latency_ms
                })

            if self.metrics.messages_sent % 10000 == 0:
                logger.info(f"Progress: {self.metrics}")

    def send(
        self,
        topic: str,
        value: Any,
        key: Optional[Any] = None,
        partition: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
        callback: Optional[Callable] = None
    ):
        """
        Send a message asynchronously.

        This method returns immediately and the message is sent asynchronously.
        Use flush() to wait for all messages to be delivered.

        Args:
            topic: Topic name
            value: Message value (dict for JSON, bytes for raw)
            key: Optional message key
            partition: Optional specific partition
            headers: Optional message headers
            callback: Optional callback(error, result)

        Raises:
            BufferError: If producer queue is full
        """
        start_time = time.time()

        # Serialize value
        if isinstance(value, dict):
            value_bytes = json.dumps(value).encode('utf-8')
        elif isinstance(value, bytes):
            value_bytes = value
        else:
            value_bytes = str(value).encode('utf-8')

        # Serialize key
        key_bytes = None
        if key is not None:
            if isinstance(key, bytes):
                key_bytes = key
            else:
                key_bytes = str(key).encode('utf-8')

        # Convert headers
        kafka_headers = None
        if headers:
            kafka_headers = [(k, v.encode('utf-8') if isinstance(v, str) else v)
                           for k, v in headers.items()]

        # Send message
        try:
            self.producer.produce(
                topic=topic,
                value=value_bytes,
                key=key_bytes,
                partition=partition,
                headers=kafka_headers,
                on_delivery=lambda err, msg: self._delivery_callback(
                    err, msg, start_time, callback
                )
            )

            # Trigger callbacks for completed sends (non-blocking)
            self.producer.poll(0)

        except BufferError as e:
            logger.warning("Producer queue full, flushing...")
            self.flush()
            # Retry after flush
            self.producer.produce(
                topic=topic,
                value=value_bytes,
                key=key_bytes,
                partition=partition,
                headers=kafka_headers,
                on_delivery=lambda err, msg: self._delivery_callback(
                    err, msg, start_time, callback
                )
            )

    def send_avro(
        self,
        topic: str,
        value: Dict[str, Any],
        schema: Dict[str, Any],
        key: Optional[Any] = None,
        callback: Optional[Callable] = None
    ):
        """
        Send a message with Avro serialization.

        Args:
            topic: Topic name
            value: Message value as dict
            schema: Avro schema definition
            key: Optional message key
            callback: Optional callback
        """
        # Cache Avro writer for schema
        schema_str = json.dumps(schema)
        if schema_str not in self.avro_writers:
            avro_schema = avro.schema.parse(schema_str)
            self.avro_writers[schema_str] = avro.io.DatumWriter(avro_schema)

        writer = self.avro_writers[schema_str]

        # Serialize to Avro binary
        bytes_writer = io.BytesIO()
        encoder = avro.io.BinaryEncoder(bytes_writer)
        writer.write(value, encoder)
        value_bytes = bytes_writer.getvalue()

        self.send(topic, value_bytes, key=key, callback=callback)

    def begin_transaction(self):
        """
        Begin a transaction for exactly-once semantics.

        All messages sent after this will be part of the transaction
        until commit_transaction() or abort_transaction() is called.

        Raises:
            RuntimeError: If transactional_id not configured
        """
        if not self.config.transactional_id:
            raise RuntimeError("Transaction support requires transactional_id")

        self.producer.begin_transaction()
        self.in_transaction = True
        logger.debug("Transaction started")

    def commit_transaction(self, timeout: float = 60.0):
        """
        Commit the current transaction.

        Args:
            timeout: Commit timeout in seconds

        Raises:
            KafkaException: If commit fails
        """
        if not self.in_transaction:
            raise RuntimeError("No active transaction")

        self.producer.commit_transaction(timeout)
        self.in_transaction = False
        logger.debug("Transaction committed")

    def abort_transaction(self, timeout: float = 60.0):
        """
        Abort the current transaction.

        Args:
            timeout: Abort timeout in seconds
        """
        if not self.in_transaction:
            raise RuntimeError("No active transaction")

        self.producer.abort_transaction(timeout)
        self.in_transaction = False
        logger.warning("Transaction aborted")

    def flush(self, timeout: float = 60.0) -> int:
        """
        Wait for all messages to be delivered.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            Number of messages still in queue
        """
        remaining = self.producer.flush(timeout)
        if remaining > 0:
            logger.warning(f"{remaining} messages still in queue after flush")
        return remaining

    def get_metrics(self) -> ProducerMetrics:
        """Get producer metrics."""
        return self.metrics

    def close(self):
        """
        Close the producer and release resources.

        Waits for all pending messages to be delivered.
        """
        logger.info("Closing producer...")
        self.flush()
        # Producer doesn't have explicit close in confluent-kafka
        logger.info(f"Producer closed. Final metrics:\n{self.metrics}")


async def async_send_batch(
    producer: HighThroughputProducer,
    topic: str,
    messages: List[Dict[str, Any]],
    key_func: Optional[Callable[[Dict[str, Any]], str]] = None
) -> int:
    """
    Send a batch of messages asynchronously with progress tracking.

    Args:
        producer: Producer instance
        topic: Topic name
        messages: List of message values
        key_func: Optional function to extract key from message

    Returns:
        Number of messages sent successfully
    """
    total = len(messages)
    success_count = 0
    failed_count = 0

    def batch_callback(err, result):
        nonlocal success_count, failed_count
        if err:
            failed_count += 1
        else:
            success_count += 1

    logger.info(f"Sending batch of {total:,} messages to {topic}")

    for i, message in enumerate(messages):
        key = key_func(message) if key_func else None
        producer.send(topic, message, key=key, callback=batch_callback)

        # Periodic progress log
        if (i + 1) % 10000 == 0:
            logger.info(f"Sent {i + 1:,}/{total:,} messages")

        # Small yield to allow other tasks
        if i % 1000 == 0:
            await asyncio.sleep(0)

    # Wait for all messages to be delivered
    producer.flush()

    logger.info(
        f"Batch complete: {success_count:,} succeeded, {failed_count:,} failed"
    )

    return success_count

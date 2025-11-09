"""
Transactional State Manager for distributed exactly-once processing.

Manages consistency across Kafka, PostgreSQL, and local state using
two-phase commit patterns with saga compensation.
"""

import logging
import asyncio
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
from uuid import uuid4, UUID
from datetime import datetime
import psycopg2
from psycopg2 import pool
import redis
from .producer import HighThroughputProducer, ProducerConfig

logger = logging.getLogger(__name__)


class TransactionStatus(Enum):
    """Transaction lifecycle states."""
    PREPARED = "PREPARED"
    COMMITTED = "COMMITTED"
    ROLLED_BACK = "ROLLED_BACK"
    FAILED = "FAILED"


@dataclass
class TransactionContext:
    """
    Context for a distributed transaction.

    Attributes:
        transaction_id: Unique transaction identifier
        kafka_topic: Source Kafka topic
        kafka_partition: Source partition
        kafka_offset: Source offset
        message_key: Message key
        status: Transaction status
        retry_count: Number of retries
        error_message: Error message if failed
    """
    transaction_id: UUID
    kafka_topic: str
    kafka_partition: int
    kafka_offset: int
    message_key: Optional[str] = None
    status: TransactionStatus = TransactionStatus.PREPARED
    retry_count: int = 0
    error_message: Optional[str] = None


class TransactionalStateManager:
    """
    Manages exactly-once processing across Kafka, PostgreSQL, and Redis.

    Problem: Kafka commit succeeds but DB write fails = inconsistent state
    Solution: Two-phase commit pattern with saga compensation

    Features:
    - Distributed transaction coordination
    - Automatic rollback on failure
    - Idempotency via deduplication
    - State recovery after crashes
    - Saga pattern for compensation

    Example:
        >>> state_mgr = TransactionalStateManager(
        ...     postgres_dsn="postgresql://user:pass@localhost/db",
        ...     redis_url="redis://localhost:6379"
        ... )
        >>>
        >>> async def process_payment(msg):
        ...     async with state_mgr.transaction(msg) as tx:
        ...         # Business logic here
        ...         await tx.save_to_db(payment_data)
        ...         await tx.publish_event(fraud_event)
        ...         # Auto-commit if no exception
        >>>
        >>> await process_payment(message)
    """

    def __init__(
        self,
        postgres_dsn: str,
        redis_url: str = "redis://localhost:6379",
        kafka_bootstrap: str = "localhost:19092",
        max_pool_size: int = 20
    ):
        """
        Initialize transactional state manager.

        Args:
            postgres_dsn: PostgreSQL connection string
            redis_url: Redis connection URL
            kafka_bootstrap: Kafka bootstrap servers
            max_pool_size: Max PostgreSQL connection pool size
        """
        # PostgreSQL connection pool
        self.pg_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=max_pool_size,
            dsn=postgres_dsn
        )

        # Redis for fast state lookups and caching
        self.redis_client = redis.from_url(redis_url, decode_responses=True)

        # Kafka producer for output events
        producer_config = ProducerConfig(
            bootstrap_servers=kafka_bootstrap,
            client_id="state-manager-producer",
            transactional_id=f"state-manager-{uuid4()}"
        )
        self.kafka_producer = HighThroughputProducer(producer_config)

        logger.info("Transactional State Manager initialized")

    def transaction(self, message: Dict[str, Any]) -> 'Transaction':
        """
        Create a new transaction context.

        Args:
            message: Source Kafka message

        Returns:
            Transaction context manager
        """
        tx_id = uuid4()
        ctx = TransactionContext(
            transaction_id=tx_id,
            kafka_topic=message['topic'],
            kafka_partition=message['partition'],
            kafka_offset=message['offset'],
            message_key=message.get('key')
        )

        return Transaction(self, ctx)

    def check_idempotency(self, idempotency_key: str) -> bool:
        """
        Check if message has already been processed.

        Args:
            idempotency_key: Unique key for the message

        Returns:
            True if already processed, False otherwise
        """
        try:
            # Check Redis cache first (fast path)
            cached = self.redis_client.get(f"processed:{idempotency_key}")
            if cached:
                logger.debug(f"Message already processed (cache): {idempotency_key}")
                return True

            # Check PostgreSQL (authoritative)
            conn = self.pg_pool.getconn()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT 1 FROM processed_messages WHERE idempotency_key = %s",
                    (idempotency_key,)
                )
                result = cursor.fetchone()
                cursor.close()

                if result:
                    # Update cache for future lookups
                    self.redis_client.setex(
                        f"processed:{idempotency_key}",
                        86400,  # 24 hours TTL
                        "1"
                    )
                    logger.debug(f"Message already processed (db): {idempotency_key}")
                    return True

                return False

            finally:
                self.pg_pool.putconn(conn)

        except Exception as e:
            logger.error(f"Error checking idempotency: {e}")
            # On error, assume not processed (at-least-once fallback)
            return False

    def mark_processed(
        self,
        idempotency_key: str,
        message_id: str,
        topic: str,
        partition: int,
        offset: int,
        kafka_timestamp: int,
        processing_duration_ms: int
    ):
        """
        Mark message as processed for idempotency.

        Args:
            idempotency_key: Unique key
            message_id: Message ID
            topic: Kafka topic
            partition: Partition
            offset: Offset
            kafka_timestamp: Message timestamp
            processing_duration_ms: Processing time
        """
        conn = self.pg_pool.getconn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO processed_messages
                (message_id, kafka_topic, kafka_partition, kafka_offset,
                 kafka_timestamp, idempotency_key, processing_duration_ms)
                VALUES (%s, %s, %s, %s, to_timestamp(%s/1000.0), %s, %s)
                ON CONFLICT (idempotency_key) DO NOTHING
                """,
                (message_id, topic, partition, offset, kafka_timestamp,
                 idempotency_key, processing_duration_ms)
            )
            conn.commit()
            cursor.close()

            # Update cache
            self.redis_client.setex(
                f"processed:{idempotency_key}",
                86400,
                "1"
            )

        except Exception as e:
            logger.error(f"Error marking as processed: {e}")
            conn.rollback()
            raise
        finally:
            self.pg_pool.putconn(conn)

    def get_consumer_offset(
        self,
        consumer_group: str,
        topic: str,
        partition: int
    ) -> Optional[int]:
        """
        Get committed offset for a consumer group.

        Args:
            consumer_group: Consumer group ID
            topic: Topic name
            partition: Partition number

        Returns:
            Committed offset or None
        """
        conn = self.pg_pool.getconn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT offset_value FROM consumer_offsets
                WHERE consumer_group = %s AND topic = %s AND partition = %s
                """,
                (consumer_group, topic, partition)
            )
            result = cursor.fetchone()
            cursor.close()
            return result[0] if result else None
        finally:
            self.pg_pool.putconn(conn)

    def commit_consumer_offset(
        self,
        consumer_group: str,
        topic: str,
        partition: int,
        offset: int,
        metadata: Optional[str] = None
    ):
        """
        Commit consumer offset to database.

        Args:
            consumer_group: Consumer group ID
            topic: Topic name
            partition: Partition number
            offset: Offset to commit
            metadata: Optional metadata
        """
        conn = self.pg_pool.getconn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO consumer_offsets
                (consumer_group, topic, partition, offset_value, metadata)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (consumer_group, topic, partition)
                DO UPDATE SET offset_value = EXCLUDED.offset_value,
                            metadata = EXCLUDED.metadata,
                            committed_at = NOW()
                """,
                (consumer_group, topic, partition, offset, metadata)
            )
            conn.commit()
            cursor.close()
        finally:
            self.pg_pool.putconn(conn)

    def close(self):
        """Close all connections."""
        logger.info("Closing state manager...")
        self.pg_pool.closeall()
        self.redis_client.close()
        self.kafka_producer.close()
        logger.info("State manager closed")


class Transaction:
    """
    Transaction context manager for exactly-once processing.

    Coordinates changes across Kafka, PostgreSQL, and Redis with
    automatic rollback on failure.

    Example:
        >>> async with state_mgr.transaction(message) as tx:
        ...     await tx.save_to_db(data)
        ...     await tx.publish_event("output-topic", event)
    """

    def __init__(
        self,
        state_manager: TransactionalStateManager,
        context: TransactionContext
    ):
        """
        Initialize transaction.

        Args:
            state_manager: Parent state manager
            context: Transaction context
        """
        self.state_manager = state_manager
        self.context = context
        self.pg_conn = None
        self.compensation_actions: list = []  # For saga pattern

    async def __aenter__(self):
        """Enter transaction context."""
        # Get PostgreSQL connection
        self.pg_conn = self.state_manager.pg_pool.getconn()
        self.pg_conn.autocommit = False

        # Begin Kafka transaction
        self.state_manager.kafka_producer.begin_transaction()

        # Record transaction start
        cursor = self.pg_conn.cursor()
        cursor.execute(
            """
            INSERT INTO transaction_state
            (transaction_id, status, kafka_topic, kafka_partition,
             kafka_offset, message_key)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                str(self.context.transaction_id),
                TransactionStatus.PREPARED.value,
                self.context.kafka_topic,
                self.context.kafka_partition,
                self.context.kafka_offset,
                self.context.message_key
            )
        )
        self.pg_conn.commit()
        cursor.close()

        logger.debug(f"Transaction started: {self.context.transaction_id}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit transaction context."""
        try:
            if exc_type is None:
                # No exception, commit everything
                await self._commit_all()
                return True
            else:
                # Exception occurred, rollback
                logger.error(
                    f"Transaction failed: {exc_val}",
                    exc_info=(exc_type, exc_val, exc_tb)
                )
                await self._rollback_all(str(exc_val))
                return False  # Re-raise exception

        finally:
            # Always release connection
            if self.pg_conn:
                self.state_manager.pg_pool.putconn(self.pg_conn)
                self.pg_conn = None

    async def save_to_db(self, table: str, data: Dict[str, Any]):
        """
        Save data to PostgreSQL within transaction.

        Args:
            table: Table name
            data: Data to insert
        """
        cursor = self.pg_conn.cursor()
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['%s'] * len(data))
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"

        cursor.execute(query, list(data.values()))
        cursor.close()

        # Add compensation action for saga
        self.compensation_actions.append(
            ('delete', table, data)
        )

    async def publish_event(
        self,
        topic: str,
        event: Dict[str, Any],
        key: Optional[str] = None
    ):
        """
        Publish event to Kafka within transaction.

        Args:
            topic: Output topic
            event: Event data
            key: Optional message key
        """
        self.state_manager.kafka_producer.send(
            topic=topic,
            value=event,
            key=key
        )

    async def _commit_all(self):
        """Commit all changes atomically."""
        try:
            # Commit PostgreSQL
            self.pg_conn.commit()

            # Commit Kafka transaction
            self.state_manager.kafka_producer.commit_transaction()

            # Update transaction status
            cursor = self.pg_conn.cursor()
            cursor.execute(
                """
                UPDATE transaction_state
                SET status = %s, processing_completed_at = NOW()
                WHERE transaction_id = %s
                """,
                (TransactionStatus.COMMITTED.value, str(self.context.transaction_id))
            )
            self.pg_conn.commit()
            cursor.close()

            logger.debug(f"Transaction committed: {self.context.transaction_id}")

        except Exception as e:
            logger.error(f"Commit failed: {e}")
            await self._rollback_all(str(e))
            raise

    async def _rollback_all(self, error_message: str):
        """Rollback all changes."""
        try:
            # Rollback PostgreSQL
            if self.pg_conn:
                self.pg_conn.rollback()

            # Rollback Kafka transaction
            try:
                self.state_manager.kafka_producer.abort_transaction()
            except Exception as e:
                logger.error(f"Kafka rollback failed: {e}")

            # Execute compensation actions (saga pattern)
            await self._compensate()

            # Update transaction status
            if self.pg_conn:
                cursor = self.pg_conn.cursor()
                cursor.execute(
                    """
                    UPDATE transaction_state
                    SET status = %s, error_message = %s,
                        processing_completed_at = NOW()
                    WHERE transaction_id = %s
                    """,
                    (
                        TransactionStatus.ROLLED_BACK.value,
                        error_message,
                        str(self.context.transaction_id)
                    )
                )
                self.pg_conn.commit()
                cursor.close()

            logger.warning(f"Transaction rolled back: {self.context.transaction_id}")

        except Exception as e:
            logger.error(f"Rollback failed: {e}")

    async def _compensate(self):
        """Execute compensation actions (saga pattern)."""
        for action in reversed(self.compensation_actions):
            try:
                action_type, table, data = action
                if action_type == 'delete':
                    # Compensate insert with delete
                    cursor = self.pg_conn.cursor()
                    # Build WHERE clause from primary key
                    # (simplified - assumes 'id' field)
                    if 'id' in data:
                        cursor.execute(
                            f"DELETE FROM {table} WHERE id = %s",
                            (data['id'],)
                        )
                    cursor.close()
                    logger.debug(f"Compensated action: {action}")
            except Exception as e:
                logger.error(f"Compensation failed: {e}")

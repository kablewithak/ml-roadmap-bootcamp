"""
Comprehensive monitoring and metrics collection for streaming infrastructure.

Provides Prometheus integration, custom business metrics tracking,
and advanced observability for production systems.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    CollectorRegistry, push_to_gateway, start_http_server
)
import psycopg2

logger = logging.getLogger(__name__)


@dataclass
class MetricLabels:
    """
    Common labels for metrics.

    Attributes:
        topic: Kafka topic
        partition: Partition number
        consumer_group: Consumer group ID
        environment: Environment (dev/staging/prod)
    """
    topic: Optional[str] = None
    partition: Optional[int] = None
    consumer_group: Optional[str] = None
    environment: str = "development"

    def as_dict(self) -> Dict[str, str]:
        """Convert to dictionary for Prometheus labels."""
        return {k: str(v) for k, v in {
            'topic': self.topic,
            'partition': self.partition,
            'consumer_group': self.consumer_group,
            'environment': self.environment
        }.items() if v is not None}


class StreamingMetrics:
    """
    Prometheus metrics for streaming infrastructure.

    Tracks technical and business metrics for production observability.

    Metrics categories:
    1. Throughput: Messages/sec, bytes/sec
    2. Latency: Processing time (p50/p95/p99)
    3. Errors: Failure rates, retry counts
    4. Consumer lag: Per-partition lag
    5. Business: Transaction volumes, fraud detection rates

    Example:
        >>> metrics = StreamingMetrics(registry=CollectorRegistry())
        >>> metrics.record_message_consumed("payments", 0)
        >>> metrics.record_processing_latency(25.5, "payments")
        >>> metrics.expose_http_server(port=8000)
    """

    def __init__(
        self,
        registry: Optional[CollectorRegistry] = None,
        namespace: str = "streaming"
    ):
        """
        Initialize streaming metrics.

        Args:
            registry: Prometheus registry (None = default)
            namespace: Metric namespace prefix
        """
        self.registry = registry or CollectorRegistry()
        self.namespace = namespace

        # Producer metrics
        self.messages_produced = Counter(
            f'{namespace}_messages_produced_total',
            'Total messages produced',
            ['topic', 'environment'],
            registry=self.registry
        )

        self.bytes_produced = Counter(
            f'{namespace}_bytes_produced_total',
            'Total bytes produced',
            ['topic', 'environment'],
            registry=self.registry
        )

        self.produce_errors = Counter(
            f'{namespace}_produce_errors_total',
            'Total produce errors',
            ['topic', 'error_type', 'environment'],
            registry=self.registry
        )

        # Consumer metrics
        self.messages_consumed = Counter(
            f'{namespace}_messages_consumed_total',
            'Total messages consumed',
            ['topic', 'consumer_group', 'environment'],
            registry=self.registry
        )

        self.messages_processed = Counter(
            f'{namespace}_messages_processed_total',
            'Total messages processed successfully',
            ['topic', 'consumer_group', 'environment'],
            registry=self.registry
        )

        self.processing_errors = Counter(
            f'{namespace}_processing_errors_total',
            'Total processing errors',
            ['topic', 'consumer_group', 'error_type', 'environment'],
            registry=self.registry
        )

        self.consumer_lag = Gauge(
            f'{namespace}_consumer_lag',
            'Current consumer lag',
            ['topic', 'partition', 'consumer_group', 'environment'],
            registry=self.registry
        )

        # Latency metrics (histogram with buckets)
        self.processing_latency = Histogram(
            f'{namespace}_processing_latency_ms',
            'Message processing latency in milliseconds',
            ['topic', 'consumer_group', 'environment'],
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
            registry=self.registry
        )

        self.end_to_end_latency = Histogram(
            f'{namespace}_end_to_end_latency_ms',
            'End-to-end latency from produce to consume',
            ['topic', 'environment'],
            buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000],
            registry=self.registry
        )

        # Offset commit metrics
        self.offset_commits = Counter(
            f'{namespace}_offset_commits_total',
            'Total offset commits',
            ['topic', 'consumer_group', 'environment'],
            registry=self.registry
        )

        self.offset_commit_errors = Counter(
            f'{namespace}_offset_commit_errors_total',
            'Total offset commit errors',
            ['topic', 'consumer_group', 'environment'],
            registry=self.registry
        )

        # Dead letter queue metrics
        self.dlq_messages = Counter(
            f'{namespace}_dlq_messages_total',
            'Total messages sent to DLQ',
            ['topic', 'error_type', 'environment'],
            registry=self.registry
        )

        # Circuit breaker metrics
        self.circuit_breaker_state = Gauge(
            f'{namespace}_circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=half_open, 2=open)',
            ['name', 'environment'],
            registry=self.registry
        )

        # Backpressure metrics
        self.consumption_rate = Gauge(
            f'{namespace}_consumption_rate_msg_per_sec',
            'Current consumption rate',
            ['consumer_group', 'environment'],
            registry=self.registry
        )

        self.backpressure_paused = Gauge(
            f'{namespace}_backpressure_paused',
            'Whether backpressure has paused consumption (0=no, 1=yes)',
            ['consumer_group', 'environment'],
            registry=self.registry
        )

        # Business metrics
        self.transaction_amount = Counter(
            f'{namespace}_transaction_amount_total',
            'Total transaction amount processed',
            ['currency', 'environment'],
            registry=self.registry
        )

        self.high_value_transactions = Counter(
            f'{namespace}_high_value_transactions_total',
            'Total high-value transactions (>10k)',
            ['currency', 'environment'],
            registry=self.registry
        )

        self.fraud_decisions = Counter(
            f'{namespace}_fraud_decisions_total',
            'Total fraud decisions',
            ['decision', 'environment'],
            registry=self.registry
        )

        self.fraud_score = Histogram(
            f'{namespace}_fraud_score',
            'Fraud score distribution',
            ['environment'],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )

        logger.info(f"Streaming metrics initialized: {namespace}")

    # Producer methods
    def record_message_produced(
        self,
        topic: str,
        size_bytes: int,
        environment: str = "development"
    ):
        """Record a produced message."""
        self.messages_produced.labels(topic=topic, environment=environment).inc()
        self.bytes_produced.labels(topic=topic, environment=environment).inc(size_bytes)

    def record_produce_error(
        self,
        topic: str,
        error_type: str,
        environment: str = "development"
    ):
        """Record a produce error."""
        self.produce_errors.labels(
            topic=topic,
            error_type=error_type,
            environment=environment
        ).inc()

    # Consumer methods
    def record_message_consumed(
        self,
        topic: str,
        consumer_group: str,
        environment: str = "development"
    ):
        """Record a consumed message."""
        self.messages_consumed.labels(
            topic=topic,
            consumer_group=consumer_group,
            environment=environment
        ).inc()

    def record_message_processed(
        self,
        topic: str,
        consumer_group: str,
        environment: str = "development"
    ):
        """Record successfully processed message."""
        self.messages_processed.labels(
            topic=topic,
            consumer_group=consumer_group,
            environment=environment
        ).inc()

    def record_processing_error(
        self,
        topic: str,
        consumer_group: str,
        error_type: str,
        environment: str = "development"
    ):
        """Record a processing error."""
        self.processing_errors.labels(
            topic=topic,
            consumer_group=consumer_group,
            error_type=error_type,
            environment=environment
        ).inc()

    def record_processing_latency(
        self,
        latency_ms: float,
        topic: str,
        consumer_group: str = "default",
        environment: str = "development"
    ):
        """Record processing latency."""
        self.processing_latency.labels(
            topic=topic,
            consumer_group=consumer_group,
            environment=environment
        ).observe(latency_ms)

    def record_end_to_end_latency(
        self,
        latency_ms: float,
        topic: str,
        environment: str = "development"
    ):
        """Record end-to-end latency."""
        self.end_to_end_latency.labels(
            topic=topic,
            environment=environment
        ).observe(latency_ms)

    def update_consumer_lag(
        self,
        topic: str,
        partition: int,
        consumer_group: str,
        lag: int,
        environment: str = "development"
    ):
        """Update consumer lag gauge."""
        self.consumer_lag.labels(
            topic=topic,
            partition=str(partition),
            consumer_group=consumer_group,
            environment=environment
        ).set(lag)

    def record_offset_commit(
        self,
        topic: str,
        consumer_group: str,
        environment: str = "development"
    ):
        """Record offset commit."""
        self.offset_commits.labels(
            topic=topic,
            consumer_group=consumer_group,
            environment=environment
        ).inc()

    def record_dlq_message(
        self,
        topic: str,
        error_type: str,
        environment: str = "development"
    ):
        """Record message sent to DLQ."""
        self.dlq_messages.labels(
            topic=topic,
            error_type=error_type,
            environment=environment
        ).inc()

    # Circuit breaker methods
    def update_circuit_breaker_state(
        self,
        name: str,
        state: str,
        environment: str = "development"
    ):
        """
        Update circuit breaker state.

        Args:
            name: Circuit breaker name
            state: State (CLOSED/HALF_OPEN/OPEN)
            environment: Environment
        """
        state_value = {"CLOSED": 0, "HALF_OPEN": 1, "OPEN": 2}.get(state, -1)
        self.circuit_breaker_state.labels(
            name=name,
            environment=environment
        ).set(state_value)

    # Backpressure methods
    def update_consumption_rate(
        self,
        rate: float,
        consumer_group: str,
        environment: str = "development"
    ):
        """Update consumption rate."""
        self.consumption_rate.labels(
            consumer_group=consumer_group,
            environment=environment
        ).set(rate)

    def update_backpressure_state(
        self,
        is_paused: bool,
        consumer_group: str,
        environment: str = "development"
    ):
        """Update backpressure pause state."""
        self.backpressure_paused.labels(
            consumer_group=consumer_group,
            environment=environment
        ).set(1 if is_paused else 0)

    # Business metrics methods
    def record_transaction(
        self,
        amount: float,
        currency: str,
        is_high_value: bool = False,
        environment: str = "development"
    ):
        """
        Record a transaction for business metrics.

        Args:
            amount: Transaction amount
            currency: Currency code
            is_high_value: Whether it's a high-value transaction
            environment: Environment
        """
        self.transaction_amount.labels(
            currency=currency,
            environment=environment
        ).inc(amount)

        if is_high_value:
            self.high_value_transactions.labels(
                currency=currency,
                environment=environment
            ).inc()

    def record_fraud_decision(
        self,
        decision: str,
        fraud_score: float,
        environment: str = "development"
    ):
        """
        Record fraud detection decision.

        Args:
            decision: Decision (APPROVE/DECLINE/REVIEW/CHALLENGE)
            fraud_score: Fraud score (0.0-1.0)
            environment: Environment
        """
        self.fraud_decisions.labels(
            decision=decision,
            environment=environment
        ).inc()

        self.fraud_score.labels(environment=environment).observe(fraud_score)

    def expose_http_server(self, port: int = 8000):
        """
        Start HTTP server to expose metrics.

        Args:
            port: Port to listen on
        """
        try:
            start_http_server(port, registry=self.registry)
            logger.info(f"Metrics HTTP server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")

    def push_to_gateway(
        self,
        gateway_url: str,
        job_name: str
    ):
        """
        Push metrics to Prometheus Pushgateway.

        Args:
            gateway_url: Pushgateway URL
            job_name: Job name for grouping
        """
        try:
            push_to_gateway(gateway_url, job=job_name, registry=self.registry)
            logger.debug(f"Pushed metrics to {gateway_url}")
        except Exception as e:
            logger.error(f"Failed to push metrics: {e}")


class BusinessMetricsCollector:
    """
    Collects and persists business-critical metrics.

    Beyond technical metrics, tracks what actually matters:
    - Money flow (transaction volumes)
    - Fraud detection effectiveness
    - SLA adherence
    - Revenue impact

    Example:
        >>> collector = BusinessMetricsCollector(postgres_dsn)
        >>> collector.track_daily_volume(10000.50, "USD")
        >>> collector.track_fraud_effectiveness(0.95, 0.02)
    """

    def __init__(self, postgres_dsn: str):
        """
        Initialize business metrics collector.

        Args:
            postgres_dsn: PostgreSQL connection string
        """
        self.postgres_dsn = postgres_dsn
        logger.info("Business metrics collector initialized")

    def _get_connection(self):
        """Get PostgreSQL connection."""
        return psycopg2.connect(self.postgres_dsn)

    def track_daily_volume(
        self,
        amount: float,
        currency: str,
        date: Optional[datetime] = None
    ):
        """
        Track daily transaction volume.

        Args:
            amount: Transaction amount
            currency: Currency code
            date: Transaction date (default: today)
        """
        if date is None:
            date = datetime.now().date()

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO processing_metrics
                (metric_name, metric_value, metric_type, tags, recorded_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    'daily_volume',
                    amount,
                    'COUNTER',
                    {'currency': currency, 'date': str(date)},
                    datetime.now()
                )
            )
            conn.commit()
            cursor.close()
        finally:
            conn.close()

    def track_fraud_effectiveness(
        self,
        true_positive_rate: float,
        false_positive_rate: float,
        model_version: Optional[str] = None
    ):
        """
        Track fraud detection model effectiveness.

        Args:
            true_positive_rate: TPR (recall)
            false_positive_rate: FPR
            model_version: Model version identifier
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO processing_metrics
                (metric_name, metric_value, metric_type, tags, recorded_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    'fraud_tpr',
                    true_positive_rate,
                    'GAUGE',
                    {'model_version': model_version} if model_version else {},
                    datetime.now()
                )
            )

            cursor.execute(
                """
                INSERT INTO processing_metrics
                (metric_name, metric_value, metric_type, tags, recorded_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    'fraud_fpr',
                    false_positive_rate,
                    'GAUGE',
                    {'model_version': model_version} if model_version else {},
                    datetime.now()
                )
            )

            conn.commit()
            cursor.close()

            logger.info(
                f"Fraud effectiveness: TPR={true_positive_rate:.2%}, "
                f"FPR={false_positive_rate:.2%}"
            )
        finally:
            conn.close()

    def get_daily_summary(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get daily processing summary.

        Args:
            date: Date to query (default: today)

        Returns:
            Summary statistics
        """
        if date is None:
            date = datetime.now().date()

        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Get transaction counts and amounts
            cursor.execute(
                """
                SELECT COUNT(*), SUM(amount), AVG(fraud_score)
                FROM payment_events
                WHERE DATE(created_at) = %s
                """,
                (date,)
            )

            result = cursor.fetchone()
            cursor.close()

            return {
                'date': str(date),
                'transaction_count': result[0] or 0,
                'total_amount': float(result[1] or 0),
                'avg_fraud_score': float(result[2] or 0)
            }
        finally:
            conn.close()

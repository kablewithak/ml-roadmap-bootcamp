"""
Kafka producer for fraud signal logging and ML training data.
"""

import json
import logging
from typing import Dict, Optional
from datetime import datetime
from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic

logger = logging.getLogger(__name__)


class FraudEventProducer:
    """
    Kafka producer for fraud detection events.
    Publishes signals to Kafka for ML training and analysis.
    """

    def __init__(
        self,
        bootstrap_servers: str,
        topics: Dict[str, str],
        compression_type: str = "snappy"
    ):
        self.bootstrap_servers = bootstrap_servers
        self.topics = topics

        # Configure producer for high throughput
        self.producer = Producer({
            'bootstrap.servers': bootstrap_servers,
            'compression.type': compression_type,
            'linger.ms': 10,  # Batch messages for 10ms
            'batch.size': 16384,  # 16KB batches
            'acks': 1,  # Leader acknowledgment only
            'retries': 3,
            'max.in.flight.requests.per.connection': 5,
        })

        # Create topics if they don't exist
        self._create_topics()

    def _create_topics(self):
        """Create Kafka topics if they don't exist."""
        admin_client = AdminClient({'bootstrap.servers': self.bootstrap_servers})

        # Check existing topics
        existing_topics = admin_client.list_topics(timeout=10).topics

        # Topics to create
        new_topics = []
        for topic_name in self.topics.values():
            if topic_name not in existing_topics:
                new_topics.append(
                    NewTopic(
                        topic=topic_name,
                        num_partitions=3,
                        replication_factor=1  # For development
                    )
                )

        if new_topics:
            fs = admin_client.create_topics(new_topics)
            for topic, f in fs.items():
                try:
                    f.result()
                    logger.info(f"Topic {topic} created")
                except Exception as e:
                    logger.warning(f"Failed to create topic {topic}: {e}")

    def publish_fraud_signal(
        self,
        transaction_id: str,
        event_data: Dict,
        callback: Optional[callable] = None
    ):
        """
        Publish fraud signal event to Kafka.

        Args:
            transaction_id: Unique transaction identifier (used as key)
            event_data: Complete fraud detection data
            callback: Optional delivery callback
        """
        topic = self.topics["fraud_signals"]

        # Serialize to JSON
        value = json.dumps(event_data, default=str).encode('utf-8')
        key = transaction_id.encode('utf-8')

        # Publish asynchronously
        self.producer.produce(
            topic=topic,
            key=key,
            value=value,
            callback=callback or self._delivery_callback
        )

        # Trigger delivery of messages
        self.producer.poll(0)

    def publish_payment_decision(
        self,
        transaction_id: str,
        decision_data: Dict,
        callback: Optional[callable] = None
    ):
        """Publish payment decision event."""
        topic = self.topics["payment_decisions"]

        value = json.dumps(decision_data, default=str).encode('utf-8')
        key = transaction_id.encode('utf-8')

        self.producer.produce(
            topic=topic,
            key=key,
            value=value,
            callback=callback or self._delivery_callback
        )

        self.producer.poll(0)

    def publish_transaction_event(
        self,
        transaction_id: str,
        transaction_data: Dict,
        callback: Optional[callable] = None
    ):
        """Publish raw transaction event."""
        topic = self.topics["transaction_events"]

        value = json.dumps(transaction_data, default=str).encode('utf-8')
        key = transaction_id.encode('utf-8')

        self.producer.produce(
            topic=topic,
            key=key,
            value=value,
            callback=callback or self._delivery_callback
        )

        self.producer.poll(0)

    def _delivery_callback(self, err, msg):
        """Callback for message delivery confirmation."""
        if err:
            logger.error(f"Message delivery failed: {err}")
        else:
            logger.debug(
                f"Message delivered to {msg.topic()} "
                f"[{msg.partition()}] at offset {msg.offset()}"
            )

    def flush(self, timeout: float = 10.0):
        """Wait for all messages to be delivered."""
        remaining = self.producer.flush(timeout)
        if remaining > 0:
            logger.warning(f"{remaining} messages failed to deliver")
        return remaining

    def close(self):
        """Close producer and flush remaining messages."""
        self.flush()
        logger.info("Kafka producer closed")

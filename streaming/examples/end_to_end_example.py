"""
End-to-end example of streaming infrastructure with exactly-once processing.

This example demonstrates:
1. High-throughput producer sending payment events
2. Exactly-once consumer processing events
3. Fraud detection with ML features
4. State management across Kafka + PostgreSQL
5. Metrics collection and monitoring
"""

import asyncio
import logging
from decimal import Decimal
from uuid import uuid4
from datetime import datetime
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Import streaming components
from streaming.core.producer import HighThroughputProducer, ProducerConfig
from streaming.core.consumer import ExactlyOnceConsumer, ConsumerConfig
from streaming.core.state_manager import TransactionalStateManager
from streaming.schemas.payment_event import PaymentEvent
from streaming.schemas.fraud_decision_event import (
    FraudDecisionEvent, RiskFactors, FraudReason
)
from streaming.ml_integration.feature_streaming import MLFeatureStreaming
from streaming.monitoring.metrics import StreamingMetrics


class PaymentProcessor:
    """
    End-to-end payment processing with fraud detection.

    This class demonstrates production-grade streaming with:
    - Exactly-once processing
    - Real-time fraud detection
    - ML feature computation
    - Metrics tracking
    """

    def __init__(self):
        """Initialize payment processor with all components."""
        # Producer configuration
        self.producer_config = ProducerConfig(
            bootstrap_servers="localhost:19092",
            client_id="payment-processor-producer",
            compression_type="zstd",
            enable_idempotence=True
        )

        self.producer = HighThroughputProducer(self.producer_config)

        # Consumer configuration
        self.consumer_config = ConsumerConfig(
            bootstrap_servers="localhost:19092",
            group_id="payment-processor-group",
            topics=["payments"],
            enable_auto_commit=False,
            isolation_level="read_committed"
        )

        self.consumer = ExactlyOnceConsumer(
            self.consumer_config,
            dead_letter_topic="payments-dlq"
        )

        # State manager for exactly-once
        self.state_manager = TransactionalStateManager(
            postgres_dsn="postgresql://streaming:streaming_pass@localhost:5432/streaming_state",
            redis_url="redis://localhost:6379",
            kafka_bootstrap="localhost:19092"
        )

        # ML feature streaming
        self.feature_stream = MLFeatureStreaming(
            redis_url="redis://localhost:6379"
        )

        # Metrics
        self.metrics = StreamingMetrics()
        self.metrics.expose_http_server(port=8000)

        logger.info("Payment processor initialized")

    async def generate_payment_events(self, num_events: int = 1000):
        """
        Generate sample payment events for testing.

        Args:
            num_events: Number of events to generate
        """
        logger.info(f"Generating {num_events} payment events...")

        for i in range(num_events):
            # Create realistic payment
            payment = PaymentEvent.create(
                user_id=f"user_{random.randint(1, 1000)}",
                merchant_id=f"merchant_{random.randint(1, 100)}",
                amount=Decimal(str(round(random.uniform(10, 1000), 2))),
                currency="USD",
                payment_method=random.choice([
                    "CREDIT_CARD", "DEBIT_CARD", "BANK_TRANSFER"
                ]),
                ip_address=f"192.168.{random.randint(0, 255)}.{random.randint(0, 255)}",
                session_id=str(uuid4()),
                country_code=random.choice(["US", "GB", "DE", "FR", "JP"]),
                device_fingerprint=str(uuid4())
            )

            # Send to Kafka
            self.producer.send(
                topic="payments",
                value=payment.to_dict(),
                key=payment.payment_id
            )

            # Record metrics
            self.metrics.record_message_produced(
                topic="payments",
                size_bytes=1024,
                environment="development"
            )

            if (i + 1) % 100 == 0:
                logger.info(f"Produced {i + 1}/{num_events} payments")

        self.producer.flush()
        logger.info(f"✓ Generated {num_events} payment events")

    async def detect_fraud(self, payment: dict) -> dict:
        """
        Perform fraud detection on payment.

        Args:
            payment: Payment event dictionary

        Returns:
            Fraud decision
        """
        # Compute streaming features
        features = await self.feature_stream.compute_streaming_features(payment)

        # Simple rule-based fraud detection (replace with ML model in production)
        fraud_score = 0.0

        # High velocity = higher fraud risk
        if features.get('tx_count_5min', 0) > 5:
            fraud_score += 0.3

        # Large amount deviation = suspicious
        if features.get('amount_deviation', 0) > 2.0:
            fraud_score += 0.2

        # Unusual time = suspicious
        if features.get('unusual_time', 0) == 1.0:
            fraud_score += 0.15

        # New merchant = slightly suspicious
        if features.get('new_merchant', 0) == 1.0:
            fraud_score += 0.1

        # High merchant risk
        if features.get('merchant_risk_score', 0.5) > 0.7:
            fraud_score += 0.25

        # Cap at 1.0
        fraud_score = min(fraud_score, 1.0)

        # Make decision
        if fraud_score >= 0.8:
            decision = "DECLINE"
        elif fraud_score >= 0.5:
            decision = "REVIEW"
        else:
            decision = "APPROVE"

        # Create fraud decision event
        risk_factors = RiskFactors(
            velocity_score=features.get('tx_count_5min', 0) / 10.0,
            behavioral_score=features.get('unusual_time', 0),
            merchant_score=features.get('merchant_risk_score', 0.5)
        )

        fraud_reasons = []
        if fraud_score >= 0.5:
            if features.get('tx_count_5min', 0) > 5:
                fraud_reasons.append(FraudReason(
                    reason_code="HIGH_VELOCITY",
                    reason_description="Too many transactions in short time",
                    confidence=0.8
                ))

            if features.get('unusual_time', 0) == 1.0:
                fraud_reasons.append(FraudReason(
                    reason_code="UNUSUAL_TIME",
                    reason_description="Transaction at unusual hour",
                    confidence=0.6
                ))

        fraud_decision = FraudDecisionEvent.create(
            payment_id=payment['payment_id'],
            user_id=payment['user_id'],
            decision=decision,
            fraud_score=fraud_score,
            model_version="1.0.0",
            model_name="rule_based_v1",
            risk_factors=risk_factors,
            processing_time_ms=5,
            idempotency_key=payment['idempotency_key'],
            fraud_reasons=fraud_reasons,
            features_used={k: str(v) for k, v in features.items()}
        )

        # Record metrics
        self.metrics.record_fraud_decision(
            decision=decision,
            fraud_score=fraud_score,
            environment="development"
        )

        return fraud_decision.to_dict()

    async def process_payments(self, max_messages: int = 1000):
        """
        Process payment events with exactly-once semantics.

        Args:
            max_messages: Maximum messages to process
        """
        logger.info("Starting payment processing...")

        processed_count = 0

        def process_payment(msg: dict) -> bool:
            """Process single payment message."""
            nonlocal processed_count

            try:
                payment = msg['value']
                idempotency_key = payment['idempotency_key']

                # Check idempotency
                if self.state_manager.check_idempotency(idempotency_key):
                    logger.debug(f"Skipping duplicate: {idempotency_key}")
                    return True

                # Detect fraud (synchronous wrapper for async)
                fraud_decision = asyncio.run(self.detect_fraud(payment))

                # Record metrics
                self.metrics.record_message_consumed(
                    topic="payments",
                    consumer_group="payment-processor-group",
                    environment="development"
                )

                # Mark as processed
                self.state_manager.mark_processed(
                    idempotency_key=idempotency_key,
                    message_id=payment['payment_id'],
                    topic=msg['topic'],
                    partition=msg['partition'],
                    offset=msg['offset'],
                    kafka_timestamp=msg['timestamp'],
                    processing_duration_ms=5
                )

                # Send fraud decision to output topic
                self.producer.send(
                    topic="fraud-decisions",
                    value=fraud_decision,
                    key=payment['payment_id']
                )

                processed_count += 1

                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count} payments")

                self.metrics.record_message_processed(
                    topic="payments",
                    consumer_group="payment-processor-group",
                    environment="development"
                )

                return True

            except Exception as e:
                logger.error(f"Error processing payment: {e}", exc_info=True)
                return False

        # Consume messages
        self.consumer.consume(
            process_func=process_payment,
            max_messages=max_messages,
            max_retries=3,
            timeout=5.0
        )

        logger.info(f"✓ Processed {processed_count} payments")

    def shutdown(self):
        """Gracefully shutdown processor."""
        logger.info("Shutting down payment processor...")

        self.producer.close()
        self.consumer.close()
        self.state_manager.close()
        self.feature_stream.close()

        logger.info("✓ Payment processor shutdown complete")


async def main():
    """Main entry point for end-to-end example."""
    processor = PaymentProcessor()

    try:
        # Step 1: Generate payment events
        await processor.generate_payment_events(num_events=1000)

        # Step 2: Process payments with fraud detection
        await processor.process_payments(max_messages=1000)

        # Step 3: Show metrics
        producer_metrics = processor.producer.get_metrics()
        consumer_metrics = processor.consumer.get_metrics()

        print("\n" + "="*60)
        print("PAYMENT PROCESSING COMPLETED")
        print("="*60)
        print(f"\nProducer Metrics:")
        print(f"  {producer_metrics}")
        print(f"\nConsumer Metrics:")
        print(f"  {consumer_metrics}")
        print(f"\nMetrics available at: http://localhost:8000/metrics")
        print("="*60 + "\n")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
    finally:
        processor.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

"""
Pytest configuration and fixtures for fraud detection tests.
"""

import pytest
import asyncio
from datetime import datetime
from typing import AsyncGenerator

from src.infrastructure.redis.velocity_tracker import VelocityTracker, create_redis_client
from src.fraud.services.signal_collector import SignalCollector
from src.fraud.services.fraud_detector import create_fraud_detector
from src.fraud.models import TransactionRequest, SignalWeights, RiskThresholds


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def redis_client():
    """Create Redis client for tests."""
    client = await create_redis_client(host="localhost", port=6379, db=1)  # Use DB 1 for tests
    yield client

    # Cleanup: flush test database
    await client.flushdb()
    await client.close()


@pytest.fixture
async def velocity_tracker(redis_client):
    """Create velocity tracker instance."""
    return VelocityTracker(redis_client)


@pytest.fixture
async def signal_collector(velocity_tracker):
    """Create signal collector instance."""
    return SignalCollector(
        velocity_tracker=velocity_tracker,
        small_tx_threshold=10.0,
        new_card_high_value=1000.0,
        merchant_switch_threshold=3
    )


@pytest.fixture
def fraud_config():
    """Fraud detection configuration for tests."""
    return {
        "velocity": {
            "transaction_count_5min": 5,
            "transaction_count_1hr": 20,
            "amount_sum_5min": 5000.0,
            "amount_sum_1hr": 20000.0,
            "new_card_high_value": 1000.0
        },
        "patterns": {
            "small_transaction_threshold": 10.0,
            "new_card_high_value": 1000.0,
            "merchant_category_switch_threshold": 3
        },
        "weights": {
            "velocity_count": 0.25,
            "velocity_amount": 0.20,
            "new_card_risk": 0.15,
            "merchant_pattern": 0.15,
            "time_pattern": 0.10,
            "card_testing_pattern": 0.15
        },
        "thresholds": {
            "approve_below": 0.30,
            "review_below": 0.70,
            "decline_above": 0.70
        }
    }


@pytest.fixture
def kafka_config():
    """Kafka configuration for tests."""
    return {
        "bootstrap_servers": "localhost:9092",
        "topics": {
            "fraud_signals": "test.fraud.signals",
            "payment_decisions": "test.fraud.decisions",
            "transaction_events": "test.payment.transactions"
        },
        "producer": {
            "compression_type": "snappy",
            "acks": 1
        }
    }


@pytest.fixture
async def fraud_detector(redis_client, kafka_config, fraud_config):
    """Create fraud detector instance."""
    detector = await create_fraud_detector(
        redis_client=redis_client,
        kafka_config=kafka_config,
        fraud_config=fraud_config
    )
    yield detector

    # Cleanup
    detector.kafka_producer.close()


def create_transaction(
    transaction_id: str = "test-tx-001",
    user_id: str = "user-123",
    card_id: str = "card-456",
    ip_address: str = "192.168.1.1",
    amount: float = 100.0,
    merchant_category: str = "retail",
    merchant_id: str = "merchant-789",
    merchant_name: str = "Test Store",
    timestamp: datetime = None
) -> TransactionRequest:
    """Helper to create transaction request."""
    return TransactionRequest(
        transaction_id=transaction_id,
        user_id=user_id,
        card_id=card_id,
        ip_address=ip_address,
        amount=amount,
        currency="USD",
        merchant_id=merchant_id,
        merchant_category=merchant_category,
        merchant_name=merchant_name,
        timestamp=timestamp or datetime.utcnow()
    )

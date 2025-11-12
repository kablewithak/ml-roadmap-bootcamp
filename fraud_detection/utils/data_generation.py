"""Data generation utilities for realistic transactions and fraud patterns."""

import random
import string
from datetime import datetime, timedelta
from typing import List
from faker import Faker

from fraud_detection.types import Transaction, TransactionType, FraudLabel


fake = Faker()


def generate_user_id() -> str:
    """Generate a realistic user ID."""
    return f"user_{fake.uuid4()[:8]}"


def generate_device_id() -> str:
    """Generate a realistic device ID."""
    return f"device_{fake.uuid4()[:12]}"


def generate_card_bin() -> str:
    """Generate realistic card BIN (first 6 digits)."""
    # Common card BINs: Visa (4), Mastercard (51-55), Amex (34, 37)
    bins = ["411111", "424242", "510510", "555555", "378282", "371449"]
    return random.choice(bins)


def generate_card_last4() -> str:
    """Generate last 4 digits of card."""
    return ''.join(random.choices(string.digits, k=4))


def generate_legitimate_transactions(
    num_transactions: int = 100,
    start_time: datetime = None
) -> List[Transaction]:
    """
    Generate realistic legitimate transactions.

    Args:
        num_transactions: Number of transactions to generate
        start_time: Starting timestamp

    Returns:
        List of legitimate Transaction objects
    """
    if start_time is None:
        start_time = datetime.utcnow() - timedelta(days=7)

    transactions = []
    users = [generate_user_id() for _ in range(num_transactions // 10)]

    for i in range(num_transactions):
        # Realistic timing distribution (not uniform)
        hours_offset = random.expovariate(1 / 24) * 168  # 7 days
        timestamp = start_time + timedelta(hours=hours_offset)

        # Pick a user (some users make multiple transactions)
        user_id = random.choice(users)

        # Realistic amounts (log-normal distribution)
        amount = round(random.lognormvariate(3.5, 1.2), 2)  # Mean ~$50
        amount = max(1.0, min(amount, 5000.0))  # Clip to reasonable range

        transaction = Transaction(
            timestamp=timestamp,
            user_id=user_id,
            merchant_id=f"merchant_{random.randint(1, 100)}",
            amount=amount,
            transaction_type=random.choices(
                [TransactionType.PURCHASE, TransactionType.REFUND, TransactionType.TRANSFER],
                weights=[0.85, 0.10, 0.05]
            )[0],
            device_id=generate_device_id(),
            ip_address=fake.ipv4(),
            user_agent=fake.user_agent(),
            card_bin=generate_card_bin(),
            card_last4=generate_card_last4(),
            latitude=float(fake.latitude()),
            longitude=float(fake.longitude()),
            country=fake.country_code(),
            is_first_transaction=random.random() < 0.05,
            time_since_account_creation_hours=random.expovariate(1 / 720),  # ~30 days mean
            transactions_last_24h=random.randint(0, 3),
            is_fraud=False,
            fraud_label=FraudLabel.LEGITIMATE,
            fraud_score=random.uniform(0.0, 0.3),  # Low fraud score
        )

        transactions.append(transaction)

    return sorted(transactions, key=lambda t: t.timestamp)


def generate_fraud_transactions(
    num_transactions: int = 50,
    start_time: datetime = None,
    attack_signature: str = "generic"
) -> List[Transaction]:
    """
    Generate realistic fraudulent transactions.

    Args:
        num_transactions: Number of transactions to generate
        start_time: Starting timestamp
        attack_signature: Type of attack pattern signature

    Returns:
        List of fraudulent Transaction objects
    """
    if start_time is None:
        start_time = datetime.utcnow()

    transactions = []

    # Fraudulent patterns typically share characteristics
    fraud_device_id = generate_device_id()
    fraud_ip = fake.ipv4()

    for i in range(num_transactions):
        # Fraud often happens quickly
        minutes_offset = random.uniform(0, 60)
        timestamp = start_time + timedelta(minutes=minutes_offset)

        # Higher amounts, round numbers
        if attack_signature == "card_testing":
            amount = 1.0  # Small test amounts
        elif attack_signature == "gift_card":
            amount = random.choice([25.0, 50.0, 100.0, 250.0])  # Gift card denominations
        else:
            amount = random.choice([99.99, 199.99, 299.99, 499.99])  # Round amounts

        transaction = Transaction(
            timestamp=timestamp,
            user_id=f"user_{fake.uuid4()[:8]}",
            merchant_id=f"merchant_{random.randint(1, 20)}",
            amount=amount,
            transaction_type=TransactionType.PURCHASE,
            device_id=fraud_device_id if random.random() < 0.7 else generate_device_id(),
            ip_address=fraud_ip if random.random() < 0.6 else fake.ipv4(),
            user_agent=fake.user_agent(),
            card_bin=generate_card_bin(),
            card_last4=generate_card_last4(),
            latitude=float(fake.latitude()),
            longitude=float(fake.longitude()),
            country=fake.country_code(),
            is_first_transaction=True,
            time_since_account_creation_hours=random.uniform(0.1, 2.0),  # Recently created
            transactions_last_24h=random.randint(5, 20),  # High velocity
            is_fraud=True,
            fraud_label=FraudLabel.FRAUDULENT,
            fraud_score=random.uniform(0.7, 1.0),  # High fraud score
        )

        transactions.append(transaction)

    return sorted(transactions, key=lambda t: t.timestamp)


def generate_synthetic_identity() -> dict:
    """
    Generate a synthetic identity with realistic but fake data.

    Returns:
        Dictionary with synthetic identity information
    """
    # Generate realistic SSN pattern (not a real SSN)
    area = random.randint(1, 899)
    group = random.randint(1, 99)
    serial = random.randint(1, 9999)
    ssn_pattern = f"{area:03d}-{group:02d}-{serial:04d}"

    return {
        "name": fake.name(),
        "email": fake.email(),
        "phone": fake.phone_number(),
        "address": fake.address(),
        "date_of_birth": fake.date_of_birth(minimum_age=18, maximum_age=90).isoformat(),
        "ssn_pattern": ssn_pattern,  # Pattern only, not real SSN
        "user_id": generate_user_id(),
        "device_id": generate_device_id(),
        "ip_address": fake.ipv4(),
        "created_at": datetime.utcnow().isoformat(),
    }

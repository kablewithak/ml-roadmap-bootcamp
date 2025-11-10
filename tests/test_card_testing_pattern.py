"""
Test card testing pattern detection.

Card testing: Fraudsters test stolen cards with multiple small charges
before making large purchases.
"""

import pytest
from datetime import datetime, timedelta

from tests.conftest import create_transaction
from src.fraud.models import RiskDecision


@pytest.mark.asyncio
async def test_card_testing_pattern_detected(fraud_detector):
    """
    Test Scenario: Card Testing Pattern

    Pattern: Multiple small transactions (<$10) in quick succession
    Expected: High risk score, DECLINE decision
    """
    card_id = "card-testing-001"
    user_id = "user-testing-001"

    # Simulate card testing: 5 small transactions in 2 minutes
    base_time = datetime.utcnow()

    transactions = []
    for i in range(5):
        tx = create_transaction(
            transaction_id=f"card-test-{i}",
            user_id=user_id,
            card_id=card_id,
            amount=1.00 + i * 0.50,  # $1.00, $1.50, $2.00, $2.50, $3.00
            merchant_category="online_retail",
            timestamp=base_time + timedelta(seconds=i * 20)  # 20 seconds apart
        )
        transactions.append(tx)

    # Process first 3 transactions (establish pattern)
    results = []
    for tx in transactions[:3]:
        result = await fraud_detector.assess_transaction(tx)
        results.append(result)

    # By the 4th transaction, card testing should be detected
    result_4 = await fraud_detector.assess_transaction(transactions[3])

    # Assertions
    assert "card_testing_pattern_detected" in result_4.signals_triggered, \
        "Card testing pattern should be detected"
    assert result_4.risk_score > 0.5, \
        f"Risk score should be high (>0.5), got {result_4.risk_score:.3f}"

    # 5th transaction should definitely be declined
    result_5 = await fraud_detector.assess_transaction(transactions[4])
    assert result_5.decision == RiskDecision.DECLINE, \
        f"5th small transaction should be declined, got {result_5.decision}"
    assert result_5.risk_score > 0.7, \
        f"Risk score should be very high (>0.7), got {result_5.risk_score:.3f}"

    print(f"\n✓ Card Testing Pattern Test PASSED")
    print(f"  Transactions: {len(transactions)}")
    print(f"  Final Risk Score: {result_5.risk_score:.3f}")
    print(f"  Decision: {result_5.decision.value}")
    print(f"  Signals: {', '.join(result_5.signals_triggered[:3])}")


@pytest.mark.asyncio
async def test_legitimate_small_transactions(fraud_detector):
    """
    Test Scenario: Legitimate small transactions

    Pattern: Few small transactions spread over time
    Expected: Low risk, APPROVE decision
    """
    card_id = "card-legit-001"
    user_id = "user-legit-001"

    # 2 small transactions over 10 minutes (legitimate)
    base_time = datetime.utcnow()

    tx1 = create_transaction(
        transaction_id="legit-1",
        user_id=user_id,
        card_id=card_id,
        amount=8.50,
        merchant_category="coffee_shop",
        timestamp=base_time
    )

    tx2 = create_transaction(
        transaction_id="legit-2",
        user_id=user_id,
        card_id=card_id,
        amount=12.75,
        merchant_category="fast_food",
        timestamp=base_time + timedelta(minutes=10)
    )

    result1 = await fraud_detector.assess_transaction(tx1)
    result2 = await fraud_detector.assess_transaction(tx2)

    # Assertions
    assert "card_testing_pattern_detected" not in result2.signals_triggered, \
        "Legitimate transactions should not trigger card testing"
    assert result2.decision == RiskDecision.APPROVE, \
        f"Legitimate small transactions should be approved, got {result2.decision}"
    assert result2.risk_score < 0.3, \
        f"Risk score should be low (<0.3), got {result2.risk_score:.3f}"

    print(f"\n✓ Legitimate Small Transactions Test PASSED")
    print(f"  Risk Score: {result2.risk_score:.3f}")
    print(f"  Decision: {result2.decision.value}")


@pytest.mark.asyncio
async def test_card_testing_then_large_purchase(fraud_detector):
    """
    Test Scenario: Card testing followed by large purchase

    Pattern: Multiple small charges followed by a large charge
    Expected: Both phases should be flagged
    """
    card_id = "card-testing-large-001"
    user_id = "user-testing-large-001"
    base_time = datetime.utcnow()

    # Phase 1: Card testing (4 small transactions)
    for i in range(4):
        tx = create_transaction(
            transaction_id=f"test-small-{i}",
            user_id=user_id,
            card_id=card_id,
            amount=2.00,
            timestamp=base_time + timedelta(seconds=i * 15)
        )
        await fraud_detector.assess_transaction(tx)

    # Phase 2: Large purchase attempt
    large_tx = create_transaction(
        transaction_id="test-large-1",
        user_id=user_id,
        card_id=card_id,
        amount=2500.00,
        timestamp=base_time + timedelta(minutes=2)
    )

    result = await fraud_detector.assess_transaction(large_tx)

    # Assertions
    assert result.decision in [RiskDecision.DECLINE, RiskDecision.REVIEW], \
        f"Large purchase after card testing should be declined/reviewed, got {result.decision}"
    assert result.risk_score > 0.6, \
        f"Risk score should be high (>0.6), got {result.risk_score:.3f}"

    # Should have multiple signals
    assert len(result.signals_triggered) >= 2, \
        "Should have multiple fraud signals triggered"

    print(f"\n✓ Card Testing -> Large Purchase Test PASSED")
    print(f"  Risk Score: {result.risk_score:.3f}")
    print(f"  Decision: {result.decision.value}")
    print(f"  Signals: {', '.join(result.signals_triggered[:5])}")

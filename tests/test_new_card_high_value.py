"""
Test new card high-value transaction scenarios.

New card risk:
- First-time card usage with high value (>$1000) is suspicious
- Could indicate stolen card being used immediately
"""

import pytest
from datetime import datetime, timedelta

from tests.conftest import create_transaction
from src.fraud.models import RiskDecision


@pytest.mark.asyncio
async def test_new_card_high_value_transaction(fraud_detector):
    """
    Test Scenario: New Card with High-Value Purchase

    Pattern: First transaction on new card is $2500
    Expected: High risk, DECLINE/REVIEW decision
    """
    card_id = "card-new-high-001"
    user_id = "user-new-high-001"

    # First transaction with new card: $2500
    tx = create_transaction(
        transaction_id="new-high-1",
        user_id=user_id,
        card_id=card_id,
        amount=2500.0,
        merchant_category="electronics"
    )

    result = await fraud_detector.assess_transaction(tx)

    # Assertions
    assert "first_card_use_high_value" in result.signals_triggered, \
        "First-time high-value signal should be triggered"
    assert result.risk_score > 0.5, \
        f"Risk score should be high (>0.5), got {result.risk_score:.3f}"
    assert result.decision in [RiskDecision.REVIEW, RiskDecision.DECLINE], \
        f"New card high-value should be reviewed/declined, got {result.decision}"

    print(f"\n✓ New Card High-Value Transaction Test PASSED")
    print(f"  Amount: ${tx.amount:.2f}")
    print(f"  Risk Score: {result.risk_score:.3f}")
    print(f"  Decision: {result.decision.value}")
    print(f"  Signals: {', '.join(result.signals_triggered)}")


@pytest.mark.asyncio
async def test_new_card_moderate_value_transaction(fraud_detector):
    """
    Test Scenario: New Card with Moderate-Value Purchase

    Pattern: First transaction on new card is $750
    Expected: Moderate risk, possibly REVIEW
    """
    card_id = "card-new-moderate-001"
    user_id = "user-new-moderate-001"

    # First transaction: $750 (50% of high-value threshold)
    tx = create_transaction(
        transaction_id="new-moderate-1",
        user_id=user_id,
        card_id=card_id,
        amount=750.0,
        merchant_category="clothing"
    )

    result = await fraud_detector.assess_transaction(tx)

    # Assertions
    assert "first_card_use" in ' '.join(result.signals_triggered), \
        "First-time card use signal should be present"
    assert 0.3 <= result.risk_score <= 0.7, \
        f"Risk score should be moderate (0.3-0.7), got {result.risk_score:.3f}"

    print(f"\n✓ New Card Moderate-Value Test PASSED")
    print(f"  Amount: ${tx.amount:.2f}")
    print(f"  Risk Score: {result.risk_score:.3f}")
    print(f"  Decision: {result.decision.value}")


@pytest.mark.asyncio
async def test_new_card_low_value_transaction(fraud_detector):
    """
    Test Scenario: New Card with Low-Value Purchase

    Pattern: First transaction on new card is $25
    Expected: Low risk, APPROVE decision
    """
    card_id = "card-new-low-001"
    user_id = "user-new-low-001"

    # First transaction: $25
    tx = create_transaction(
        transaction_id="new-low-1",
        user_id=user_id,
        card_id=card_id,
        amount=25.0,
        merchant_category="grocery"
    )

    result = await fraud_detector.assess_transaction(tx)

    # Assertions
    assert result.decision == RiskDecision.APPROVE, \
        f"New card low-value should be approved, got {result.decision}"
    assert result.risk_score < 0.4, \
        f"Risk score should be low (<0.4), got {result.risk_score:.3f}"

    print(f"\n✓ New Card Low-Value Test PASSED")
    print(f"  Amount: ${tx.amount:.2f}")
    print(f"  Risk Score: {result.risk_score:.3f}")
    print(f"  Decision: {result.decision.value}")


@pytest.mark.asyncio
async def test_established_card_high_value_transaction(fraud_detector):
    """
    Test Scenario: Established Card with High-Value Purchase

    Pattern: Card has transaction history, then makes $2500 purchase
    Expected: Lower risk than new card, possibly APPROVE
    """
    card_id = "card-established-001"
    user_id = "user-established-001"
    base_time = datetime.utcnow()

    # Build transaction history (3 normal transactions)
    for i in range(3):
        tx = create_transaction(
            transaction_id=f"established-history-{i}",
            user_id=user_id,
            card_id=card_id,
            amount=50.0 + i * 25.0,
            timestamp=base_time + timedelta(minutes=i * 30)
        )
        await fraud_detector.assess_transaction(tx)

    # Now make high-value purchase
    high_value_tx = create_transaction(
        transaction_id="established-high-1",
        user_id=user_id,
        card_id=card_id,
        amount=2500.0,
        merchant_category="electronics",
        timestamp=base_time + timedelta(hours=2)
    )

    result = await fraud_detector.assess_transaction(high_value_tx)

    # Assertions
    assert "first_card_use_high_value" not in result.signals_triggered, \
        "Should not trigger first-time high-value for established card"
    assert result.risk_score < 0.6, \
        f"Established card should have lower risk (<0.6), got {result.risk_score:.3f}"

    print(f"\n✓ Established Card High-Value Test PASSED")
    print(f"  Amount: ${high_value_tx.amount:.2f}")
    print(f"  Risk Score: {result.risk_score:.3f}")
    print(f"  Decision: {result.decision.value}")


@pytest.mark.asyncio
async def test_new_card_multiple_high_value_attempts(fraud_detector):
    """
    Test Scenario: New Card with Multiple High-Value Attempts

    Pattern: New card tries multiple high-value transactions quickly
    Expected: Very high risk, DECLINE decision
    """
    card_id = "card-new-multi-high-001"
    user_id = "user-new-multi-high-001"
    base_time = datetime.utcnow()

    # Attempt 3 high-value transactions with new card in quick succession
    results = []
    for i in range(3):
        tx = create_transaction(
            transaction_id=f"new-multi-high-{i}",
            user_id=user_id,
            card_id=card_id,
            amount=1500.0,
            timestamp=base_time + timedelta(minutes=i * 2)
        )
        result = await fraud_detector.assess_transaction(tx)
        results.append(result)

    final_result = results[-1]

    # Assertions
    assert final_result.decision == RiskDecision.DECLINE, \
        f"Multiple new card high-value should be declined, got {final_result.decision}"
    assert final_result.risk_score > 0.7, \
        f"Risk score should be very high (>0.7), got {final_result.risk_score:.3f}"

    # Should have multiple signals
    assert len(final_result.signals_triggered) >= 3, \
        f"Should have multiple signals, got {len(final_result.signals_triggered)}"

    print(f"\n✓ New Card Multiple High-Value Test PASSED")
    print(f"  Attempts: {len(results)}")
    print(f"  Final Risk Score: {final_result.risk_score:.3f}")
    print(f"  Decision: {final_result.decision.value}")
    print(f"  Signals: {', '.join(final_result.signals_triggered[:5])}")

"""
Test velocity breach scenarios.

Velocity limits:
- 5 transactions per 5 minutes
- 20 transactions per hour
- $5000 per 5 minutes
- $20000 per hour
"""

import pytest
from datetime import datetime, timedelta

from tests.conftest import create_transaction
from src.fraud.models import RiskDecision


@pytest.mark.asyncio
async def test_transaction_count_velocity_breach(fraud_detector):
    """
    Test Scenario: Transaction Count Velocity Breach

    Pattern: 6+ transactions in 5 minutes (exceeds limit of 5)
    Expected: High risk, DECLINE/REVIEW decision
    """
    card_id = "card-velocity-count-001"
    user_id = "user-velocity-001"
    base_time = datetime.utcnow()

    # Send 6 transactions in 5 minutes
    results = []
    for i in range(6):
        tx = create_transaction(
            transaction_id=f"velocity-count-{i}",
            user_id=user_id,
            card_id=card_id,
            amount=50.0,
            timestamp=base_time + timedelta(seconds=i * 45)  # 45 seconds apart
        )
        result = await fraud_detector.assess_transaction(tx)
        results.append(result)

    # 6th transaction should trigger velocity breach
    final_result = results[-1]

    # Assertions
    assert any("velocity" in signal for signal in final_result.signals_triggered), \
        "Velocity breach signal should be triggered"
    assert final_result.risk_score > 0.5, \
        f"Risk score should be high (>0.5), got {final_result.risk_score:.3f}"
    assert final_result.decision in [RiskDecision.REVIEW, RiskDecision.DECLINE], \
        f"6th transaction should be reviewed/declined, got {final_result.decision}"

    print(f"\n✓ Transaction Count Velocity Breach Test PASSED")
    print(f"  Transactions: {len(results)}")
    print(f"  Final Risk Score: {final_result.risk_score:.3f}")
    print(f"  Decision: {final_result.decision.value}")
    print(f"  Velocity Signals: {[s for s in final_result.signals_triggered if 'velocity' in s]}")


@pytest.mark.asyncio
async def test_amount_velocity_breach(fraud_detector):
    """
    Test Scenario: Amount Velocity Breach

    Pattern: $6000+ in 5 minutes (exceeds limit of $5000)
    Expected: High risk, DECLINE/REVIEW decision
    """
    card_id = "card-velocity-amount-001"
    user_id = "user-velocity-amount-001"
    base_time = datetime.utcnow()

    # Send transactions totaling $6500 in 5 minutes
    amounts = [1500.0, 2000.0, 1500.0, 1500.0]  # Total: $6500
    results = []

    for i, amount in enumerate(amounts):
        tx = create_transaction(
            transaction_id=f"velocity-amount-{i}",
            user_id=user_id,
            card_id=card_id,
            amount=amount,
            timestamp=base_time + timedelta(seconds=i * 60)  # 1 minute apart
        )
        result = await fraud_detector.assess_transaction(tx)
        results.append(result)

    # Last transaction should trigger amount velocity breach
    final_result = results[-1]

    # Assertions
    assert any("amount" in signal for signal in final_result.signals_triggered), \
        "Amount velocity breach signal should be triggered"
    assert final_result.risk_score > 0.5, \
        f"Risk score should be high (>0.5), got {final_result.risk_score:.3f}"

    print(f"\n✓ Amount Velocity Breach Test PASSED")
    print(f"  Total Amount: ${sum(amounts):.2f}")
    print(f"  Final Risk Score: {final_result.risk_score:.3f}")
    print(f"  Decision: {final_result.decision.value}")
    print(f"  Amount Signals: {[s for s in final_result.signals_triggered if 'amount' in s]}")


@pytest.mark.asyncio
async def test_ip_velocity_breach(fraud_detector):
    """
    Test Scenario: IP Address Velocity Breach

    Pattern: Multiple cards from same IP in short time
    Expected: Moderate risk, potential review
    """
    ip_address = "192.168.1.100"
    base_time = datetime.utcnow()

    # Send 6 transactions from different cards/users but same IP
    results = []
    for i in range(6):
        tx = create_transaction(
            transaction_id=f"ip-velocity-{i}",
            user_id=f"user-{i}",
            card_id=f"card-{i}",
            ip_address=ip_address,
            amount=100.0,
            timestamp=base_time + timedelta(seconds=i * 30)  # 30 seconds apart
        )
        result = await fraud_detector.assess_transaction(tx)
        results.append(result)

    final_result = results[-1]

    # Assertions
    assert any("ip_velocity" in signal for signal in final_result.signals_triggered), \
        "IP velocity breach signal should be triggered"
    assert final_result.risk_score > 0.3, \
        f"Risk score should be elevated (>0.3), got {final_result.risk_score:.3f}"

    print(f"\n✓ IP Velocity Breach Test PASSED")
    print(f"  Transactions from IP: {len(results)}")
    print(f"  Final Risk Score: {final_result.risk_score:.3f}")
    print(f"  Decision: {final_result.decision.value}")


@pytest.mark.asyncio
async def test_normal_velocity_approved(fraud_detector):
    """
    Test Scenario: Normal Transaction Velocity

    Pattern: Within all velocity limits
    Expected: Low risk, APPROVE decision
    """
    card_id = "card-normal-001"
    user_id = "user-normal-001"
    base_time = datetime.utcnow()

    # Send 3 transactions in 10 minutes (well within limits)
    amounts = [50.0, 75.0, 100.0]
    results = []

    for i, amount in enumerate(amounts):
        tx = create_transaction(
            transaction_id=f"normal-{i}",
            user_id=user_id,
            card_id=card_id,
            amount=amount,
            timestamp=base_time + timedelta(minutes=i * 3)  # 3 minutes apart
        )
        result = await fraud_detector.assess_transaction(tx)
        results.append(result)

    final_result = results[-1]

    # Assertions
    assert final_result.decision == RiskDecision.APPROVE, \
        f"Normal velocity should be approved, got {final_result.decision}"
    assert final_result.risk_score < 0.3, \
        f"Risk score should be low (<0.3), got {final_result.risk_score:.3f}"

    print(f"\n✓ Normal Velocity Test PASSED")
    print(f"  Transactions: {len(results)}")
    print(f"  Final Risk Score: {final_result.risk_score:.3f}")
    print(f"  Decision: {final_result.decision.value}")


@pytest.mark.asyncio
async def test_combined_velocity_breach(fraud_detector):
    """
    Test Scenario: Combined Velocity Breach

    Pattern: Both count and amount velocity exceeded
    Expected: Very high risk, DECLINE decision
    """
    card_id = "card-combined-001"
    user_id = "user-combined-001"
    base_time = datetime.utcnow()

    # Send 7 transactions totaling $7000 in 5 minutes
    results = []
    for i in range(7):
        tx = create_transaction(
            transaction_id=f"combined-{i}",
            user_id=user_id,
            card_id=card_id,
            amount=1000.0,
            timestamp=base_time + timedelta(seconds=i * 40)  # 40 seconds apart
        )
        result = await fraud_detector.assess_transaction(tx)
        results.append(result)

    final_result = results[-1]

    # Assertions
    assert final_result.decision == RiskDecision.DECLINE, \
        f"Combined velocity breach should be declined, got {final_result.decision}"
    assert final_result.risk_score > 0.7, \
        f"Risk score should be very high (>0.7), got {final_result.risk_score:.3f}"

    # Should have both count and amount signals
    signals_str = ' '.join(final_result.signals_triggered)
    assert "count" in signals_str or "velocity" in signals_str, \
        "Should have count velocity signal"
    assert "amount" in signals_str, \
        "Should have amount velocity signal"

    print(f"\n✓ Combined Velocity Breach Test PASSED")
    print(f"  Transactions: {len(results)}")
    print(f"  Total Amount: ${1000.0 * len(results):.2f}")
    print(f"  Final Risk Score: {final_result.risk_score:.3f}")
    print(f"  Decision: {final_result.decision.value}")
    print(f"  Signals: {len(final_result.signals_triggered)}")

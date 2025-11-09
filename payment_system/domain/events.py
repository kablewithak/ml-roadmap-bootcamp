"""
Domain Events - Immutable Facts About What Happened

CRITICAL CONCEPT: Events are the source of truth, not database rows.

Why this matters:
- Traditional DB: UPDATE payments SET status='failed' (lost history)
- Event sourcing: PaymentFailed event stored forever (complete audit trail)

Business impact:
- Chargeback disputes: Can prove exactly what happened 6 months ago
- Debugging: Replay events to reproduce any bug
- Compliance: Automatic audit trail for PCI-DSS, SOX, GDPR
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class EventMetadata(BaseModel):
    """
    Metadata attached to every event.

    CRITICAL: event_id must be deterministic for idempotency.
    If we retry a failed operation, we generate the same event_id.
    This prevents duplicate charges when networks lie to us.
    """

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str
    aggregate_id: str  # Which payment/chargeback/merchant does this belong to?
    aggregate_type: Literal["payment", "chargeback", "merchant", "fx_conversion"]
    sequence_number: int  # Order of events for this aggregate (prevents reordering)
    occurred_at: datetime = Field(default_factory=datetime.utcnow)
    causation_id: str | None = None  # What event caused this? (for tracing)
    correlation_id: str | None = None  # Original request ID (for distributed tracing)
    user_id: str | None = None  # Who triggered this? (for audit)

    @field_validator("occurred_at", mode="before")
    @classmethod
    def ensure_utc(cls, v: datetime | str) -> datetime:
        """
        CRITICAL: Always UTC to prevent timezone disasters.

        Real bug prevented: Merchant in Tokyo disputes $50K transaction.
        Without UTC: "Transaction happened at 3am" (whose timezone?)
        With UTC: "Transaction happened at 2023-10-15T18:30:00Z" (unambiguous)
        """
        if isinstance(v, str):
            v = datetime.fromisoformat(v)
        return v if v.tzinfo else v.replace(tzinfo=None)  # Force naive UTC


class DomainEvent(BaseModel):
    """
    Base class for all domain events.

    Design principle: Events are IMMUTABLE and describe PAST FACTS.
    - Good: PaymentCompleted (past tense, immutable fact)
    - Bad: CompletePayment (command, not event)
    - Bad: PaymentStatus (mutable state, not event)
    """

    metadata: EventMetadata

    class Config:
        frozen = True  # Immutable

    def with_metadata(self, **kwargs: Any) -> DomainEvent:
        """Update metadata (useful for adding correlation IDs in middleware)."""
        metadata_dict = self.metadata.model_dump()
        metadata_dict.update(kwargs)
        return self.model_copy(update={"metadata": EventMetadata(**metadata_dict)})


# ============================================================================
# PAYMENT LIFECYCLE EVENTS
# ============================================================================

class Currency(str, Enum):
    """ISO 4217 currency codes."""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CAD = "CAD"
    AUD = "AUD"
    CHF = "CHF"
    CNY = "CNY"
    # Add more as needed


class PaymentMethod(str, Enum):
    """Payment rails supported."""
    CARD = "card"
    ACH = "ach"
    WIRE = "wire"
    WALLET = "wallet"  # Future: Apple Pay, Google Pay


class PaymentInitiated(DomainEvent):
    """
    Payment process started.

    This is the FIRST event in every payment saga.
    Everything else flows from this.
    """

    payment_id: str
    merchant_id: str
    customer_id: str
    amount: Decimal
    currency: Currency
    payment_method: PaymentMethod
    description: str
    idempotency_key: str  # CRITICAL: Prevents double-charging on retry

    # Risk signals (for fraud detection)
    customer_ip: str | None = None
    user_agent: str | None = None

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, v: Decimal) -> Decimal:
        """
        CRITICAL: Prevent negative or zero payments.

        Real attack prevented: Negative amount = refund without original charge.
        Attacker could drain merchant accounts.
        """
        if v <= 0:
            raise ValueError(f"Payment amount must be positive, got {v}")
        if v > Decimal("100000"):  # $100K limit
            raise ValueError(f"Payment amount exceeds limit: {v}")
        return v.quantize(Decimal("0.01"))  # Always 2 decimal places


class PaymentMethodValidated(DomainEvent):
    """
    Payment method passed validation checks.

    For cards: Luhn check, expiry date, CVV
    For ACH: Routing number validation
    For wire: SWIFT code validation
    """

    payment_id: str
    payment_method: PaymentMethod
    validation_checks: dict[str, bool]  # Which checks passed


class FraudCheckCompleted(DomainEvent):
    """
    Fraud detection completed.

    Business impact: Preventing $1M in fraud saves:
    - $1M in direct losses
    - $100K in chargeback fees (10% chargeback rate)
    - $500K in reputation damage (customers scared to buy)
    Total saved: $1.6M
    """

    payment_id: str
    risk_score: Decimal  # 0-100, higher = riskier
    risk_level: Literal["low", "medium", "high", "critical"]
    checks_performed: list[str]
    flags_raised: list[str]
    recommended_action: Literal["approve", "review", "decline", "require_3ds"]


class PaymentAuthorized(DomainEvent):
    """
    Payment gateway (Stripe/bank) authorized the charge.

    CRITICAL DISTINCTION:
    - Authorized: Bank reserved the funds (can still fail)
    - Captured: Money actually moved (irreversible without refund)

    Why separate events?
    - Hotels: Authorize $500, capture $350 (customer didn't raid minibar)
    - Fraud: Authorize low amount, if succeeds, try higher amount (fraud pattern)
    """

    payment_id: str
    authorization_id: str  # Gateway's reference (e.g., Stripe charge ID)
    authorized_amount: Decimal
    authorized_currency: Currency
    gateway: Literal["stripe", "ach", "wire"]
    gateway_response: dict[str, Any]  # Raw response for debugging


class PaymentCaptured(DomainEvent):
    """
    Money actually moved from customer to platform.

    This is when revenue is recognized (accounting).
    """

    payment_id: str
    captured_amount: Decimal
    captured_currency: Currency
    capture_id: str  # Gateway reference
    fee_amount: Decimal  # Gateway fee (e.g., Stripe's 2.9% + $0.30)
    net_amount: Decimal  # What we actually get


class FXConversionRequired(DomainEvent):
    """
    Payment needs currency conversion.

    Example: Customer pays in USD, merchant settles in EUR.
    This triggers the FX conversion saga.
    """

    payment_id: str
    from_amount: Decimal
    from_currency: Currency
    to_currency: Currency
    conversion_reason: Literal["customer_currency", "merchant_settlement", "cross_border"]


class FXRateLocked(DomainEvent):
    """
    Exchange rate locked in for this transaction.

    CRITICAL: Must lock rate to prevent slippage.

    Without locking:
    - T+0: Customer pays $100, rate is 0.85 = €85 expected
    - T+1: Rate drops to 0.80 = €80 actual (lost €5)
    - At $1B volume: 1% slippage = $10M loss

    With locking:
    - Rate locked at quote time
    - We honor that rate even if market moves
    - We hedge the risk separately
    """

    payment_id: str
    from_currency: Currency
    to_currency: Currency
    rate: Decimal  # Exchange rate
    markup_bps: int  # Our markup in basis points (e.g., 150 = 1.5%)
    effective_rate: Decimal  # Rate after markup (our revenue)
    locked_until: datetime
    rate_source: str  # Which API gave us this rate
    quote_id: str  # For rate guarantee disputes


class FXConverted(DomainEvent):
    """
    Currency conversion completed.

    Business impact: This is pure margin.
    - $1B annual volume
    - 30% cross-border (need FX)
    - 1.5% FX markup
    - Revenue: $300M * 0.015 = $4.5M annually
    """

    payment_id: str
    from_amount: Decimal
    from_currency: Currency
    to_amount: Decimal
    to_currency: Currency
    rate_used: Decimal
    markup_earned: Decimal  # Our revenue on this conversion
    conversion_id: str


class PaymentCompleted(DomainEvent):
    """
    Payment saga completed successfully.

    This is the HAPPY PATH end state.
    Money moved, books balanced, everyone happy.
    """

    payment_id: str
    final_amount: Decimal
    final_currency: Currency
    settled_amount: Decimal  # After fees
    total_fees: Decimal
    completed_at: datetime


class PaymentFailed(DomainEvent):
    """
    Payment saga failed.

    CRITICAL: This event triggers compensation (refunds, reversals).
    """

    payment_id: str
    failure_stage: Literal[
        "validation",
        "fraud_check",
        "authorization",
        "capture",
        "fx_conversion",
        "settlement",
    ]
    failure_reason: str
    failure_code: str  # Machine-readable (e.g., "insufficient_funds")
    gateway_error: dict[str, Any] | None = None
    should_retry: bool  # Can customer retry or is this permanent?


class PaymentRefunded(DomainEvent):
    """
    Payment reversed (full or partial).

    Reasons:
    - Customer requested refund
    - Fraud detected after capture
    - Merchant requested reversal
    - Chargeback pre-emption (refund before customer disputes)
    """

    payment_id: str
    refund_id: str
    refund_amount: Decimal
    refund_currency: Currency
    refund_reason: str
    refunded_by: str  # User ID who initiated
    is_partial: bool


# ============================================================================
# CHARGEBACK LIFECYCLE EVENTS
# ============================================================================

class ChargebackReason(str, Enum):
    """
    Reason codes from card networks.

    Win rates vary dramatically by reason:
    - Fraud: 20% win rate (customer claims unauthorized)
    - Product not received: 60% win rate (we have tracking)
    - Product not as described: 40% win rate (subjective)
    - Duplicate charge: 90% win rate (we have logs)
    """

    FRAUDULENT = "fraudulent"
    PRODUCT_NOT_RECEIVED = "product_not_received"
    PRODUCT_UNACCEPTABLE = "product_unacceptable"
    DUPLICATE = "duplicate"
    SUBSCRIPTION_CANCELLED = "subscription_cancelled"
    CREDIT_NOT_PROCESSED = "credit_not_processed"
    GENERAL = "general"


class ChargebackInitiated(DomainEvent):
    """
    Customer disputed transaction with their bank.

    Business impact:
    - Lost transaction amount
    - $15-25 chargeback fee
    - 1% chargeback rate = lose Stripe account
    - High rate = banned from payment processing
    """

    chargeback_id: str
    payment_id: str
    amount: Decimal
    currency: Currency
    reason: ChargebackReason
    reason_code: str  # Network code (e.g., "4863" for Visa)
    dispute_deadline: datetime  # When we must respond
    evidence_required: list[str]  # What docs we need


class ChargebackEvidenceCollected(DomainEvent):
    """
    We gathered evidence to fight the chargeback.

    Winning strategy:
    - Proof of delivery (tracking number, signature)
    - Customer communication logs
    - IP address match
    - AVS/CVV match
    - Previous successful transactions (shows pattern)
    """

    chargeback_id: str
    evidence_type: str
    evidence_data: dict[str, Any]
    collected_by: str
    confidence_score: Decimal  # ML model prediction of win rate


class ChargebackDisputed(DomainEvent):
    """
    We submitted evidence to fight the chargeback.

    Process:
    1. Collect evidence (7-14 days)
    2. Submit to network (Visa/Mastercard/Amex)
    3. Wait for decision (30-90 days)

    During this time: Money is held, can't settle to merchant.
    """

    chargeback_id: str
    evidence_submitted: dict[str, Any]
    submitted_at: datetime
    expected_decision_date: datetime


class ChargebackWon(DomainEvent):
    """
    We won the chargeback dispute!

    Money returned to merchant, fee refunded.
    This is tracked for win rate analytics.
    """

    chargeback_id: str
    won_amount: Decimal
    network_decision: str
    won_at: datetime


class ChargebackLost(DomainEvent):
    """
    We lost the chargeback dispute.

    Money permanently gone, fee charged.
    Analyze WHY we lost to improve future win rate.
    """

    chargeback_id: str
    lost_amount: Decimal
    network_decision: str
    loss_reason: str
    lost_at: datetime

    # ML training data
    should_have_refunded: bool  # Would pre-emptive refund been cheaper?
    win_probability_predicted: Decimal  # What did our model predict?


class ChargebackAccepted(DomainEvent):
    """
    We chose not to fight the chargeback.

    Decision factors:
    - Low win probability (< 30%)
    - Small amount (not worth effort)
    - Customer goodwill matters more
    - Evidence collection cost > potential recovery
    """

    chargeback_id: str
    accepted_reason: str
    accepted_by: str


# ============================================================================
# RECONCILIATION EVENTS
# ============================================================================

class ReconciliationStarted(DomainEvent):
    """
    Daily reconciliation process started.

    We compare:
    - Our event store (source of truth)
    - Stripe dashboard
    - Bank statements
    - Merchant payouts

    Any mismatch triggers investigation.
    """

    reconciliation_id: str
    reconciliation_date: datetime
    period_start: datetime
    period_end: datetime
    expected_transaction_count: int
    expected_total_amount: Decimal


class ReconciliationDiscrepancy(DomainEvent):
    """
    Found mismatch between our records and external systems.

    CRITICAL: This can indicate:
    - Lost webhook (Stripe charged, we didn't record)
    - Double recording (we recorded twice)
    - FX rate mismatch
    - Fee calculation error

    Cost of ignoring: $8K manual investigation + potential losses
    """

    reconciliation_id: str
    discrepancy_type: Literal[
        "missing_payment",
        "duplicate_payment",
        "amount_mismatch",
        "fx_variance",
        "fee_mismatch",
    ]
    expected_value: Any
    actual_value: Any
    variance_amount: Decimal
    severity: Literal["low", "medium", "high", "critical"]
    auto_remediation_attempted: bool


class ReconciliationCompleted(DomainEvent):
    """
    Reconciliation finished, books balanced.
    """

    reconciliation_id: str
    total_transactions: int
    total_amount: Decimal
    discrepancies_found: int
    discrepancies_resolved: int
    manual_review_required: int
    completed_at: datetime


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_event_metadata(
    event_type: str,
    aggregate_id: str,
    aggregate_type: Literal["payment", "chargeback", "merchant", "fx_conversion"],
    sequence_number: int,
    correlation_id: str | None = None,
    causation_id: str | None = None,
) -> EventMetadata:
    """
    Factory for creating consistent event metadata.

    This ensures every event has proper tracing IDs.
    """
    return EventMetadata(
        event_type=event_type,
        aggregate_id=aggregate_id,
        aggregate_type=aggregate_type,
        sequence_number=sequence_number,
        correlation_id=correlation_id or str(uuid.uuid4()),
        causation_id=causation_id,
    )

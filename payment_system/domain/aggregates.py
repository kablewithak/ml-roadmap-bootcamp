"""
Aggregates - Consistency Boundaries

An aggregate is a cluster of domain objects that must be consistent.

Key concepts:
1. Aggregate Root: The entry point (e.g., Payment)
2. Invariants: Rules that must ALWAYS be true
3. Events: Aggregates produce events, don't mutate state directly
4. Consistency: One aggregate = one transaction boundary

Example invariant:
"A payment cannot be refunded for more than the captured amount"

If this breaks, we lose money. Aggregates enforce this.

Why event sourcing in aggregates?
- State is derived from events (replay to get current state)
- No UPDATE statements (only INSERT events)
- Complete history preserved
- Easy to test (given events, expect behavior)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

import structlog

from payment_system.domain.events import (
    ChargebackIdentifier,
    ChargebackInitiated,
    ChargebackReason,
    Currency,
    DomainEvent,
    FXConversionRequired,
    FXConverted,
    FXRateLocked,
    FraudCheckCompleted,
    PaymentAuthorized,
    PaymentCaptured,
    PaymentCompleted,
    PaymentFailed,
    PaymentIdentifier,
    PaymentInitiated,
    PaymentMethod,
    PaymentMethodValidated,
    PaymentRefunded,
    create_event_metadata,
)
from payment_system.domain.value_objects import FXRate, IdempotencyKey, Money

logger = structlog.get_logger()


class PaymentStatus(str, Enum):
    """
    Payment lifecycle states.

    State machine:
    INITIATED → VALIDATED → FRAUD_CHECKED → AUTHORIZED → CAPTURED → COMPLETED
                                                ↓
                                              FAILED
                                                ↓
                                          REFUND_PENDING → REFUNDED
    """

    INITIATED = "initiated"
    VALIDATED = "validated"
    FRAUD_CHECKED = "fraud_checked"
    AUTHORIZED = "authorized"
    CAPTURED = "captured"
    FX_CONVERTING = "fx_converting"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUND_PENDING = "refund_pending"
    REFUNDED = "refunded"


@dataclass
class Payment:
    """
    Payment Aggregate Root.

    This aggregate enforces payment business rules and produces events.

    CRITICAL DESIGN: State is derived from events, not stored directly.

    Traditional approach:
    payment.status = "completed"  # Lost history
    payment.save()

    Event sourcing approach:
    payment.apply(PaymentCompleted(...))  # Event recorded
    payment.status = "completed"  # State derived from event

    Benefits:
    - Can rebuild state from events (time-travel)
    - Complete audit trail
    - Can add new projections later
    """

    # Identifier
    payment_id: PaymentIdentifier
    idempotency_key: IdempotencyKey

    # Participants
    merchant_id: str
    customer_id: str

    # Money
    amount: Money
    payment_method: PaymentMethod

    # State (derived from events)
    status: PaymentStatus = PaymentStatus.INITIATED
    version: int = 0  # For optimistic concurrency control

    # Event sourcing
    _uncommitted_events: list[DomainEvent] = field(default_factory=list)
    _event_history: list[DomainEvent] = field(default_factory=list)

    # Tracking
    correlation_id: str | None = None

    # Optional fields (populated during lifecycle)
    authorization_id: str | None = None
    captured_amount: Money | None = None
    fx_conversion: dict[str, Any] | None = None
    failure_reason: str | None = None

    @classmethod
    def initiate(
        cls,
        payment_id: str,
        merchant_id: str,
        customer_id: str,
        amount: Money,
        payment_method: PaymentMethod,
        idempotency_key: str,
        description: str = "",
        customer_ip: str | None = None,
    ) -> Payment:
        """
        Factory method: Start a new payment.

        This is the ONLY way to create a payment (enforces invariants).

        Why factory method?
        - Ensures payment always starts with PaymentInitiated event
        - Validates amount > 0
        - Generates correlation ID for tracing
        """
        payment = cls(
            payment_id=PaymentIdentifier(value=payment_id),
            idempotency_key=IdempotencyKey(value=idempotency_key),
            merchant_id=merchant_id,
            customer_id=customer_id,
            amount=amount,
            payment_method=payment_method,
        )

        # Produce PaymentInitiated event
        event = PaymentInitiated(
            metadata=create_event_metadata(
                event_type="PaymentInitiated",
                aggregate_id=payment_id,
                aggregate_type="payment",
                sequence_number=0,
            ),
            payment_id=payment_id,
            merchant_id=merchant_id,
            customer_id=customer_id,
            amount=amount.amount,
            currency=amount.currency,
            payment_method=payment_method,
            description=description,
            idempotency_key=idempotency_key,
            customer_ip=customer_ip,
        )

        payment._apply_event(event)
        return payment

    def validate_payment_method(self, validation_checks: dict[str, bool]) -> None:
        """
        Validate payment method (card checks, bank account, etc.).

        Invariant: Can only validate if status is INITIATED.
        """
        if self.status != PaymentStatus.INITIATED:
            raise PaymentError(
                f"Cannot validate payment in status {self.status}",
                payment_id=str(self.payment_id),
            )

        event = PaymentMethodValidated(
            metadata=create_event_metadata(
                event_type="PaymentMethodValidated",
                aggregate_id=str(self.payment_id),
                aggregate_type="payment",
                sequence_number=self.version + 1,
            ),
            payment_id=str(self.payment_id),
            payment_method=self.payment_method,
            validation_checks=validation_checks,
        )

        self._apply_event(event)

    def complete_fraud_check(
        self,
        risk_score: Decimal,
        risk_level: str,
        checks_performed: list[str],
        recommended_action: str,
    ) -> None:
        """
        Record fraud check results.

        Business logic: If risk is CRITICAL, automatically fail payment.
        """
        event = FraudCheckCompleted(
            metadata=create_event_metadata(
                event_type="FraudCheckCompleted",
                aggregate_id=str(self.payment_id),
                aggregate_type="payment",
                sequence_number=self.version + 1,
            ),
            payment_id=str(self.payment_id),
            risk_score=risk_score,
            risk_level=risk_level,  # type: ignore
            checks_performed=checks_performed,
            flags_raised=[],
            recommended_action=recommended_action,  # type: ignore
        )

        self._apply_event(event)

        # Business rule: Auto-decline critical risk
        if risk_level == "critical":
            self.fail("Fraud risk too high", "fraud_detected_critical")

    def authorize(
        self,
        authorization_id: str,
        authorized_amount: Money,
        gateway: str,
        gateway_response: dict[str, Any],
    ) -> None:
        """
        Record successful payment authorization.

        Invariant: Amount authorized must match requested amount.
        """
        if self.status not in [PaymentStatus.VALIDATED, PaymentStatus.FRAUD_CHECKED]:
            raise PaymentError(
                f"Cannot authorize payment in status {self.status}",
                payment_id=str(self.payment_id),
            )

        # Business rule: Authorized amount must match requested
        if authorized_amount != self.amount:
            raise PaymentError(
                f"Authorized amount {authorized_amount} != requested {self.amount}",
                payment_id=str(self.payment_id),
            )

        event = PaymentAuthorized(
            metadata=create_event_metadata(
                event_type="PaymentAuthorized",
                aggregate_id=str(self.payment_id),
                aggregate_type="payment",
                sequence_number=self.version + 1,
            ),
            payment_id=str(self.payment_id),
            authorization_id=authorization_id,
            authorized_amount=authorized_amount.amount,
            authorized_currency=authorized_amount.currency,
            gateway=gateway,  # type: ignore
            gateway_response=gateway_response,
        )

        self._apply_event(event)

    def capture(
        self,
        captured_amount: Money,
        capture_id: str,
        fee_amount: Money,
    ) -> None:
        """
        Capture the payment (actually move money).

        Invariant: Can only capture authorized payments.
        Invariant: Captured amount <= authorized amount.
        """
        if self.status != PaymentStatus.AUTHORIZED:
            raise PaymentError(
                f"Cannot capture payment in status {self.status}",
                payment_id=str(self.payment_id),
            )

        # Business rule: Can't capture more than authorized
        if captured_amount > self.amount:
            raise PaymentError(
                f"Cannot capture {captured_amount}, authorized was {self.amount}",
                payment_id=str(self.payment_id),
            )

        net_amount = captured_amount - fee_amount

        event = PaymentCaptured(
            metadata=create_event_metadata(
                event_type="PaymentCaptured",
                aggregate_id=str(self.payment_id),
                aggregate_type="payment",
                sequence_number=self.version + 1,
            ),
            payment_id=str(self.payment_id),
            captured_amount=captured_amount.amount,
            captured_currency=captured_amount.currency,
            capture_id=capture_id,
            fee_amount=fee_amount.amount,
            net_amount=net_amount.amount,
        )

        self._apply_event(event)

    def require_fx_conversion(
        self, to_currency: Currency, reason: str = "merchant_settlement"
    ) -> None:
        """
        Mark that this payment needs FX conversion.

        This triggers the FX conversion saga.
        """
        event = FXConversionRequired(
            metadata=create_event_metadata(
                event_type="FXConversionRequired",
                aggregate_id=str(self.payment_id),
                aggregate_type="payment",
                sequence_number=self.version + 1,
            ),
            payment_id=str(self.payment_id),
            from_amount=self.amount.amount,
            from_currency=self.amount.currency,
            to_currency=to_currency,
            conversion_reason=reason,  # type: ignore
        )

        self._apply_event(event)

    def lock_fx_rate(
        self, rate: FXRate, quote_id: str, locked_until: datetime
    ) -> None:
        """
        Lock in FX rate for this transaction.

        CRITICAL: Prevents slippage between quote and execution.
        """
        event = FXRateLocked(
            metadata=create_event_metadata(
                event_type="FXRateLocked",
                aggregate_id=str(self.payment_id),
                aggregate_type="payment",
                sequence_number=self.version + 1,
            ),
            payment_id=str(self.payment_id),
            from_currency=rate.from_currency,
            to_currency=rate.to_currency,
            rate=rate.rate,
            markup_bps=rate.markup_bps,
            effective_rate=rate.effective_rate,
            locked_until=locked_until,
            rate_source=rate.rate_source,
            quote_id=quote_id,
        )

        self._apply_event(event)

    def convert_currency(
        self, to_amount: Money, rate_used: FXRate, conversion_id: str
    ) -> None:
        """
        Record completed FX conversion.

        Business impact: This records our FX markup revenue.
        """
        markup_earned = rate_used.markup_amount(self.amount)

        event = FXConverted(
            metadata=create_event_metadata(
                event_type="FXConverted",
                aggregate_id=str(self.payment_id),
                aggregate_type="payment",
                sequence_number=self.version + 1,
            ),
            payment_id=str(self.payment_id),
            from_amount=self.amount.amount,
            from_currency=self.amount.currency,
            to_amount=to_amount.amount,
            to_currency=to_amount.currency,
            rate_used=rate_used.effective_rate,
            markup_earned=markup_earned.amount,
            conversion_id=conversion_id,
        )

        self._apply_event(event)

    def complete(self, settled_amount: Money, total_fees: Money) -> None:
        """
        Complete the payment (final state).

        This is when revenue is recognized.
        """
        if self.status not in [PaymentStatus.CAPTURED, PaymentStatus.FX_CONVERTING]:
            raise PaymentError(
                f"Cannot complete payment in status {self.status}",
                payment_id=str(self.payment_id),
            )

        event = PaymentCompleted(
            metadata=create_event_metadata(
                event_type="PaymentCompleted",
                aggregate_id=str(self.payment_id),
                aggregate_type="payment",
                sequence_number=self.version + 1,
            ),
            payment_id=str(self.payment_id),
            final_amount=self.amount.amount,
            final_currency=self.amount.currency,
            settled_amount=settled_amount.amount,
            total_fees=total_fees.amount,
            completed_at=datetime.utcnow(),
        )

        self._apply_event(event)

    def fail(self, reason: str, error_code: str, should_retry: bool = False) -> None:
        """
        Fail the payment.

        This triggers compensation (refund if already captured).
        """
        event = PaymentFailed(
            metadata=create_event_metadata(
                event_type="PaymentFailed",
                aggregate_id=str(self.payment_id),
                aggregate_type="payment",
                sequence_number=self.version + 1,
            ),
            payment_id=str(self.payment_id),
            failure_stage=self._current_stage(),
            failure_reason=reason,
            failure_code=error_code,
            should_retry=should_retry,
        )

        self._apply_event(event)

    def refund(
        self,
        refund_amount: Money,
        refund_id: str,
        reason: str,
        refunded_by: str,
    ) -> None:
        """
        Refund payment (full or partial).

        Invariant: Cannot refund more than captured.
        Invariant: Can only refund completed payments.
        """
        if self.status != PaymentStatus.COMPLETED:
            raise PaymentError(
                f"Cannot refund payment in status {self.status}",
                payment_id=str(self.payment_id),
            )

        if not self.captured_amount:
            raise PaymentError(
                "Cannot refund payment without captured amount",
                payment_id=str(self.payment_id),
            )

        # Business rule: Can't refund more than captured
        if refund_amount > self.captured_amount:
            raise PaymentError(
                f"Cannot refund {refund_amount}, only captured {self.captured_amount}",
                payment_id=str(self.payment_id),
            )

        is_partial = refund_amount < self.captured_amount

        event = PaymentRefunded(
            metadata=create_event_metadata(
                event_type="PaymentRefunded",
                aggregate_id=str(self.payment_id),
                aggregate_type="payment",
                sequence_number=self.version + 1,
            ),
            payment_id=str(self.payment_id),
            refund_id=refund_id,
            refund_amount=refund_amount.amount,
            refund_currency=refund_amount.currency,
            refund_reason=reason,
            refunded_by=refunded_by,
            is_partial=is_partial,
        )

        self._apply_event(event)

    def _current_stage(self) -> str:
        """Determine current stage for failure reporting."""
        stage_map = {
            PaymentStatus.INITIATED: "validation",
            PaymentStatus.VALIDATED: "fraud_check",
            PaymentStatus.FRAUD_CHECKED: "authorization",
            PaymentStatus.AUTHORIZED: "capture",
            PaymentStatus.CAPTURED: "settlement",
            PaymentStatus.FX_CONVERTING: "fx_conversion",
        }
        return stage_map.get(self.status, "unknown")

    def _apply_event(self, event: DomainEvent) -> None:
        """
        Apply event to aggregate state.

        This is the CORE of event sourcing:
        - Event is added to uncommitted events (for persistence)
        - State is updated based on event (state derived from events)
        - Version incremented (for optimistic locking)
        """
        self._uncommitted_events.append(event)
        self._event_history.append(event)
        self.version += 1

        # Update state based on event type
        self._mutate(event)

    def _mutate(self, event: DomainEvent) -> None:
        """
        Update aggregate state based on event.

        CRITICAL: This must be DETERMINISTIC.
        Same events → same state (always).

        This is how we rebuild state from events (time-travel debugging).
        """
        if isinstance(event, PaymentInitiated):
            self.status = PaymentStatus.INITIATED

        elif isinstance(event, PaymentMethodValidated):
            self.status = PaymentStatus.VALIDATED

        elif isinstance(event, FraudCheckCompleted):
            self.status = PaymentStatus.FRAUD_CHECKED

        elif isinstance(event, PaymentAuthorized):
            self.status = PaymentStatus.AUTHORIZED
            self.authorization_id = event.authorization_id

        elif isinstance(event, PaymentCaptured):
            self.status = PaymentStatus.CAPTURED
            self.captured_amount = Money(
                amount=event.captured_amount, currency=event.captured_currency
            )

        elif isinstance(event, FXConversionRequired):
            self.status = PaymentStatus.FX_CONVERTING

        elif isinstance(event, FXConverted):
            self.fx_conversion = {
                "to_amount": event.to_amount,
                "to_currency": event.to_currency,
                "rate": event.rate_used,
                "markup_earned": event.markup_earned,
            }

        elif isinstance(event, PaymentCompleted):
            self.status = PaymentStatus.COMPLETED

        elif isinstance(event, PaymentFailed):
            self.status = PaymentStatus.FAILED
            self.failure_reason = event.failure_reason

        elif isinstance(event, PaymentRefunded):
            self.status = PaymentStatus.REFUNDED if not event.is_partial else self.status

    @classmethod
    def from_events(cls, events: list[DomainEvent]) -> Payment:
        """
        Rebuild payment from event history (time-travel debugging).

        This is THE KEY feature of event sourcing.

        Use case:
        1. Customer disputes charge from 6 months ago
        2. Load all events for that payment_id
        3. Replay events to see exact state at any point
        4. "At 3:42pm, payment was authorized for $100.50..."
        """
        if not events:
            raise ValueError("Cannot rebuild payment from empty event list")

        first_event = events[0]
        if not isinstance(first_event, PaymentInitiated):
            raise ValueError("First event must be PaymentInitiated")

        # Create payment from first event
        payment = cls(
            payment_id=PaymentIdentifier(value=first_event.payment_id),
            idempotency_key=IdempotencyKey(value=first_event.idempotency_key),
            merchant_id=first_event.merchant_id,
            customer_id=first_event.customer_id,
            amount=Money(
                amount=first_event.amount, currency=first_event.currency
            ),
            payment_method=first_event.payment_method,
        )

        # Replay all events
        for event in events:
            payment._mutate(event)
            payment._event_history.append(event)
            payment.version += 1

        return payment

    def get_uncommitted_events(self) -> list[DomainEvent]:
        """Get events that haven't been persisted yet."""
        return self._uncommitted_events.copy()

    def mark_events_committed(self) -> None:
        """Clear uncommitted events after persistence."""
        self._uncommitted_events.clear()


class PaymentError(Exception):
    """Domain error for payment operations."""

    def __init__(self, message: str, payment_id: str | None = None):
        self.payment_id = payment_id
        super().__init__(message)

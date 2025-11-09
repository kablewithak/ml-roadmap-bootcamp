"""
Value Objects - Immutable Domain Concepts

Value objects have no identity - two value objects are equal if their values are equal.

Example:
- Money(100, "USD") == Money(100, "USD") ✓
- Payment("abc") != Payment("abc") if different transactions ✗

Why this matters:
- Immutability prevents bugs (can't accidentally change amount mid-transaction)
- Type safety (can't add Money + String)
- Business logic encapsulated (Money knows how to add Money)
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Self

from pydantic import BaseModel, Field, field_validator

from payment_system.domain.events import Currency


class Money(BaseModel):
    """
    Money value object with currency.

    CRITICAL: Never use raw Decimal for money - always include currency!

    Bug prevented:
    amount = Decimal("100")  # Is this USD? EUR? JPY? 💥
    money = Money(amount=Decimal("100"), currency=Currency.USD)  # Clear! ✓
    """

    amount: Decimal
    currency: Currency

    class Config:
        frozen = True  # Immutable

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, v: Decimal) -> Decimal:
        """Ensure proper decimal precision for currency."""
        return v.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def __add__(self, other: Money) -> Money:
        """
        Add money (only same currency).

        CRITICAL: Can't add USD + EUR without conversion.
        """
        if self.currency != other.currency:
            raise ValueError(
                f"Cannot add {self.currency} to {other.currency} - convert first"
            )
        return Money(amount=self.amount + other.amount, currency=self.currency)

    def __sub__(self, other: Money) -> Money:
        """Subtract money (only same currency)."""
        if self.currency != other.currency:
            raise ValueError(
                f"Cannot subtract {other.currency} from {self.currency} - convert first"
            )
        return Money(amount=self.amount - other.amount, currency=self.currency)

    def __mul__(self, multiplier: Decimal | int | float) -> Money:
        """Multiply money by scalar (e.g., apply fees)."""
        return Money(
            amount=self.amount * Decimal(str(multiplier)), currency=self.currency
        )

    def __truediv__(self, divisor: Decimal | int | float) -> Money:
        """Divide money by scalar."""
        return Money(amount=self.amount / Decimal(str(divisor)), currency=self.currency)

    def apply_percentage(self, percentage: Decimal) -> Money:
        """
        Apply percentage (e.g., fee, markup).

        Example: Money(100, USD).apply_percentage(Decimal("2.9")) = Money(2.90, USD)
        """
        return Money(
            amount=self.amount * (percentage / Decimal("100")), currency=self.currency
        )

    def apply_basis_points(self, bps: int) -> Money:
        """
        Apply basis points (1 bps = 0.01%).

        Why use bps? Financial industry standard for small percentages.
        Example: 150 bps = 1.5%

        FX markup example:
        Money(100, USD).apply_basis_points(150) = Money(1.50, USD) markup
        """
        return Money(
            amount=self.amount * (Decimal(bps) / Decimal("10000")), currency=self.currency
        )

    def __eq__(self, other: object) -> bool:
        """Value equality."""
        if not isinstance(other, Money):
            return False
        return self.amount == other.amount and self.currency == other.currency

    def __lt__(self, other: Money) -> bool:
        """Compare money (same currency only)."""
        if self.currency != other.currency:
            raise ValueError(f"Cannot compare {self.currency} to {other.currency}")
        return self.amount < other.amount

    def __repr__(self) -> str:
        return f"Money({self.amount}, {self.currency.value})"


class FXRate(BaseModel):
    """
    Foreign exchange rate.

    Design: Separate from Money because FX rates have additional properties:
    - Timestamp (rates change every second)
    - Source (which API)
    - Markup (our revenue)
    """

    from_currency: Currency
    to_currency: Currency
    rate: Decimal  # How many units of to_currency per 1 unit of from_currency
    markup_bps: int = Field(default=0, ge=0, le=10000)  # 0-100% in basis points
    rate_source: str = "internal"
    quoted_at: str  # ISO timestamp

    class Config:
        frozen = True

    @property
    def effective_rate(self) -> Decimal:
        """
        Rate after applying markup.

        Example:
        Market rate: 1 USD = 0.85 EUR
        Markup: 150 bps (1.5%)
        Effective rate: 0.85 * (1 - 0.015) = 0.83725

        Customer pays: $100
        Customer gets: €83.73 (not €85)
        We keep: €1.27 as profit
        """
        markup_multiplier = Decimal("1") - (Decimal(self.markup_bps) / Decimal("10000"))
        return (self.rate * markup_multiplier).quantize(
            Decimal("0.000001"), rounding=ROUND_HALF_UP
        )

    def convert(self, money: Money) -> Money:
        """
        Convert money using this rate.

        CRITICAL: Always use effective_rate (includes markup), not raw rate.
        """
        if money.currency != self.from_currency:
            raise ValueError(
                f"Rate is for {self.from_currency}, got {money.currency}"
            )

        converted_amount = money.amount * self.effective_rate
        return Money(amount=converted_amount, currency=self.to_currency)

    def markup_amount(self, money: Money) -> Money:
        """
        Calculate how much markup we earn on this conversion.

        This is pure profit.
        """
        market_converted = money.amount * self.rate
        actual_converted = money.amount * self.effective_rate
        markup = market_converted - actual_converted

        return Money(amount=markup, currency=self.to_currency)

    def inverse(self) -> FXRate:
        """
        Get inverse rate (for reverse conversions).

        Example: USD->EUR rate is 0.85, EUR->USD rate is 1.176
        """
        inverse_rate = Decimal("1") / self.rate
        return FXRate(
            from_currency=self.to_currency,
            to_currency=self.from_currency,
            rate=inverse_rate,
            markup_bps=self.markup_bps,
            rate_source=self.rate_source,
            quoted_at=self.quoted_at,
        )


class PaymentIdentifier(BaseModel):
    """
    Unique identifier for a payment.

    Why not just use UUID string?
    - Type safety: Can't accidentally pass merchant_id where payment_id expected
    - Validation: Can encode business rules (e.g., format, checksum)
    - Parsing: Can extract metadata (e.g., timestamp, shard ID)
    """

    value: str

    class Config:
        frozen = True

    @field_validator("value")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """
        Validate payment ID format.

        Format: pay_{timestamp}_{random}
        Example: pay_20231015_a3f7k9m2

        Why this format?
        - Sortable by creation time
        - Shardable by timestamp
        - Human-readable prefix
        """
        if not v.startswith("pay_"):
            raise ValueError(f"Payment ID must start with 'pay_', got {v}")
        if len(v) < 20:
            raise ValueError(f"Payment ID too short: {v}")
        return v

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"PaymentIdentifier({self.value})"


class IdempotencyKey(BaseModel):
    """
    Idempotency key for preventing duplicate operations.

    CRITICAL: This prevents double-charging when networks lie.

    How it works:
    1. Client generates unique key (e.g., UUID)
    2. Client sends: charge($100, key="abc123")
    3. Network times out
    4. Client retries: charge($100, key="abc123")
    5. Server sees same key → returns cached result, doesn't charge again

    Without this: Customer charged $200
    With this: Customer charged $100 (second request ignored)

    Cost saved: Infinite (prevents catastrophic bugs)
    """

    value: str

    class Config:
        frozen = True

    @field_validator("value")
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Idempotency key cannot be empty")
        if len(v) > 255:
            raise ValueError(f"Idempotency key too long: {len(v)} characters")
        return v.strip()

    def __str__(self) -> str:
        return self.value

    def __hash__(self) -> int:
        return hash(self.value)


class MerchantIdentifier(BaseModel):
    """Unique identifier for a merchant."""

    value: str

    class Config:
        frozen = True

    @field_validator("value")
    @classmethod
    def validate_format(cls, v: str) -> str:
        if not v.startswith("merch_"):
            raise ValueError(f"Merchant ID must start with 'merch_', got {v}")
        return v

    def __str__(self) -> str:
        return self.value


class ChargebackIdentifier(BaseModel):
    """Unique identifier for a chargeback."""

    value: str

    class Config:
        frozen = True

    @field_validator("value")
    @classmethod
    def validate_format(cls, v: str) -> str:
        if not v.startswith("cb_"):
            raise ValueError(f"Chargeback ID must start with 'cb_', got {v}")
        return v

    def __str__(self) -> str:
        return self.value


class ReconciliationIdentifier(BaseModel):
    """Unique identifier for a reconciliation run."""

    value: str

    class Config:
        frozen = True

    @field_validator("value")
    @classmethod
    def validate_format(cls, v: str) -> str:
        if not v.startswith("recon_"):
            raise ValueError(f"Reconciliation ID must start with 'recon_', got {v}")
        return v

    def __str__(self) -> str:
        return self.value

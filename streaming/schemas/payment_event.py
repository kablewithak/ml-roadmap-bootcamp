"""
PaymentEvent Avro schema and Python dataclass.

Represents a payment transaction event in the system. This is the core event
for payment processing with fraud detection.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime
from decimal import Decimal
import uuid

# Avro schema definition for PaymentEvent
PAYMENT_EVENT_SCHEMA = {
    "type": "record",
    "name": "PaymentEvent",
    "namespace": "com.streaming.events.payment",
    "doc": "Payment transaction event with fraud detection metadata",
    "fields": [
        {
            "name": "payment_id",
            "type": "string",
            "doc": "Unique payment identifier (UUID)",
            "logicalType": "uuid"
        },
        {
            "name": "user_id",
            "type": "string",
            "doc": "User/customer identifier"
        },
        {
            "name": "merchant_id",
            "type": "string",
            "doc": "Merchant identifier"
        },
        {
            "name": "amount",
            "type": {
                "type": "bytes",
                "logicalType": "decimal",
                "precision": 15,
                "scale": 2
            },
            "doc": "Payment amount"
        },
        {
            "name": "currency",
            "type": "string",
            "doc": "ISO 4217 currency code (e.g., USD, EUR)",
            "default": "USD"
        },
        {
            "name": "payment_method",
            "type": {
                "type": "enum",
                "name": "PaymentMethod",
                "symbols": ["CREDIT_CARD", "DEBIT_CARD", "BANK_TRANSFER", "WALLET", "CRYPTO"]
            },
            "doc": "Payment method used"
        },
        {
            "name": "card_last_four",
            "type": ["null", "string"],
            "doc": "Last 4 digits of card (if applicable)",
            "default": None
        },
        {
            "name": "status",
            "type": {
                "type": "enum",
                "name": "PaymentStatus",
                "symbols": ["PENDING", "AUTHORIZED", "CAPTURED", "DECLINED", "REFUNDED", "FAILED"]
            },
            "doc": "Current payment status"
        },
        {
            "name": "fraud_score",
            "type": ["null", "double"],
            "doc": "Fraud risk score (0.0-1.0), null if not yet evaluated",
            "default": None
        },
        {
            "name": "ip_address",
            "type": "string",
            "doc": "Client IP address"
        },
        {
            "name": "device_fingerprint",
            "type": ["null", "string"],
            "doc": "Device fingerprint for fraud detection",
            "default": None
        },
        {
            "name": "session_id",
            "type": "string",
            "doc": "User session identifier"
        },
        {
            "name": "merchant_category",
            "type": ["null", "string"],
            "doc": "Merchant category code (MCC)",
            "default": None
        },
        {
            "name": "country_code",
            "type": "string",
            "doc": "ISO 3166-1 alpha-2 country code"
        },
        {
            "name": "metadata",
            "type": ["null", {
                "type": "map",
                "values": "string"
            }],
            "doc": "Additional metadata as key-value pairs",
            "default": None
        },
        {
            "name": "idempotency_key",
            "type": "string",
            "doc": "Idempotency key for exactly-once processing"
        },
        {
            "name": "timestamp",
            "type": "long",
            "logicalType": "timestamp-millis",
            "doc": "Event timestamp in milliseconds since epoch"
        },
        {
            "name": "schema_version",
            "type": "string",
            "doc": "Schema version for evolution tracking",
            "default": "1.0.0"
        }
    ]
}


@dataclass
class PaymentEvent:
    """
    Python representation of PaymentEvent.

    This class provides a type-safe interface for working with payment events,
    with automatic validation and serialization/deserialization.

    Attributes:
        payment_id: Unique payment identifier
        user_id: Customer identifier
        merchant_id: Merchant identifier
        amount: Payment amount (Decimal for precision)
        currency: ISO 4217 currency code
        payment_method: Payment method used
        status: Current payment status
        ip_address: Client IP address
        session_id: User session identifier
        country_code: ISO country code
        idempotency_key: Key for exactly-once processing
        timestamp: Event timestamp
        fraud_score: Optional fraud risk score (0.0-1.0)
        card_last_four: Optional last 4 digits of card
        device_fingerprint: Optional device fingerprint
        merchant_category: Optional merchant category code
        metadata: Optional additional metadata
        schema_version: Schema version for evolution
    """

    payment_id: str
    user_id: str
    merchant_id: str
    amount: Decimal
    currency: str
    payment_method: str
    status: str
    ip_address: str
    session_id: str
    country_code: str
    idempotency_key: str
    timestamp: int
    fraud_score: Optional[float] = None
    card_last_four: Optional[str] = None
    device_fingerprint: Optional[str] = None
    merchant_category: Optional[str] = None
    metadata: Optional[Dict[str, str]] = field(default_factory=dict)
    schema_version: str = "1.0.0"

    @classmethod
    def create(
        cls,
        user_id: str,
        merchant_id: str,
        amount: Decimal,
        currency: str,
        payment_method: str,
        ip_address: str,
        session_id: str,
        country_code: str,
        **kwargs
    ) -> 'PaymentEvent':
        """
        Factory method to create a PaymentEvent with auto-generated fields.

        Args:
            user_id: Customer identifier
            merchant_id: Merchant identifier
            amount: Payment amount
            currency: ISO 4217 currency code
            payment_method: Payment method
            ip_address: Client IP address
            session_id: User session identifier
            country_code: ISO country code
            **kwargs: Additional optional fields

        Returns:
            New PaymentEvent instance with generated payment_id, idempotency_key, and timestamp
        """
        payment_id = str(uuid.uuid4())
        idempotency_key = kwargs.pop('idempotency_key', f"{user_id}:{payment_id}")
        timestamp = kwargs.pop('timestamp', int(datetime.utcnow().timestamp() * 1000))

        return cls(
            payment_id=payment_id,
            user_id=user_id,
            merchant_id=merchant_id,
            amount=amount,
            currency=currency,
            payment_method=payment_method,
            status="PENDING",
            ip_address=ip_address,
            session_id=session_id,
            country_code=country_code,
            idempotency_key=idempotency_key,
            timestamp=timestamp,
            **kwargs
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for Avro serialization.

        Returns:
            Dictionary representation compatible with Avro schema
        """
        return {
            'payment_id': self.payment_id,
            'user_id': self.user_id,
            'merchant_id': self.merchant_id,
            'amount': str(self.amount),  # Convert Decimal to string for Avro
            'currency': self.currency,
            'payment_method': self.payment_method,
            'card_last_four': self.card_last_four,
            'status': self.status,
            'fraud_score': self.fraud_score,
            'ip_address': self.ip_address,
            'device_fingerprint': self.device_fingerprint,
            'session_id': self.session_id,
            'merchant_category': self.merchant_category,
            'country_code': self.country_code,
            'metadata': self.metadata,
            'idempotency_key': self.idempotency_key,
            'timestamp': self.timestamp,
            'schema_version': self.schema_version
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PaymentEvent':
        """
        Create PaymentEvent from dictionary (Avro deserialization).

        Args:
            data: Dictionary from Avro deserialization

        Returns:
            PaymentEvent instance
        """
        # Convert string amount back to Decimal
        if isinstance(data['amount'], str):
            data['amount'] = Decimal(data['amount'])

        return cls(**data)

    def is_high_value(self, threshold: Decimal = Decimal('10000.00')) -> bool:
        """Check if this is a high-value transaction."""
        return self.amount > threshold

    def is_high_risk(self, threshold: float = 0.7) -> bool:
        """Check if this transaction is high risk based on fraud score."""
        return self.fraud_score is not None and self.fraud_score > threshold

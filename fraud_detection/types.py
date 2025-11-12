"""Core types and data structures for the fraud detection system."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4


class TransactionType(Enum):
    """Types of financial transactions."""
    PURCHASE = "purchase"
    REFUND = "refund"
    TRANSFER = "transfer"
    WITHDRAWAL = "withdrawal"
    DEPOSIT = "deposit"


class AttackType(Enum):
    """Types of fraud attacks."""
    CARD_TESTING = "card_testing"
    ACCOUNT_TAKEOVER = "account_takeover"
    VELOCITY_EVASION = "velocity_evasion"
    SYNTHETIC_IDENTITY = "synthetic_identity"
    DEVICE_ROTATION = "device_rotation"
    IP_ROTATION = "ip_rotation"
    BLEND_ATTACK = "blend_attack"
    SLOW_BURN = "slow_burn"
    BIN_ATTACK = "bin_attack"
    GIFT_CARD_CASHOUT = "gift_card_cashout"


class FraudLabel(Enum):
    """Fraud classification labels."""
    LEGITIMATE = 0
    SUSPICIOUS = 1
    FRAUDULENT = 2


@dataclass
class Transaction:
    """Represents a financial transaction."""
    transaction_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user_id: str = ""
    merchant_id: str = ""
    amount: float = 0.0
    currency: str = "USD"
    transaction_type: TransactionType = TransactionType.PURCHASE

    # Device and network info
    device_id: str = ""
    ip_address: str = ""
    user_agent: str = ""

    # Card info (masked)
    card_bin: str = ""  # First 6 digits
    card_last4: str = ""  # Last 4 digits

    # Location
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    country: str = ""

    # Behavioral features
    is_first_transaction: bool = False
    time_since_account_creation_hours: float = 0.0
    transactions_last_24h: int = 0

    # Labels and predictions
    is_fraud: bool = False
    fraud_label: FraudLabel = FraudLabel.LEGITIMATE
    fraud_score: float = 0.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert transaction to dictionary."""
        return {
            "transaction_id": self.transaction_id,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "merchant_id": self.merchant_id,
            "amount": self.amount,
            "currency": self.currency,
            "transaction_type": self.transaction_type.value,
            "device_id": self.device_id,
            "ip_address": self.ip_address,
            "card_bin": self.card_bin,
            "card_last4": self.card_last4,
            "country": self.country,
            "is_fraud": self.is_fraud,
            "fraud_score": self.fraud_score,
        }


@dataclass
class AttackPattern:
    """Configuration for an attack pattern."""
    attack_type: AttackType
    name: str
    description: str
    num_transactions: int
    duration_hours: float
    success_threshold: float
    adaptive: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackResult:
    """Results from executing an attack."""
    attack_type: AttackType
    start_time: datetime
    end_time: datetime
    total_transactions: int
    successful_transactions: int
    blocked_transactions: int
    success_rate: float
    estimated_loss: float
    detection_rate: float
    transactions: List[Transaction] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DefenseMetrics:
    """Metrics for defense system performance."""
    timestamp: datetime
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    avg_detection_latency_ms: float
    throughput_tps: float

    @property
    def false_positive_rate(self) -> float:
        """Calculate false positive rate."""
        if self.false_positives + self.true_negatives == 0:
            return 0.0
        return self.false_positives / (self.false_positives + self.true_negatives)

    @property
    def false_negative_rate(self) -> float:
        """Calculate false negative rate."""
        if self.false_negatives + self.true_positives == 0:
            return 0.0
        return self.false_negatives / (self.false_negatives + self.true_positives)


@dataclass
class BusinessMetrics:
    """Business impact metrics."""
    timestamp: datetime
    prevented_loss: float
    false_positive_cost: float
    operational_cost: float
    net_savings: float
    roi: float
    customer_lifetime_value_impact: float
    transactions_processed: int
    cost_per_transaction: float

"""
Data models for fraud detection system.
"""

from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field, validator


class RiskDecision(str, Enum):
    """Risk decision types."""
    APPROVE = "approve"
    REVIEW = "review"
    DECLINE = "decline"


class TransactionRequest(BaseModel):
    """Incoming transaction request."""
    transaction_id: str
    user_id: str
    card_id: str
    ip_address: str
    amount: float = Field(gt=0, description="Transaction amount in USD")
    currency: str = Field(default="USD")
    merchant_id: str
    merchant_category: str
    merchant_name: str
    timestamp: Optional[datetime] = None

    @validator('timestamp', pre=True, always=True)
    def set_timestamp(cls, v):
        return v or datetime.utcnow()

    @validator('amount')
    def validate_amount(cls, v):
        if v < 0:
            raise ValueError("Amount must be positive")
        if v > 100000:
            raise ValueError("Amount exceeds maximum allowed")
        return round(v, 2)


class VelocitySignals(BaseModel):
    """Velocity-based fraud signals."""
    card_count_5min: int = 0
    card_count_1hr: int = 0
    card_amount_5min: float = 0.0
    card_amount_1hr: float = 0.0

    user_count_5min: int = 0
    user_count_1hr: int = 0
    user_amount_5min: float = 0.0
    user_amount_1hr: float = 0.0

    ip_count_5min: int = 0
    ip_count_1hr: int = 0
    ip_amount_5min: float = 0.0
    ip_amount_1hr: float = 0.0

    lookup_latency_ms: float = 0.0


class PatternSignals(BaseModel):
    """Pattern-based fraud signals."""
    is_first_card_use: bool = False
    merchant_category_count: int = 0
    current_hour_tx_count: int = 0
    hour_patterns: Dict[int, int] = Field(default_factory=dict)
    is_card_testing: bool = False
    card_testing_details: Dict = Field(default_factory=dict)
    is_unusual_hour: bool = False
    is_merchant_category_switch: bool = False


class SignalWeights(BaseModel):
    """Configurable weights for risk scoring."""
    velocity_count: float = 0.25
    velocity_amount: float = 0.20
    new_card_risk: float = 0.15
    merchant_pattern: float = 0.15
    time_pattern: float = 0.10
    card_testing_pattern: float = 0.15

    @validator('*')
    def validate_weight(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Weight must be between 0 and 1")
        return v


class RiskThresholds(BaseModel):
    """Risk score thresholds for decisions."""
    approve_below: float = 0.30
    review_below: float = 0.70
    decline_above: float = 0.70

    @validator('*')
    def validate_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        return v


class RiskScore(BaseModel):
    """Calculated risk score with details."""
    risk_score: float = Field(ge=0, le=1, description="Overall risk score 0-1")
    decision: RiskDecision
    signals_triggered: List[str] = Field(default_factory=list)
    signal_scores: Dict[str, float] = Field(default_factory=dict)
    velocity_signals: VelocitySignals
    pattern_signals: PatternSignals
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class FraudDecision(BaseModel):
    """Complete fraud detection decision with all context."""
    transaction_id: str
    decision: RiskDecision
    risk_score: float
    signals_triggered: List[str]
    should_process_payment: bool
    requires_manual_review: bool
    decline_reason: Optional[str] = None
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # For logging and ML training
    raw_signals: Dict = Field(default_factory=dict)
    thresholds_used: Dict = Field(default_factory=dict)


class KafkaFraudEvent(BaseModel):
    """Event published to Kafka for ML training."""
    event_type: str = "fraud_signal_collection"
    transaction_id: str
    user_id: str
    card_id: str
    ip_address: str
    amount: float
    merchant_category: str

    # Risk assessment
    risk_score: float
    decision: str
    signals_triggered: List[str]

    # All signals for ML training
    velocity_signals: Dict
    pattern_signals: Dict
    signal_scores: Dict

    # Outcome (to be updated later)
    actual_fraud: Optional[bool] = None
    chargeback_occurred: Optional[bool] = None

    timestamp: datetime
    processing_time_ms: float

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

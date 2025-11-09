"""
FraudDecisionEvent Avro schema and Python dataclass.

Represents a fraud detection decision for a payment transaction.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime
import uuid

# Avro schema definition for FraudDecisionEvent
FRAUD_DECISION_EVENT_SCHEMA = {
    "type": "record",
    "name": "FraudDecisionEvent",
    "namespace": "com.streaming.events.fraud",
    "doc": "Fraud detection decision event for payment transactions",
    "fields": [
        {
            "name": "decision_id",
            "type": "string",
            "doc": "Unique decision identifier (UUID)",
            "logicalType": "uuid"
        },
        {
            "name": "payment_id",
            "type": "string",
            "doc": "Associated payment identifier",
            "logicalType": "uuid"
        },
        {
            "name": "user_id",
            "type": "string",
            "doc": "User identifier for the payment"
        },
        {
            "name": "decision",
            "type": {
                "type": "enum",
                "name": "FraudDecision",
                "symbols": ["APPROVE", "DECLINE", "REVIEW", "CHALLENGE"]
            },
            "doc": "Fraud detection decision"
        },
        {
            "name": "fraud_score",
            "type": "double",
            "doc": "Overall fraud risk score (0.0-1.0)"
        },
        {
            "name": "model_version",
            "type": "string",
            "doc": "ML model version used for scoring"
        },
        {
            "name": "model_name",
            "type": "string",
            "doc": "ML model name (e.g., 'gradient_boosting_v2')"
        },
        {
            "name": "fraud_reasons",
            "type": {
                "type": "array",
                "items": {
                    "type": "record",
                    "name": "FraudReason",
                    "fields": [
                        {
                            "name": "reason_code",
                            "type": "string",
                            "doc": "Fraud reason code (e.g., 'VELOCITY_CHECK', 'BLACKLIST')"
                        },
                        {
                            "name": "reason_description",
                            "type": "string",
                            "doc": "Human-readable description"
                        },
                        {
                            "name": "confidence",
                            "type": "double",
                            "doc": "Confidence score for this reason (0.0-1.0)"
                        }
                    ]
                }
            },
            "doc": "List of fraud indicators detected",
            "default": []
        },
        {
            "name": "risk_factors",
            "type": {
                "type": "record",
                "name": "RiskFactors",
                "fields": [
                    {
                        "name": "velocity_score",
                        "type": ["null", "double"],
                        "doc": "Transaction velocity risk score",
                        "default": None
                    },
                    {
                        "name": "device_score",
                        "type": ["null", "double"],
                        "doc": "Device fingerprint risk score",
                        "default": None
                    },
                    {
                        "name": "location_score",
                        "type": ["null", "double"],
                        "doc": "Geographic location risk score",
                        "default": None
                    },
                    {
                        "name": "behavioral_score",
                        "type": ["null", "double"],
                        "doc": "User behavioral pattern risk score",
                        "default": None
                    },
                    {
                        "name": "merchant_score",
                        "type": ["null", "double"],
                        "doc": "Merchant risk score",
                        "default": None
                    }
                ]
            },
            "doc": "Breakdown of risk factors"
        },
        {
            "name": "features_used",
            "type": {
                "type": "map",
                "values": "string"
            },
            "doc": "Feature values used in the decision (for explainability)",
            "default": {}
        },
        {
            "name": "processing_time_ms",
            "type": "long",
            "doc": "Time taken to make the fraud decision (milliseconds)"
        },
        {
            "name": "challenge_type",
            "type": ["null", {
                "type": "enum",
                "name": "ChallengeType",
                "symbols": ["SMS_OTP", "EMAIL_OTP", "BIOMETRIC", "SECURITY_QUESTIONS"]
            }],
            "doc": "Type of challenge if decision is CHALLENGE",
            "default": None
        },
        {
            "name": "manual_review_required",
            "type": "boolean",
            "doc": "Whether manual review is required",
            "default": False
        },
        {
            "name": "idempotency_key",
            "type": "string",
            "doc": "Idempotency key matching the payment event"
        },
        {
            "name": "timestamp",
            "type": "long",
            "logicalType": "timestamp-millis",
            "doc": "Decision timestamp in milliseconds since epoch"
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
class FraudReason:
    """
    Represents a single fraud indicator.

    Attributes:
        reason_code: Code identifying the fraud reason
        reason_description: Human-readable description
        confidence: Confidence score (0.0-1.0)
    """
    reason_code: str
    reason_description: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'reason_code': self.reason_code,
            'reason_description': self.reason_description,
            'confidence': self.confidence
        }


@dataclass
class RiskFactors:
    """
    Breakdown of individual risk factor scores.

    Attributes:
        velocity_score: Transaction velocity risk
        device_score: Device fingerprint risk
        location_score: Geographic location risk
        behavioral_score: User behavior pattern risk
        merchant_score: Merchant risk
    """
    velocity_score: Optional[float] = None
    device_score: Optional[float] = None
    location_score: Optional[float] = None
    behavioral_score: Optional[float] = None
    merchant_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'velocity_score': self.velocity_score,
            'device_score': self.device_score,
            'location_score': self.location_score,
            'behavioral_score': self.behavioral_score,
            'merchant_score': self.merchant_score
        }


@dataclass
class FraudDecisionEvent:
    """
    Python representation of FraudDecisionEvent.

    This event is produced by the fraud detection system after analyzing
    a payment transaction. It contains the decision and detailed reasoning.

    Attributes:
        decision_id: Unique decision identifier
        payment_id: Associated payment ID
        user_id: User identifier
        decision: Fraud decision (APPROVE, DECLINE, REVIEW, CHALLENGE)
        fraud_score: Overall fraud score (0.0-1.0)
        model_version: ML model version
        model_name: ML model name
        fraud_reasons: List of detected fraud indicators
        risk_factors: Breakdown of risk scores
        features_used: Features used in the decision
        processing_time_ms: Processing time in milliseconds
        challenge_type: Type of challenge if applicable
        manual_review_required: Whether manual review is needed
        idempotency_key: Key for exactly-once processing
        timestamp: Decision timestamp
        schema_version: Schema version
    """

    decision_id: str
    payment_id: str
    user_id: str
    decision: str
    fraud_score: float
    model_version: str
    model_name: str
    risk_factors: RiskFactors
    processing_time_ms: int
    idempotency_key: str
    timestamp: int
    fraud_reasons: List[FraudReason] = field(default_factory=list)
    features_used: Dict[str, str] = field(default_factory=dict)
    challenge_type: Optional[str] = None
    manual_review_required: bool = False
    schema_version: str = "1.0.0"

    @classmethod
    def create(
        cls,
        payment_id: str,
        user_id: str,
        decision: str,
        fraud_score: float,
        model_version: str,
        model_name: str,
        risk_factors: RiskFactors,
        processing_time_ms: int,
        idempotency_key: str,
        **kwargs
    ) -> 'FraudDecisionEvent':
        """
        Factory method to create a FraudDecisionEvent.

        Args:
            payment_id: Associated payment ID
            user_id: User identifier
            decision: Fraud decision
            fraud_score: Overall fraud score
            model_version: ML model version
            model_name: ML model name
            risk_factors: Risk factor breakdown
            processing_time_ms: Processing time
            idempotency_key: Idempotency key
            **kwargs: Additional optional fields

        Returns:
            New FraudDecisionEvent instance
        """
        decision_id = str(uuid.uuid4())
        timestamp = kwargs.pop('timestamp', int(datetime.utcnow().timestamp() * 1000))

        return cls(
            decision_id=decision_id,
            payment_id=payment_id,
            user_id=user_id,
            decision=decision,
            fraud_score=fraud_score,
            model_version=model_version,
            model_name=model_name,
            risk_factors=risk_factors,
            processing_time_ms=processing_time_ms,
            idempotency_key=idempotency_key,
            timestamp=timestamp,
            **kwargs
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Avro serialization."""
        return {
            'decision_id': self.decision_id,
            'payment_id': self.payment_id,
            'user_id': self.user_id,
            'decision': self.decision,
            'fraud_score': self.fraud_score,
            'model_version': self.model_version,
            'model_name': self.model_name,
            'fraud_reasons': [r.to_dict() for r in self.fraud_reasons],
            'risk_factors': self.risk_factors.to_dict(),
            'features_used': self.features_used,
            'processing_time_ms': self.processing_time_ms,
            'challenge_type': self.challenge_type,
            'manual_review_required': self.manual_review_required,
            'idempotency_key': self.idempotency_key,
            'timestamp': self.timestamp,
            'schema_version': self.schema_version
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FraudDecisionEvent':
        """Create FraudDecisionEvent from dictionary."""
        # Convert nested structures
        if 'fraud_reasons' in data:
            data['fraud_reasons'] = [
                FraudReason(**r) for r in data['fraud_reasons']
            ]
        if 'risk_factors' in data:
            data['risk_factors'] = RiskFactors(**data['risk_factors'])

        return cls(**data)

    def should_block_payment(self) -> bool:
        """Determine if payment should be blocked based on decision."""
        return self.decision in ['DECLINE', 'REVIEW']

    def requires_additional_verification(self) -> bool:
        """Check if additional verification is required."""
        return self.decision == 'CHALLENGE' or self.manual_review_required

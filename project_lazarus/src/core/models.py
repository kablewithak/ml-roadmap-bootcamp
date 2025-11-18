"""Pydantic models for Project Lazarus."""

from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Any


class Decision(str, Enum):
    """Loan decision outcomes."""
    APPROVE = "APPROVE"
    REJECT = "REJECT"


class DecisionReason(str, Enum):
    """Reasons for loan decisions."""
    COMPLIANCE_BLOCK = "COMPLIANCE_BLOCK"
    PROJECT_LAZARUS_EXPLORE = "PROJECT_LAZARUS_EXPLORE"
    MODEL_QUALIFIED = "MODEL_QUALIFIED"
    MODEL_HIGH_RISK = "MODEL_HIGH_RISK"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


class TreatmentGroup(str, Enum):
    """Treatment groups for causal inference."""
    EXPLORE = "explore"
    EXPLOIT = "exploit"


class UserFeatures(BaseModel):
    """Features describing a loan applicant."""

    user_id: str = Field(..., description="Unique user identifier")
    age: int = Field(..., ge=0, le=120, description="User age in years")
    income: float = Field(..., ge=0, description="Annual income in dollars")
    debt: float = Field(..., ge=0, description="Total debt in dollars")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score")
    employment_years: float = Field(..., ge=0, description="Years at current job")
    recent_bankruptcy: bool = Field(default=False, description="Bankruptcy in last 7 years")
    num_credit_lines: int = Field(..., ge=0, description="Number of open credit lines")
    avg_txn_amt_30d: float = Field(default=0.0, ge=0, description="Average transaction amount last 30 days")

    # Features for thin-file detection
    credit_history_months: int = Field(default=0, ge=0, description="Length of credit history in months")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "usr_12345",
                "age": 35,
                "income": 75000.0,
                "debt": 15000.0,
                "credit_score": 720,
                "employment_years": 5.0,
                "recent_bankruptcy": False,
                "num_credit_lines": 3,
                "avg_txn_amt_30d": 250.0,
                "credit_history_months": 120
            }
        }


class LoanApplication(BaseModel):
    """A loan application with user features and request details."""

    application_id: str = Field(..., description="Unique application identifier")
    user_features: UserFeatures
    requested_amount: float = Field(..., gt=0, description="Requested loan amount")
    loan_purpose: str = Field(default="general", description="Purpose of the loan")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DecisionResult(BaseModel):
    """Result of a loan decision."""

    application_id: str
    user_id: str
    decision: Decision
    reason: DecisionReason
    risk_score: float = Field(..., ge=0.0, le=1.0)
    treatment_group: TreatmentGroup
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Additional metadata
    exploration_budget_remaining: float | None = None
    model_version: str = "v1"


class ExperimentLog(BaseModel):
    """Log entry for experiment tracking."""

    user_id: str
    application_id: str
    variant: TreatmentGroup
    decision: Decision
    risk_score: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    features_snapshot: dict[str, Any] = Field(default_factory=dict)


class LoanOutcome(BaseModel):
    """Actual outcome of an approved loan."""

    application_id: str
    user_id: str
    defaulted: bool
    days_to_default: int | None = None
    amount_recovered: float = 0.0
    treatment_group: TreatmentGroup
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ModelMetrics(BaseModel):
    """Metrics for model performance tracking."""

    model_version: str
    timestamp: datetime
    auc_roc: float
    precision: float
    recall: float
    f1_score: float
    blindness_score: float = Field(
        ...,
        description="Measure of model confidence on rejected population"
    )

    # Causal metrics
    ate_estimate: float | None = Field(
        None,
        description="Average Treatment Effect estimate"
    )
    exploration_cost: float = 0.0
    projected_revenue_lift: float = 0.0

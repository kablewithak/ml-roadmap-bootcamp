"""PostgreSQL database models and connections for Project Lazarus."""

from datetime import datetime
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Float,
    Integer,
    Boolean,
    DateTime,
    Enum,
    JSON,
    Index,
)
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool

from config.settings import get_settings

Base = declarative_base()


class LoanApplication(Base):
    """Record of a loan application."""

    __tablename__ = "loan_applications"

    application_id = Column(String(50), primary_key=True)
    user_id = Column(String(50), nullable=False, index=True)
    requested_amount = Column(Float, nullable=False)
    loan_purpose = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)

    # User features snapshot
    age = Column(Integer, nullable=False)
    income = Column(Float, nullable=False)
    debt = Column(Float, nullable=False)
    credit_score = Column(Integer, nullable=False)
    employment_years = Column(Float)
    recent_bankruptcy = Column(Boolean, default=False)
    num_credit_lines = Column(Integer)
    avg_txn_amt_30d = Column(Float)
    credit_history_months = Column(Integer)

    __table_args__ = (
        Index("idx_applications_created", "created_at"),
    )


class LoanDecision(Base):
    """Record of a loan decision."""

    __tablename__ = "loan_decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    application_id = Column(String(50), nullable=False, index=True)
    user_id = Column(String(50), nullable=False, index=True)

    # Decision details
    decision = Column(String(20), nullable=False)  # APPROVE/REJECT
    reason = Column(String(50), nullable=False)
    risk_score = Column(Float, nullable=False)
    treatment_group = Column(String(20), nullable=False)  # explore/exploit

    # Metadata
    model_version = Column(String(20), default="v1")
    exploration_budget_remaining = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_decisions_treatment", "treatment_group"),
        Index("idx_decisions_created", "created_at"),
    )


class LoanOutcome(Base):
    """Actual outcome of an approved loan (ground truth)."""

    __tablename__ = "loan_outcomes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    application_id = Column(String(50), nullable=False, unique=True, index=True)
    user_id = Column(String(50), nullable=False, index=True)

    # Outcome details
    defaulted = Column(Boolean, nullable=False)
    days_to_default = Column(Integer)
    amount_recovered = Column(Float, default=0.0)

    # Treatment group for causal analysis
    treatment_group = Column(String(20), nullable=False)

    # Timestamps
    loan_disbursed_at = Column(DateTime)
    outcome_recorded_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_outcomes_defaulted", "defaulted"),
        Index("idx_outcomes_treatment", "treatment_group"),
    )


class ExperimentExposure(Base):
    """Experiment exposure log for causal inference."""

    __tablename__ = "experiment_exposures"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(50), nullable=False, index=True)
    application_id = Column(String(50), nullable=False, index=True)

    # Experiment details
    variant = Column(String(20), nullable=False)  # explore/exploit
    risk_score = Column(Float, nullable=False)
    decision = Column(String(20), nullable=False)

    # Features snapshot for reproducibility
    features_snapshot = Column(JSON)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_exposures_variant", "variant"),
        Index("idx_exposures_created", "created_at"),
    )


class ModelMetrics(Base):
    """Model performance metrics over time."""

    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_version = Column(String(20), nullable=False)

    # Performance metrics
    auc_roc = Column(Float)
    precision_score = Column(Float)
    recall_score = Column(Float)
    f1 = Column(Float)

    # Causal metrics
    blindness_score = Column(Float)
    ate_estimate = Column(Float)
    exploration_cost = Column(Float)
    projected_revenue_lift = Column(Float)

    # Training details
    training_samples = Column(Integer)
    explore_samples = Column(Integer)
    exploit_samples = Column(Integer)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_metrics_version", "model_version"),
        Index("idx_metrics_created", "created_at"),
    )


def get_engine():
    """Get SQLAlchemy engine."""
    settings = get_settings()
    return create_engine(
        settings.postgres_url,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
    )


def get_session():
    """Get database session."""
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def init_database():
    """Initialize database tables."""
    engine = get_engine()
    Base.metadata.create_all(engine)

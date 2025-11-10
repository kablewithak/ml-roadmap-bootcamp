"""
API routes for fraud detection and payment processing.
"""

from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from .main import app, get_fraud_detector, get_payment_service
from ..fraud.models import TransactionRequest, FraudDecision
from ..fraud.services.fraud_detector import FraudDetector
from ..payments.service import PaymentService

# Create routers
fraud_router = APIRouter(prefix="/fraud", tags=["Fraud Detection"])
payment_router = APIRouter(prefix="/payments", tags=["Payments"])


# Request/Response Models
class FraudCheckRequest(BaseModel):
    """Request for fraud check only (no payment)."""
    transaction_id: str
    user_id: str
    card_id: str
    ip_address: str
    amount: float = Field(gt=0)
    currency: str = "USD"
    merchant_id: str
    merchant_category: str
    merchant_name: str


class PaymentRequest(BaseModel):
    """Request for payment processing with fraud check."""
    transaction_id: str
    user_id: str
    card_id: str
    card_token: str  # Stripe token
    ip_address: str
    amount: float = Field(gt=0)
    currency: str = "USD"
    merchant_id: str
    merchant_category: str
    merchant_name: str
    description: Optional[str] = None


# Fraud Detection Routes
@fraud_router.post("/check", response_model=FraudDecision)
async def check_fraud(
    request: FraudCheckRequest,
    detector: FraudDetector = Depends(get_fraud_detector)
):
    """
    Check transaction for fraud signals without processing payment.

    Use this endpoint to get a fraud assessment before payment processing.
    """
    transaction = TransactionRequest(**request.dict())
    decision = await detector.assess_transaction(transaction)
    decision.transaction_id = request.transaction_id
    return decision


@fraud_router.get("/health")
async def fraud_health(detector: FraudDetector = Depends(get_fraud_detector)):
    """Check fraud detection system health."""
    redis_health = await detector.signal_collector.velocity_tracker.get_health_metrics()
    return {
        "status": "healthy",
        "redis": redis_health,
        "timestamp": datetime.utcnow().isoformat()
    }


# Payment Routes
@payment_router.post("/process")
async def process_payment(
    request: PaymentRequest,
    service: PaymentService = Depends(get_payment_service)
):
    """
    Process payment with integrated fraud detection.

    Flow:
    1. Check fraud signals
    2. If approved: charge via Stripe
    3. If declined: reject payment
    4. If review: process but flag for review

    All signals logged to Kafka for ML training.
    """
    result = await service.process_payment(
        transaction_id=request.transaction_id,
        user_id=request.user_id,
        card_id=request.card_id,
        card_token=request.card_token,
        ip_address=request.ip_address,
        amount=request.amount,
        currency=request.currency,
        merchant_id=request.merchant_id,
        merchant_category=request.merchant_category,
        merchant_name=request.merchant_name,
        description=request.description
    )
    return result


@payment_router.post("/refund/{stripe_charge_id}")
async def refund_payment(
    stripe_charge_id: str,
    amount: Optional[float] = None,
    reason: str = "requested_by_customer",
    service: PaymentService = Depends(get_payment_service)
):
    """Refund a payment."""
    result = await service.refund_payment(
        stripe_charge_id=stripe_charge_id,
        amount=amount,
        reason=reason
    )
    return result


# Health check
@app.get("/health")
async def health():
    """Overall API health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "fraud-detection-api",
        "version": "1.0.0"
    }


# Register routers
app.include_router(fraud_router)
app.include_router(payment_router)

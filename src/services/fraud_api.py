"""
Fraud Detection API

This is the production ML service that:
1. Receives payment requests
2. Fetches features from cache
3. Runs ML model prediction
4. Returns fraud decision
5. Emits metrics/traces/logs

CRITICAL: This is where ALL observability patterns come together:
- Auto-instrumented HTTP spans (OpenTelemetry + FastAPI)
- Custom ML prediction spans
- Business metrics (fraud_blocked_total, revenue_total)
- Structured logs with correlation IDs
- Error handling with retries

BUSINESS SLA:
- p95 latency < 100ms (or customers abandon checkout)
- Availability > 99.9% (43 minutes downtime/month)
- Fraud detection recall > 85% (catch 85% of fraud)
- Precision > 90% (false positive rate < 10%)
"""

import logging
import time
from datetime import datetime
from typing import Optional
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from prometheus_client import generate_latest, REGISTRY
from opentelemetry import trace

# Our observability modules
from observability.instrumentation import (
    initialize_observability,
    business_metrics,
    get_trace_context
)
from observability.logging_config import setup_logging, get_logger

# Our business logic
from services.feature_store import FeatureStore
from models.fraud_model import FraudDetectionModel

# Initialize structured logging FIRST (before any logging calls)
setup_logging(level="INFO", service_name="fraud-detection-api")

logger = get_logger(__name__)

# Initialize OpenTelemetry
tracer, meter = initialize_observability(
    service_name="fraud-detection-api",
    otlp_endpoint=os.getenv("OTLP_ENDPOINT", "http://otel-collector:4317")
)

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Fraud Detection API",
    description="ML-powered fraud detection for payment processing",
    version="2.0.0"
)

# ============================================================================
# CRITICAL: Global State
# ============================================================================
# In production, you'd use dependency injection (FastAPI's Depends())
# But for clarity, we use global singletons here.
# ============================================================================

# Load ML model (on startup)
MODEL_PATH = os.getenv("MODEL_PATH", "models/fraud_model.pkl")
model: Optional[FraudDetectionModel] = None

# Feature store (Redis connection)
feature_store: Optional[FeatureStore] = None


@app.on_event("startup")
async def startup_event():
    """
    Initialize services on startup.

    FAILURE MODE:
    If model fails to load, the app crashes (fail-fast).
    Kubernetes will restart the pod.

    ALTERNATIVE: Lazy loading (load on first request)
    - ✅ App starts even if model is missing
    - ❌ First request fails (poor UX)

    We choose fail-fast because:
    - K8s readiness probe will detect the failure
    - Better than serving errors to customers
    """
    global model, feature_store

    logger.info("🚀 Starting fraud detection API...")

    # Initialize feature store
    try:
        feature_store = FeatureStore(
            redis_host=os.getenv("REDIS_HOST", "redis"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
        )
        logger.info("✓ Feature store initialized")
    except Exception as e:
        logger.error(f"✗ Failed to initialize feature store: {e}")
        raise

    # Load ML model
    try:
        if os.path.exists(MODEL_PATH):
            model = FraudDetectionModel.load(MODEL_PATH)
            logger.info(f"✓ Model loaded from {MODEL_PATH}")
        else:
            logger.warning(f"Model not found at {MODEL_PATH}, training demo model...")
            from models.fraud_model import train_demo_model
            train_demo_model(MODEL_PATH)
            model = FraudDetectionModel.load(MODEL_PATH)
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        raise

    logger.info("✓ Fraud detection API ready")


# ============================================================================
# Request/Response Models
# ============================================================================

class PaymentRequest(BaseModel):
    """
    Payment transaction request.

    In production, this would have 50+ fields.
    We keep it simple for demo purposes.
    """
    transaction_id: str = Field(..., description="Unique transaction ID")
    user_id: int = Field(..., description="User ID", gt=0)
    amount: float = Field(..., description="Transaction amount (USD)", gt=0)
    payment_method: str = Field(..., description="card | bank_transfer | crypto")
    merchant_id: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "txn_abc123",
                "user_id": 12345,
                "amount": 499.99,
                "payment_method": "card",
                "merchant_id": "merchant_xyz"
            }
        }


class FraudResponse(BaseModel):
    """Fraud detection result."""
    transaction_id: str
    decision: str = Field(..., description="approve | review | block")
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: str = Field(..., description="low | medium | high")
    latency_ms: float = Field(..., description="Processing latency")
    model_version: str

    # For debugging (correlation)
    trace_id: Optional[str] = None


# ============================================================================
# CORE ENDPOINT: Fraud Detection
# ============================================================================

@app.post("/v1/predict", response_model=FraudResponse)
async def predict_fraud(payment: PaymentRequest, request: Request):
    """
    Predict fraud for a payment transaction.

    BUSINESS FLOW:
    1. Validate request
    2. Fetch features from Redis (2ms)
    3. Enrich with request data
    4. Run ML model (10ms)
    5. Apply business rules
    6. Return decision

    OBSERVABILITY:
    - Span: predict_fraud (auto-created by FastAPI instrumentation)
    - Custom spans: fetch_features, model_inference, apply_business_rules
    - Metrics: predictions_total, prediction_latency_seconds
    - Logs: Decision logged with trace_id

    SLO: p95 latency < 100ms
    """
    start_time = time.time()
    trace_ctx = get_trace_context()

    # Create request-scoped logger with context
    request_logger = get_logger(
        __name__,
        transaction_id=payment.transaction_id,
        user_id=payment.user_id,
        **trace_ctx
    )

    request_logger.info(
        f"Fraud prediction request: ${payment.amount} from user {payment.user_id}"
    )

    try:
        # ==================================================================
        # STEP 1: Fetch features from cache
        # ==================================================================
        with tracer.start_as_current_span("fetch_features") as span:
            span.set_attribute("user.id", payment.user_id)

            features = feature_store.get_features(payment.user_id)

            # Enrich with request data
            features['transaction_amount'] = payment.amount
            features['transaction_hour'] = datetime.now().hour

            # If cache miss, use defaults (this degrades model accuracy!)
            if 'days_since_signup' not in features or features['days_since_signup'] == 0:
                request_logger.warning(
                    "Using default features due to cache miss (accuracy may degrade)"
                )
                features['days_since_signup'] = 365  # Default: 1 year old account
                features['transaction_count_24h'] = 5
                features['avg_transaction_amount'] = 100
                features['is_international'] = 0

            span.set_attribute("features.count", len(features))

        # ==================================================================
        # STEP 2: Run ML model
        # ==================================================================
        with tracer.start_as_current_span("model_inference") as span:
            span.set_attribute("model.name", "fraud_detector_v1")
            span.set_attribute("model.version", "1.0.0")

            model_start = time.time()
            prediction = model.predict(features)
            model_latency = time.time() - model_start

            # Record metrics
            business_metrics.prediction_latency_seconds.labels(
                model_name="fraud_detector_v1"
            ).observe(model_latency)

            business_metrics.prediction_score.labels(
                model_name="fraud_detector_v1"
            ).observe(prediction['fraud_probability'])

            span.set_attribute("prediction.fraud_probability", prediction['fraud_probability'])
            span.set_attribute("prediction.decision", prediction['decision'])
            span.set_attribute("latency_ms", model_latency * 1000)

        # ==================================================================
        # STEP 3: Apply business rules
        # ==================================================================
        with tracer.start_as_current_span("apply_business_rules") as span:
            decision = prediction['decision']

            # BUSINESS RULE: Always block transactions > $10,000
            if payment.amount > 10000:
                decision = "block"
                prediction['risk_level'] = "high"
                span.set_attribute("rule.triggered", "high_amount_block")

            # BUSINESS RULE: Auto-approve crypto < $100 (lower fraud rate)
            if payment.payment_method == "crypto" and payment.amount < 100:
                decision = "approve"
                prediction['risk_level'] = "low"
                span.set_attribute("rule.triggered", "crypto_small_amount")

            span.set_attribute("final_decision", decision)

        # ==================================================================
        # STEP 4: Record metrics and logs
        # ==================================================================

        # Increment prediction counter
        business_metrics.predictions_total.labels(
            model_name="fraud_detector_v1",
            model_version="1.0.0",
            prediction=decision
        ).inc()

        # Record payment attempt
        business_metrics.payments_total.labels(
            status=decision,  # approve/review/block
            payment_method=payment.payment_method
        ).inc()

        business_metrics.payment_amount_usd.labels(
            payment_method=payment.payment_method
        ).observe(payment.amount)

        # Track revenue and blocked fraud
        if decision == "approve":
            business_metrics.revenue_usd_total.inc(payment.amount)
        elif decision == "block":
            business_metrics.fraud_blocked_usd_total.inc(payment.amount)

        # Calculate total latency
        total_latency = time.time() - start_time

        # Log decision
        request_logger.info(
            f"Fraud decision: {decision} (probability: {prediction['fraud_probability']:.2%}, latency: {total_latency*1000:.1f}ms)",
            extra={
                "decision": decision,
                "fraud_probability": prediction['fraud_probability'],
                "latency_ms": total_latency * 1000,
                "amount": payment.amount
            }
        )

        # Return response
        return FraudResponse(
            transaction_id=payment.transaction_id,
            decision=decision,
            fraud_probability=prediction['fraud_probability'],
            risk_level=prediction['risk_level'],
            latency_ms=round(total_latency * 1000, 2),
            model_version="1.0.0",
            trace_id=trace_ctx.get('trace_id')
        )

    except Exception as e:
        # ==================================================================
        # FAILURE HANDLING
        # ==================================================================
        # Record error metric
        business_metrics.error_total.labels(
            error_type=type(e).__name__,
            service="fraud_api"
        ).inc()

        # Log error with full context
        request_logger.error(
            f"Fraud prediction failed: {str(e)}",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                **trace_ctx
            },
            exc_info=True  # Include stack trace
        )

        # Set error on span
        span = trace.get_current_span()
        span.set_attribute("error", True)
        span.set_attribute("error.message", str(e))

        # Return error to client
        raise HTTPException(
            status_code=500,
            detail=f"Fraud prediction failed: {str(e)}"
        )


# ============================================================================
# HEALTH CHECKS (for Kubernetes)
# ============================================================================

@app.get("/health/live")
async def liveness():
    """
    Liveness probe: Is the app running?

    Kubernetes uses this to know if it should restart the pod.

    Return 200 if:
    - Process is running
    - Not deadlocked

    DO NOT check:
    - Database connectivity (that's readiness, not liveness)
    - Model loaded (that's readiness)
    """
    return {"status": "alive"}


@app.get("/health/ready")
async def readiness():
    """
    Readiness probe: Is the app ready to serve traffic?

    Kubernetes uses this to know if it should send traffic to this pod.

    Return 200 if:
    - Model is loaded
    - Redis is reachable
    - All dependencies are healthy

    If this returns 503, Kubernetes will NOT send traffic to this pod.
    """
    checks = {}

    # Check model loaded
    if model is None:
        checks['model'] = 'not_loaded'
        return {"status": "not_ready", "checks": checks}, 503

    checks['model'] = 'loaded'

    # Check Redis connection
    try:
        feature_store.redis_client.ping()
        checks['redis'] = 'connected'
    except Exception as e:
        checks['redis'] = f'error: {str(e)}'
        return {"status": "not_ready", "checks": checks}, 503

    return {"status": "ready", "checks": checks}


# ============================================================================
# METRICS ENDPOINT (for Prometheus)
# ============================================================================

@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.

    Prometheus will scrape this endpoint every 15 seconds.

    CRITICAL: This returns ALL metrics from prometheus_client.REGISTRY:
    - Auto-instrumented metrics (HTTP request count, latency)
    - Custom business metrics (fraud_detected_total, revenue_total)

    Example output:
        # HELP predictions_total Total ML predictions
        # TYPE predictions_total counter
        predictions_total{model_name="fraud_v1",prediction="block"} 127.0
        predictions_total{model_name="fraud_v1",prediction="approve"} 8473.0
    """
    return PlainTextResponse(
        generate_latest(REGISTRY),
        media_type="text/plain; charset=utf-8"
    )


# ============================================================================
# DEBUG ENDPOINTS (Development only)
# ============================================================================

@app.get("/debug/cache-stats")
async def cache_stats():
    """Get Redis cache statistics (for debugging)."""
    return feature_store.get_cache_stats()


@app.post("/debug/populate-cache/{user_id}")
async def populate_cache(user_id: int):
    """
    Populate cache with sample features (for testing).

    In production, this would be done by batch jobs, not API calls.
    """
    import random

    features = {
        'transaction_amount': 0,  # Will be overridden by request
        'transaction_hour': 0,    # Will be overridden by request
        'days_since_signup': random.randint(1, 1000),
        'transaction_count_24h': random.randint(1, 50),
        'avg_transaction_amount': random.uniform(50, 500),
        'is_international': random.choice([0, 1])
    }

    feature_store.set_features_batch(user_id, features)

    return {
        "user_id": user_id,
        "features": features,
        "message": "Features cached successfully"
    }


# ============================================================================
# Root endpoint
# ============================================================================

@app.get("/")
async def root():
    """API information."""
    return {
        "service": "fraud-detection-api",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "predict": "/v1/predict",
            "health": "/health/ready",
            "metrics": "/metrics"
        }
    }


if __name__ == "__main__":
    import uvicorn

    # Run with: python -m services.fraud_api
    uvicorn.run(
        "services.fraud_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes (dev only)
        log_config=None,  # Use our custom logging config
    )

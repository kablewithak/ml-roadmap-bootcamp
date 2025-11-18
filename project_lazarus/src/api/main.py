"""
FastAPI Inference Service for Project Lazarus.

This is the main API entry point that handles loan application decisions.
"""

import structlog
from contextlib import asynccontextmanager
from typing import Any

import redis
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.core.models import (
    LoanApplication,
    DecisionResult,
    UserFeatures,
    Decision,
)
from src.core.policy import decide_application
from src.core.router import TrafficRouter
from src.models.risk_model import RiskModel
from config.settings import get_settings, Settings

logger = structlog.get_logger(__name__)

# Prometheus metrics
DECISION_COUNTER = Counter(
    "lazarus_decisions_total",
    "Total loan decisions",
    ["decision", "reason", "treatment_group"]
)

DECISION_LATENCY = Histogram(
    "lazarus_decision_latency_seconds",
    "Latency of loan decisions",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

EXPLORATION_BUDGET = Counter(
    "lazarus_exploration_budget_spent",
    "Total exploration budget spent"
)


# Global instances
_redis_client: redis.Redis | None = None
_risk_model: RiskModel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global _redis_client, _risk_model

    settings = get_settings()

    # Initialize Redis
    logger.info("initializing_redis", url=settings.redis_url)
    _redis_client = redis.from_url(settings.redis_url)

    # Initialize exploration budget
    router = TrafficRouter(redis_client=_redis_client)
    router.initialize_budget()

    # Initialize risk model
    logger.info("initializing_risk_model")
    _risk_model = RiskModel()
    _risk_model.load_or_create_default()

    logger.info("application_started", app_name=settings.app_name)

    yield

    # Cleanup
    if _redis_client:
        _redis_client.close()

    logger.info("application_shutdown")


def get_redis() -> redis.Redis:
    """Dependency to get Redis client."""
    if _redis_client is None:
        raise HTTPException(status_code=503, detail="Redis not initialized")
    return _redis_client


def get_risk_model() -> RiskModel:
    """Dependency to get risk model."""
    if _risk_model is None:
        raise HTTPException(status_code=503, detail="Risk model not initialized")
    return _risk_model


# Create FastAPI app
app = FastAPI(
    title="Project Lazarus - Causal Rejection Inference System",
    description="""
    A closed-loop active learning system for credit decisioning.

    This API implements an epsilon-greedy exploration strategy to combat
    rejection inference bias in credit models.

    Key Features:
    - Compliance-first safety valve
    - Atomic budget management with Redis
    - Causal inference through Inverse Probability Weighting
    """,
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "project-lazarus"}


@app.get("/ready")
async def readiness_check(
    redis_client: redis.Redis = Depends(get_redis)
) -> dict[str, Any]:
    """Readiness check with dependency verification."""
    try:
        redis_client.ping()
        router = TrafficRouter(redis_client=redis_client)
        status = router.get_budget_status()
        return {
            "status": "ready",
            "redis": "connected",
            "exploration_budget": status["remaining_budget"]
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/decide", response_model=DecisionResult)
async def make_decision(
    application: LoanApplication,
    redis_client: redis.Redis = Depends(get_redis),
    risk_model: RiskModel = Depends(get_risk_model),
) -> DecisionResult:
    """
    Make a loan decision for an application.

    This endpoint implements the full Lazarus decision pipeline:
    1. Feature extraction
    2. Risk scoring
    3. Safety valve checks
    4. Exploration/exploitation decision
    """
    with DECISION_LATENCY.time():
        # Get risk score from model
        risk_score = risk_model.predict(application.user_features)

        # Make decision using policy engine
        result = decide_application(
            user_features=application.user_features,
            risk_score=risk_score,
            application_id=application.application_id,
            redis_client=redis_client,
        )

        # Update Prometheus metrics
        DECISION_COUNTER.labels(
            decision=result.decision.value,
            reason=result.reason.value,
            treatment_group=result.treatment_group.value
        ).inc()

        if result.treatment_group.value == "explore":
            EXPLORATION_BUDGET.inc()

        logger.info(
            "decision_made",
            application_id=application.application_id,
            decision=result.decision.value,
            reason=result.reason.value,
            treatment_group=result.treatment_group.value,
            risk_score=risk_score
        )

        return result


@app.get("/budget")
async def get_budget_status(
    redis_client: redis.Redis = Depends(get_redis)
) -> dict[str, Any]:
    """Get current exploration budget status."""
    router = TrafficRouter(redis_client=redis_client)
    return router.get_budget_status()


@app.post("/budget/reset")
async def reset_budget(
    budget: float | None = None,
    redis_client: redis.Redis = Depends(get_redis)
) -> dict[str, Any]:
    """Reset exploration budget."""
    router = TrafficRouter(redis_client=redis_client)
    router.initialize_budget(budget)
    return router.get_budget_status()


@app.get("/metrics")
async def get_metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/exploration/metrics")
async def get_exploration_metrics(
    redis_client: redis.Redis = Depends(get_redis)
) -> dict[str, Any]:
    """Get detailed exploration metrics."""
    router = TrafficRouter(redis_client=redis_client)
    return router.get_exploration_metrics()


def run():
    """Run the API server."""
    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )


if __name__ == "__main__":
    run()

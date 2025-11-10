"""
FastAPI application with fraud detection and payment processing.
"""

import logging
import yaml
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ..fraud.services.fraud_detector import create_fraud_detector
from ..payments.service import PaymentService
from ..infrastructure.redis.velocity_tracker import create_redis_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
fraud_detector = None
payment_service = None
redis_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global fraud_detector, payment_service, redis_client

    # Load configuration
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize Redis
    redis_config = config["redis"]
    redis_client = await create_redis_client(
        host=redis_config["host"],
        port=redis_config["port"],
        db=redis_config["db"]
    )
    logger.info("Redis client initialized")

    # Initialize fraud detector
    fraud_detector = await create_fraud_detector(
        redis_client=redis_client,
        kafka_config=config["kafka"],
        fraud_config=config["fraud_detection"]
    )
    logger.info("Fraud detector initialized")

    # Initialize payment service
    stripe_api_key = config["stripe"]["api_key"]
    payment_service = PaymentService(
        fraud_detector=fraud_detector,
        stripe_api_key=stripe_api_key
    )
    logger.info("Payment service initialized")

    yield

    # Cleanup
    if redis_client:
        await redis_client.close()
    if fraud_detector:
        fraud_detector.kafka_producer.close()
    logger.info("Application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Fraud Detection & Payment Processing API",
    description="Production-ready fraud detection integrated with payment processing",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency injection
def get_fraud_detector():
    if fraud_detector is None:
        raise HTTPException(status_code=503, detail="Fraud detector not initialized")
    return fraud_detector


def get_payment_service():
    if payment_service is None:
        raise HTTPException(status_code=503, detail="Payment service not initialized")
    return payment_service


# Import routes
from . import routes

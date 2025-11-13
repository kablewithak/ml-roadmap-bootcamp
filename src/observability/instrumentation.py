"""
OpenTelemetry Instrumentation Setup

This module initializes the "auto-instrumentation" for:
- FastAPI (HTTP request spans, metrics)
- Redis (cache operation spans)
- HTTP clients (outbound API call spans)
- Logging (correlation IDs injected into every log line)

BUSINESS IMPACT:
Without this, you're debugging blind:
- "Why is this payment slow?" → No trace to show DB query took 2s
- "Did Redis go down?" → No span to show cache.get() timed out
- "Which user hit the error?" → No correlation ID linking logs to traces

ARCHITECTURAL PATTERN: The "Observability Facade"
Rather than having each service configure OTel directly, we centralize it here.
This means:
- ✅ Consistency: All services emit the same span attributes
- ✅ DRY: Change sampling rate in ONE place
- ✅ Testing: Mock this module to disable instrumentation in tests
- ❌ Trade-off: Less flexibility per-service (acceptable for most cases)
"""

import logging
import os
from typing import Optional

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION, DEPLOYMENT_ENVIRONMENT
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor

# Prometheus client for custom business metrics
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY

logger = logging.getLogger(__name__)


# ============================================================================
# CRITICAL: Service Resource Attributes
# ============================================================================
# These labels appear on EVERY metric/trace/log from this service.
# This is how Grafana can filter "show me only fraud-service metrics"
#
# FAILURE MODE: If you forget to set these, all services look identical in Grafana.
# You'll see "latency = 500ms" but won't know if it's from fraud-service or payment-service.
# ============================================================================

def create_resource(service_name: str, service_version: str = "1.0.0") -> Resource:
    """
    Creates an OpenTelemetry Resource with service metadata.

    Resource attributes are KEY-VALUE pairs attached to all telemetry data.
    Think of them as "global tags" for this service instance.

    Why this matters for ML systems:
    - service.name: "fraud-model-v2" vs "fraud-model-v1" (A/B testing)
    - deployment.environment: "canary" vs "production" (phased rollouts)
    - service.version: Track metrics across model versions

    Args:
        service_name: Unique identifier for this service (e.g., "fraud-detection-api")
        service_version: Semantic version (e.g., "2.1.0")

    Returns:
        OpenTelemetry Resource object
    """
    return Resource(attributes={
        SERVICE_NAME: service_name,
        SERVICE_VERSION: service_version,
        DEPLOYMENT_ENVIRONMENT: os.getenv("ENVIRONMENT", "development"),
        # CUSTOM: Add your ML-specific attributes here
        "ml.model.name": os.getenv("MODEL_NAME", "fraud_model_v1"),
        "ml.model.version": os.getenv("MODEL_VERSION", "1.0.0"),
    })


# ============================================================================
# TRACING SETUP (Pillar #1)
# ============================================================================

def setup_tracing(
    service_name: str,
    otlp_endpoint: str = "http://otel-collector:4317",
    sample_rate: float = 1.0,
) -> trace.Tracer:
    """
    Initializes distributed tracing with OpenTelemetry.

    HOW IT WORKS:
    1. Your code calls `with tracer.start_as_current_span("predict"):`
    2. OTel SDK creates a Span (start time, operation name, attributes)
    3. BatchSpanProcessor buffers spans in memory (batch of 512)
    4. Every 5 seconds, exports batch to OTel Collector via gRPC
    5. Collector forwards to Jaeger for storage/visualization

    CRITICAL: Sampling Strategy
    -----------------------------
    sample_rate=1.0 means "trace 100% of requests"

    When to sample less:
    - High traffic: 1M requests/day × 1.0 = 1M traces = 10GB storage/day
    - Sample 0.01 (1%) = 10K traces = 100MB storage/day

    When to sample more:
    - Development: 1.0 (trace everything for debugging)
    - Production errors: Use "ParentBased" sampler (always trace if parent traced)

    FAILURE MODE:
    If OTel Collector is unreachable, BatchSpanProcessor will:
    1. Retry with exponential backoff (2s, 4s, 8s...)
    2. After 30s, drop spans and log a warning
    3. Your application continues running (traces lost, but no crash)

    This is a TRADE-OFF:
    - ✅ Resilience: App doesn't crash if monitoring is down
    - ❌ Blind spot: You lose visibility during the outage

    Alternative: Use a sidecar OTel Collector (runs on localhost) for reliability.

    Args:
        service_name: Service identifier
        otlp_endpoint: OTel Collector gRPC endpoint
        sample_rate: Fraction of requests to trace (0.0 to 1.0)

    Returns:
        Configured Tracer instance
    """
    resource = create_resource(service_name)

    # CRITICAL: BatchSpanProcessor vs SimpleSpanProcessor
    # SimpleSpanProcessor exports EVERY span immediately (blocks request thread)
    # BatchSpanProcessor buffers in memory, exports in background (5ms overhead)
    # ALWAYS use Batch in production!
    provider = TracerProvider(resource=resource)

    try:
        # Export to OTel Collector (which forwards to Jaeger)
        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        logger.info(f"✓ Tracing initialized: {service_name} → {otlp_endpoint}")
    except Exception as e:
        # FALLBACK: If OTel Collector is down, use no-op exporter
        logger.warning(f"⚠ Failed to connect to OTel Collector: {e}. Traces will be discarded.")

    trace.set_tracer_provider(provider)

    # Auto-instrument common libraries
    # This adds spans for:
    # - FastAPI: HTTP request spans (method, path, status_code)
    # - Redis: cache.get(), cache.set() spans
    # - Requests: Outbound HTTP call spans
    FastAPIInstrumentor().instrument()
    RedisInstrumentor().instrument()
    RequestsInstrumentor().instrument()

    # Inject trace IDs into logs (correlation!)
    LoggingInstrumentor().instrument(set_logging_format=True)

    return trace.get_tracer(__name__)


# ============================================================================
# METRICS SETUP (Pillar #2)
# ============================================================================

def setup_metrics(
    service_name: str,
    otlp_endpoint: str = "http://otel-collector:4317",
    export_interval_ms: int = 10000,  # Export every 10 seconds
) -> metrics.Meter:
    """
    Initializes Prometheus-compatible metrics export.

    ARCHITECTURAL DECISION: Push vs Pull
    --------------------------------------
    Prometheus uses a PULL model:
    - Prometheus scrapes /metrics endpoint every 15s (you configure the interval)
    - Your app exposes prometheus_client.generate_latest(REGISTRY)

    But OpenTelemetry uses a PUSH model:
    - Your app pushes metrics to OTel Collector every 10s
    - OTel Collector exposes /metrics for Prometheus to scrape

    Why this indirection?
    - ✅ Decoupling: App doesn't need to expose HTTP endpoint (good for internal services)
    - ✅ Batching: Collector can aggregate metrics from multiple instances
    - ✅ Transformation: Collector can relabel/filter metrics before Prometheus
    - ❌ Complexity: One more hop in the pipeline

    ALTERNATIVE: Direct Prometheus Exporter
    You can skip OTel Collector and use:
        from opentelemetry.exporter.prometheus import PrometheusMetricReader
        provider = MeterProvider(metric_readers=[PrometheusMetricReader()])

    This exposes /metrics directly from your app (simpler, but less flexible).

    Args:
        service_name: Service identifier
        otlp_endpoint: OTel Collector gRPC endpoint
        export_interval_ms: How often to push metrics (default: 10s)

    Returns:
        Configured Meter instance
    """
    resource = create_resource(service_name)

    try:
        otlp_exporter = OTLPMetricExporter(endpoint=otlp_endpoint, insecure=True)
        reader = PeriodicExportingMetricReader(otlp_exporter, export_interval_millis=export_interval_ms)
        provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(provider)
        logger.info(f"✓ Metrics initialized: {service_name} → {otlp_endpoint}")
    except Exception as e:
        logger.warning(f"⚠ Failed to connect to OTel Collector for metrics: {e}")

    return metrics.get_meter(__name__)


# ============================================================================
# BUSINESS METRICS (Custom Prometheus Metrics)
# ============================================================================
# These are metrics YOU define based on your business logic.
# Auto-instrumentation gives you HTTP request counts, but not "fraud_detected_total"
# ============================================================================

class BusinessMetrics:
    """
    Custom business metrics for ML fraud detection system.

    METRIC TYPES:
    1. Counter: Monotonically increasing (requests_total, errors_total)
       - Use for: Event counts, cumulative values
       - Query: rate(requests_total[5m]) → requests per second

    2. Histogram: Samples observations into buckets (latency_seconds)
       - Use for: Latency, request size, prediction scores
       - Query: histogram_quantile(0.95, latency_seconds) → p95 latency

    3. Gauge: Can go up or down (active_users, queue_size)
       - Use for: Current state, capacity
       - Query: avg(queue_size) → average queue depth

    CRITICAL: Cardinality
    ----------------------
    DO NOT add high-cardinality labels (user_id, transaction_id, etc.)

    BAD:  prediction_total{user_id="12345"}  ← 1M users = 1M time series
    GOOD: prediction_total{model="fraud_v1"} ← 5 models = 5 time series

    High cardinality KILLS Prometheus:
    - 1M time series × 8 bytes/sample × 60 samples/min = 480MB/min RAM
    - Your Prometheus pod OOMs and crashes

    WHEN THIS PATTERN IS WRONG:
    If you need per-user metrics, use logs + analytics DB (ClickHouse, BigQuery)
    """

    def __init__(self):
        # Payment processing
        self.payments_total = Counter(
            'payments_total',
            'Total payment attempts',
            ['status', 'payment_method']  # Labels: success/failed, card/bank
        )

        self.payment_amount_usd = Histogram(
            'payment_amount_usd',
            'Payment amount in USD',
            ['payment_method'],
            buckets=[10, 50, 100, 500, 1000, 5000, 10000]  # Histogram buckets
        )

        # ML Model metrics
        self.predictions_total = Counter(
            'predictions_total',
            'Total ML predictions',
            ['model_name', 'model_version', 'prediction']  # fraud/legitimate
        )

        self.prediction_latency_seconds = Histogram(
            'prediction_latency_seconds',
            'Model inference latency',
            ['model_name'],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]  # 10ms to 5s
        )

        self.prediction_score = Histogram(
            'prediction_score',
            'Model prediction probability (0-1)',
            ['model_name'],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        )

        # Feature store metrics
        self.feature_cache_hits_total = Counter(
            'feature_cache_hits_total',
            'Redis cache hits',
            ['feature_name']
        )

        self.feature_cache_misses_total = Counter(
            'feature_cache_misses_total',
            'Redis cache misses',
            ['feature_name']
        )

        self.feature_fetch_latency_seconds = Histogram(
            'feature_fetch_latency_seconds',
            'Time to fetch features from cache',
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1]  # 1ms to 100ms
        )

        # Business metrics
        self.revenue_usd_total = Counter(
            'revenue_usd_total',
            'Total revenue processed (cumulative)'
        )

        self.fraud_blocked_usd_total = Counter(
            'fraud_blocked_usd_total',
            'Total fraud amount blocked'
        )

        # System health
        self.error_total = Counter(
            'errors_total',
            'Application errors',
            ['error_type', 'service']
        )

        # ML-Specific: Data Drift (Updated by batch job)
        self.feature_psi = Gauge(
            'feature_psi',
            'Population Stability Index for features',
            ['feature_name']
        )

        self.model_performance = Gauge(
            'model_performance',
            'Model performance metrics',
            ['model_name', 'metric']  # metric: precision, recall, f1
        )


# Global singleton
business_metrics = BusinessMetrics()


# ============================================================================
# HELPER: Get current trace context
# ============================================================================

def get_trace_context() -> dict:
    """
    Extract current trace ID and span ID for correlation.

    USE CASE: Logging
    -----------------
    When you log an error, include the trace_id so you can:
    1. Search logs: "Find all logs for trace_id=abc123"
    2. Jump to trace: Click trace_id in Grafana → opens Jaeger UI

    Example:
        logger.error("Payment failed", extra=get_trace_context())
        # Output: {"msg": "Payment failed", "trace_id": "abc123", "span_id": "xyz789"}

    Returns:
        Dict with trace_id and span_id (or empty if no active trace)
    """
    span = trace.get_current_span()
    ctx = span.get_span_context()

    if ctx.is_valid:
        return {
            "trace_id": format(ctx.trace_id, "032x"),  # Convert to hex string
            "span_id": format(ctx.span_id, "016x"),
        }
    return {}


# ============================================================================
# INITIALIZATION FUNCTION (Called by main.py)
# ============================================================================

def initialize_observability(
    service_name: str,
    otlp_endpoint: Optional[str] = None,
) -> tuple[trace.Tracer, metrics.Meter]:
    """
    One-line setup for all observability.

    Usage in your FastAPI app:
        from observability.instrumentation import initialize_observability

        tracer, meter = initialize_observability("fraud-detection-api")

    Args:
        service_name: Unique service identifier
        otlp_endpoint: OTel Collector endpoint (defaults to env var or localhost)

    Returns:
        (tracer, meter) tuple for creating custom spans/metrics
    """
    endpoint = otlp_endpoint or os.getenv("OTLP_ENDPOINT", "http://otel-collector:4317")

    tracer = setup_tracing(service_name, endpoint)
    meter = setup_metrics(service_name, endpoint)

    logger.info(f"🔭 Observability initialized for {service_name}")
    return tracer, meter

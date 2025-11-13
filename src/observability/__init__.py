"""
Observability Package

This module provides a centralized configuration for the "Three Pillars of Observability":
1. TRACES: Distributed tracing via OpenTelemetry → Jaeger
2. METRICS: Time-series data via Prometheus
3. LOGS: Structured JSON logs via Loki

CRITICAL DESIGN DECISION:
We use OpenTelemetry (OTel) as the instrumentation layer because:
- Vendor-neutral (works with Jaeger, Zipkin, DataDog, etc.)
- Auto-instrumentation for common libraries (FastAPI, Redis, HTTP)
- Future-proof (CNCF standard, backed by all major vendors)

FAILURE MODE:
If OTel Collector goes down, this module falls back to no-op exporters
(metrics/traces are discarded, but the application keeps running).
"""

"""
Infrastructure Layer - External Dependencies

This layer contains:
- Event store (PostgreSQL + Kafka)
- Payment gateways (Stripe, banks)
- External APIs (FX rates, fraud detection)
- Caching (Redis)
- Observability (Prometheus, OpenTelemetry)

Key principle: All infrastructure is REPLACEABLE.
Domain layer knows nothing about this layer (dependency inversion).
"""

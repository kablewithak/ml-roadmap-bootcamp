"""
Domain Layer - Pure Business Logic

This layer contains:
- Domain events (immutable facts about what happened)
- Aggregates (consistency boundaries)
- Value objects (immutable domain concepts)
- Domain services (business logic that doesn't belong to entities)

Key principle: ZERO dependencies on infrastructure
This allows us to test business logic without databases, APIs, or external services.
"""

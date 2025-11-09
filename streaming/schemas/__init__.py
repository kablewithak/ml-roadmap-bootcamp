"""
Avro schemas and schema management for streaming events.

This module defines the event schemas used throughout the streaming infrastructure,
including PaymentEvent, FraudDecisionEvent, and UserActionEvent.
"""

from .payment_event import PAYMENT_EVENT_SCHEMA, PaymentEvent
from .fraud_decision_event import FRAUD_DECISION_EVENT_SCHEMA, FraudDecisionEvent
from .user_action_event import USER_ACTION_EVENT_SCHEMA, UserActionEvent
from .schema_registry import SchemaEvolutionManager, SchemaRegistryClient

__all__ = [
    'PAYMENT_EVENT_SCHEMA',
    'PaymentEvent',
    'FRAUD_DECISION_EVENT_SCHEMA',
    'FraudDecisionEvent',
    'USER_ACTION_EVENT_SCHEMA',
    'UserActionEvent',
    'SchemaEvolutionManager',
    'SchemaRegistryClient',
]

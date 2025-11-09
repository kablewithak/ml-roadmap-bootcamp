"""
Core streaming infrastructure components.

Provides high-throughput producers, exactly-once consumers, transactional
state management, and resilience patterns for production Kafka/Redpanda systems.
"""

from .producer import HighThroughputProducer, ProducerConfig
from .consumer import ExactlyOnceConsumer, ConsumerConfig
from .state_manager import TransactionalStateManager
from .exactly_once import ExactlyOnceProcessor

__all__ = [
    'HighThroughputProducer',
    'ProducerConfig',
    'ExactlyOnceConsumer',
    'ConsumerConfig',
    'TransactionalStateManager',
    'ExactlyOnceProcessor',
]

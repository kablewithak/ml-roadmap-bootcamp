# Kafka/Redpanda Streaming Infrastructure with Exactly-Once Semantics

A production-grade streaming infrastructure implementing exactly-once processing semantics for high-throughput event streaming with fraud detection and ML feature serving.

## 🎯 Features

### Core Capabilities
- **Exactly-Once Processing**: Guaranteed message processing with zero duplicates
- **High Throughput**: 100k+ messages/second with optimized batching and compression
- **Transactional State Management**: Distributed transactions across Kafka, PostgreSQL, and Redis
- **Schema Evolution**: Backward-compatible Avro schemas with registry integration
- **Resilience Patterns**: Circuit breakers, backpressure, and graceful degradation
- **Production Monitoring**: Prometheus metrics, business KPIs, and alerting

### Advanced Features
- **ML Feature Streaming**: Real-time feature computation for fraud detection
- **Cost Optimization**: Automated partition and retention management
- **Chaos Testing**: Comprehensive failure scenario testing
- **Dead Letter Queue**: Automatic poison message handling

## 📁 Project Structure

```
streaming/
├── core/                           # Core streaming components
│   ├── producer.py                 # High-throughput idempotent producer
│   ├── consumer.py                 # Exactly-once consumer
│   ├── state_manager.py            # Transactional state management
│   ├── exactly_once.py             # End-to-end processor
│   └── resilience.py               # Circuit breaker & backpressure
├── schemas/                        # Avro schemas
│   ├── payment_event.py            # Payment transaction schema
│   ├── fraud_decision_event.py     # Fraud detection schema
│   ├── user_action_event.py        # User action schema
│   └── schema_registry.py          # Schema evolution management
├── monitoring/                     # Metrics and monitoring
│   └── metrics.py                  # Prometheus metrics collection
├── ml_integration/                 # ML feature serving
│   └── feature_streaming.py        # Real-time feature computation
├── optimization/                   # Cost optimization
│   └── cost_manager.py             # Partition/retention optimization
├── tests/                          # Comprehensive test suite
│   ├── test_exactly_once.py        # Exactly-once verification
│   ├── test_throughput.py          # Performance testing
│   └── chaos_tests.py              # Chaos engineering
├── examples/                       # Usage examples
│   └── end_to_end_example.py       # Complete payment processing
└── docker/                         # Infrastructure
    ├── docker-compose.yml          # Redpanda cluster + dependencies
    ├── prometheus.yml              # Prometheus configuration
    └── init.sql                    # PostgreSQL schema
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker and Docker Compose
- 8GB+ RAM for local cluster

### 1. Start Infrastructure

```bash
cd streaming/docker
docker-compose up -d
```

This starts:
- **Redpanda cluster** (3 brokers) on ports 19092, 29092, 39092
- **Schema Registry** on port 18081
- **PostgreSQL** on port 5432
- **Redis** on port 6379
- **Prometheus** on port 9090
- **Grafana** on port 3000
- **Redpanda Console** on port 8080

### 2. Install Python Dependencies

```bash
cd streaming
pip install -r requirements.txt
```

### 3. Initialize Database Schema

```bash
# Database initialization happens automatically via Docker init.sql
# Verify:
psql postgresql://streaming:streaming_pass@localhost:5432/streaming_state -c "\dt"
```

### 4. Run End-to-End Example

```bash
python examples/end_to_end_example.py
```

This demonstrates:
- Producing 1000 payment events
- Consuming with exactly-once processing
- Real-time fraud detection
- ML feature computation
- Metrics collection

## 📊 Architecture

### Exactly-Once Processing Flow

```
┌─────────────┐
│   Producer  │
│ (Idempotent)│
└──────┬──────┘
       │ Transactional Send
       ▼
┌─────────────┐      ┌──────────────┐
│   Kafka/    │◄────►│    Schema    │
│  Redpanda   │      │   Registry   │
└──────┬──────┘      └──────────────┘
       │ Read Committed
       ▼
┌─────────────┐
│  Consumer   │
│  (Manual    │
│   Offsets)  │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────┐
│  Transactional State Mgr     │
│  ┌────────┐ ┌──────────┐    │
│  │ Kafka  │ │PostgreSQL│    │
│  │  Tx    │ │    Tx    │    │
│  └────────┘ └──────────┘    │
│                              │
│  Two-Phase Commit (2PC)      │
└──────────────────────────────┘
       │
       ▼
┌─────────────┐      ┌──────────────┐
│    Redis    │      │   ML Model   │
│  Features   │◄────►│   Serving    │
└─────────────┘      └──────────────┘
```

### Key Design Patterns

1. **Idempotent Producer**: Automatic deduplication at broker level
2. **Transactional Consumer**: Manual offset commits within distributed transactions
3. **State Manager**: Two-phase commit across Kafka, PostgreSQL, Redis
4. **Circuit Breaker**: Prevents cascade failures when downstream services fail
5. **Backpressure**: Adaptive rate limiting based on processing latency

## 💻 Usage Examples

### Basic Producer

```python
from streaming.core.producer import HighThroughputProducer, ProducerConfig

# Configure producer for 100k+ msg/sec
config = ProducerConfig(
    bootstrap_servers="localhost:19092",
    compression_type="zstd",
    batch_size=1000000,  # 1MB batches
    enable_idempotence=True
)

producer = HighThroughputProducer(config)

# Send message
producer.send(
    topic="payments",
    value={"user_id": "user_123", "amount": 100.00},
    key="payment_id_456"
)

producer.flush()
producer.close()
```

### Exactly-Once Consumer

```python
from streaming.core.consumer import ExactlyOnceConsumer, ConsumerConfig

config = ConsumerConfig(
    bootstrap_servers="localhost:19092",
    group_id="payment-processor",
    topics=["payments"],
    enable_auto_commit=False,  # Manual commits for exactly-once
    isolation_level="read_committed"
)

consumer = ExactlyOnceConsumer(config, dead_letter_topic="payments-dlq")

def process_message(msg):
    # Your processing logic
    print(f"Processing: {msg['value']}")
    return True  # Success

consumer.consume(process_message, max_messages=1000)
consumer.close()
```

### Transactional Processing

```python
from streaming.core.state_manager import TransactionalStateManager

state_mgr = TransactionalStateManager(
    postgres_dsn="postgresql://streaming:streaming_pass@localhost:5432/streaming_state",
    redis_url="redis://localhost:6379"
)

# Atomic processing across Kafka + DB
async with state_mgr.transaction(message) as tx:
    # Produce output event
    await tx.publish_event("fraud-decisions", fraud_result)

    # Write to database
    await tx.save_to_db("payment_events", payment_data)

    # Both commit or both rollback atomically
```

### ML Feature Streaming

```python
from streaming.ml_integration.feature_streaming import MLFeatureStreaming

feature_stream = MLFeatureStreaming(redis_url="redis://localhost:6379")

# Compute real-time features
features = await feature_stream.compute_streaming_features(payment_event)

# Features include:
# - tx_count_5min: Transaction velocity
# - tx_amount_24hr: Daily spending
# - unusual_time: Behavioral anomaly
# - merchant_risk_score: Merchant reputation
```

## 🧪 Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Exactly-Once Semantics Tests

```bash
pytest tests/test_exactly_once.py -v -s
```

Tests:
- No duplicates after consumer restart
- Idempotency key deduplication
- Transactional atomicity
- Consumer rebalancing safety

### Throughput Tests

```bash
pytest tests/test_throughput.py -v -s
```

Tests:
- Producer: 100k+ msg/sec
- Consumer: 50k+ msg/sec
- End-to-end latency: p99 < 250ms
- Compression effectiveness

### Chaos Tests

```bash
pytest tests/chaos_tests.py -v -s
```

Tests:
- Broker failure recovery
- Consumer crash recovery
- Network partition handling
- Poison message handling

## 📈 Monitoring

### Prometheus Metrics

Access metrics at `http://localhost:8000/metrics`

Key metrics:
- `streaming_messages_produced_total`: Messages produced
- `streaming_messages_processed_total`: Messages processed
- `streaming_processing_latency_ms`: Processing latency histogram
- `streaming_consumer_lag`: Current consumer lag
- `streaming_circuit_breaker_state`: Circuit breaker state
- `streaming_transaction_amount_total`: Business KPI - transaction volume
- `streaming_fraud_score`: Fraud score distribution

### Prometheus UI

Open `http://localhost:9090` and query:

```promql
# Throughput
rate(streaming_messages_processed_total[5m])

# p99 latency
histogram_quantile(0.99, rate(streaming_processing_latency_ms_bucket[5m]))

# Consumer lag
streaming_consumer_lag

# Fraud detection rate
rate(streaming_fraud_decisions_total{decision="DECLINE"}[5m])
```

### Grafana Dashboards

Open `http://localhost:3000` (admin/admin)

Pre-configured dashboards:
- Streaming Overview
- Consumer Lag Monitoring
- Fraud Detection Analytics
- Cost Optimization

### Redpanda Console

Open `http://localhost:8080`

Features:
- Topic management
- Consumer group monitoring
- Message inspection
- Schema registry browser

## 🎯 Performance Tuning

### Producer Optimization

For maximum throughput:

```python
config = ProducerConfig(
    compression_type="zstd",      # Best compression ratio
    linger_ms=10,                  # Batch for 10ms
    batch_size=1000000,            # 1MB batches
    buffer_memory=67108864,        # 64MB buffer
    max_in_flight=5,               # Pipeline requests
    enable_idempotence=True        # Exactly-once
)
```

### Consumer Optimization

For low latency:

```python
config = ConsumerConfig(
    max_poll_records=500,          # Smaller batches
    session_timeout_ms=10000,      # 10s timeout
    enable_auto_commit=False,      # Manual control
    isolation_level="read_committed"
)
```

### Partition Sizing

Rule of thumb:
- **1 partition**: Up to 10k msg/sec
- **10 partitions**: Up to 100k msg/sec
- **50+ partitions**: 500k+ msg/sec

Use cost optimizer to right-size:

```python
from streaming.optimization.cost_manager import StreamingCostOptimizer

optimizer = StreamingCostOptimizer("localhost:19092")
recommendations = optimizer.analyze_topic("payments")
print(recommendations)
```

## 🔧 Configuration

### Environment Variables

```bash
# Kafka
export KAFKA_BOOTSTRAP_SERVERS="localhost:19092"

# Database
export POSTGRES_DSN="postgresql://streaming:streaming_pass@localhost:5432/streaming_state"

# Redis
export REDIS_URL="redis://localhost:6379"

# Schema Registry
export SCHEMA_REGISTRY_URL="http://localhost:18081"

# Monitoring
export PROMETHEUS_PORT="9090"
export METRICS_PORT="8000"
```

### Production Recommendations

1. **Replication Factor**: 3 for production
2. **Min In-Sync Replicas**: 2
3. **Retention**: 7 days default, adjust per topic
4. **Compression**: ZSTD for best ratio
5. **Monitoring**: 1-minute scrape interval

## 🐛 Troubleshooting

### Consumer Lag Growing

```python
# Check lag
consumer.get_lag()

# Apply backpressure
from streaming.core.resilience import AdaptiveBackpressure

backpressure = AdaptiveBackpressure(consumer)
# Automatically adjusts consumption rate
```

### Circuit Breaker Open

```python
from streaming.core.resilience import StreamingCircuitBreaker

breaker = StreamingCircuitBreaker()
state = breaker.get_state()
print(f"Circuit state: {state['state']}")
print(f"Failure rate: {state['failure_rate']:.2%}")
```

### Database Connection Issues

```bash
# Check PostgreSQL
docker-compose ps postgres
docker-compose logs postgres

# Verify schema
psql postgresql://streaming:streaming_pass@localhost:5432/streaming_state -c "\dt"
```

### Schema Registry Issues

```bash
# Check registry
curl http://localhost:18081/subjects

# Register schema manually
python -c "from streaming.schemas.schema_registry import SchemaEvolutionManager; \
mgr = SchemaEvolutionManager(); \
from streaming.schemas.payment_event import PAYMENT_EVENT_SCHEMA; \
mgr.register_event_schemas('payments', None, PAYMENT_EVENT_SCHEMA)"
```

## 📚 Documentation

### Core Concepts

- **Exactly-Once Semantics**: [streaming/core/exactly_once.py](streaming/core/exactly_once.py:1)
- **State Management**: [streaming/core/state_manager.py](streaming/core/state_manager.py:1)
- **Resilience Patterns**: [streaming/core/resilience.py](streaming/core/resilience.py:1)

### API Reference

All modules include comprehensive docstrings. Generate docs:

```bash
pydoc streaming.core.producer
pydoc streaming.core.consumer
pydoc streaming.core.state_manager
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest tests/`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open Pull Request

## 📄 License

This project is part of the ML Roadmap Bootcamp educational materials.

## 🙏 Acknowledgments

Built with:
- [Redpanda](https://redpanda.com/) - Kafka-compatible streaming platform
- [confluent-kafka-python](https://github.com/confluentinc/confluent-kafka-python) - Python Kafka client
- [Prometheus](https://prometheus.io/) - Monitoring system
- [PostgreSQL](https://www.postgresql.org/) - Relational database
- [Redis](https://redis.io/) - In-memory data store

## 📞 Support

For questions and support:
- GitHub Issues: [Create an issue](../../issues)
- Documentation: This README
- Examples: [examples/](examples/)

---

**Built for production. Designed for learning. Ready for ML at scale.**

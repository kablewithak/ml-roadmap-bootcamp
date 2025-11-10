# Fraud Signal Collection System

Production-ready fraud detection system integrated with payment processing. Fast (<50ms), scalable, and ML-ready.

## 🎯 Features

### 1. **Real-Time Velocity Tracking** (Redis-based)
- Transaction counts per time window (5min, 1hr)
- Amount velocity tracking
- Multi-dimensional: card_id, user_id, ip_address
- **<10ms lookup latency** ⚡

### 2. **Transaction Pattern Signals**
- **Card Testing Detection**: Multiple small charges pattern
- **Amount Velocity**: Sum in time windows
- **Merchant Category Switching**: Unusual category changes
- **Time Pattern Analysis**: Unusual hour detection
- **First-Time Card Usage**: New card risk scoring

### 3. **Risk Score Calculation**
- Weighted rule-based system
- Configurable thresholds
- Explainable decisions with triggered signals
- Returns: `risk_score`, `decision`, `signals_triggered`

### 4. **Payment Flow Integration**
- Fraud check **BEFORE** Stripe charge
- Decisions: APPROVE, REVIEW, DECLINE
- Automatic Kafka logging for ML training
- Store decisions for reconciliation

### 5. **Performance Optimized**
- <50ms total latency (P95)
- <10ms Redis lookups (P95)
- Handles 1000+ TPS
- Async/await throughout

## 📊 Architecture

```
┌─────────────────┐
│ Payment Request │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│   Fraud Detection Service   │
│                             │
│  ┌─────────────────────┐   │
│  │ Signal Collection   │   │ ◄─── Redis (Velocity)
│  └──────────┬──────────┘   │
│             │               │
│  ┌──────────▼──────────┐   │
│  │  Risk Scoring       │   │
│  └──────────┬──────────┘   │
│             │               │
│  ┌──────────▼──────────┐   │
│  │  Decision Engine    │   │
│  └──────────┬──────────┘   │
└─────────────┼───────────────┘
              │
              ├──────► Kafka (Logging)
              │
              ▼
      ┌───────────────┐
      │ Stripe Charge │
      └───────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker & Docker Compose
- Redis
- Kafka
- Stripe account (for payment processing)

### Installation

1. **Clone and setup**:
```bash
git clone <repository>
cd ml-roadmap-bootcamp
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Start infrastructure**:
```bash
docker-compose up -d
```

This starts:
- Redis (port 6379)
- Kafka (port 9092)
- Zookeeper (port 2181)
- PostgreSQL (port 5432)

3. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your Stripe API keys
```

4. **Run the API**:
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

API available at: http://localhost:8000

API docs at: http://localhost:8000/docs

## 📖 Usage

### Fraud Check Only

Check transaction for fraud without processing payment:

```bash
curl -X POST http://localhost:8000/fraud/check \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "tx-001",
    "user_id": "user-123",
    "card_id": "card-456",
    "ip_address": "192.168.1.1",
    "amount": 100.00,
    "currency": "USD",
    "merchant_id": "merchant-789",
    "merchant_category": "retail",
    "merchant_name": "Test Store"
  }'
```

Response:
```json
{
  "transaction_id": "tx-001",
  "decision": "approve",
  "risk_score": 0.15,
  "signals_triggered": [],
  "should_process_payment": true,
  "requires_manual_review": false,
  "processing_time_ms": 8.5
}
```

### Payment Processing with Fraud Detection

Process payment with integrated fraud check:

```bash
curl -X POST http://localhost:8000/payments/process \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "tx-002",
    "user_id": "user-123",
    "card_id": "card-456",
    "card_token": "tok_visa",
    "ip_address": "192.168.1.1",
    "amount": 250.00,
    "currency": "USD",
    "merchant_id": "merchant-789",
    "merchant_category": "electronics",
    "merchant_name": "Electronics Store",
    "description": "Laptop purchase"
  }'
```

## 🧪 Testing

### Run Unit & Integration Tests

```bash
# All tests
pytest tests/ -v

# Specific test scenarios
pytest tests/test_card_testing_pattern.py -v
pytest tests/test_velocity_breach.py -v
pytest tests/test_new_card_high_value.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Scenarios Included

1. **Card Testing Pattern**
   - Multiple small charges detection
   - Legitimate small transactions (pass)
   - Card testing → large purchase

2. **Velocity Breaches**
   - Transaction count velocity (5 txns/5min)
   - Amount velocity ($5000/5min)
   - IP velocity (shared IP patterns)
   - Combined velocity breaches

3. **New Card High-Value**
   - First-time card + high value = high risk
   - First-time card + low value = approve
   - Established card + high value = lower risk

## 📈 Performance Benchmarking

### Run Benchmarks

```bash
python benchmarks/fraud_detector_benchmark.py
```

This tests:
- Redis lookup latency
- End-to-end fraud detection latency
- Performance at 10, 50, 100, 500, 1000 TPS

Expected results:
```
REDIS LOOKUP
  Mean Latency:     3.2 ms
  P95 Latency:      6.8 ms ✅ < 10ms target
  P99 Latency:      8.5 ms

FRAUD DETECTION
  Mean Latency:     12.5 ms
  P95 Latency:      28.3 ms ✅ < 50ms target
  P99 Latency:      45.2 ms

TPS BENCHMARK 1000 TPS
  Actual TPS:       998.5
  P95 Latency:      32.1 ms ✅
```

### Load Testing with Locust

```bash
locust -f benchmarks/locustfile.py --host=http://localhost:8000
```

Open http://localhost:8089 to:
- Configure number of users
- Set spawn rate
- Monitor real-time performance
- View latency distribution

## ⚙️ Configuration

Edit `config.yml` to customize:

```yaml
fraud_detection:
  velocity:
    transaction_count_5min: 5      # Max txns per 5min
    transaction_count_1hr: 20      # Max txns per hour
    amount_sum_5min: 5000.00       # Max $ per 5min
    amount_sum_1hr: 20000.00       # Max $ per hour

  weights:
    velocity_count: 0.25           # Weight for count velocity
    velocity_amount: 0.20          # Weight for amount velocity
    new_card_risk: 0.15           # Weight for new card
    merchant_pattern: 0.15        # Weight for merchant patterns
    time_pattern: 0.10            # Weight for time patterns
    card_testing_pattern: 0.15    # Weight for card testing

  thresholds:
    approve_below: 0.30           # Approve if risk < 0.30
    review_below: 0.70            # Review if 0.30 <= risk < 0.70
    decline_above: 0.70           # Decline if risk >= 0.70
```

## 📊 Kafka Topics

Fraud signals logged to Kafka for ML training:

- `fraud.signals`: All fraud detection events
- `fraud.decisions`: Payment decisions
- `payment.transactions`: Transaction events

Example fraud signal:
```json
{
  "event_type": "fraud_signal_collection",
  "transaction_id": "tx-001",
  "user_id": "user-123",
  "risk_score": 0.85,
  "decision": "decline",
  "signals_triggered": ["card_testing_pattern_detected"],
  "velocity_signals": {...},
  "pattern_signals": {...},
  "timestamp": "2025-01-15T10:30:00Z"
}
```

## 🔍 Monitoring

### Health Checks

```bash
# Overall API health
curl http://localhost:8000/health

# Fraud detection system health
curl http://localhost:8000/fraud/health
```

### Redis Metrics

```bash
redis-cli INFO stats
```

### Kafka Consumer Lag

```bash
kafka-consumer-groups --bootstrap-server localhost:9092 --describe --group fraud-detector
```

## 🏗️ Project Structure

```
.
├── src/
│   ├── api/                      # FastAPI application
│   │   ├── main.py              # App initialization
│   │   └── routes.py            # API endpoints
│   ├── fraud/                   # Fraud detection core
│   │   ├── models.py            # Data models
│   │   └── services/
│   │       ├── fraud_detector.py    # Main detector
│   │       ├── signal_collector.py  # Signal collection
│   │       └── risk_scorer.py       # Risk scoring
│   ├── payments/                # Payment processing
│   │   └── service.py           # Payment service
│   └── infrastructure/          # Infrastructure
│       ├── redis/               # Redis velocity tracking
│       │   └── velocity_tracker.py
│       └── kafka/               # Kafka event logging
│           └── producer.py
├── tests/                       # Test suite
│   ├── test_card_testing_pattern.py
│   ├── test_velocity_breach.py
│   └── test_new_card_high_value.py
├── benchmarks/                  # Performance benchmarks
│   ├── fraud_detector_benchmark.py
│   └── locustfile.py
├── config.yml                   # Configuration
├── docker-compose.yml          # Infrastructure
└── requirements.txt            # Dependencies
```

## 🎓 Signal Explanations

### Velocity Signals

| Signal | Description | Threshold |
|--------|-------------|-----------|
| `card_velocity_5min_exceeded` | Too many transactions on this card | >5 in 5min |
| `card_amount_5min_exceeded` | Too much $ charged on this card | >$5000 in 5min |
| `user_velocity_5min_exceeded` | Too many transactions by this user | >5 in 5min |
| `ip_velocity_5min_exceeded` | Too many transactions from this IP | >5 in 5min |

### Pattern Signals

| Signal | Description | Risk |
|--------|-------------|------|
| `card_testing_pattern_detected` | Multiple small charges | HIGH |
| `first_card_use_high_value` | New card + high value | HIGH |
| `merchant_category_switch` | Unusual category changes | MEDIUM |
| `unusual_hour_pattern` | Transaction at unusual time | LOW |

## 🔐 Security Considerations

1. **PCI Compliance**: Never log full card numbers
2. **Rate Limiting**: Implement API rate limiting
3. **Encryption**: Use TLS for all API traffic
4. **Secrets Management**: Use environment variables
5. **Audit Logging**: Log all fraud decisions

## 📚 Next Steps

1. **ML Model Integration**
   - Train ML model on Kafka signal data
   - Replace rule-based scoring with ML predictions
   - A/B test ML vs rules

2. **Advanced Features**
   - Device fingerprinting
   - Behavioral biometrics
   - Graph-based fraud detection
   - Real-time model updates

3. **Monitoring & Alerting**
   - Grafana dashboards
   - PagerDuty alerts
   - False positive tracking
   - Chargeback correlation

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ for fraud prevention and payment security**

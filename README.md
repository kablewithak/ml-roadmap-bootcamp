# Production ML Observability Stack v2.0

**A complete, production-grade observability system for ML services**, demonstrating the "Three Pillars" of observability (Metrics, Traces, Logs) with real ML-specific monitoring patterns.

Built as a **learning-first resource** for engineers developing senior-level ML platform competence.

---

## 🎯 What You'll Learn

This project demonstrates **production ML engineering patterns** that separate senior engineers from junior ones:

### 1. **The "Three Pillars" Architecture**
- **Metrics** (Prometheus): "Are we meeting our SLOs?" (availability, latency, accuracy)
- **Traces** (Jaeger): "Why is THIS specific request slow?" (debugging)
- **Logs** (Loki): "What was the exact error message?" (forensics)
- **Correlation**: Jump from metric spike → trace → logs (2 min MTTR vs 2 hours)

### 2. **ML-Specific Observability** (Not in Standard APM Tools)
- Model performance drift (precision drops from 92% → 75% over time)
- Feature store monitoring (cache hit rate, staleness, data quality)
- Data drift detection (PSI > 0.25 triggers retraining)
- Business impact metrics (fraud blocked, revenue per model version)
- Prediction explainability (log feature importances for audits)

### 3. **Production Reliability Patterns**
- **SLO/Error Budget Management**: 99.9% availability = 43 min downtime/month
- **Multi-window Alerting**: Fast burn (5 min) vs slow burn (24 hour)
- **Chaos Engineering**: Kill Redis in prod → verify graceful degradation
- **Incident Runbooks**: MTTR < 10 minutes with step-by-step playbooks

### 4. **Trade-Off Thinking** (Critical for Senior Engineers)
Every decision documents:
- ✅ **Why this approach**: Business impact of doing it right
- ❌ **When NOT to use**: Anti-patterns and failure modes
- 💰 **Cost analysis**: $500/mo (self-hosted) vs $5K/mo (DataDog)
- 🔄 **Alternatives**: 2-3 other approaches with pros/cons

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FRAUD DETECTION API (FastAPI)                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │ HTTP Handler │───▶│ Feature Store│───▶│  ML Model    │     │
│  │  /v1/predict │    │   (Redis)    │    │ (Fraud Det.) │     │
│  └──────┬───────┘    └──────────────┘    └──────────────┘     │
│         │ OpenTelemetry SDK (auto-instrumentation)            │
│         ├─────────────────────┬───────────────┬───────────┐   │
└─────────┼─────────────────────┼───────────────┼───────────┼───┘
          ▼                     ▼               ▼           ▼
     ┌────────┐           ┌──────────┐    ┌───────┐   ┌────────┐
     │METRICS │           │  TRACES  │    │ LOGS  │   │BUSINESS│
     │(Prom.) │           │ (Jaeger) │    │(Loki) │   │EVENTS  │
     │        │           │          │    │       │   │(Kafka) │
     └────┬───┘           └─────┬────┘    └───┬───┘   └────┬───┘
          │                     │              │            │
          │           ┌─────────▼──────────────▼────────────┘
          │           │  OpenTelemetry Collector (batching, sampling)
          │           └─────────┬──────────────┬
          └─────────────────────┘              │
                                                ▼
                                          ┌──────────┐
                                          │ Grafana  │
                                          │ + Alerts │
                                          └──────────┘
```

**Key Components**:
- **Fraud API**: ML service with OpenTelemetry instrumentation
- **Feature Store**: Redis-backed feature cache (95% hit rate SLO)
- **OTel Collector**: Telemetry aggregation hub (batching, sampling, routing)
- **Prometheus**: Metrics storage (30-day retention)
- **Jaeger**: Distributed tracing (10% sampling)
- **Loki**: Log aggregation (structured JSON logs)
- **Grafana**: Unified visualization (metrics + traces + logs)
- **Alertmanager**: Alert routing and deduplication

---

## 🚀 Quick Start (5 Minutes)

### Prerequisites
- Docker & Docker Compose (v2.0+)
- **8GB RAM minimum** (16GB recommended for load testing)
- 10GB free disk space

### 1. Start the Stack

```bash
# Clone repository
git clone <repo-url>
cd ml-roadmap-bootcamp

# Start all services
docker-compose up -d

# Wait for services to be healthy (~60 seconds)
docker-compose ps

# Expected output: All services "Up" and "healthy"
```

### 2. Access UIs

| Service | URL | Purpose |
|---------|-----|---------|
| **Fraud API** | http://localhost:8000 | ML prediction endpoint |
| **API Docs** | http://localhost:8000/docs | Interactive Swagger UI |
| **Grafana** | http://localhost:3000 | Dashboards (admin/admin) |
| **Prometheus** | http://localhost:9090 | Metrics query interface |
| **Jaeger** | http://localhost:16686 | Distributed tracing UI |
| **Alertmanager** | http://localhost:9093 | Alert management |

### 3. Generate Traffic

```bash
# Populate feature cache for user 12345
curl -X POST http://localhost:8000/debug/populate-cache/12345

# Make fraud prediction
curl -X POST http://localhost:8000/v1/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "transaction_id": "txn_demo_001",
    "user_id": 12345,
    "amount": 149.99,
    "payment_method": "card",
    "merchant_id": "merchant_42"
  }'

# Expected response:
# {
#   "transaction_id": "txn_demo_001",
#   "decision": "approve",  # or "review" or "block"
#   "fraud_probability": 0.12,
#   "risk_level": "low",
#   "latency_ms": 45.2,
#   "model_version": "1.0.0",
#   "trace_id": "abc123..."  # Use this to find trace in Jaeger!
# }
```

### 4. Explore Observability

#### **Metrics** → Prometheus
```bash
# Open http://localhost:9090
# Query: rate(predictions_total[5m])
# See: Prediction rate over time (requests/second)

# Query: histogram_quantile(0.95, sum(rate(prediction_latency_seconds_bucket[5m])) by (le))
# See: p95 prediction latency
```

#### **Traces** → Jaeger
```bash
# Open http://localhost:16686
# 1. Service: fraud-detection-api
# 2. Operation: POST /v1/predict
# 3. Click any trace

# You'll see:
# - Total request duration: 47ms
#   ├─ fetch_features: 3ms (Redis lookup)
#   ├─ model_inference: 12ms (ML prediction)
#   └─ apply_business_rules: 1ms
```

#### **Logs** → Grafana + Loki
```bash
# Open http://localhost:3000 → Explore
# 1. Select datasource: Loki
# 2. Query: {container="fraud-api"} | json | level="INFO"
# 3. Click any log line
# 4. Click trace_id → Opens Jaeger trace!

# Advanced queries:
# Errors only:
{container="fraud-api"} | json | level="ERROR"

# High-value transactions:
{container="fraud-api"} | json | amount > 1000

# Specific user:
{container="fraud-api"} | json | user_id="12345"
```

#### **Dashboards** → Grafana
```bash
# Open http://localhost:3000
# Dashboards → ML Observability folder

# Pre-built dashboards:
# 1. ML Model Performance
#    - Prediction volume, latency, accuracy
#    - Feature drift (PSI metrics)
# 2. Business Metrics
#    - Revenue, fraud blocked, approval rates
# 3. Service Health
#    - SLO compliance, error budget, cache hit rate
```

---

## 📊 Example: Debugging a Production Issue

**Scenario**: Customer reports "Payment failed at 2:34 PM"

### Traditional Debugging (2 hours):
1. grep logs for user_id → 10K log lines
2. Try to reconstruct request flow from timestamps
3. Guess which service failed
4. Still not sure what happened

### With Observability (2 minutes):
1. **Grafana Logs**: `{container="fraud-api"} | json | user_id="12345" | timestamp > 2024-01-15T14:30:00Z`
2. **Find error**: "Redis connection timeout"
3. **Click trace_id** in log → Opens Jaeger
4. **See span**: `feature_store.get` took 5 seconds (timeout)
5. **Root cause**: Redis was restarting at 2:34 PM
6. **Fix**: Increase Redis restart timeout, add circuit breaker

**Time to root cause**: 2 minutes vs 2 hours = **60x faster MTTR**

---

## 📁 Project Structure (Annotated)

```
ml-roadmap-bootcamp/
├── src/
│   ├── observability/                    # 🔭 Core instrumentation
│   │   ├── instrumentation.py            # OpenTelemetry setup (read this first!)
│   │   │   ├── initialize_observability() # One-line setup
│   │   │   ├── BusinessMetrics class     # Custom Prometheus metrics
│   │   │   └── get_trace_context()       # Correlation IDs
│   │   └── logging_config.py             # Structured JSON logging
│   │       ├── CorrelationJsonFormatter  # Injects trace_id into logs
│   │       └── scrub_sensitive_data()    # PII redaction
│   ├── services/
│   │   ├── fraud_api.py                  # 🎯 Main FastAPI service
│   │   │   ├── /v1/predict endpoint      # Fraud detection
│   │   │   ├── /health/* endpoints       # K8s probes
│   │   │   └── /metrics endpoint         # Prometheus scrape
│   │   └── feature_store.py              # Redis-backed cache
│   │       ├── get_features()            # Batch feature fetch
│   │       ├── Cache miss handling       # Graceful degradation
│   │       └── Observability spans       # Custom tracing
│   └── models/
│       └── fraud_model.py                # 🤖 Logistic regression model
│           ├── predict_proba()           # Fraud probability (0-1)
│           ├── train()                   # Model training
│           └── Threshold tuning          # Precision/recall trade-off
├── config/
│   ├── prometheus/
│   │   ├── prometheus.yml                # Scrape configuration
│   │   ├── alerts.yml                    # 🚨 20+ alerting rules
│   │   │   ├── ML-specific alerts        # FeatureDriftDetected, ModelDegraded
│   │   │   ├── SLO alerts                # Fast/slow burn rates
│   │   │   └── Business alerts           # RevenueDrop, FraudSpike
│   │   └── alertmanager.yml              # Alert routing (Slack/PagerDuty)
│   ├── grafana/
│   │   ├── provisioning/
│   │   │   └── datasources/              # Auto-provisioned (Prom, Loki, Jaeger)
│   │   └── dashboards/
│   │       └── README.md                 # 📊 Dashboard definitions (PromQL)
│   ├── loki/
│   │   ├── loki-config.yaml              # Log storage (30-day retention)
│   │   └── promtail-config.yaml          # Log shipping (Docker containers)
│   └── otel-collector-config.yaml        # Telemetry pipeline
│       ├── Receivers (OTLP gRPC)
│       ├── Processors (batching, sampling)
│       └── Exporters (Jaeger, Prometheus)
├── docker/
│   └── Dockerfile.fraud-api              # Multi-stage build (400MB image)
├── docs/
│   ├── slo-definitions.md                # 📐 SLOs and error budgets
│   │   ├── 99.9% availability SLO        # 43 min/month downtime
│   │   ├── p95 latency <100ms SLO        # Customer experience
│   │   └── Error budget policy           # When to freeze deploys
│   └── runbooks/                         # 🚒 Incident response
│       ├── high-error-rate.md            # 5-whys, rollback procedure
│       └── (template for more runbooks)
├── chaos/
│   ├── chaos-scenarios.md                # 💥 Deliberate failure injection
│   │   ├── Redis failure                 # Verify graceful degradation
│   │   ├── OTel Collector down           # App continues, no observability
│   │   ├── Memory pressure               # OOMKill recovery
│   │   └── Network partition             # Fail-fast vs timeout
│   └── load_test.py                      # 🔥 Locust load testing
│       ├── Realistic traffic model       # 70% approve, 5% fraud
│       ├── SLO validation                # Fails CI if p95 >500ms
│       └── Business logic tests          # False positive detection
├── docker-compose.yml                    # 🐳 Full stack (10 services)
│   ├── fraud-api                         # ML service
│   ├── redis                             # Feature store
│   ├── otel-collector                    # Telemetry hub
│   ├── jaeger                            # Tracing
│   ├── prometheus                        # Metrics
│   ├── alertmanager                      # Alerts
│   ├── loki + promtail                   # Logs
│   ├── grafana                           # Dashboards
│   └── kafka + zookeeper (optional)      # Event streaming
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```

---

## 🎓 Learning Path (20 Hours to Production Competence)

### Day 1: Foundations (4 hours)
**Goal**: Understand the three pillars

- [ ] **Hour 1**: Start stack, generate traffic
  - Run `docker-compose up -d`
  - Make 100 API calls (see Quick Start)
  - View each UI (Grafana, Prometheus, Jaeger)

- [ ] **Hour 2**: Read core instrumentation
  - `src/observability/instrumentation.py` (600 lines, heavily commented)
  - Understand: Tracer, Meter, SpanProcessor, MetricReader
  - Question: "Why BatchSpanProcessor vs SimpleSpanProcessor?"

- [ ] **Hour 3**: Add custom metric
  - Add: `transaction_country` label to `predictions_total`
  - Query in Prometheus: `sum by (transaction_country) (rate(predictions_total[5m]))`
  - Create Grafana panel showing predictions by country

- [ ] **Hour 4**: Debug with traces
  - Trigger slow request (kill Redis, make API call, restart Redis)
  - Find slow trace in Jaeger
  - See which span took longest
  - Correlate trace_id with logs in Loki

**Learning Check**: Explain to a colleague: "How does OpenTelemetry correlation work?"

---

### Day 2: ML-Specific Monitoring (4 hours)
**Goal**: Instrument ML-specific metrics

- [ ] **Hour 5**: Feature drift detection
  - Read: `docs/slo-definitions.md` (PSI explanation)
  - Add PSI calculation batch job
  - Push `feature_psi` gauge to Prometheus
  - Create Grafana alert: PSI > 0.25

- [ ] **Hour 6**: Model performance tracking
  - Implement: Weekly batch job that labels 1000 random predictions
  - Calculate: Precision, recall, F1
  - Push: `model_performance{metric="precision"}` gauge
  - Dashboard: Model performance over time (detect degradation)

- [ ] **Hour 7**: Business impact metrics
  - Add: `revenue_per_model_version` (A/B test new model)
  - Add: `false_positive_cost` (blocked legitimate transactions)
  - Calculate: ROI of improving model from 90% → 95% precision

- [ ] **Hour 8**: Prediction explainability
  - Log: Top 3 features that contributed to fraud decision
  - Store in structured logs: `{"feature_importances": {"amount": 0.4, "velocity": 0.3, ...}}`
  - Use for: Customer inquiries ("Why was I flagged?")

---

### Day 3: Reliability Engineering (4 hours)
**Goal**: Build confidence in system resilience

- [ ] **Hour 9**: Define SLOs
  - Read: `docs/slo-definitions.md`
  - Define: Your own SLOs (availability, latency, accuracy)
  - Calculate: Error budget (how much downtime can you afford?)

- [ ] **Hour 10**: Set up alerting
  - Review: `config/prometheus/alerts.yml` (20+ alerts)
  - Customize: Thresholds for your use case
  - Test: Trigger alert (kill service, verify Alertmanager fires)

- [ ] **Hour 11**: Write runbook
  - Pick: One alert (e.g., HighPredictionLatency)
  - Write: Runbook in `docs/runbooks/high-latency.md`
  - Include: Triage steps, common causes, rollback procedure

- [ ] **Hour 12**: Practice incident response
  - Simulate: Redis failure (stop container)
  - Follow: Runbook you wrote
  - Measure: MTTR (time from alert → mitigation)
  - Goal: <10 minutes

---

### Day 4: Chaos Engineering (4 hours)
**Goal**: Find weaknesses before customers do

- [ ] **Hour 13**: Read chaos philosophy
  - Read: `chaos/chaos-scenarios.md`
  - Understand: Hypothesis → Experiment → Measure → Learn

- [ ] **Hour 14**: Run Scenario 1 (Redis failure)
  - Hypothesis: "API degrades gracefully (returns 200, uses defaults)"
  - Execute: `docker stop redis`
  - Observe: Error rate, latency, logs
  - Validate: Does API stay under 500ms p95?

- [ ] **Hour 15**: Run Scenario 3 (Memory pressure)
  - Trigger: Memory leak endpoint
  - Observe: Alert fires at 85% memory
  - Observe: Container OOMKilled, auto-restarts
  - Fix: Add memory leak detection

- [ ] **Hour 16**: Run load test
  - Execute: `locust -f chaos/load_test.py --users 1000`
  - Find: Breaking point (at what RPS does latency spike?)
  - Optimize: Hot paths (profiling shows bottlenecks)
  - Result: "System handles 500 RPS at p95 <100ms"

---

### Day 5: Production Deployment (4 hours)
**Goal**: Deploy to production safely

- [ ] **Hour 17**: Security hardening
  - Enable: TLS (HTTPS, not HTTP)
  - Add: Authentication (API keys)
  - Scrub: PII from logs (credit cards, SSNs)
  - Rotate: Secrets (database passwords)

- [ ] **Hour 18**: Scale configuration
  - Add: Horizontal Pod Autoscaler (K8s)
  - Configure: Prometheus remote write (Thanos/Cortex)
  - Set up: Loki S3 backend (not local filesystem)
  - Enable: Redis persistence (AOF + RDB)

- [ ] **Hour 19**: Set up on-call
  - Configure: Alertmanager → PagerDuty
  - Define: Escalation policy (30 min → page manager)
  - Create: Runbooks for all critical alerts
  - Schedule: Chaos tests (weekly, during business hours)

- [ ] **Hour 20**: Disaster recovery drill
  - Simulate: Multi-service failure (Redis + Prometheus down)
  - Practice: Runbook execution
  - Measure: RTO (Recovery Time Objective): <10 min
  - Measure: RPO (Recovery Point Objective): <5 min data loss

---

## 🔧 Common Issues & Solutions

### "Services won't start"
```bash
# Check logs
docker-compose logs -f fraud-api

# Common causes:
# 1. Port conflict (8000 already in use)
lsof -i :8000  # Find process
docker-compose down && docker-compose up -d  # Restart

# 2. Out of memory
# Increase Docker memory: Settings → Resources → 8GB

# 3. Model file missing
docker-compose exec fraud-api python -m models.fraud_model  # Train model
```

### "No metrics in Prometheus"
```bash
# Check metrics endpoint
curl http://localhost:8000/metrics

# Should return: predictions_total{...} 0.0

# If empty, check OTel Collector
docker-compose logs otel-collector | grep error

# Common: OTLP_ENDPOINT misconfigured
docker-compose exec fraud-api env | grep OTLP
# Should be: http://otel-collector:4317
```

### "No traces in Jaeger"
```bash
# Verify trace sampling
# By default, 10% of requests are traced

# Force sampling: Set env var
OTEL_TRACES_SAMPLER=always_on

# Check OTel Collector → Jaeger connection
curl http://localhost:16686/api/services
# Should include: "fraud-detection-api"
```

### "Grafana can't reach Prometheus"
```bash
# Check datasource
docker-compose exec grafana cat /etc/grafana/provisioning/datasources/datasources.yaml

# Test connection
curl http://prometheus:9090/api/v1/query?query=up
# Should return: {"status":"success", ...}

# If fails, check network
docker network inspect ml-observability
# All services should be connected
```

---

## 💰 Cost Analysis: Self-Hosted vs Vendors

### Self-Hosted (This Stack)

**Infrastructure** (AWS t3.medium × 3):
- Compute: 3 × $30/month = $90/month
- Storage: 100GB × $0.10/GB = $10/month
- **Total: $100/month**

**Engineering Time**:
- Setup: 20 hours (one-time)
- Maintenance: 4 hours/month
- At $100/hour: $400/month ongoing

**Total Cost**: $500/month

---

### DataDog (APM Vendor)

**Pricing**:
- Infrastructure monitoring: 3 hosts × $15/host = $45/month
- APM: 3 hosts × $31/host = $93/month
- Custom metrics: 10K metrics × $0.05/metric = $500/month
- Logs: 100GB × $0.10/GB = $10/month (no retention beyond 15 days)
- **Total: ~$650/month base**

**But**:
- Every new custom metric: +$0.05/month
- At 10K predictions/day with 20 features: 200K unique time series
- Custom metrics: 200K × $0.05 = **$10,000/month** 😱

**Typical ML company**: $5K-15K/month (grows with usage)

---

### Verdict

| Criteria | Self-Hosted | DataDog |
|----------|-------------|---------|
| **Cost at scale** | ~$500/mo | $5K-15K/mo |
| **Setup time** | 20 hours | 2 hours |
| **Maintenance** | 4 hours/mo | 0 hours |
| **ML-specific features** | Custom (unlimited) | Limited |
| **Data control** | Full (GDPR compliant) | Vendor stores data |
| **Learning value** | High (senior skill) | Low (black box) |

**When to self-host**:
- ✅ >$5M ARR (cost savings justify eng time)
- ✅ Compliance requirements (HIPAA, PCI-DSS)
- ✅ High-cardinality ML metrics (per-feature PSI)
- ✅ Learning goal (this project!)

**When to use vendor**:
- ✅ <$1M ARR (speed > cost)
- ✅ Team <5 engineers (no bandwidth for ops)
- ✅ Non-ML use case (standard metrics)

---

## 📚 Deep Dives (for Curious Engineers)

### How does trace correlation work?

**The Magic**: Every log line has a `trace_id` field

1. **Request arrives**: OpenTelemetry creates `TraceContext` (trace_id, span_id)
2. **Context propagates**: Passed to all function calls (threading.local)
3. **Logger injects**: `CorrelationJsonFormatter` adds trace_id to every log
4. **Loki indexes**: Logs queryable by `{trace_id="abc123"}`
5. **Grafana links**: Click trace_id → Opens Jaeger

**Code**:
```python
# observability/logging_config.py:CorrelationJsonFormatter
span = trace.get_current_span()
log_record['trace_id'] = format(span.get_span_context().trace_id, '032x')
```

**Business Impact**: Jump from log → trace in 1 click (vs 30 min grepping logs)

---

### Why OpenTelemetry vs vendor SDKs?

**Vendor Lock-in Problem**:
- Start with DataDog → `import datadog`
- Switch to New Relic → Rewrite all instrumentation
- 100 hours of eng time

**OpenTelemetry Solution**:
- One instrumentation: `import opentelemetry`
- Swap backends in config (no code changes)
- Export to: Jaeger, DataDog, New Relic, Honeycomb, etc.

**Standard Adoption**:
- CNCF project (like Kubernetes)
- Supported by: Google, Microsoft, AWS, Grafana, DataDog

**Learning Value**: Understanding OTel is portable across companies

---

### When is Loki better than Elasticsearch?

**Loki Wins**:
- ✅ Structured logs (JSON with consistent labels)
- ✅ Cost-sensitive (10x cheaper storage)
- ✅ Simple ops (no shard management)
- ✅ Grafana integration (unified UI)

**Elasticsearch Wins**:
- ✅ Full-text search (grep across all fields)
- ✅ Unstructured logs (Apache access logs, Java stack traces)
- ✅ Complex aggregations (Kibana analytics)

**This Project Uses Loki Because**:
- We emit structured JSON logs (easy to query)
- Labels are low-cardinality (container, level, logger)
- Query pattern: "Show me errors for trace_id X" (label-based, fast)

---

## 🤝 Contributing

This is a **learning resource**. Contributions that improve educational value are welcome:

### High-Value Contributions
- ✅ New chaos scenarios (with hypothesis + expected behavior)
- ✅ Runbooks for additional alerts
- ✅ Grafana dashboard exports (with PromQL explanations)
- ✅ Trade-off analyses ("When NOT to use X")

### What We Won't Merge
- ❌ "Upgrade dependency X" (unless security fix)
- ❌ "Replace Y with Z" (without trade-off analysis)
- ❌ Removing comments (code is intentionally over-commented for learning)

### Contribution Checklist
- [ ] Includes "Why" explanation (not just "What")
- [ ] Documents trade-offs (pros + cons)
- [ ] Adds test case (if code change)
- [ ] Updates relevant documentation

---

## 📜 License

MIT License - Use for learning, production, or teaching

---

## 🙏 Acknowledgments

**Patterns learned from**:
- **Google SRE**: SLO/error budget framework, incident management
- **Netflix**: Chaos engineering, failure injection in production
- **Grafana Labs**: Loki design, unified observability
- **OpenTelemetry Community**: Instrumentation best practices

**Books that influenced this**:
- *Site Reliability Engineering* (Google)
- *Designing Data-Intensive Applications* (Martin Kleppmann)
- *The Phoenix Project* (Gene Kim)

---

**Built with ❤️ for ML engineers learning production systems**

*"The difference between junior and senior engineers is that seniors know WHY, not just HOW."*

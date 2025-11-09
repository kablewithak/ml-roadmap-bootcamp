# Load Testing & Chaos Engineering Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LOAD TESTING FRAMEWORK                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐         ┌──────────────────┐                 │
│  │  Test Orchestrator│────────▶│  Ground Truth    │                 │
│  │                  │         │  Ledger          │                 │
│  │  - Generate IDs  │         │  (JSON/Redis)    │                 │
│  │  - Schedule tests│         └──────────────────┘                 │
│  │  - Reconcile     │                                               │
│  └────────┬─────────┘                                               │
│           │                                                          │
│           ▼                                                          │
│  ┌──────────────────────────────────────────────────────┐          │
│  │         LOAD GENERATORS (Locust Workers)              │          │
│  │                                                        │          │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐│          │
│  │  │Worker 1 │  │Worker 2 │  │Worker 3 │  │Worker N ││          │
│  │  │ (CPU 1) │  │ (CPU 2) │  │ (CPU 3) │  │ (CPU N) ││          │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘│          │
│  └───────┼────────────┼────────────┼────────────┼──────┘          │
│          │            │            │            │                   │
└──────────┼────────────┼────────────┼────────────┼──────────────────┘
           │            │            │            │
           ▼            ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    SYSTEMS UNDER TEST                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────┐        ┌──────────────────────┐          │
│  │  PAYMENT SYSTEM      │        │  KAFKA STREAMING     │          │
│  │                      │        │                      │          │
│  │  FastAPI             │        │  Producer            │          │
│  │    ↓                 │        │    ↓                 │          │
│  │  Stripe API          │        │  Redpanda/Kafka      │          │
│  │    ↓                 │        │    ↓                 │          │
│  │  PostgreSQL          │        │  Consumer            │          │
│  │    ↓                 │        │    ↓                 │          │
│  │  Redis (idempotency) │        │  PostgreSQL/Redis    │          │
│  └──────────────────────┘        └──────────────────────┘          │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
           │            │            │            │
           ▼            ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    CHAOS INJECTION LAYER                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │
│  │ DB Killer   │  │ Network     │  │ Disk Fill   │                │
│  │ (pg_ctl)    │  │ Latency     │  │ (dd/fallocate)│              │
│  │             │  │ (tc qdisc)  │  │             │                │
│  └─────────────┘  └─────────────┘  └─────────────┘                │
│                                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │
│  │ Kafka Killer│  │ Memory      │  │ Clock Skew  │                │
│  │ (kill)      │  │ Pressure    │  │ (libfaketime)│               │
│  │             │  │ (stress-ng) │  │             │                │
│  └─────────────┘  └─────────────┘  └─────────────┘                │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    MONITORING & VERIFICATION                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │ Prometheus   │  │ Grafana      │  │ Reconciliation│            │
│  │              │  │              │  │ Engine        │            │
│  │ - Latency    │  │ - Dashboards │  │               │            │
│  │ - Throughput │  │ - Alerts     │  │ - Zero loss   │            │
│  │ - Error rate │  │ - Reports    │  │ - Zero dup    │            │
│  └──────────────┘  └──────────────┘  │ - Consistency │            │
│                                       └──────────────┘             │
└──────────────────────────────────────────────────────────────────────┘
```

## Component Breakdown

### 1. Test Orchestrator
**Purpose:** Centralized control plane for test execution

**Responsibilities:**
- Generate ground-truth ledger of test message IDs
- Schedule test execution phases (ramp-up, steady-state, chaos, ramp-down)
- Coordinate chaos injection timing
- Execute post-test reconciliation
- Generate reports

**Technology:** Python script + Redis for distributed state

**Key Files:**
- `orchestrator/test_runner.py` - Main orchestration logic
- `orchestrator/ledger.py` - Ground truth management
- `orchestrator/reconciliation.py` - Post-test verification

### 2. Ground Truth Ledger
**Purpose:** Source of truth for verifying exactly-once processing

**Why Redis + JSON?**
- **Redis:** Fast lookups during test (O(1) for checking if ID processed)
- **JSON file:** Persistent audit trail (survives Redis restart)
- **Redundancy:** Can rebuild Redis from JSON if needed

**Data Structure:**
```json
{
  "test_run_id": "test-2024-01-15-001",
  "total_messages": 10000,
  "message_ids": [
    "pay_001_a1b2c3d4",
    "pay_002_e5f6g7h8",
    ...
  ],
  "metadata": {
    "target_tps": 1000,
    "duration_seconds": 60,
    "test_type": "payment_system_spike"
  }
}
```

**Reconciliation Algorithm:**
```python
# After test completes
ground_truth = set(load_from_ledger())
db_records = set(query_postgresql())
stripe_records = set(query_stripe_api())

# Verification
missing = ground_truth - db_records  # Should be empty
duplicates = find_duplicates(db_records)  # Should be empty
mismatches = db_records ^ stripe_records  # Should be empty
```

### 3. Load Generators (Locust Workers)

**Why Locust?**
- **Python-based:** Reuse FastAPI/Stripe client code
- **Distributed:** Run multiple workers for high TPS
- **Real-time metrics:** See p50/p95/p99 latency during test

**Worker Scaling:**
| Target TPS | Workers | vCPUs | RAM | Estimated Cost |
|-----------|---------|-------|-----|----------------|
| 100       | 1       | 2     | 2GB | $0 (local)     |
| 500       | 2       | 4     | 4GB | $20/mo (VPS)   |
| 1000      | 4       | 8     | 8GB | $40/mo (VPS)   |
| 5000      | 10      | 20    | 20GB| $200/mo (cloud)|

**Architecture Pattern: Master-Worker**
```
┌──────────────┐
│ Locust Master│  ← Web UI (localhost:8089)
│              │  ← Aggregates metrics
└──────┬───────┘
       │
       ├──────────┬──────────┬──────────┐
       │          │          │          │
┌──────▼───┐ ┌───▼──────┐ ┌─▼────────┐ │
│ Worker 1 │ │ Worker 2 │ │ Worker N │ │
│ (333 TPS)│ │ (333 TPS)│ │ (334 TPS)│ │
└──────────┘ └──────────┘ └──────────┘ │
                                        │
            Total: 1000 TPS ────────────┘
```

### 4. Chaos Injection Layer

**Design Principle: Controlled Chaos**

**NOT:** Random failures at random times (too risky)
**YES:** Scheduled failures with defined recovery expectations

**Chaos Scenario Example:**
```yaml
test_phases:
  - phase: warmup
    duration: 60s
    tps: 100
    chaos: none

  - phase: steady_state
    duration: 120s
    tps: 1000
    chaos: none

  - phase: database_failure  # ← Inject chaos here
    duration: 30s
    tps: 1000
    chaos:
      type: postgres_kill
      timing: 15s  # Kill DB after 15s into phase
      recovery_time: 10s  # DB should auto-restart in 10s

  - phase: recovery_verification
    duration: 60s
    tps: 1000
    chaos: none
    expected:
      - zero_message_loss  # All messages from chaos phase should be in DB
      - max_p99_latency: 2000ms  # Some requests will retry, but should complete
```

**Chaos Tools:**

| Failure Type | Tool | Command | Blast Radius |
|--------------|------|---------|--------------|
| DB kill | Docker | `docker pause postgres` | Test env only |
| Kafka kill | Docker | `docker pause redpanda` | Test env only |
| Network latency | `tc` | `tc qdisc add dev eth0 root netem delay 100ms` | Container only |
| Disk full | `dd` | `dd if=/dev/zero of=/var/lib/postgres/fill bs=1G count=10` | Test volume only |
| Memory pressure | `stress-ng` | `stress-ng --vm 2 --vm-bytes 80%` | Container only |

**Safety Mechanisms:**
1. **Isolated Docker network** - Can't affect host or other services
2. **Resource limits** - Each container capped (e.g., max 2GB RAM)
3. **Automatic cleanup** - Chaos reverted after test (or timeout)
4. **Kill switch** - `Ctrl+C` stops all chaos and cleans up

### 5. Monitoring & Verification

**Three-Tier Monitoring:**

**Tier 1: Real-Time (During Test)**
- Locust web UI: http://localhost:8089
- Metrics: Current TPS, response times, error rate
- Purpose: Detect failures immediately

**Tier 2: Time-Series (Post-Test Analysis)**
- Prometheus + Grafana
- Metrics: p50/p95/p99/p999 latency, throughput over time
- Purpose: Identify performance degradation patterns

**Tier 3: Correctness (Reconciliation)**
- Custom Python scripts
- Verify: Zero loss, zero duplicates, consistency
- Purpose: Prove exactly-once guarantees

**Critical Metrics:**

| Metric | Target | Alert Threshold | Business Impact |
|--------|--------|-----------------|-----------------|
| **p95 latency** | <500ms | >1000ms | Customer frustration → cart abandonment |
| **p99 latency** | <1000ms | >2000ms | Worst-case UX, impacts reviews |
| **Error rate** | <0.1% | >1% | Direct revenue loss (failed payments) |
| **Duplicate rate** | 0% | >0.01% | Chargebacks, customer trust loss |
| **Message loss** | 0% | >0% | Audit failures, compliance violations |
| **Kafka lag** | <1s | >10s | Fraud goes undetected |

## Business Impact Model

### Cost of Failure by Scenario

**Assumptions:**
- Average transaction: $100
- Peak TPS: 1000
- Business hours: 16 hrs/day
- Fraud rate (baseline): 0.3%

#### Scenario 1: Database Connection Pool Exhaustion
**Failure:** Connection pool maxes out at 50 connections, new requests rejected

**Impact Calculation:**
```
Outage duration: 15 minutes (time to detect + restart)
Lost transactions: 1000 TPS × 60 sec × 15 min = 900,000 transactions
Revenue loss: 900,000 × $100 = $90M

Realistic (accounting for retries): ~10% actual loss = $9M
```

**Prevention Cost:** Adding connection pool monitoring + auto-scaling: $0 (just config)

**ROI:** Infinite (free to fix, prevents $9M loss)

#### Scenario 2: Idempotency Failure → Duplicate Charges
**Failure:** Race condition in Redis lock causes 0.1% duplicate rate

**Impact Calculation:**
```
Daily transactions: 1000 TPS × 60 × 60 × 16 hrs = 57.6M
Duplicates: 57.6M × 0.001 = 57,600 duplicates/day
Duplicate charges: 57,600 × $100 = $5.76M/day

Stripe chargeback fee: 57,600 × $15 = $864K/day
Customer support: 57,600 × 5 min × $25/hr = $1.2M/day

Total daily cost: $7.8M
```

**Prevention Cost:** Load testing to find the bug: $50 (this framework)

**ROI:** 156,000x return

#### Scenario 3: Kafka Consumer Lag → Delayed Fraud Detection
**Failure:** Consumer can't keep up, lag grows to 60 seconds

**Impact Calculation:**
```
Transactions in 60s: 1000 TPS × 60 = 60,000
Fraud rate: 0.3% = 180 fraudulent transactions
Average fraud amount: $500 (fraudsters target high-value)
Loss: 180 × $500 = $90K per lag spike

If lag happens 3×/day during peaks: $270K/day
Annual: $98M
```

**Prevention Cost:** Load test to find consumer bottleneck: $50

**ROI:** 1,960,000x return

### ROI Summary Table

| Investment | Prevention | Annual Savings | ROI |
|------------|-----------|----------------|-----|
| $50 (this framework) | Duplicate charges | $2.8B | 56M× |
| $50 | Kafka lag fraud | $98M | 1.96M× |
| $50 | DB outages | $500M (conservative) | 10M× |

**Conclusion:** Even if this framework only prevents ONE incident, ROI is massive.

## Scaling Roadmap

### Phase 1: Local Testing ($0-20/month) ← YOU ARE HERE
- Single Locust worker on laptop
- Target: 100-500 TPS
- Purpose: Functional correctness, find obvious bugs
- **Timeline:** Week 1

### Phase 2: Single-Node Distributed ($20-50/month)
- Locust master + 2-4 workers on single VPS
- Target: 500-1000 TPS
- Purpose: Realistic load, find race conditions
- **Timeline:** Week 2-3

### Phase 3: Multi-Node Distributed ($200/month)
- Kubernetes cluster with 10+ workers
- Target: 5000+ TPS
- Purpose: Pre-production validation
- **Timeline:** Before launch

### Phase 4: Production Chaos ($500/test)
- AWS/K6 Cloud for one-time validation
- Target: 50,000+ TPS
- Purpose: Black Friday prep, investor demos
- **Timeline:** Major launches only

## When to Move to Next Phase

**Phase 1 → 2:**
- ✅ All tests pass locally
- ✅ Found and fixed >3 bugs
- ✅ You understand every component

**Phase 2 → 3:**
- ✅ Consistently hit 1000 TPS
- ✅ Zero duplicates/losses for 1 hour
- ✅ Have real users (>1000 DAU)

**Phase 3 → 4:**
- ✅ Raising funding (need impressive metrics)
- ✅ Launching major feature (Black Friday)
- ✅ Customer requires proof (enterprise sales)

## Next Steps

1. **Read:** `/load-testing/docs/BUSINESS_IMPACT.md` - Understand the "why"
2. **Build:** `/load-testing/setup/` - Set up infrastructure
3. **Test:** `/load-testing/tests/` - Run your first load test
4. **Learn:** Complete verification questions in each module

Let's build this framework step by step, with full understanding at each stage.

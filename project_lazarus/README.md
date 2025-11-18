# Project Lazarus: Causal Rejection Inference System

> "The loop that heals itself" - A closed-loop active learning system for credit decisioning

## Overview

Project Lazarus addresses the **rejection inference problem** in credit modeling: we never observe outcomes for rejected applicants, leading to models that are increasingly "blind" to certain population segments.

This system implements an **epsilon-greedy exploration strategy** with causal inference techniques to:
- Systematically explore the rejected population
- Weight observations using Inverse Probability Weighting (IPW)
- Retrain models to "heal" their blindness

## Architecture

```
[User App] → [API Gateway] → [Feature Store (Feast)] → [Traffic Router (Redis)] → [Inference Service] → [Decision]
```

### Components

1. **Traffic Router (The "Brain")** - Redis + Lua scripts for atomic budget management
2. **Safety Valve (The "Brakes")** - Compliance rules that override exploration
3. **Causal Training Pipeline (The "Healer")** - IPW-weighted model training

## Key Innovation: The Math

**Normal Loss Function:**
```
L = (y - ŷ)²
```

**Causal Loss Function (IPW):**
```
L = (1/P(approval)) × (y - ŷ)²
```

A loan approved in the "Explore" bucket (1% probability) is weighted **100x higher** than a normal loan, forcing the model to pay massive attention to these rare data points.

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+

### Run the System

```bash
# Start all services
docker-compose up -d

# Initialize the database
docker-compose exec api python -c "from src.models.database import init_database; init_database()"

# Run simulation (100k applications)
docker-compose exec api python scripts/lazarus_sim.py

# Train causal model
docker-compose exec api python -m src.training.causal_trainer
```

### Access Services

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Grafana**: http://localhost:3000 (admin/lazarus)
- **Prometheus**: http://localhost:9090

## The "Undeniable" Artifacts

### 1. Cost of Knowledge Dashboard

![Grafana Dashboard](artifacts/figures/grafana_dashboard.png)

Three key panels:
- **Current Loss Budget**: Starts at $1,000, ticks down as explore loans default
- **Model Blindness Score**: Confidence metric on rejected regions
- **Projected Revenue Lift**: `(New Good Loans × $500 LTV) - Cost of Exploration`

### 2. Shadow Mode Report

Comparison between Model_V1 (Standard) and Model_V2 (Causal):

> **Headline**: "How spending $450 on random approvals unlocked $12,000 in safe revenue"

### 3. The policy.py Code Block

```python
def decide_application(user_features, risk_score):
    # 1. SAFETY VALVE (Compliance)
    if is_legally_prohibited(user_features):
        return Decision.REJECT, reason="COMPLIANCE_BLOCK"

    # 2. EXPLORATION LOGIC (The "IQ Test" part)
    # Atomic decrement of exploration budget in Redis
    current_budget = redis.get("loss_budget")

    if current_budget > 0 and random.random() < EPSILON:
        # We are BUYING data. We force approval to learn.
        log_experiment_exposure(user_id, variant="EXPLORE")
        return Decision.APPROVE, reason="PROJECT_LAZARUS_EXPLORE"

    # 3. EXPLOITATION (Standard Business Logic)
    if risk_score > THRESHOLD:
        return Decision.APPROVE, reason="MODEL_QUALIFIED"

    return Decision.REJECT, reason="MODEL_HIGH_RISK"
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Language | Python 3.11 | Type-hinted, Pydantic models |
| API | FastAPI | Inference endpoint |
| State | Redis | Traffic routing, budget management |
| Database | PostgreSQL | Ground truth storage |
| Features | Feast | Feature serving |
| Monitoring | Grafana + Prometheus | Visualization |
| Infrastructure | Docker Compose | Microservices simulation |

## Project Structure

```
project_lazarus/
├── src/
│   ├── api/           # FastAPI inference service
│   ├── core/          # Policy engine, safety valve, router
│   ├── models/        # Risk model, database models
│   ├── training/      # Causal training pipeline
│   └── features/      # Feast feature definitions
├── scripts/
│   └── lazarus_sim.py # Simulation script
├── dashboards/        # Grafana dashboard configs
├── docker/            # Docker configurations
├── config/            # Application settings
└── tests/             # Unit tests
```

## Simulation Strategy

Since we can't wait 12 months for loan defaults, we simulate time:

1. Generate 100,000 synthetic applicants using Faker
2. **The Trick**: Inject a hidden variable (e.g., gambling addiction) the model doesn't see
3. Send each to the API for decision
4. Simulate default outcomes based on hidden + visible features

This generates ~1,000 "Explore" data points for causal analysis.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/decide` | POST | Make loan decision |
| `/health` | GET | Health check |
| `/ready` | GET | Readiness check |
| `/budget` | GET | Get exploration budget |
| `/budget/reset` | POST | Reset exploration budget |
| `/metrics` | GET | Prometheus metrics |
| `/exploration/metrics` | GET | Detailed exploration stats |

## Configuration

Environment variables (prefix `LAZARUS_`):

```bash
LAZARUS_EPSILON=0.01              # Exploration probability
LAZARUS_INITIAL_LOSS_BUDGET=1000  # Starting budget ($)
LAZARUS_DEFAULT_LTV=500           # Lifetime value per good loan
LAZARUS_RISK_THRESHOLD=0.5        # Risk score threshold
```

## Why This Matters

Traditional credit models suffer from **survivorship bias** - they only learn from approved loans. This leads to:

1. **Increasing conservatism** over time
2. **Lost revenue** from safe applicants who were rejected
3. **Unfairness** to certain demographic groups

Project Lazarus systematically addresses this by:
- Quantifying the "cost of knowledge"
- Making exploration a first-class business metric
- Using causal inference to debias the training data

## References

- Rejection Inference in Credit Scoring (Hand & Henley, 1993)
- Counterfactual Fairness (Kusner et al., 2017)
- Inverse Probability Weighting (Rosenbaum & Rubin, 1983)

## License

MIT License - See LICENSE file for details.

---

*Project Lazarus - Because every rejected applicant is a lost opportunity to learn.*

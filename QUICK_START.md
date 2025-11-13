# Quick Start Guide (5 Minutes)

## Prerequisites
- Docker & Docker Compose
- 8GB RAM minimum
- 10GB free disk

## Option 1: Automated Setup (Recommended)

```bash
# Make script executable (if not already)
chmod +x scripts/start.sh

# Start everything
./scripts/start.sh

# Or rebuild containers first
./scripts/start.sh --build
```

## Option 2: Manual Setup

```bash
# Start all services
docker-compose up -d

# Wait 60 seconds for services to be healthy
docker-compose ps

# Check logs if any service failed
docker-compose logs fraud-api
```

## Verify Stack is Running

```bash
# Check all services are up
docker-compose ps

# Expected: All services show "Up" and "healthy"
```

## Make Your First Prediction

```bash
# 1. Populate feature cache
curl -X POST http://localhost:8000/debug/populate-cache/12345

# 2. Make fraud prediction
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "quick_start_001",
    "user_id": 12345,
    "amount": 99.99,
    "payment_method": "card",
    "merchant_id": "merchant_1"
  }'

# Expected response:
# {
#   "transaction_id": "quick_start_001",
#   "decision": "approve",
#   "fraud_probability": 0.15,
#   "risk_level": "low",
#   "latency_ms": 42.5,
#   "model_version": "1.0.0",
#   "trace_id": "..."
# }
```

## Explore Observability

### 1. View Metrics (Prometheus)
- Open: http://localhost:9090
- Query: `rate(predictions_total[5m])`
- See: Prediction requests per second

### 2. View Traces (Jaeger)
- Open: http://localhost:16686
- Service: `fraud-detection-api`
- Click any trace to see request flow

### 3. View Logs (Grafana + Loki)
- Open: http://localhost:3000 (admin/admin)
- Go to: Explore → Select Loki
- Query: `{container="fraud-api"} | json | level="INFO"`

### 4. View Dashboards (Grafana)
- Open: http://localhost:3000
- Dashboards → ML Observability folder
- Pre-built dashboards for ML, business, and system metrics

## Troubleshooting

### Services won't start
```bash
# Check logs
docker-compose logs -f fraud-api

# Common fixes:
# 1. Increase Docker memory to 8GB
#    Docker Desktop → Settings → Resources
# 2. Stop conflicting services
lsof -i :8000  # Check if port 8000 is in use
```

### No metrics showing
```bash
# Verify metrics endpoint works
curl http://localhost:8000/metrics

# Should return Prometheus metrics
# If not, check fraud-api logs
```

### Model file missing
```bash
# Train demo model
docker-compose exec fraud-api python -m models.fraud_model

# Restart service
docker-compose restart fraud-api
```

## Stop Everything

```bash
# Stop services (keeps data)
./scripts/start.sh --stop

# Or manually
docker-compose down

# Stop and delete all data
./scripts/start.sh --clean
```

## Next Steps

1. **Read the README**: Full documentation in [README.md](./README.md)
2. **Learn Observability**: Follow the 20-hour learning path
3. **Run Load Tests**: `locust -f chaos/load_test.py`
4. **Try Chaos Scenarios**: See [chaos/chaos-scenarios.md](./chaos/chaos-scenarios.md)

## Quick Reference

| What | URL |
|------|-----|
| Fraud API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Grafana | http://localhost:3000 (admin/admin) |
| Prometheus | http://localhost:9090 |
| Jaeger | http://localhost:16686 |
| Alertmanager | http://localhost:9093 |

**Need Help?** Check [README.md](./README.md) troubleshooting section

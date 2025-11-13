# Runbook: High Error Rate

## Alert Definition
```
Alert: HighErrorRate
Severity: CRITICAL
Threshold: >5% of requests returning 5xx errors for 3 minutes
```

## Business Impact
- **Customer Impact**: Users unable to complete payments
- **Revenue Impact**: $10,000/hour (based on $100K/month revenue)
- **SLO Impact**: Burns 5% of monthly error budget per hour

## Triage Steps (First 5 Minutes)

### Step 1: Confirm the Alert
```bash
# Check current error rate
curl -s "http://localhost:9090/api/v1/query?query=sum(rate(errors_total[1m]))/sum(rate(predictions_total[1m]))" | jq

# Expected: <0.01 (1%)
# If >0.05 (5%), alert is valid
```

### Step 2: Check Scope
**Question**: Is this affecting ALL requests or specific subset?

```promql
# Error rate by error type
sum by (error_type) (rate(errors_total[5m]))

# Error rate by service component
sum by (service) (rate(errors_total[5m]))
```

**Interpretation**:
- **All error types**: Systemic issue (database down, service crash)
- **Specific error type**: Code bug or dependency failure

### Step 3: Check Recent Changes
```bash
# Last 3 deployments
git log --oneline -3

# Check if error started after deploy
# Compare error rate timeline in Grafana with deployment annotations
```

---

## Common Causes & Solutions

### Cause 1: Redis Connection Failure
**Symptoms**:
- Error type: `RedisConnectionError`
- Cache miss rate: 100%
- Logs: "Failed to connect to Redis"

**Fix**:
```bash
# Check Redis health
docker exec redis redis-cli ping
# Expected: PONG

# If not responding, restart Redis
docker restart redis

# Verify recovery
watch "curl -s http://localhost:8000/health/ready | jq"
# Wait for {"status": "ready"}
```

**Prevention**:
- Add circuit breaker (stop trying Redis after 3 failures)
- Use Redis Sentinel (auto-failover)

---

### Cause 2: Model Loading Failure
**Symptoms**:
- Error type: `ModelNotLoadedError`
- Logs: "Model not found at /app/models/fraud_model.pkl"
- Health check: `{"model": "not_loaded"}`

**Fix**:
```bash
# Check if model file exists
docker exec fraud-api ls -lh /app/models/
# If missing, restore from backup

# Copy model into container
docker cp models/fraud_model.pkl fraud-api:/app/models/

# Restart service to reload model
docker restart fraud-api
```

**Prevention**:
- Add model validation on startup (fail-fast)
- Store models in S3/GCS (not local filesystem)

---

### Cause 3: Memory Exhaustion (OOMKill)
**Symptoms**:
- Container restarts every few minutes
- Logs: Last message before restart at exact memory limit
- Grafana: Memory usage spiked to 100%

**Fix**:
```bash
# Check container memory limit
docker inspect fraud-api | jq '.[0].HostConfig.Memory'

# Check current memory usage
docker stats fraud-api --no-stream

# If legitimate workload, increase limit
# Edit docker-compose.yml:
# deploy.resources.limits.memory: 2G  # was 1G

# Restart with new limit
docker-compose up -d fraud-api
```

**Prevention**:
- Profile memory usage (find leaks)
- Add memory alerts at 80% (before OOMKill)

---

### Cause 4: Database Connection Pool Exhausted
**Symptoms**:
- Error type: `TimeoutError: Could not acquire connection`
- Traces: Long wait time in database connection span
- Logs: "Connection pool exhausted"

**Fix**:
```bash
# Check active connections
docker exec fraud-api python -c "
from services.feature_store import feature_store
print(f'Active connections: {feature_store.redis_client.connection_pool._available_connections}')
"

# Increase pool size
# Edit feature_store.py:
# ConnectionPool(max_connections=50)  # was 10

# Restart
docker restart fraud-api
```

---

## Rollback Procedure (If cause unknown)

```bash
# 1. Identify last known good version
git log --oneline --since="2 hours ago"

# 2. Find commit before error started (check Grafana annotations)
GOOD_COMMIT=abc123

# 3. Rollback code
git checkout $GOOD_COMMIT

# 4. Rebuild and deploy
docker-compose build fraud-api
docker-compose up -d fraud-api

# 5. Verify recovery (error rate should drop to <1% within 2 minutes)
watch "curl -s 'http://localhost:9090/api/v1/query?query=sum(rate(errors_total[1m]))/sum(rate(predictions_total[1m]))' | jq '.data.result[0].value[1]'"
```

---

## Investigation (After mitigation)

### Collect Evidence
```bash
# 1. Export error logs (last 1 hour)
curl -G "http://localhost:3100/loki/api/v1/query_range" \\
  --data-urlencode 'query={container="fraud-api"} | json | level="ERROR"' \\
  --data-urlencode "start=$(date -u -d '1 hour ago' +%s)000000000" \\
  --data-urlencode "end=$(date -u +%s)000000000" \\
  > /tmp/error_logs.json

# 2. Export sample traces (failed requests)
# Open Jaeger: http://localhost:16686
# Search: service=fraud-detection-api, tags=error:true
# Export 10 sample traces

# 3. Export metrics snapshot
curl "http://localhost:9090/api/v1/query_range?query=errors_total&start=$(date -u -d '2 hours ago' +%s)&end=$(date -u +%s)&step=15s" \\
  > /tmp/error_metrics.json
```

### Root Cause Analysis (5 Whys)
1. **Why did errors occur?** Database connection timeout
2. **Why did database timeout?** Connection pool exhausted
3. **Why was pool exhausted?** Traffic spike (10x normal)
4. **Why didn't we handle traffic spike?** No auto-scaling configured
5. **Why no auto-scaling?** Never tested under high load

**Action Items**:
- [ ] Add Horizontal Pod Autoscaler (scale at 80% CPU)
- [ ] Run monthly load tests (find capacity limits)
- [ ] Increase connection pool size from 10 → 50

---

## Communication Template

### Internal (Engineering Slack)
```
🚨 INCIDENT: High Error Rate
Status: Investigating / Mitigated / Resolved
Impact: 15% of payments failing since 14:23 UTC
Root Cause: TBD / Redis connection failure
ETA: Mitigated in 5 min / Full resolution in 30 min

Next Update: 15:00 UTC
Incident Commander: @alice
```

### External (Status Page)
```
We are currently experiencing elevated error rates affecting payment processing.
Our team is actively investigating and working on a fix.

Impact: Some customers may see errors during checkout
Workaround: Please retry after 5 minutes

Last Updated: 14:30 UTC
Next Update: 15:00 UTC
```

---

## Post-Incident Review Template

**Incident**: High Error Rate - 2024-01-15
**Duration**: 14:23 - 14:45 UTC (22 minutes)
**Impact**: 15% error rate, ~500 failed payments, $3,600 lost revenue

**Timeline**:
- 14:23: Alert fired (HighErrorRate)
- 14:25: On-call paged
- 14:28: Identified Redis connection failures
- 14:30: Restarted Redis
- 14:32: Service recovered
- 14:45: Confirmed stable

**Root Cause**: Redis container OOMKilled due to memory leak

**Action Items**:
1. [ ] Upgrade Redis memory limit 512MB → 1GB (owner: @bob, due: 2024-01-17)
2. [ ] Add Redis memory monitoring (owner: @alice, due: 2024-01-18)
3. [ ] Implement circuit breaker for Redis (owner: @charlie, due: 2024-01-22)

**What Went Well**:
- Alert fired immediately (within 3 min of error spike)
- Recovery was fast (22 min total)
- Clear runbook helped triage

**What Went Poorly**:
- No monitoring for Redis memory usage
- No circuit breaker (cascading failures)
- Unclear ownership (took 5 min to page right person)

---

## Related Runbooks
- [High Latency](./high-latency.md)
- [Service Down](./service-down.md)
- [Database Failover](./database-failover.md)

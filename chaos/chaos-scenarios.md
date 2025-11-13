# Chaos Engineering Scenarios

## WHY CHAOS ENGINEERING?

**The Problem**: You don't know if your system handles failures gracefully until failures happen.

**The Solution**: Deliberately inject failures in production (controlled experiments) to:
1. **Discover weaknesses** before customers do
2. **Build confidence** in system resilience
3. **Practice incident response** (muscle memory for outages)

**BUSINESS IMPACT**:
- **Without chaos testing**: First Redis failure in production → 2 hours of downtime → $20K lost
- **With chaos testing**: Discovered Redis dependency 3 months ago → Added fallback → 0 downtime

---

## CHAOS TESTING PRINCIPLES

### 1. Start with Hypothesis
**Example**: "If Redis goes down, the API should degrade gracefully (slower, but still functional)"

### 2. Minimize Blast Radius
- Test in dev/staging first
- Test during business hours (when team is ready)
- Start with 1% traffic, then 10%, then 100%

### 3. Measure Impact
- **What to measure**: Latency, error rate, user experience
- **Acceptance criteria**: Error rate <1%, latency <500ms

### 4. Automate & Repeat
- Run chaos tests in CI/CD (every deploy)
- Schedule weekly chaos days (Netflix-style)

---

## SCENARIO 1: Redis Failure (Feature Store Down)

### Hypothesis
"If Redis goes down, predictions continue (using default features) with <10% accuracy degradation"

### Business Impact if Fails
- **Without fallback**: 100% API errors → $10K/hour revenue loss
- **With fallback**: 5% accuracy drop → 10 extra customer complaints/day

### Execution

```bash
# Kill Redis container
docker stop redis

# Expected behavior:
# - Feature store returns None for all features
# - API uses default features (account_age=365, transaction_count=5, etc.)
# - Predictions continue with degraded accuracy
# - Cache miss metric spikes to 100%
# - Alert fires: "HighCacheMissRate"

# Verify observability:
# 1. Check Grafana: Cache hit rate drops to 0%
# 2. Check logs: "Cache miss: feature:12345:transaction_velocity"
# 3. Check traces: feature_store.get span shows "cache.hit=false"
# 4. Check Alertmanager: Alert firing after 5 minutes

# Restore
docker start redis

# Recovery time: <10 seconds (cache warming)
```

### Success Criteria
✅ API returns 200 OK (not 500)
✅ Latency <500ms (p95)
✅ Alert fires within 5 minutes
✅ Logs show graceful degradation

### If Test Fails
→ **Add fallback logic** in `feature_store.py:get_feature()`
→ **Add circuit breaker** (after 3 failures, stop trying Redis for 60s)

---

## SCENARIO 2: OTel Collector Failure (Observability Blind Spot)

### Hypothesis
"If OTel Collector crashes, the API continues (metrics/traces dropped, but no functional impact)"

### Business Impact if Fails
- **Without resilience**: API crashes because it can't send metrics → 100% downtime
- **With resilience**: Blind spot for 5 minutes (no metrics), but API runs fine

### Execution

```bash
# Kill OTel Collector
docker stop otel-collector

# Expected behavior:
# - API continues processing requests
# - Metrics/traces buffered in memory (512MB limit)
# - After 30s, buffer full → start dropping telemetry
# - No customer impact

# Verify:
# 1. Send requests to API: curl http://localhost:8000/v1/predict
# 2. Check API logs: "⚠ Failed to connect to OTel Collector"
# 3. Prometheus stops receiving metrics (stale data)
# 4. Jaeger stops receiving traces

# Restore
docker start otel-collector

# Recovery: Metrics/traces resume immediately (buffered data is lost)
```

### Success Criteria
✅ API continues serving requests
✅ No customer-facing errors
✅ Logs show warning (not error)

### If Test Fails
→ **OTel SDK should use no-op exporters** (silent failure)
→ **Add retry logic** in OTel Collector config

---

## SCENARIO 3: Memory Pressure (OOM Scenario)

### Hypothesis
"Under memory pressure, container is throttled (slow) but doesn't crash"

### Business Impact if Fails
- **Without limits**: Container uses all host RAM → host crashes → ALL services down
- **With limits**: Container OOMKilled → Kubernetes restarts → 10s downtime

### Execution

```bash
# Simulate memory leak (add to fraud_api.py temporarily)
memory_leak = []
@app.get("/debug/leak")
def leak():
    memory_leak.append("X" * 10_000_000)  # 10MB per call
    return {"leaked_mb": len(memory_leak) * 10}

# Trigger leak
for i in {1..100}; do
  curl http://localhost:8000/debug/leak
done

# Expected behavior:
# 1. Memory usage climbs to container limit (1GB)
# 2. Alert fires: "HighMemoryUsage" at 850MB
# 3. Container is OOMKilled at 1GB
# 4. Docker restarts container (health check fails)
# 5. Service resumes within 30 seconds

# Observe in Grafana:
# - Memory usage graph spikes to limit
# - Container restart event (downtime annotation)
# - Request error rate spike during restart
```

### Success Criteria
✅ Alert fires before OOMKill
✅ Container restarts automatically
✅ Service recovers within 60 seconds

### If Test Fails
→ **Increase memory limit** (if workload is legitimate)
→ **Add memory leak detection** (profile memory growth)
→ **Implement graceful shutdown** (finish in-flight requests before killing)

---

## SCENARIO 4: Network Partition (Split Brain)

### Hypothesis
"If fraud-api can't reach Redis, it fails fast (doesn't hang waiting)"

### Execution

```bash
# Block network traffic to Redis (using Docker network)
docker network disconnect ml-observability redis

# Expected behavior:
# - Redis connection timeout after 1 second
# - Requests fail immediately (not after 30s timeout)
# - Circuit breaker opens (stop trying Redis)
# - Alert fires: "ServiceDown - redis"

# Verify latency:
time curl http://localhost:8000/v1/predict
# Should return in <2 seconds (not hang for 30s)

# Restore
docker network connect ml-observability redis
```

### Success Criteria
✅ Requests fail fast (<2s)
✅ Error message is clear ("Redis unreachable")
✅ Circuit breaker prevents cascading failures

---

## SCENARIO 5: High Traffic Spike (Load Testing)

### Hypothesis
"System handles 10x traffic spike for 5 minutes without degradation"

### Execution

See `load-test.py` below for Locust-based load testing.

```bash
# Run load test
locust -f chaos/load-test.py --host=http://localhost:8000

# Ramp up:
# - Start: 10 users
# - 1 min: 100 users
# - 2 min: 500 users
# - 3 min: 1000 users (10x normal)

# Observe in Grafana:
# - Request rate climbs to 1000 req/s
# - Latency stays <100ms (p95)
# - Error rate stays <1%
# - CPU usage climbs to 80%
# - Memory usage stable
```

### Success Criteria
✅ p95 latency <500ms at 1000 req/s
✅ Error rate <1%
✅ No container restarts
✅ Auto-scaling triggers (if enabled)

### If Test Fails
→ **Scale horizontally** (add more pods)
→ **Optimize hot paths** (profiling shows bottlenecks)
→ **Add rate limiting** (protect against abuse)

---

## SCENARIO 6: Cascading Failure (Retry Storm)

### Hypothesis
"If fraud-api retries failed requests, it doesn't overwhelm Redis during recovery"

### Execution

```bash
# Simulate flaky Redis (intermittent failures)
# Use toxiproxy or manual restart loop

while true; do
  docker stop redis
  sleep 5
  docker start redis
  sleep 10
done

# Without backoff:
# - API retries immediately → 1000 req/s to Redis
# - Redis can't handle load → crashes again
# - Death spiral (never recovers)

# With exponential backoff:
# - Retry after 1s, then 2s, then 4s, then 8s
# - Redis has time to recover
# - System self-heals
```

### Success Criteria
✅ Retry backoff visible in logs
✅ Redis recovers successfully
✅ No infinite retry loops

---

## SCENARIO 7: Slow Dependency (Latency Injection)

### Hypothesis
"If Redis becomes slow (100ms latency), API stays under 500ms SLO"

### Execution

```bash
# Use tc (traffic control) to add latency
docker exec redis tc qdisc add dev eth0 root netem delay 100ms

# Expected:
# - Feature fetch: 2ms → 100ms
# - Prediction latency: 50ms → 150ms (still under 500ms SLO)
# - Warning alert fires (not critical)

# Remove latency
docker exec redis tc qdisc del dev eth0 root
```

---

## CHAOS TESTING SCHEDULE

### CI/CD Pipeline (Every Deploy)
- ✅ Scenario 2: OTel Collector failure
- ✅ Scenario 5: Load test (100 users)

### Weekly Chaos Day (Friday 2-3pm)
- Rotate through Scenarios 1-4
- Team on standby to observe
- Postmortem if anything unexpected

### Monthly Game Day
- Multi-service failure (Redis + Prometheus)
- Practice incident response
- Measure MTTR (Mean Time To Recovery)

---

## CHAOS ENGINEERING MATURITY

### Level 1: Manual Chaos (You are here)
- Run scripts manually
- Document results in Google Docs

### Level 2: Automated Chaos
- CI/CD runs chaos tests
- Failures block deploy if SLO violated

### Level 3: Continuous Chaos
- Background chaos (1% of prod traffic)
- Automated rollback on SLO breach

### Level 4: GameDay Cadence
- Monthly disaster recovery drills
- RTO <10 min, RPO <5 min

---

## FURTHER READING

- [Principles of Chaos Engineering](https://principlesofchaos.org/)
- [Netflix Chaos Engineering](https://netflixtechblog.com/tagged/chaos-engineering)
- [Chaos Mesh (Kubernetes chaos)](https://chaos-mesh.org/)

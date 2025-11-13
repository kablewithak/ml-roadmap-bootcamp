# Service Level Objectives (SLOs) and Error Budgets

## WHY SLOs MATTER

**The Problem**: "We need 100% uptime" is:
- ❌ Impossible (even AWS has outages)
- ❌ Expensive (99.99% → 99.999% costs 10x more)
- ❌ Slows innovation (teams afraid to deploy)

**The Solution**: SLOs quantify "good enough" reliability
- ✅ 99.9% uptime = 43 minutes downtime/month (acceptable for most businesses)
- ✅ Error budget = 43 minutes you can "spend" on deployments, experiments
- ✅ When budget exhausted → freeze deployments, focus on reliability

---

## BUSINESS IMPACT OF SLOs

### Revenue Impact Table

| Availability | Downtime/Month | Lost Revenue (at $100K/month) |
|--------------|----------------|-------------------------------|
| 90% (1 nine) | 72 hours       | $10,000 😱                     |
| 99% (2 nines)| 7.2 hours      | $1,000 😐                      |
| 99.9% (3 nines)| 43 minutes   | $100 ✅                        |
| 99.99% (4 nines)| 4.3 minutes | $10 ✅✅                        |
| 99.999% (5 nines)| 26 seconds | $1 💰                          |

**Sweet Spot for Most Businesses**: 99.9% (3 nines)
- Reasonable cost (no expensive multi-region active-active)
- Acceptable customer impact (43 min/month = most customers never notice)
- Allows innovation (can deploy risky features without fear)

---

## SLO #1: API AVAILABILITY

### Objective
**99.9% of API requests succeed (return 2xx or 4xx)**

### Why This Matters
- **Customer Impact**: Failed requests = failed payments = lost revenue
- **Business Metric**: $1,000 lost revenue per hour of downtime

### Measurement

**SLI (Service Level Indicator)**: Success rate
```promql
# Success rate (last 30 days)
sum(rate(http_server_requests_total{status!~"5.."}[30d])) /
sum(rate(http_server_requests_total[30d]))
```

**Error Budget**: 0.1% = 43.2 minutes/month

```promql
# Error budget remaining (percentage)
1 - (
  sum(rate(http_server_requests_total{status=~"5.."}[30d])) /
  sum(rate(http_server_requests_total[30d]))
) / 0.001
```

### What Counts as "Success"

✅ **Counts toward SLO**:
- 200 OK (successful prediction)
- 201 Created
- 400 Bad Request (user error, not our fault)
- 401 Unauthorized (auth failure, expected)
- 404 Not Found

❌ **Counts against SLO**:
- 500 Internal Server Error (our bug)
- 502 Bad Gateway (dependency failure we should handle)
- 503 Service Unavailable (overload we should scale for)
- 504 Gateway Timeout (slow response we should optimize)

### Alert Thresholds

**Fast Burn** (uses 5% of monthly budget in 1 hour):
```promql
# If this fires, you'll exhaust budget in 20 hours
sum(rate(http_server_requests_total{status=~"5.."}[1h])) /
sum(rate(http_server_requests_total[1h])) > 0.05
```
→ **Page on-call immediately**

**Slow Burn** (uses 10% of monthly budget in 1 day):
```promql
sum(rate(http_server_requests_total{status=~"5.."}[24h])) /
sum(rate(http_server_requests_total[24h])) > 0.0003
```
→ **Create ticket for tomorrow**

---

## SLO #2: API LATENCY

### Objective
**95% of API requests complete in <100ms**

### Why This Matters
- **Customer Impact**: Slow checkout → abandoned carts
- **Business Metric**: Every 100ms delay = 1% conversion drop = $1,000/day lost

### Measurement

**SLI**: p95 latency
```promql
histogram_quantile(0.95,
  sum(rate(http_server_duration_seconds_bucket[30d])) by (le)
) < 0.1
```

**Error Budget**: 5% of requests can be >100ms

### What Counts as "Within Budget"

✅ **Good** (counts toward SLO):
- <100ms: ✅

❌ **Bad** (counts against budget):
- 100-500ms: ⚠️  Degraded
- >500ms: ❌ Unacceptable

### Alert Thresholds

**Critical**: p95 >500ms for 5 minutes
```promql
histogram_quantile(0.95, sum(rate(http_server_duration_seconds_bucket[5m])) by (le)) > 0.5
```

**Warning**: p95 >100ms for 15 minutes
```promql
histogram_quantile(0.95, sum(rate(http_server_duration_seconds_bucket[15m])) by (le)) > 0.1
```

---

## SLO #3: ML MODEL ACCURACY

### Objective
**Model precision >90% (measured weekly)**

### Why This Matters
- **Customer Impact**: Low precision = false positives = blocked legitimate customers
- **Business Metric**: 1% precision drop = 100 complaints/week = $5K support costs

### Measurement

**SLI**: Precision on labeled data
```promql
model_performance{metric="precision"}
```

**Error Budget**: Can drop to 85% for 1 week before action required

### Ground Truth Labeling

```python
# Weekly batch job
labeled_samples = get_manually_reviewed_transactions(last_week)
predictions = model.predict(labeled_samples.features)
precision = calculate_precision(predictions, labeled_samples.labels)

# Push to Prometheus
model_performance.labels(metric="precision").set(precision)
```

### Alert Thresholds

**Critical**: Precision <85%
→ **Roll back to previous model version**

**Warning**: Precision 85-90%
→ **Schedule model retraining**

---

## SLO #4: FEATURE STORE CACHE HIT RATE

### Objective
**95% cache hit rate for feature fetches**

### Why This Matters
- **Performance**: Cache miss = 50ms → 500ms (10x slower)
- **Cost**: Database queries cost $0.001 each, Redis $0.0001 (10x cheaper)

### Measurement

**SLI**: Cache hit percentage
```promql
sum(rate(feature_cache_hits_total[30d])) /
(sum(rate(feature_cache_hits_total[30d])) + sum(rate(feature_cache_misses_total[30d])))
```

**Error Budget**: 5% cache misses allowed

### Alert Thresholds

**Critical**: Hit rate <80% for 10 minutes
```promql
# Redis likely down or memory full
sum(rate(feature_cache_hits_total[10m])) /
(sum(rate(feature_cache_hits_total[10m])) + sum(rate(feature_cache_misses_total[10m]))) < 0.8
```

---

## ERROR BUDGET POLICY

### When Budget is Healthy (>50% remaining)

✅ **ALLOWED**:
- Deploy new features freely
- Experiment with new ML models (A/B testing)
- Refactor code for performance
- Scale down during off-hours (cost savings)

### When Budget is Low (<25% remaining)

⚠️ **RESTRICTED**:
- Freeze non-critical deployments
- Increase testing rigor (manual QA before prod)
- Defer risky experiments to next month
- Focus on reliability improvements

❌ **EMERGENCY MODE** (Budget exhausted):
- **FREEZE ALL DEPLOYMENTS** (except hotfixes)
- Cancel planned experiments
- On-call engineer assigned full-time to reliability
- Post-mortem required before unfreezing

### Calculating Budget Burn Rate

```promql
# What % of monthly budget did we use in the last hour?
(
  sum(increase(http_server_requests_total{status=~"5.."}[1h]))
  /
  sum(increase(http_server_requests_total[1h]))
) / 0.001 * 100
```

**Example**:
- Month starts with 100% budget (43.2 min downtime allowed)
- Week 1: Deploy breaks prod for 10 min → 23% budget used
- Week 2: Database failover takes 5 min → 35% budget used
- Week 3: **FREEZE** (65% budget used, risky to continue)

---

## SLO REVIEW CADENCE

### Weekly
- Check current SLO compliance (dashboard review)
- Estimate remaining error budget
- Decide: Can we deploy risky feature X this week?

### Monthly
- Full SLO report (availability, latency, accuracy)
- Error budget retrospective: Why did we burn budget?
- Adjust SLOs if business needs changed

### Quarterly
- Recalibrate SLOs based on business growth
- Example: "We're now processing $1M/month, need 99.95% SLO"

---

## WHEN TO RELAX SLOs

You DON'T need 99.99% if:
- ❌ You're a startup (speed > reliability)
- ❌ Your competitors have 99.9% (no competitive advantage)
- ❌ Customers don't pay more for higher reliability

You DO need stricter SLOs if:
- ✅ Regulated industry (healthcare, finance)
- ✅ Contractual obligations (enterprise SLAs)
- ✅ High revenue per minute (losing $10K/min during outage)

---

## SLO TOOLING

### Prometheus Recording Rules
```yaml
# Pre-calculate SLIs for fast querying
groups:
  - name: slo_recording_rules
    interval: 30s
    rules:
      - record: slo:api_availability:30d
        expr: |
          sum(rate(http_server_requests_total{status!~"5.."}[30d])) /
          sum(rate(http_server_requests_total[30d]))

      - record: slo:api_latency_p95:30d
        expr: |
          histogram_quantile(0.95,
            sum(rate(http_server_duration_seconds_bucket[30d])) by (le)
          )
```

### Grafana SLO Dashboard
- **Availability Gauge**: Green >99.9%, Red <99.9%
- **Error Budget Bar**: Shows % remaining
- **Burn Rate Graph**: Shows budget consumption rate
- **Incident Timeline**: Annotations for outages

---

## FURTHER READING

- [Google SRE Book - SLOs](https://sre.google/sre-book/service-level-objectives/)
- [Implementing SLOs (Google Cloud)](https://cloud.google.com/blog/products/devops-sre/sre-fundamentals-slis-slas-and-slos)
- [Error Budget Policy (Grafana)](https://grafana.com/blog/2022/07/14/a-practical-guide-to-setting-slos/)

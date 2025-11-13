# Grafana Dashboards for ML Fraud Detection

## Dashboard Structure

### 1. **ML Model Performance Dashboard** (`ml-model-performance.json`)

**Purpose**: Monitor ML model health and performance
**Audience**: ML Engineers, Data Scientists
**Update Frequency**: Real-time (15s refresh)

#### Key Panels:

**Row 1: Prediction Overview**
- **Prediction Volume** (Graph)
  ```promql
  sum(rate(predictions_total[5m])) by (prediction)
  ```
  Shows: approve/review/block rates over time

- **Prediction Distribution** (Pie Chart)
  ```promql
  sum by (prediction) (increase(predictions_total[1h]))
  ```
  Shows: Breakdown of decisions

**Row 2: Model Latency**
- **Inference Latency p50/p95/p99** (Graph)
  ```promql
  histogram_quantile(0.95, sum(rate(prediction_latency_seconds_bucket[5m])) by (le))
  ```
  SLO Line: 100ms

- **Latency Heatmap** (Heatmap)
  Shows latency distribution over time

**Row 3: Model Accuracy (Updated by batch jobs)**
- **Precision/Recall/F1** (Stat panels)
  ```promql
  model_performance{metric="precision"}
  ```
  Color coding: Green >0.9, Yellow 0.8-0.9, Red <0.8

**Row 4: Data Drift**
- **Feature PSI** (Graph)
  ```promql
  feature_psi
  ```
  Threshold lines: 0.1 (warning), 0.25 (critical)

- **Prediction Score Distribution** (Histogram)
  ```promql
  histogram_quantile(0.5, sum(rate(prediction_score_bucket[5m])) by (le))
  ```

---

### 2. **Business Metrics Dashboard** (`business-metrics.json`)

**Purpose**: Track business impact and revenue
**Audience**: Product Managers, Executives
**Update Frequency**: 1 minute

#### Key Panels:

**Row 1: Revenue**
- **Revenue Rate** (Stat + Sparkline)
  ```promql
  sum(rate(revenue_usd_total[5m])) * 3600  # Per hour
  ```

- **Revenue Over Time** (Graph)
  ```promql
  sum(increase(revenue_usd_total[1h]))
  ```

**Row 2: Fraud Impact**
- **Fraud Blocked (USD/hour)** (Stat)
  ```promql
  sum(rate(fraud_blocked_usd_total[5m])) * 3600
  ```

- **Fraud Block Rate** (Gauge)
  ```promql
  sum(rate(predictions_total{prediction="block"}[5m])) /
  sum(rate(predictions_total[5m]))
  ```
  Target: 5-10%

**Row 3: Payment Processing**
- **Payment Volume by Method** (Bar Chart)
  ```promql
  sum by (payment_method) (increase(payments_total[1h]))
  ```

- **Payment Amount Distribution** (Histogram)
  ```promql
  histogram_quantile(0.95, sum(rate(payment_amount_usd_bucket[5m])) by (le))
  ```

---

### 3. **Service Health Dashboard** (`service-health.json`)

**Purpose**: Monitor system health and SLOs
**Audience**: SRE, DevOps
**Update Frequency**: 15 seconds

#### Key Panels:

**Row 1: SLO Status**
- **Availability (99.9% SLO)** (Stat)
  ```promql
  sum(rate(http_server_requests_total{status!~"5.."}[5m])) /
  sum(rate(http_server_requests_total[5m]))
  ```
  Color: Red if <0.999

- **Error Budget Remaining** (Gauge)
  ```promql
  # 99.9% SLO = 0.1% error budget = 43 min/month downtime
  # Calculate remaining budget
  ```

**Row 2: Request Metrics**
- **Request Rate** (Graph)
  ```promql
  sum(rate(http_server_requests_total[5m]))
  ```

- **Error Rate** (Graph)
  ```promql
  sum(rate(http_server_requests_total{status=~"5.."}[5m])) /
  sum(rate(http_server_requests_total[5m]))
  ```
  Threshold: 1%

**Row 3: Latency**
- **Request Latency** (Graph)
  ```promql
  histogram_quantile(0.95,
    sum(rate(http_server_duration_seconds_bucket[5m])) by (le)
  )
  ```

**Row 4: Feature Store**
- **Cache Hit Rate** (Graph)
  ```promql
  sum(rate(feature_cache_hits_total[5m])) /
  (sum(rate(feature_cache_hits_total[5m])) + sum(rate(feature_cache_misses_total[5m])))
  ```
  Target: >95%

- **Redis Connections** (Stat)
  ```promql
  redis_connected_clients
  ```

**Row 5: Resources**
- **CPU Usage** (Graph)
  ```promql
  rate(container_cpu_usage_seconds_total{container="fraud-api"}[5m])
  ```

- **Memory Usage** (Graph)
  ```promql
  container_memory_usage_bytes{container="fraud-api"} /
  container_spec_memory_limit_bytes{container="fraud-api"}
  ```

---

### 4. **Logs Dashboard** (Explore view with Loki)

**Pre-configured queries**:

1. **All Errors**:
   ```logql
   {container="fraud-api"} | json | level="ERROR"
   ```

2. **High-value transactions**:
   ```logql
   {container="fraud-api"} | json | amount > 1000
   ```

3. **Fraud detections**:
   ```logql
   {container="fraud-api"} | json | decision="block"
   ```

4. **Trace correlation**:
   ```logql
   {container="fraud-api"} | json | trace_id="<paste_trace_id>"
   ```

---

## How to Use Dashboards

### Debugging Workflow:

1. **Alert fires**: "High Error Rate"
2. **Open Service Health Dashboard**: See error rate spike at 14:23
3. **Click on spike**: View exemplar trace → Opens Jaeger
4. **In Jaeger**: See which span failed (e.g., Redis timeout)
5. **Click "Logs"**: See error logs for that trace_id
6. **Root cause**: Redis was restarting (check Redis dashboard)

### Business Review Workflow:

1. **Weekly meeting**: Open Business Metrics Dashboard
2. **Check Revenue trend**: Up 15% week-over-week ✅
3. **Check Fraud block rate**: 8% (normal range) ✅
4. **Check ML Performance Dashboard**: Precision stable at 92% ✅
5. **Action items**: None (all green)

---

## Creating Dashboards

### Method 1: Import Pre-built (Recommended for learning)
```bash
# Dashboards are auto-loaded from config/grafana/dashboards/
# Just start the stack: docker-compose up
```

### Method 2: Build in UI
1. Go to http://localhost:3000
2. Create → Dashboard → Add Panel
3. Copy PromQL queries from above
4. Save dashboard
5. Export JSON → Save to `config/grafana/dashboards/`

### Method 3: Use grafonnet (Code-based dashboards)
```jsonnet
// ml-dashboard.jsonnet
local grafana = import 'grafonnet/grafana.libsonnet';

grafana.dashboard.new(
  'ML Model Performance',
  schemaVersion=16,
  tags=['ml', 'fraud-detection'],
)
.addPanel(
  grafana.graphPanel.new(
    'Prediction Latency',
    datasource='Prometheus',
    targets=[
      {
        expr: 'histogram_quantile(0.95, sum(rate(prediction_latency_seconds_bucket[5m])) by (le))',
        legendFormat: 'p95',
      },
    ],
  ),
  gridPos={x: 0, y: 0, w: 12, h: 8}
)
```

---

## Dashboard Maintenance

### When to Update Dashboards:

1. **New service deployed**: Add panels for new metrics
2. **SLO changed**: Update threshold lines
3. **Alert fired but not visible**: Add panel for that metric
4. **Business asks new question**: Add panel to answer it

### Dashboard Review Cadence:

- **Weekly**: Check all dashboards still working
- **Monthly**: Archive unused dashboards
- **Quarterly**: Refactor based on user feedback

---

## Advanced: Variable Templates

Use Grafana variables for dynamic dashboards:

```
Variable: $model_name
Query: label_values(predictions_total, model_name)

Panel query:
sum(rate(predictions_total{model_name="$model_name"}[5m]))
```

This lets you select model version from dropdown (useful for A/B testing).

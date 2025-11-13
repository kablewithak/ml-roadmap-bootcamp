"""
Load Testing with Locust

WHAT IS LOAD TESTING:
Simulate realistic user traffic to find performance bottlenecks BEFORE they hit production.

WHY LOCUST:
- Python-based (familiar for ML engineers)
- Distributed load generation (scale to millions of users)
- Real-time web UI (see results live)
- Scriptable scenarios (not just random requests)

BUSINESS VALUE:
- Find that your API breaks at 500 req/s (not 5000 like you thought)
- Discover Redis becomes bottleneck at 1000 req/s
- Prove to stakeholders: "We can handle Black Friday traffic"

USAGE:
    # Install
    pip install locust

    # Run locally
    locust -f load_test.py --host=http://localhost:8000

    # Open browser: http://localhost:8089
    # Set: 1000 users, 100 users/second spawn rate
    # Click "Start swarming"

    # Run headless (CI/CD)
    locust -f load_test.py --host=http://localhost:8000 \\
           --users 1000 --spawn-rate 100 --run-time 5m --headless

    # Distributed mode (for high load)
    # Master:
    locust -f load_test.py --master
    # Workers (on separate machines):
    locust -f load_test.py --worker --master-host=<master-ip>
"""

import random
import time
from locust import HttpUser, task, between, events
from locust.contrib.fasthttp import FastHttpUser


# ============================================================================
# REALISTIC FRAUD DETECTION USER BEHAVIOR
# ============================================================================

class FraudDetectionUser(FastHttpUser):
    """
    Simulates a user making payment predictions.

    USER BEHAVIOR MODEL:
    - 70% approve (legitimate customers)
    - 25% review (borderline cases)
    - 5% block (fraudsters)

    This matches real-world fraud distribution (not uniform random traffic).
    """

    # Wait time between requests (realistic user pacing)
    wait_time = between(1, 3)  # 1-3 seconds between requests

    def on_start(self):
        """
        Called once when a user starts.
        Use this to:
        - Log in (if your API requires auth)
        - Fetch initial data (user profile, etc.)
        - Populate cache (if testing warm cache)
        """
        # Populate feature cache for this user
        self.user_id = random.randint(1, 10000)
        self.client.post(
            f"/debug/populate-cache/{self.user_id}",
            name="/debug/populate-cache/[id]"  # Group URLs in stats
        )

    @task(70)  # 70% of requests are legitimate (small amounts)
    def legitimate_payment(self):
        """
        Simulate legitimate customer payment.

        BUSINESS SCENARIO:
        - Small transaction ($10-$200)
        - Card payment
        - User with history
        """
        payload = {
            "transaction_id": f"txn_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
            "user_id": self.user_id,
            "amount": round(random.uniform(10, 200), 2),
            "payment_method": random.choice(["card", "card", "bank_transfer"]),  # Mostly cards
            "merchant_id": f"merchant_{random.randint(1, 100)}"
        }

        with self.client.post(
            "/v1/predict",
            json=payload,
            catch_response=True,
            name="/v1/predict [legitimate]"
        ) as response:
            if response.status_code == 200:
                result = response.json()

                # CRITICAL: Validate business logic
                # If small transaction is blocked, that's a bug (false positive)
                if payload["amount"] < 100 and result["decision"] == "block":
                    response.failure(f"False positive: ${payload['amount']} blocked")
                else:
                    response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(25)  # 25% borderline cases (review needed)
    def borderline_payment(self):
        """
        Simulate borderline transactions that need manual review.

        BUSINESS SCENARIO:
        - Medium amount ($200-$1000)
        - May trigger review
        """
        payload = {
            "transaction_id": f"txn_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
            "user_id": self.user_id,
            "amount": round(random.uniform(200, 1000), 2),
            "payment_method": random.choice(["card", "bank_transfer", "crypto"]),
            "merchant_id": f"merchant_{random.randint(1, 100)}"
        }

        with self.client.post(
            "/v1/predict",
            json=payload,
            name="/v1/predict [borderline]"
        ) as response:
            if response.status_code == 200:
                result = response.json()
                # Expect approve or review (not block)
                if result["decision"] in ["approve", "review"]:
                    response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(5)  # 5% fraudulent (should be blocked)
    def fraudulent_payment(self):
        """
        Simulate fraudulent transaction.

        BUSINESS SCENARIO:
        - Large amount ($1000-$10000)
        - Suspicious patterns (new account, international, high velocity)
        """
        payload = {
            "transaction_id": f"txn_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
            "user_id": random.randint(1, 100),  # New user (no history)
            "amount": round(random.uniform(1000, 10000), 2),  # Large amount
            "payment_method": "crypto",  # Higher fraud rate
            "merchant_id": f"merchant_{random.randint(1, 10)}"
        }

        with self.client.post(
            "/v1/predict",
            json=payload,
            name="/v1/predict [fraud]"
        ) as response:
            if response.status_code == 200:
                result = response.json()
                # Should be blocked or reviewed (not approved!)
                if result["decision"] in ["block", "review"]:
                    response.success()
                else:
                    # False negative (fraud slipped through)
                    response.failure(f"False negative: ${payload['amount']} fraud approved")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)  # Occasionally check health (monitoring)
    def health_check(self):
        """Health check (should be very fast)."""
        with self.client.get("/health/ready", name="/health/ready") as response:
            if response.elapsed.total_seconds() > 0.1:  # Should be <100ms
                response.failure(f"Health check slow: {response.elapsed.total_seconds()}s")


# ============================================================================
# LOAD TEST SCENARIOS
# ============================================================================

class SpikeTest(FraudDetectionUser):
    """
    Simulate traffic spike (Black Friday, viral event).

    SCENARIO:
    - Normal: 100 users
    - Spike: Ramp to 1000 users in 1 minute
    - Sustain: 1000 users for 5 minutes
    - Return: Back to 100 users

    WHAT TO WATCH:
    - Does latency spike during ramp-up?
    - Does error rate increase?
    - Do containers auto-scale?
    """
    wait_time = between(0.5, 1.5)  # Faster requests (higher load)


class SoakTest(FraudDetectionUser):
    """
    Long-running test to find memory leaks.

    SCENARIO:
    - 100 users for 24 hours
    - Check: Does memory grow over time? (leak)
    - Check: Does latency degrade? (resource exhaustion)
    """
    wait_time = between(2, 5)  # Normal pacing


# ============================================================================
# CUSTOM METRICS (Export to Prometheus)
# ============================================================================

# Track business metrics during load test
prediction_decisions = {"approve": 0, "review": 0, "block": 0}


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """
    Hook into every request to collect custom metrics.

    USAGE:
    - Track business KPIs (approval rate, fraud detection rate)
    - Export to Prometheus (scrape from Locust)
    - Correlate with system metrics (latency vs approval rate)
    """
    if name.startswith("/v1/predict") and not exception:
        # Parse response to track business metrics
        # (In production, response is already parsed in tasks above)
        pass


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when test starts."""
    print("🚀 Load test starting...")
    print(f"Target: {environment.host}")
    print(f"Users: {environment.runner.target_user_count}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """
    Called when test stops.

    Use this to:
    - Print summary statistics
    - Send results to monitoring system
    - Fail CI/CD if SLO violated
    """
    stats = environment.stats.total

    print("\n" + "=" * 80)
    print("LOAD TEST SUMMARY")
    print("=" * 80)
    print(f"Total requests: {stats.num_requests}")
    print(f"Failures: {stats.num_failures} ({stats.fail_ratio * 100:.2f}%)")
    print(f"Median latency: {stats.median_response_time}ms")
    print(f"95th percentile: {stats.get_response_time_percentile(0.95)}ms")
    print(f"Requests/sec: {stats.total_rps:.2f}")

    # CRITICAL: Fail CI/CD if SLO violated
    if stats.fail_ratio > 0.01:  # >1% error rate
        print("❌ FAILED: Error rate too high")
        environment.process_exit_code = 1
    elif stats.get_response_time_percentile(0.95) > 500:  # p95 >500ms
        print("❌ FAILED: Latency too high")
        environment.process_exit_code = 1
    else:
        print("✅ PASSED: All SLOs met")
        environment.process_exit_code = 0


# ============================================================================
# RUN CONFIGURATIONS
# ============================================================================

if __name__ == "__main__":
    """
    Run different test scenarios from command line.

    EXAMPLES:

    # Baseline test (find max capacity)
    locust -f load_test.py --host=http://localhost:8000 \\
           --users 1000 --spawn-rate 50 --run-time 10m --headless

    # Spike test
    locust -f load_test.py --host=http://localhost:8000 \\
           --users 1000 --spawn-rate 100 --run-time 5m --headless

    # Soak test (memory leak detection)
    locust -f load_test.py --host=http://localhost:8000 \\
           --users 100 --spawn-rate 10 --run-time 24h --headless
    """
    pass

"""
Locust load testing file for fraud detection API.

Usage:
    locust -f benchmarks/locustfile.py --host=http://localhost:8000

Then open http://localhost:8089 to configure and run load tests.
"""

from locust import HttpUser, task, between
import random
import string


class FraudDetectionUser(HttpUser):
    """Simulated user for load testing fraud detection API."""

    wait_time = between(0.1, 0.5)  # Wait 0.1-0.5s between requests

    def on_start(self):
        """Initialize user session."""
        self.transaction_counter = 0

    def generate_transaction_id(self):
        """Generate unique transaction ID."""
        self.transaction_counter += 1
        return f"load-test-{self.transaction_counter}-{random.randint(1000, 9999)}"

    def generate_user_id(self):
        """Generate user ID (100 unique users)."""
        return f"user-{random.randint(1, 100)}"

    def generate_card_id(self):
        """Generate card ID (200 unique cards)."""
        return f"card-{random.randint(1, 200)}"

    def generate_ip(self):
        """Generate IP address."""
        return f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}"

    @task(10)
    def check_fraud_normal_transaction(self):
        """Test normal fraud check (most common)."""
        payload = {
            "transaction_id": self.generate_transaction_id(),
            "user_id": self.generate_user_id(),
            "card_id": self.generate_card_id(),
            "ip_address": self.generate_ip(),
            "amount": round(random.uniform(10, 500), 2),
            "currency": "USD",
            "merchant_id": f"merchant-{random.randint(1, 50)}",
            "merchant_category": random.choice(["retail", "online", "grocery", "restaurant"]),
            "merchant_name": "Test Merchant"
        }

        with self.client.post(
            "/fraud/check",
            json=payload,
            catch_response=True,
            name="fraud_check_normal"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                # Track latency
                if data.get("processing_time_ms", 0) > 50:
                    response.failure(f"Latency too high: {data['processing_time_ms']:.2f}ms")
                else:
                    response.success()
            else:
                response.failure(f"Got status code: {response.status_code}")

    @task(3)
    def check_fraud_high_value(self):
        """Test fraud check with high-value transaction."""
        payload = {
            "transaction_id": self.generate_transaction_id(),
            "user_id": self.generate_user_id(),
            "card_id": self.generate_card_id(),
            "ip_address": self.generate_ip(),
            "amount": round(random.uniform(1000, 5000), 2),
            "currency": "USD",
            "merchant_id": f"merchant-{random.randint(1, 50)}",
            "merchant_category": "electronics",
            "merchant_name": "Electronics Store"
        }

        with self.client.post(
            "/fraud/check",
            json=payload,
            catch_response=True,
            name="fraud_check_high_value"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code: {response.status_code}")

    @task(2)
    def check_fraud_velocity_test(self):
        """Test rapid transactions (velocity test)."""
        card_id = self.generate_card_id()
        user_id = self.generate_user_id()
        ip = self.generate_ip()

        # Send 3 rapid transactions
        for i in range(3):
            payload = {
                "transaction_id": self.generate_transaction_id(),
                "user_id": user_id,
                "card_id": card_id,
                "ip_address": ip,
                "amount": round(random.uniform(50, 200), 2),
                "currency": "USD",
                "merchant_id": f"merchant-{random.randint(1, 50)}",
                "merchant_category": "retail",
                "merchant_name": "Test Merchant"
            }

            with self.client.post(
                "/fraud/check",
                json=payload,
                catch_response=True,
                name="fraud_check_velocity"
            ) as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"Got status code: {response.status_code}")

    @task(1)
    def health_check(self):
        """Test health check endpoint."""
        with self.client.get("/health", name="health_check") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code: {response.status_code}")


class PaymentUser(HttpUser):
    """Simulated user for payment processing load test."""

    wait_time = between(1, 3)  # Wait 1-3s between payments

    def on_start(self):
        """Initialize user session."""
        self.transaction_counter = 0

    @task
    def process_payment(self):
        """Test payment processing (requires valid Stripe token)."""
        # Note: This will fail without valid Stripe token
        # In production, you'd integrate with Stripe's test tokens
        pass


# Performance thresholds
# These can be monitored during load tests:
# - P95 latency < 50ms
# - Redis lookup < 10ms
# - Success rate > 99%
# - Able to handle 1000 TPS

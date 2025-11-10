"""
Performance benchmarking suite for fraud detection system.

Measures:
- Latency at different TPS rates (10, 50, 100, 500, 1000 TPS)
- Redis lookup performance
- End-to-end fraud detection latency
- Throughput under load

Target: <50ms total latency, <10ms Redis lookup
"""

import asyncio
import time
import statistics
from datetime import datetime
from typing import List, Dict
import yaml

from src.infrastructure.redis.velocity_tracker import create_redis_client, VelocityTracker
from src.fraud.services.fraud_detector import create_fraud_detector
from src.fraud.models import TransactionRequest


class FraudDetectorBenchmark:
    """Performance benchmark suite for fraud detection."""

    def __init__(self):
        self.redis_client = None
        self.fraud_detector = None
        self.results = []

    async def setup(self):
        """Initialize components for benchmarking."""
        # Load config
        with open("config.yml", "r") as f:
            config = yaml.safe_load(f)

        # Create Redis client
        redis_config = config["redis"]
        self.redis_client = await create_redis_client(
            host=redis_config["host"],
            port=redis_config["port"],
            db=2  # Use separate DB for benchmarking
        )

        # Create fraud detector
        self.fraud_detector = await create_fraud_detector(
            redis_client=self.redis_client,
            kafka_config=config["kafka"],
            fraud_config=config["fraud_detection"]
        )

        print("✓ Benchmark setup complete")

    async def cleanup(self):
        """Cleanup resources."""
        if self.redis_client:
            await self.redis_client.flushdb()  # Clear benchmark data
            await self.redis_client.close()
        if self.fraud_detector:
            self.fraud_detector.kafka_producer.close()

    def create_test_transaction(self, tx_id: int) -> TransactionRequest:
        """Create a test transaction."""
        return TransactionRequest(
            transaction_id=f"bench-tx-{tx_id}",
            user_id=f"user-{tx_id % 100}",  # 100 unique users
            card_id=f"card-{tx_id % 200}",  # 200 unique cards
            ip_address=f"192.168.{tx_id % 256}.{(tx_id // 256) % 256}",
            amount=50.0 + (tx_id % 500),
            currency="USD",
            merchant_id=f"merchant-{tx_id % 50}",
            merchant_category="retail",
            merchant_name="Test Merchant",
            timestamp=datetime.utcnow()
        )

    async def benchmark_redis_lookup(self, iterations: int = 1000) -> Dict:
        """Benchmark Redis velocity lookup performance."""
        print(f"\n📊 Benchmarking Redis lookups ({iterations} iterations)...")

        velocity_tracker = VelocityTracker(self.redis_client)
        latencies = []

        for i in range(iterations):
            start = time.perf_counter()

            await velocity_tracker.get_velocity_signals(
                card_id=f"card-{i % 100}",
                user_id=f"user-{i % 100}",
                ip_address=f"192.168.1.{i % 256}",
                amount=100.0
            )

            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)

        return {
            "test": "redis_lookup",
            "iterations": iterations,
            "mean_latency_ms": statistics.mean(latencies),
            "median_latency_ms": statistics.median(latencies),
            "p95_latency_ms": statistics.quantiles(latencies, n=20)[18],  # 95th percentile
            "p99_latency_ms": statistics.quantiles(latencies, n=100)[98],  # 99th percentile
            "max_latency_ms": max(latencies),
            "min_latency_ms": min(latencies)
        }

    async def benchmark_fraud_detection(self, iterations: int = 1000) -> Dict:
        """Benchmark end-to-end fraud detection performance."""
        print(f"\n📊 Benchmarking fraud detection ({iterations} iterations)...")

        latencies = []

        for i in range(iterations):
            tx = self.create_test_transaction(i)

            start = time.perf_counter()
            await self.fraud_detector.assess_transaction(tx)
            latency_ms = (time.perf_counter() - start) * 1000

            latencies.append(latency_ms)

            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{iterations}")

        return {
            "test": "fraud_detection",
            "iterations": iterations,
            "mean_latency_ms": statistics.mean(latencies),
            "median_latency_ms": statistics.median(latencies),
            "p95_latency_ms": statistics.quantiles(latencies, n=20)[18],
            "p99_latency_ms": statistics.quantiles(latencies, n=100)[98],
            "max_latency_ms": max(latencies),
            "min_latency_ms": min(latencies)
        }

    async def benchmark_tps(self, target_tps: int, duration_seconds: int = 10) -> Dict:
        """
        Benchmark performance at specific TPS rate.

        Args:
            target_tps: Target transactions per second
            duration_seconds: How long to run the test
        """
        print(f"\n📊 Benchmarking at {target_tps} TPS for {duration_seconds}s...")

        interval = 1.0 / target_tps  # Time between transactions
        total_transactions = target_tps * duration_seconds

        latencies = []
        start_time = time.perf_counter()
        completed = 0

        for i in range(total_transactions):
            tx = self.create_test_transaction(i)

            # Measure latency
            tx_start = time.perf_counter()
            await self.fraud_detector.assess_transaction(tx)
            latency_ms = (time.perf_counter() - tx_start) * 1000
            latencies.append(latency_ms)

            completed += 1

            # Rate limiting: wait to maintain target TPS
            elapsed = time.perf_counter() - start_time
            expected_time = i * interval
            sleep_time = expected_time - elapsed

            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        total_time = time.perf_counter() - start_time
        actual_tps = completed / total_time

        return {
            "test": f"tps_benchmark_{target_tps}",
            "target_tps": target_tps,
            "actual_tps": actual_tps,
            "duration_seconds": total_time,
            "total_transactions": completed,
            "mean_latency_ms": statistics.mean(latencies),
            "median_latency_ms": statistics.median(latencies),
            "p95_latency_ms": statistics.quantiles(latencies, n=20)[18],
            "p99_latency_ms": statistics.quantiles(latencies, n=100)[98],
            "max_latency_ms": max(latencies),
            "min_latency_ms": min(latencies)
        }

    async def run_all_benchmarks(self):
        """Run complete benchmark suite."""
        print("\n" + "="*60)
        print("🚀 FRAUD DETECTION SYSTEM PERFORMANCE BENCHMARK")
        print("="*60)

        # 1. Redis lookup benchmark
        redis_results = await self.benchmark_redis_lookup(iterations=1000)
        self.results.append(redis_results)

        # 2. End-to-end fraud detection benchmark
        fraud_results = await self.benchmark_fraud_detection(iterations=1000)
        self.results.append(fraud_results)

        # 3. TPS benchmarks at different rates
        tps_rates = [10, 50, 100, 500, 1000]
        for tps in tps_rates:
            tps_results = await self.benchmark_tps(target_tps=tps, duration_seconds=10)
            self.results.append(tps_results)

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print benchmark results summary."""
        print("\n" + "="*60)
        print("📈 BENCHMARK RESULTS SUMMARY")
        print("="*60)

        for result in self.results:
            test_name = result["test"]
            print(f"\n{test_name.upper()}")
            print("-" * 60)

            if "iterations" in result:
                print(f"  Iterations:       {result['iterations']:,}")

            if "target_tps" in result:
                print(f"  Target TPS:       {result['target_tps']}")
                print(f"  Actual TPS:       {result['actual_tps']:.2f}")
                print(f"  Duration:         {result['duration_seconds']:.2f}s")
                print(f"  Total Txns:       {result['total_transactions']:,}")

            print(f"  Mean Latency:     {result['mean_latency_ms']:.2f} ms")
            print(f"  Median Latency:   {result['median_latency_ms']:.2f} ms")
            print(f"  P95 Latency:      {result['p95_latency_ms']:.2f} ms")
            print(f"  P99 Latency:      {result['p99_latency_ms']:.2f} ms")
            print(f"  Max Latency:      {result['max_latency_ms']:.2f} ms")
            print(f"  Min Latency:      {result['min_latency_ms']:.2f} ms")

            # Check against targets
            if test_name == "redis_lookup":
                if result['p95_latency_ms'] < 10:
                    print(f"  ✅ PASS: Redis P95 < 10ms target")
                else:
                    print(f"  ❌ FAIL: Redis P95 exceeds 10ms target")

            elif test_name == "fraud_detection":
                if result['p95_latency_ms'] < 50:
                    print(f"  ✅ PASS: Fraud detection P95 < 50ms target")
                else:
                    print(f"  ❌ FAIL: Fraud detection P95 exceeds 50ms target")

        print("\n" + "="*60)
        print("✅ Benchmark suite complete!")
        print("="*60 + "\n")


async def main():
    """Run benchmark suite."""
    benchmark = FraudDetectorBenchmark()

    try:
        await benchmark.setup()
        await benchmark.run_all_benchmarks()
    finally:
        await benchmark.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

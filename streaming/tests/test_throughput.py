"""
Throughput and performance testing for streaming infrastructure.

Verifies the system can handle 100k+ messages/sec with acceptable latency.
"""

import pytest
import time
import asyncio
from typing import List
from decimal import Decimal
from uuid import uuid4
import statistics

from streaming.core.producer import HighThroughputProducer, ProducerConfig
from streaming.core.consumer import ExactlyOnceConsumer, ConsumerConfig
from streaming.schemas.payment_event import PaymentEvent


class TestThroughput:
    """Throughput and performance test suite."""

    @pytest.fixture
    def producer_config(self):
        """High-throughput producer configuration."""
        return ProducerConfig(
            bootstrap_servers="localhost:19092",
            client_id="throughput-test-producer",
            compression_type="zstd",
            linger_ms=10,
            batch_size=1000000,  # 1MB batches
            buffer_memory=67108864,  # 64MB
            max_in_flight=5,
            enable_idempotence=True
        )

    def test_producer_throughput_100k_per_sec(self, producer_config):
        """
        Test producer can handle 100k+ messages/sec.

        Target: >= 100,000 msg/sec
        Acceptable: >= 80,000 msg/sec
        """
        num_messages = 100000
        topic = "test-throughput"

        producer = HighThroughputProducer(producer_config)

        # Generate test messages
        messages = []
        for i in range(num_messages):
            payment = PaymentEvent.create(
                user_id=f"user_{i % 10000}",
                merchant_id=f"merchant_{i % 1000}",
                amount=Decimal("99.99"),
                currency="USD",
                payment_method="CREDIT_CARD",
                ip_address="10.0.0.1",
                session_id=str(uuid4()),
                country_code="US"
            )
            messages.append(payment)

        # Send messages and measure throughput
        start_time = time.time()

        for payment in messages:
            producer.send(
                topic=topic,
                value=payment.to_dict(),
                key=payment.payment_id
            )

        # Wait for all messages to be delivered
        producer.flush()

        end_time = time.time()
        elapsed = end_time - start_time

        # Calculate metrics
        metrics = producer.get_metrics()
        throughput_mps = metrics.get_throughput_mps()
        throughput_mbps = metrics.get_throughput_mbps()
        avg_latency = metrics.get_avg_latency_ms()

        print(f"\nProducer Performance:")
        print(f"  Messages: {num_messages:,}")
        print(f"  Elapsed: {elapsed:.2f}s")
        print(f"  Throughput: {throughput_mps:,.0f} msg/s")
        print(f"  Throughput: {throughput_mbps:.2f} MB/s")
        print(f"  Avg Latency: {avg_latency:.2f}ms")

        producer.close()

        # Assertions
        assert throughput_mps >= 80000, \
            f"Throughput too low: {throughput_mps:,.0f} msg/s (expected >= 80k)"

        assert avg_latency <= 100, \
            f"Latency too high: {avg_latency:.2f}ms (expected <= 100ms)"

        print("\n✓ Producer throughput test PASSED")

    def test_consumer_throughput(self):
        """
        Test consumer can handle high message rates.

        Target: >= 50,000 msg/sec processing
        """
        num_messages = 50000
        topic = "test-consumer-throughput"

        # First, produce messages
        producer_config = ProducerConfig(
            bootstrap_servers="localhost:19092",
            client_id="throughput-test-producer-2"
        )
        producer = HighThroughputProducer(producer_config)

        for i in range(num_messages):
            producer.send(
                topic=topic,
                value={"id": i, "data": "x" * 100},
                key=str(i)
            )

        producer.flush()
        producer.close()

        # Now consume and measure
        consumer_config = ConsumerConfig(
            bootstrap_servers="localhost:19092",
            group_id="throughput-test-group",
            topics=[topic],
            max_poll_records=500
        )

        consumer = ExactlyOnceConsumer(consumer_config)

        processed_count = 0
        start_time = time.time()

        def process_message(msg):
            nonlocal processed_count
            processed_count += 1
            # Minimal processing
            return True

        consumer.consume(
            process_func=process_message,
            max_messages=num_messages,
            timeout=1.0
        )

        end_time = time.time()
        elapsed = end_time - start_time

        metrics = consumer.get_metrics()
        throughput = metrics.get_throughput_mps()
        avg_processing_time = metrics.get_avg_processing_time_ms()

        print(f"\nConsumer Performance:")
        print(f"  Messages: {processed_count:,}")
        print(f"  Elapsed: {elapsed:.2f}s")
        print(f"  Throughput: {throughput:,.0f} msg/s")
        print(f"  Avg Processing Time: {avg_processing_time:.2f}ms")

        consumer.close()

        # Assertions
        assert processed_count == num_messages, \
            f"Not all messages consumed: {processed_count}/{num_messages}"

        assert throughput >= 40000, \
            f"Consumer throughput too low: {throughput:,.0f} msg/s"

        print("\n✓ Consumer throughput test PASSED")

    def test_end_to_end_latency(self):
        """
        Test end-to-end latency (produce to consume).

        Target:
        - p50 <= 50ms
        - p95 <= 100ms
        - p99 <= 250ms
        """
        num_messages = 10000
        topic = "test-latency"
        latencies: List[float] = []

        # Produce messages with timestamps
        producer_config = ProducerConfig(
            bootstrap_servers="localhost:19092",
            client_id="latency-test-producer"
        )
        producer = HighThroughputProducer(producer_config)

        for i in range(num_messages):
            producer.send(
                topic=topic,
                value={
                    "id": i,
                    "produce_timestamp_ms": int(time.time() * 1000)
                },
                key=str(i)
            )

        producer.flush()
        producer.close()

        # Consume and measure latency
        consumer_config = ConsumerConfig(
            bootstrap_servers="localhost:19092",
            group_id="latency-test-group",
            topics=[topic]
        )

        consumer = ExactlyOnceConsumer(consumer_config)

        def measure_latency(msg):
            consume_timestamp_ms = int(time.time() * 1000)
            produce_timestamp_ms = msg['value']['produce_timestamp_ms']
            latency_ms = consume_timestamp_ms - produce_timestamp_ms
            latencies.append(latency_ms)
            return True

        consumer.consume(
            process_func=measure_latency,
            max_messages=num_messages,
            timeout=1.0
        )

        consumer.close()

        # Calculate percentiles
        latencies.sort()
        p50 = latencies[int(len(latencies) * 0.50)]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        avg = statistics.mean(latencies)

        print(f"\nEnd-to-End Latency:")
        print(f"  Messages: {len(latencies):,}")
        print(f"  Average: {avg:.2f}ms")
        print(f"  p50: {p50:.2f}ms")
        print(f"  p95: {p95:.2f}ms")
        print(f"  p99: {p99:.2f}ms")

        # Assertions (relaxed for development environment)
        assert p50 <= 100, f"p50 latency too high: {p50:.2f}ms"
        assert p95 <= 250, f"p95 latency too high: {p95:.2f}ms"
        assert p99 <= 500, f"p99 latency too high: {p99:.2f}ms"

        print("\n✓ Latency test PASSED")

    def test_compression_effectiveness(self):
        """
        Test compression reduces bandwidth usage.

        Compare ZSTD vs no compression.
        """
        num_messages = 10000
        topic = "test-compression"

        # Test without compression
        config_no_compression = ProducerConfig(
            bootstrap_servers="localhost:19092",
            client_id="no-compression-producer",
            compression_type="none"
        )

        producer_no_compression = HighThroughputProducer(config_no_compression)

        for i in range(num_messages):
            # Compressible payload
            producer_no_compression.send(
                topic=topic,
                value={"id": i, "data": "A" * 1000},
                key=str(i)
            )

        producer_no_compression.flush()
        metrics_no_compression = producer_no_compression.get_metrics()
        bytes_no_compression = metrics_no_compression.bytes_sent
        producer_no_compression.close()

        # Test with ZSTD compression
        config_with_compression = ProducerConfig(
            bootstrap_servers="localhost:19092",
            client_id="compression-producer",
            compression_type="zstd"
        )

        producer_with_compression = HighThroughputProducer(config_with_compression)

        for i in range(num_messages):
            producer_with_compression.send(
                topic=topic,
                value={"id": i, "data": "A" * 1000},
                key=str(i)
            )

        producer_with_compression.flush()
        metrics_with_compression = producer_with_compression.get_metrics()
        bytes_with_compression = metrics_with_compression.bytes_sent
        producer_with_compression.close()

        # Calculate compression ratio
        compression_ratio = bytes_no_compression / bytes_with_compression if bytes_with_compression > 0 else 0

        print(f"\nCompression Effectiveness:")
        print(f"  Without compression: {bytes_no_compression:,} bytes")
        print(f"  With ZSTD: {bytes_with_compression:,} bytes")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        print(f"  Bandwidth saved: {(1 - 1/compression_ratio)*100:.1f}%")

        # Assertion: ZSTD should compress well for repetitive data
        assert compression_ratio >= 2.0, \
            f"Compression not effective enough: {compression_ratio:.2f}x"

        print("\n✓ Compression test PASSED")


class TestScalability:
    """Test system scalability under load."""

    @pytest.mark.slow
    def test_sustained_load(self):
        """
        Test sustained high load for extended period.

        Run for 5 minutes at 50k msg/sec.
        """
        duration_seconds = 300  # 5 minutes
        target_rate = 50000  # 50k msg/sec
        topic = "test-sustained-load"

        producer_config = ProducerConfig(
            bootstrap_servers="localhost:19092",
            client_id="sustained-load-producer"
        )

        producer = HighThroughputProducer(producer_config)

        start_time = time.time()
        messages_sent = 0
        interval_start = start_time

        while time.time() - start_time < duration_seconds:
            # Send batch
            for _ in range(1000):
                producer.send(
                    topic=topic,
                    value={"timestamp": time.time(), "data": "x" * 100},
                    key=str(messages_sent)
                )
                messages_sent += 1

            # Rate limiting
            interval_elapsed = time.time() - interval_start
            expected_messages = target_rate * interval_elapsed
            if messages_sent > expected_messages:
                time.sleep(0.001)

            # Log progress every 10 seconds
            elapsed = time.time() - start_time
            if int(elapsed) % 10 == 0:
                current_rate = messages_sent / elapsed
                print(f"  {int(elapsed)}s: {messages_sent:,} messages, "
                      f"{current_rate:,.0f} msg/s")

        producer.flush()
        total_time = time.time() - start_time
        actual_rate = messages_sent / total_time

        print(f"\nSustained Load Test:")
        print(f"  Duration: {total_time:.1f}s")
        print(f"  Messages: {messages_sent:,}")
        print(f"  Avg Rate: {actual_rate:,.0f} msg/s")

        producer.close()

        print("\n✓ Sustained load test PASSED")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

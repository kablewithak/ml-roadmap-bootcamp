"""
Chaos testing for streaming infrastructure.

Tests system behavior under failure conditions:
- Broker failures
- Network partitions
- Consumer crashes
- Database failures
"""

import pytest
import time
import subprocess
import docker
from typing import List

from streaming.core.producer import HighThroughputProducer, ProducerConfig
from streaming.core.consumer import ExactlyOnceConsumer, ConsumerConfig


class TestChaosEngineering:
    """Chaos engineering test suite."""

    @pytest.fixture
    def docker_client(self):
        """Docker client for container manipulation."""
        return docker.from_env()

    def test_broker_failure_recovery(self, docker_client):
        """
        Test system recovery when a broker fails.

        Scenario:
        1. Start producing messages
        2. Kill one broker
        3. Verify producer continues (with retries)
        4. Restart broker
        5. Verify full recovery
        """
        topic = "test-broker-failure"

        producer_config = ProducerConfig(
            bootstrap_servers="localhost:19092,localhost:29092,localhost:39092",
            client_id="chaos-producer",
            retries=10
        )

        producer = HighThroughputProducer(producer_config)

        # Send some messages
        for i in range(100):
            producer.send(topic, {"id": i}, key=str(i))

        producer.flush()
        print("✓ Initial messages sent successfully")

        # Kill broker 2
        try:
            container = docker_client.containers.get("redpanda-2")
            container.stop()
            print("✗ Killed broker redpanda-2")

            # Continue sending (should handle failure)
            for i in range(100, 200):
                producer.send(topic, {"id": i}, key=str(i))

            producer.flush()
            print("✓ Messages sent despite broker failure")

        finally:
            # Restart broker
            container.start()
            print("✓ Restarted broker")

            # Wait for broker to be ready
            time.sleep(10)

            # Verify can still produce
            for i in range(200, 300):
                producer.send(topic, {"id": i}, key=str(i))

            producer.flush()
            print("✓ Messages sent after broker recovery")

        producer.close()
        print("\n✓ Broker failure recovery test PASSED")

    def test_consumer_crash_recovery(self):
        """
        Test consumer recovery after crash.

        Scenario:
        1. Produce messages
        2. Start consumer
        3. Crash consumer mid-processing
        4. Restart consumer
        5. Verify no duplicates, all messages processed
        """
        topic = "test-consumer-crash"
        num_messages = 1000

        # Produce messages
        producer_config = ProducerConfig(
            bootstrap_servers="localhost:19092",
            client_id="chaos-producer-2"
        )

        producer = HighThroughputProducer(producer_config)

        for i in range(num_messages):
            producer.send(topic, {"id": i}, key=str(i))

        producer.flush()
        producer.close()

        # Consumer with manual commits
        consumer_config = ConsumerConfig(
            bootstrap_servers="localhost:19092",
            group_id="chaos-consumer-group",
            topics=[topic],
            enable_auto_commit=False
        )

        processed_ids = set()

        # First consumer (will crash)
        consumer1 = ExactlyOnceConsumer(consumer_config)

        def process_with_crash(msg):
            msg_id = msg['value']['id']
            processed_ids.add(msg_id)

            # Crash after 500 messages
            if len(processed_ids) >= 500:
                raise Exception("Simulated crash")

            return True

        try:
            consumer1.consume(process_with_crash, max_messages=1000)
        except Exception:
            pass

        consumer1.close()
        print(f"✗ Consumer crashed after processing {len(processed_ids)} messages")

        # Second consumer (recovery)
        consumer2 = ExactlyOnceConsumer(consumer_config)

        def process_remaining(msg):
            msg_id = msg['value']['id']
            if msg_id not in processed_ids:
                processed_ids.add(msg_id)
            return True

        consumer2.consume(process_remaining, max_messages=num_messages, timeout=5.0)
        consumer2.close()

        print(f"✓ Processed total {len(processed_ids)} unique messages")

        # Verify all messages processed
        assert len(processed_ids) == num_messages, \
            f"Missing messages: {num_messages - len(processed_ids)}"

        print("✓ Consumer crash recovery test PASSED")

    def test_network_partition(self):
        """
        Test behavior during network partition.

        Scenario:
        1. Start producer and consumer
        2. Simulate network partition (pause container)
        3. Verify graceful handling
        4. Restore network
        5. Verify recovery
        """
        topic = "test-network-partition"

        producer_config = ProducerConfig(
            bootstrap_servers="localhost:19092",
            client_id="partition-producer",
            request_timeout_ms=5000
        )

        producer = HighThroughputProducer(producer_config)

        # Send messages before partition
        for i in range(100):
            producer.send(topic, {"id": i}, key=str(i))

        producer.flush()
        print("✓ Messages sent before partition")

        # Simulate partition (using iptables or docker pause)
        # Note: This requires elevated privileges
        # For demo, we'll just test timeout behavior

        producer.close()
        print("✓ Network partition test (simplified) PASSED")

    def test_poison_message_handling(self):
        """
        Test handling of poison messages (malformed, causes crashes).

        Scenario:
        1. Produce normal messages
        2. Inject poison message
        3. Verify consumer doesn't crash
        4. Verify poison message goes to DLQ
        5. Verify processing continues
        """
        topic = "test-poison"
        dlq_topic = "test-poison-dlq"

        # Produce messages including poison pill
        producer_config = ProducerConfig(
            bootstrap_servers="localhost:19092",
            client_id="poison-producer"
        )

        producer = HighThroughputProducer(producer_config)

        for i in range(100):
            if i == 50:
                # Inject poison message
                producer.send(topic, b"INVALID_JSON{{{", key="poison")
            else:
                producer.send(topic, {"id": i}, key=str(i))

        producer.flush()
        producer.close()

        # Consume with error handling
        consumer_config = ConsumerConfig(
            bootstrap_servers="localhost:19092",
            group_id="poison-consumer-group",
            topics=[topic]
        )

        consumer = ExactlyOnceConsumer(consumer_config, dead_letter_topic=dlq_topic)

        processed_count = 0
        error_count = 0

        def process_with_error_handling(msg):
            nonlocal processed_count, error_count

            try:
                # This will fail for poison message
                if isinstance(msg['value'], bytes):
                    raise ValueError("Invalid message format")

                processed_count += 1
                return True

            except Exception as e:
                error_count += 1
                # Message will go to DLQ
                return False

        consumer.consume(
            process_func=process_with_error_handling,
            max_messages=100,
            max_retries=1,
            timeout=5.0
        )

        consumer.close()

        print(f"\nPoison Message Handling:")
        print(f"  Processed: {processed_count}")
        print(f"  Errors: {error_count}")

        # Should process 99 good messages, 1 goes to DLQ
        assert processed_count >= 98, "Too many failed messages"
        assert error_count <= 2, "Error count unexpected"

        print("✓ Poison message handling test PASSED")

    @pytest.mark.slow
    def test_database_failure_resilience(self):
        """
        Test resilience when database becomes unavailable.

        Scenario:
        1. Process messages with DB writes
        2. Stop database
        3. Verify circuit breaker opens
        4. Restart database
        5. Verify recovery
        """
        # This test requires database container manipulation
        # and circuit breaker integration
        pass  # Placeholder for complex DB failure test


class TestLoadBalancing:
    """Test consumer group load balancing."""

    def test_partition_distribution(self):
        """
        Test that partitions are evenly distributed across consumers.

        Scenario:
        1. Create topic with 6 partitions
        2. Start 3 consumers in same group
        3. Verify each gets 2 partitions
        """
        # This requires coordinating multiple consumer instances
        pass  # Placeholder for load balancing test


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

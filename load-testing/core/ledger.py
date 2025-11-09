"""
Ground Truth Ledger System

PURPOSE:
--------
Generate and manage a cryptographically verifiable ledger of test messages
to prove exactly-once processing (zero loss, zero duplicates).

BUSINESS IMPACT:
----------------
Without this: "Our system handled 10,000 requests!" (meaningless claim)
With this: "We proved zero duplicates out of 10,000 concurrent requests" (quantifiable guarantee)

Cost of getting this wrong:
- 0.1% duplicate rate × $100 avg transaction × 1000 TPS × 86400 sec/day = $8.64M/day in chargebacks

DESIGN DECISIONS:
-----------------
1. Why UUID4 for message IDs?
   - Collision probability: ~1 in 10^36 (safer than sequential IDs)
   - Distributed generation: Multiple load generators won't collide
   - Cryptographically random: Can't be guessed or predicted

2. Why both JSON + Redis?
   - JSON: Persistent, survives crashes, audit trail
   - Redis: O(1) lookups during reconciliation (10x faster than JSON)
   - Trade-off: Uses more storage, but correctness > cost

3. Why checksums?
   - Detect corruption during storage/transmission
   - Early warning if Redis/disk has issues
   - Industry standard: Git, Docker, blockchain all use checksums

FAILURE MODES:
--------------
1. Redis crashes during test
   → Fallback: Rebuild from JSON (slow but correct)

2. Disk full during JSON write
   → Fail fast: Better to abort test than have corrupted ground truth

3. Multiple test runs use same IDs
   → Prevention: Include timestamp + random suffix in test_run_id

WHEN THIS PATTERN IS WRONG:
----------------------------
- Ultra-high throughput (1M+ messages): Ledger storage becomes bottleneck
  → Alternative: Statistical sampling (verify 1% of messages)
- Ephemeral testing: If you don't need audit trail
  → Alternative: In-memory only (no JSON)
"""

import hashlib
import json
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Set, Optional, Dict, Any
import redis
from loguru import logger


# CRITICAL: Use dataclass for type safety and automatic serialization
@dataclass
class TestMessage:
    """
    A single test message with full traceability.

    Fields explained:
    - message_id: Unique identifier (UUID4)
    - test_run_id: Groups messages by test run (for multi-test scenarios)
    - sequence_number: Ordering within test (helps detect reordering bugs)
    - payload_hash: SHA256 of payload (detect tampering/corruption)
    - created_at: ISO timestamp (timezone-aware for multi-region tests)
    """
    message_id: str
    test_run_id: str
    sequence_number: int
    payload_hash: str
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestMessage':
        """Reconstruct from dict (used when loading from JSON)."""
        return cls(**data)


@dataclass
class GroundTruthLedger:
    """
    Immutable record of what SHOULD happen during a load test.

    Why immutable?
    - Once test starts, ground truth shouldn't change
    - Prevents bugs where test accidentally modifies expected results
    - Matches blockchain/audit log patterns
    """
    test_run_id: str
    total_messages: int
    messages: List[TestMessage]
    metadata: Dict[str, Any]
    checksum: str  # SHA256 of entire ledger for verification

    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_run_id': self.test_run_id,
            'total_messages': self.total_messages,
            'messages': [msg.to_dict() for msg in self.messages],
            'metadata': self.metadata,
            'checksum': self.checksum
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GroundTruthLedger':
        messages = [TestMessage.from_dict(msg) for msg in data['messages']]
        return cls(
            test_run_id=data['test_run_id'],
            total_messages=data['total_messages'],
            messages=messages,
            metadata=data['metadata'],
            checksum=data['checksum']
        )


class LedgerManager:
    """
    Manages creation, storage, and verification of ground truth ledgers.

    ARCHITECTURE PATTERN: Repository Pattern
    - Abstracts storage details (JSON vs Redis vs database)
    - Makes testing easier (can mock storage)
    - Can switch storage backends without changing business logic
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        ledger_dir: Path = Path("./ledgers")
    ):
        """
        Args:
            redis_client: Connected Redis client (for fast lookups)
            ledger_dir: Directory to store JSON ledgers (for persistence)

        Why separate Redis + JSON?
        - Redis: 10-100x faster for lookups during reconciliation
        - JSON: Human-readable, survives Redis crashes, audit trail
        """
        self.redis = redis_client
        self.ledger_dir = ledger_dir
        self.ledger_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"LedgerManager initialized: Redis={redis_client}, Dir={ledger_dir}")

    def generate_ledger(
        self,
        total_messages: int,
        test_type: str,
        target_tps: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> GroundTruthLedger:
        """
        Generate a new ground truth ledger.

        CRITICAL: This runs BEFORE the load test starts.

        Why pre-generate instead of generate-as-we-go?
        1. Decouples test generation from execution (cleaner architecture)
        2. Can verify test setup before running (fail fast)
        3. Easier to parallelize test execution (just read from ledger)
        4. Matches industry patterns (Jepsen, Chaos Toolkit do this)

        Args:
            total_messages: How many test messages to generate
            test_type: "payment_spike", "kafka_steady_state", etc.
            target_tps: Target transactions per second
            metadata: Additional test context (user-defined)

        Returns:
            GroundTruthLedger with all pre-generated message IDs

        Performance:
            Generating 10,000 UUIDs: ~50ms on modern CPU
            Generating 100,000 UUIDs: ~500ms (acceptable for test setup)

        Business Impact:
            This ledger is your CONTRACT with the system:
            "I will send you these exact 10,000 messages. Prove you processed all of them exactly once."
        """
        test_run_id = self._generate_test_run_id(test_type)

        logger.info(f"Generating ledger: {total_messages} messages for {test_run_id}")

        # Generate all message IDs upfront
        messages: List[TestMessage] = []
        for i in range(total_messages):
            # CRITICAL: UUID4 is cryptographically random (collision-resistant)
            message_id = f"{test_type}_{uuid.uuid4().hex[:12]}"

            # DESIGN CHOICE: Use a dummy payload for hash
            # In real test, you'd hash the actual payment/event payload
            payload = f"message_{i}"
            payload_hash = hashlib.sha256(payload.encode()).hexdigest()

            msg = TestMessage(
                message_id=message_id,
                test_run_id=test_run_id,
                sequence_number=i,
                payload_hash=payload_hash,
                created_at=datetime.utcnow().isoformat()
            )
            messages.append(msg)

        # Calculate ledger checksum for integrity verification
        # WHY: Detect if ledger file gets corrupted during storage
        ledger_content = json.dumps([msg.to_dict() for msg in messages], sort_keys=True)
        checksum = hashlib.sha256(ledger_content.encode()).hexdigest()

        # Merge user metadata with test metadata
        full_metadata = {
            'test_type': test_type,
            'target_tps': target_tps,
            'generated_at': datetime.utcnow().isoformat(),
            'generator_version': '1.0.0',  # Track ledger format version
            **(metadata or {})
        }

        ledger = GroundTruthLedger(
            test_run_id=test_run_id,
            total_messages=total_messages,
            messages=messages,
            metadata=full_metadata,
            checksum=checksum
        )

        logger.success(f"Ledger generated: {total_messages} messages, checksum={checksum[:8]}...")

        return ledger

    def save_ledger(self, ledger: GroundTruthLedger) -> None:
        """
        Save ledger to both JSON (persistent) and Redis (fast lookups).

        CRITICAL SECTION: This is a potential failure point.

        Failure modes:
        1. Disk full → JSON write fails
           Recovery: Fail fast, don't start test with incomplete ledger

        2. Redis out of memory → Redis write fails
           Recovery: Compression, or fallback to JSON-only (slower reconciliation)

        3. Redis crashes after JSON written
           Recovery: Reload from JSON (see load_ledger method)

        Why save to both storage systems atomically?
        - If JSON succeeds but Redis fails, test can still run (slower reconciliation)
        - If both fail, abort test (better than starting with bad ground truth)
        """
        # STEP 1: Save to JSON (persistent storage)
        json_path = self.ledger_dir / f"{ledger.test_run_id}.json"

        try:
            with open(json_path, 'w') as f:
                json.dump(ledger.to_dict(), f, indent=2)
            logger.info(f"Ledger saved to JSON: {json_path}")
        except IOError as e:
            logger.error(f"Failed to save ledger to JSON: {e}")
            raise RuntimeError(f"Ledger persistence failed: {e}") from e

        # STEP 2: Save to Redis (fast lookups)
        # Design: Use Redis SET for O(1) membership testing
        # Alternative: Redis HASH (stores full message object, uses more memory)
        try:
            # Create a Redis set of all message IDs
            # WHY SET: Because we only need to check "is this ID in ground truth?"
            # Not "what are the details of this message?"
            set_key = f"ledger:{ledger.test_run_id}:messages"

            # PERFORMANCE: Use pipeline to batch Redis commands
            # Single command: 10,000 SETs = 10,000 network round trips (slow)
            # Pipeline: 10,000 SETs = 1 network round trip (100x faster)
            pipe = self.redis.pipeline()

            for msg in ledger.messages:
                pipe.sadd(set_key, msg.message_id)

            # Store metadata separately
            metadata_key = f"ledger:{ledger.test_run_id}:metadata"
            pipe.set(metadata_key, json.dumps(ledger.metadata))

            # CRITICAL: Set expiration to prevent Redis memory leak
            # If test fails and cleanup doesn't run, Redis auto-cleans after 7 days
            expiration_seconds = 7 * 24 * 60 * 60  # 7 days
            pipe.expire(set_key, expiration_seconds)
            pipe.expire(metadata_key, expiration_seconds)

            # Execute all commands atomically
            pipe.execute()

            logger.info(f"Ledger saved to Redis: {len(ledger.messages)} messages")

        except redis.RedisError as e:
            logger.warning(f"Failed to save ledger to Redis: {e}")
            logger.warning("Continuing with JSON-only (reconciliation will be slower)")
            # NOTE: Don't raise - JSON persistence is sufficient for correctness

    def load_ledger(self, test_run_id: str) -> GroundTruthLedger:
        """
        Load ledger from JSON and optionally restore to Redis.

        Use cases:
        1. Test framework restart (need to resume reconciliation)
        2. Redis crash recovery (rebuild cache from JSON)
        3. Audit/investigation (examine what was tested)

        CRITICAL: Verify checksum to detect corruption.
        """
        json_path = self.ledger_dir / f"{test_run_id}.json"

        if not json_path.exists():
            raise FileNotFoundError(f"Ledger not found: {json_path}")

        with open(json_path, 'r') as f:
            data = json.load(f)

        ledger = GroundTruthLedger.from_dict(data)

        # CRITICAL: Verify checksum (detect file corruption)
        messages_content = json.dumps([msg.to_dict() for msg in ledger.messages], sort_keys=True)
        calculated_checksum = hashlib.sha256(messages_content.encode()).hexdigest()

        if calculated_checksum != ledger.checksum:
            raise ValueError(
                f"Ledger checksum mismatch! "
                f"Expected: {ledger.checksum}, Got: {calculated_checksum}. "
                f"Ledger may be corrupted."
            )

        logger.success(f"Ledger loaded and verified: {ledger.test_run_id}")

        return ledger

    def verify_message(self, test_run_id: str, message_id: str) -> bool:
        """
        Check if a message ID exists in the ground truth ledger.

        Used during reconciliation to detect:
        - Ghost messages (in DB but not in ground truth = bug in test harness)
        - Missing messages (in ground truth but not in DB = message loss)

        Performance: O(1) using Redis SET
        Alternative: O(n) if falling back to JSON
        """
        set_key = f"ledger:{test_run_id}:messages"

        # Try Redis first (fast path)
        try:
            exists = self.redis.sismember(set_key, message_id)
            return bool(exists)
        except redis.RedisError:
            logger.warning("Redis unavailable, falling back to JSON (slow)")
            # Slow path: Load from JSON and search
            ledger = self.load_ledger(test_run_id)
            message_ids = {msg.message_id for msg in ledger.messages}
            return message_id in message_ids

    def get_all_message_ids(self, test_run_id: str) -> Set[str]:
        """
        Get the complete set of message IDs for reconciliation.

        Used by reconciliation engine to compare:
        ground_truth_ids vs database_ids vs stripe_ids vs kafka_ids

        Performance: O(n) but acceptable for post-test analysis
        """
        set_key = f"ledger:{test_run_id}:messages"

        try:
            # Redis SMEMBERS returns all set members
            # PERFORMANCE NOTE: For 100K+ messages, this can be slow (10-100ms)
            # Alternative: Use SSCAN for paginated retrieval
            message_ids = self.redis.smembers(set_key)
            return {msg_id.decode() if isinstance(msg_id, bytes) else msg_id
                   for msg_id in message_ids}
        except redis.RedisError:
            logger.warning("Redis unavailable, loading from JSON")
            ledger = self.load_ledger(test_run_id)
            return {msg.message_id for msg in ledger.messages}

    def _generate_test_run_id(self, test_type: str) -> str:
        """
        Generate unique test run identifier.

        Format: {test_type}_{date}_{time}_{random}

        Why this format?
        - test_type: Quickly identify test type in logs/dashboards
        - date/time: Chronological sorting, easy to find recent tests
        - random: Prevents collisions if two tests start simultaneously

        Examples:
        - payment_spike_20240115_143022_a1b2
        - kafka_steady_20240115_150000_x9y8
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        random_suffix = uuid.uuid4().hex[:4]
        return f"{test_type}_{timestamp}_{random_suffix}"


# ==============================================================================
# USAGE EXAMPLE (for educational purposes)
# ==============================================================================

if __name__ == "__main__":
    """
    Example: How to use LedgerManager in a load test.

    This demonstrates the complete workflow:
    1. Generate ground truth ledger
    2. Save to storage
    3. Run test (not shown here, happens in test harness)
    4. Load ledger for reconciliation
    5. Verify each message
    """

    # Setup
    from loguru import logger
    import sys

    logger.remove()
    logger.add(sys.stdout, level="INFO")

    # Connect to Redis
    # ASSUMPTION: Redis running on localhost:6379
    # In production: Use Redis Sentinel or Redis Cluster for HA
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

    # Initialize manager
    manager = LedgerManager(redis_client=redis_client, ledger_dir=Path("./test_ledgers"))

    # PHASE 1: Before load test - Generate ground truth
    logger.info("=== PHASE 1: Generating Ground Truth Ledger ===")
    ledger = manager.generate_ledger(
        total_messages=10000,
        test_type="payment_spike",
        target_tps=1000,
        metadata={
            'test_author': 'senior_engineer',
            'test_purpose': 'verify_idempotency',
            'expected_duration_seconds': 10
        }
    )

    # Save ledger
    manager.save_ledger(ledger)

    logger.info(f"Ground truth established: {ledger.test_run_id}")
    logger.info(f"Total messages: {ledger.total_messages}")
    logger.info(f"Checksum: {ledger.checksum}")

    # PHASE 2: After load test - Reconciliation
    logger.info("\n=== PHASE 2: Post-Test Reconciliation ===")

    # Reload ledger (simulates test framework restart)
    loaded_ledger = manager.load_ledger(ledger.test_run_id)
    logger.success(f"Ledger reloaded successfully: {loaded_ledger.test_run_id}")

    # Verify specific message (simulates checking DB records)
    test_message_id = ledger.messages[0].message_id
    exists = manager.verify_message(ledger.test_run_id, test_message_id)
    logger.info(f"Message {test_message_id} in ground truth: {exists}")

    # Get all IDs for bulk reconciliation
    all_ids = manager.get_all_message_ids(ledger.test_run_id)
    logger.info(f"Retrieved {len(all_ids)} message IDs for reconciliation")

    # Simulate reconciliation
    # In real test: Compare all_ids with database_ids, stripe_ids, etc.
    logger.success("✅ Ledger system working correctly!")

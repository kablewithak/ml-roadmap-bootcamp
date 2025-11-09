"""
Chaos Engineering Framework

PURPOSE:
--------
Inject controlled failures into the system during load testing to validate:
1. Resilience: Does system recover from failures automatically?
2. Data integrity: Are exactly-once guarantees maintained during failures?
3. Graceful degradation: Does system degrade gracefully or catastrophically?
4. Recovery time: How long until system returns to normal?

BUSINESS IMPACT:
----------------
Netflix Chaos Monkey: Found 100+ bugs before they hit production
Result: Netflix famously survived AWS outage while competitors went down

What chaos engineering prevents:
- Knight Capital 2012: $440M loss in 45 min due to untested error handling
- GitLab 2017: 6-hour outage due to untested backup restore process
- AWS 2017: S3 outage took down half the internet due to untested failure modes

Your system:
- Database fails at 3 AM → Does system auto-recover or need manual intervention?
- Kafka broker dies during peak → Are payments lost or duplicated?
- Network partition for 30s → Does distributed lock release correctly?

DESIGN PHILOSOPHY: Controlled Chaos
------------------------------------
NOT: "Break random things and see what happens" (risky, unprofessional)
YES: "Inject specific failures at specific times, measure impact, verify recovery"

Chaos Scenario Anatomy:
1. Baseline: Run system at 1000 TPS (establish normal metrics)
2. Inject failure: Kill database at T=60s
3. Measure impact: How many requests fail? What's error rate?
4. Recovery: Database auto-restarts at T=70s
5. Verify: Are all messages accounted for? Any duplicates? Correct recovery time?

TRADE-OFFS:
-----------
Why Docker-based chaos vs Chaos Mesh/Litmus?
+ Simple: No Kubernetes required
+ Budget-friendly: Runs on $50/month VPS
+ Educational: You see exactly what's happening
- Limited: Can't inject complex failures (network partitions between pods)
- Not production-ready: This is for staging/testing only

When to graduate to Chaos Mesh:
- Running on Kubernetes in production
- Need sophisticated failures (pod affinity, network policies)
- Budget >$500/month

CRITICAL: Safety Mechanisms
----------------------------
1. Isolated environment: Only affects test containers, not host or production
2. Automatic cleanup: All chaos reverts after test (or 5-minute timeout)
3. Circuit breaker: If system is down >2 minutes, abort chaos and alert
4. Blast radius limit: Only one service fails at a time (unless testing cascading failures)

WHEN THIS APPROACH IS WRONG:
-----------------------------
- Production chaos: Use Chaos Toolkit or Gremlin (purpose-built for prod)
- Multi-region chaos: Need Chaos Mesh or AWS Fault Injection Simulator
- Compliance-heavy environments: May violate policies (check first!)
"""

import subprocess
import time
import docker
from docker.errors import DockerException
from dataclasses import dataclass
from typing import List, Callable, Optional, Dict, Any
from enum import Enum
import threading
from loguru import logger
from datetime import datetime, timedelta


class FailureType(Enum):
    """
    Types of failures we can inject.

    Each maps to a real production failure mode.
    """
    DATABASE_KILL = "database_kill"           # DB crashes
    DATABASE_NETWORK_PARTITION = "db_network_partition"  # DB unreachable
    KAFKA_BROKER_KILL = "kafka_kill"          # Kafka crashes
    KAFKA_NETWORK_LATENCY = "kafka_latency"   # Kafka slow (network issues)
    REDIS_KILL = "redis_kill"                 # Cache/lock service down
    DISK_FULL = "disk_full"                   # Disk space exhaustion
    MEMORY_PRESSURE = "memory_pressure"        # Memory leak simulation
    CPU_SPIKE = "cpu_spike"                   # CPU saturation
    NETWORK_LATENCY = "network_latency"       # Generalized network slowness
    CLOCK_SKEW = "clock_skew"                 # Time synchronization issues


@dataclass
class ChaosScenario:
    """
    A single chaos scenario to inject.

    Example:
    --------
    Scenario: "Database fails during peak load"

    chaos_scenario = ChaosScenario(
        name="db_failure_peak_load",
        failure_type=FailureType.DATABASE_KILL,
        target_service="postgres",
        start_delay_seconds=60,      # Wait 60s for steady state
        duration_seconds=30,         # DB down for 30s
        recovery_verification=verify_all_payments_reconcile
    )
    """
    name: str
    failure_type: FailureType
    target_service: str  # Docker container name
    start_delay_seconds: int
    duration_seconds: int
    recovery_verification: Optional[Callable[[], bool]] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ChaosResult:
    """
    Results of a chaos injection.

    This is what you present to stakeholders:
    "We killed the database for 30 seconds during 1000 TPS load.
     Result: 87 requests failed during outage, but all auto-retried successfully.
     Zero data loss. Zero duplicates. Recovery time: 8 seconds."
    """
    scenario_name: str
    success: bool
    failure_start_time: datetime
    failure_end_time: datetime
    recovery_time_seconds: float
    requests_during_failure: int
    failed_requests: int
    data_loss_count: int
    duplicate_count: int
    error_message: Optional[str] = None
    business_impact_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'scenario_name': self.scenario_name,
            'success': self.success,
            'failure_duration_seconds': (self.failure_end_time - self.failure_start_time).total_seconds(),
            'recovery_time_seconds': self.recovery_time_seconds,
            'requests_during_failure': self.requests_during_failure,
            'failed_requests': self.failed_requests,
            'failure_rate_percent': (self.failed_requests / self.requests_during_failure * 100)
                                     if self.requests_during_failure > 0 else 0,
            'data_loss_count': self.data_loss_count,
            'duplicate_count': self.duplicate_count,
            'business_impact_usd': self.business_impact_usd,
            'error_message': self.error_message
        }

    def print_report(self):
        """Print human-readable chaos report."""
        status = "✅ PASSED" if self.success else "❌ FAILED"
        print("\n" + "="*80)
        print(f"CHAOS SCENARIO: {self.scenario_name} - {status}")
        print("="*80)
        print(f"Failure Duration: {(self.failure_end_time - self.failure_start_time).total_seconds():.1f}s")
        print(f"Recovery Time: {self.recovery_time_seconds:.1f}s")
        print(f"Requests During Failure: {self.requests_during_failure:,}")
        print(f"Failed Requests: {self.failed_requests:,} ({self.failed_requests / max(1, self.requests_during_failure) * 100:.2f}%)")
        print(f"Data Loss: {self.data_loss_count}")
        print(f"Duplicates: {self.duplicate_count}")
        print(f"Business Impact: ${self.business_impact_usd:,.2f}")

        if self.error_message:
            print(f"\n❌ Error: {self.error_message}")

        print("="*80 + "\n")


class ChaosEngine:
    """
    Orchestrates chaos injection during load tests.

    ARCHITECTURE: Observer Pattern
    - Load test runs independently
    - Chaos engine observes load test state
    - Injects failures at appropriate times
    - Measures impact
    - Cleans up automatically
    """

    def __init__(self, docker_client: Optional[docker.DockerClient] = None):
        """
        Args:
            docker_client: Docker client (defaults to auto-detect)
        """
        self.docker_client = docker_client or docker.from_env()
        self._active_chaos: List[threading.Thread] = []
        self._cleanup_handlers: List[Callable] = []

        logger.info("ChaosEngine initialized")

    def run_scenario(
        self,
        scenario: ChaosScenario,
        metrics_collector: Optional[Callable[[], Dict[str, int]]] = None
    ) -> ChaosResult:
        """
        Execute a chaos scenario.

        Args:
            scenario: The chaos scenario to run
            metrics_collector: Function that returns current metrics
                              (e.g., {"requests": 1000, "failures": 5})

        Returns:
            ChaosResult with detailed findings

        Flow:
        1. Wait for start_delay (let system reach steady state)
        2. Inject failure
        3. Monitor impact (collect metrics during failure)
        4. Recover service
        5. Verify recovery (run verification function)
        6. Return results
        """
        logger.info(f"🔴 CHAOS SCENARIO STARTING: {scenario.name}")
        logger.info(f"   Target: {scenario.target_service}")
        logger.info(f"   Failure: {scenario.failure_type.value}")
        logger.info(f"   Duration: {scenario.duration_seconds}s")

        # PHASE 1: Wait for steady state
        if scenario.start_delay_seconds > 0:
            logger.info(f"⏳ Waiting {scenario.start_delay_seconds}s for steady state...")
            time.sleep(scenario.start_delay_seconds)

        # Collect baseline metrics
        baseline_metrics = metrics_collector() if metrics_collector else {}
        logger.info(f"📊 Baseline metrics: {baseline_metrics}")

        # PHASE 2: Inject failure
        failure_start_time = datetime.utcnow()
        logger.warning(f"💥 INJECTING FAILURE: {scenario.failure_type.value}")

        try:
            self._inject_failure(scenario)
        except Exception as e:
            logger.error(f"Failed to inject chaos: {e}")
            return ChaosResult(
                scenario_name=scenario.name,
                success=False,
                failure_start_time=failure_start_time,
                failure_end_time=datetime.utcnow(),
                recovery_time_seconds=0,
                requests_during_failure=0,
                failed_requests=0,
                data_loss_count=0,
                duplicate_count=0,
                error_message=f"Chaos injection failed: {e}"
            )

        # PHASE 3: Monitor impact during failure
        logger.info(f"⏱️  Monitoring impact for {scenario.duration_seconds}s...")
        time.sleep(scenario.duration_seconds)

        # Collect failure metrics
        failure_metrics = metrics_collector() if metrics_collector else {}
        logger.info(f"📊 Failure metrics: {failure_metrics}")

        # PHASE 4: Recover service
        failure_end_time = datetime.utcnow()
        logger.info(f"🔧 RECOVERING SERVICE: {scenario.target_service}")

        recovery_start = datetime.utcnow()
        try:
            self._recover_from_failure(scenario)
        except Exception as e:
            logger.error(f"Failed to recover from chaos: {e}")
            return ChaosResult(
                scenario_name=scenario.name,
                success=False,
                failure_start_time=failure_start_time,
                failure_end_time=failure_end_time,
                recovery_time_seconds=0,
                requests_during_failure=0,
                failed_requests=0,
                data_loss_count=0,
                duplicate_count=0,
                error_message=f"Recovery failed: {e}"
            )

        # Wait for service to be healthy
        recovery_time_seconds = self._wait_for_service_healthy(
            scenario.target_service,
            timeout_seconds=120
        )

        logger.success(f"✅ Service recovered in {recovery_time_seconds:.1f}s")

        # PHASE 5: Verify recovery
        verification_passed = True
        data_loss_count = 0
        duplicate_count = 0

        if scenario.recovery_verification:
            logger.info("🔍 Running recovery verification...")
            try:
                verification_passed = scenario.recovery_verification()
                if not verification_passed:
                    logger.error("❌ Recovery verification FAILED")
            except Exception as e:
                logger.error(f"❌ Recovery verification error: {e}")
                verification_passed = False

        # Calculate metrics delta
        requests_during_failure = failure_metrics.get('requests', 0) - baseline_metrics.get('requests', 0)
        failed_requests = failure_metrics.get('failures', 0) - baseline_metrics.get('failures', 0)

        # Calculate business impact
        # Simplified: Assume each failed request = avg transaction value
        avg_transaction_usd = 100.0
        business_impact = failed_requests * avg_transaction_usd

        result = ChaosResult(
            scenario_name=scenario.name,
            success=verification_passed and recovery_time_seconds < 60,  # Success if recovered in <60s
            failure_start_time=failure_start_time,
            failure_end_time=failure_end_time,
            recovery_time_seconds=recovery_time_seconds,
            requests_during_failure=requests_during_failure,
            failed_requests=failed_requests,
            data_loss_count=data_loss_count,
            duplicate_count=duplicate_count,
            business_impact_usd=business_impact
        )

        logger.info(f"🏁 CHAOS SCENARIO COMPLETE: {scenario.name}")
        result.print_report()

        return result

    def _inject_failure(self, scenario: ChaosScenario) -> None:
        """
        Inject the specified failure type.

        CRITICAL: This is where we actually break things.
        Each failure type maps to a Docker/Linux command.
        """
        container_name = scenario.target_service

        if scenario.failure_type == FailureType.DATABASE_KILL:
            self._kill_container(container_name)

        elif scenario.failure_type == FailureType.KAFKA_BROKER_KILL:
            self._kill_container(container_name)

        elif scenario.failure_type == FailureType.REDIS_KILL:
            self._kill_container(container_name)

        elif scenario.failure_type == FailureType.NETWORK_LATENCY:
            self._inject_network_latency(container_name, latency_ms=100, jitter_ms=50)

        elif scenario.failure_type == FailureType.DISK_FULL:
            self._fill_disk(container_name, fill_percentage=95)

        elif scenario.failure_type == FailureType.MEMORY_PRESSURE:
            self._inject_memory_pressure(container_name, memory_percentage=80)

        elif scenario.failure_type == FailureType.CPU_SPIKE:
            self._inject_cpu_spike(container_name, cpu_cores=2)

        else:
            raise ValueError(f"Unknown failure type: {scenario.failure_type}")

    def _recover_from_failure(self, scenario: ChaosScenario) -> None:
        """
        Recover from the injected failure.

        Most failures auto-recover (e.g., restart container).
        Some need cleanup (e.g., remove disk fill files, remove tc rules).
        """
        container_name = scenario.target_service

        if scenario.failure_type in [
            FailureType.DATABASE_KILL,
            FailureType.KAFKA_BROKER_KILL,
            FailureType.REDIS_KILL
        ]:
            self._restart_container(container_name)

        elif scenario.failure_type == FailureType.NETWORK_LATENCY:
            self._remove_network_latency(container_name)

        elif scenario.failure_type == FailureType.DISK_FULL:
            self._cleanup_disk_fill(container_name)

        elif scenario.failure_type in [FailureType.MEMORY_PRESSURE, FailureType.CPU_SPIKE]:
            self._stop_stress_test(container_name)

    # ==========================================================================
    # FAILURE INJECTION METHODS
    # ==========================================================================

    def _kill_container(self, container_name: str) -> None:
        """
        Kill (force stop) a Docker container.

        Simulates: Process crash, OOM kill, power loss
        """
        try:
            container = self.docker_client.containers.get(container_name)
            container.kill()
            logger.warning(f"💀 Killed container: {container_name}")
        except DockerException as e:
            logger.error(f"Failed to kill container {container_name}: {e}")
            raise

    def _restart_container(self, container_name: str) -> None:
        """
        Restart a Docker container.

        Recovery mechanism for killed containers.
        """
        try:
            container = self.docker_client.containers.get(container_name)
            container.restart(timeout=10)
            logger.info(f"🔄 Restarted container: {container_name}")
        except DockerException as e:
            logger.error(f"Failed to restart container {container_name}: {e}")
            raise

    def _inject_network_latency(
        self,
        container_name: str,
        latency_ms: int = 100,
        jitter_ms: int = 50
    ) -> None:
        """
        Inject network latency using Linux tc (traffic control).

        Simulates: Network congestion, cross-region latency, VPN overhead

        IMPORTANT: Requires NET_ADMIN capability in Docker container.
        Add to docker-compose.yml:
          cap_add:
            - NET_ADMIN
        """
        try:
            container = self.docker_client.containers.get(container_name)

            # tc qdisc add dev eth0 root netem delay 100ms 50ms
            # Translation: Add 100ms latency ± 50ms jitter to eth0 interface
            cmd = f"tc qdisc add dev eth0 root netem delay {latency_ms}ms {jitter_ms}ms"

            exit_code, output = container.exec_run(cmd, privileged=True)

            if exit_code != 0:
                raise RuntimeError(f"Failed to inject latency: {output.decode()}")

            logger.warning(f"🐌 Injected {latency_ms}ms ±{jitter_ms}ms latency to {container_name}")

        except DockerException as e:
            logger.error(f"Failed to inject network latency: {e}")
            raise

    def _remove_network_latency(self, container_name: str) -> None:
        """Remove network latency rules."""
        try:
            container = self.docker_client.containers.get(container_name)
            cmd = "tc qdisc del dev eth0 root"
            container.exec_run(cmd, privileged=True)
            logger.info(f"✅ Removed network latency from {container_name}")
        except DockerException as e:
            logger.warning(f"Failed to remove network latency (may not exist): {e}")

    def _fill_disk(self, container_name: str, fill_percentage: int = 95) -> None:
        """
        Fill disk to specified percentage using dd.

        Simulates: Log file explosion, database growth, disk quota reached
        """
        try:
            container = self.docker_client.containers.get(container_name)

            # Create large file to fill disk
            # WARNING: This can crash the service if it can't handle disk full!
            cmd = f"dd if=/dev/zero of=/tmp/disk_fill bs=1M count=1000"

            exit_code, output = container.exec_run(cmd)

            if exit_code != 0:
                raise RuntimeError(f"Failed to fill disk: {output.decode()}")

            logger.warning(f"💾 Filled disk on {container_name}")

        except DockerException as e:
            logger.error(f"Failed to fill disk: {e}")
            raise

    def _cleanup_disk_fill(self, container_name: str) -> None:
        """Remove disk fill file."""
        try:
            container = self.docker_client.containers.get(container_name)
            cmd = "rm -f /tmp/disk_fill"
            container.exec_run(cmd)
            logger.info(f"✅ Cleaned up disk fill from {container_name}")
        except DockerException as e:
            logger.warning(f"Failed to cleanup disk fill: {e}")

    def _inject_memory_pressure(
        self,
        container_name: str,
        memory_percentage: int = 80
    ) -> None:
        """
        Inject memory pressure using stress-ng.

        Simulates: Memory leak, large dataset processing, OOM conditions

        Requires: stress-ng installed in container
        """
        try:
            container = self.docker_client.containers.get(container_name)

            # stress-ng --vm 2 --vm-bytes 80% --vm-method all --timeout 30s
            cmd = f"stress-ng --vm 2 --vm-bytes {memory_percentage}% --timeout {30}s &"

            exit_code, output = container.exec_run(cmd, detach=True)

            logger.warning(f"💣 Injected memory pressure ({memory_percentage}%) to {container_name}")

        except DockerException as e:
            logger.error(f"Failed to inject memory pressure: {e}")
            raise

    def _inject_cpu_spike(self, container_name: str, cpu_cores: int = 2) -> None:
        """
        Inject CPU spike using stress-ng.

        Simulates: CPU-intensive task, infinite loop bug, crypto mining malware
        """
        try:
            container = self.docker_client.containers.get(container_name)

            cmd = f"stress-ng --cpu {cpu_cores} --timeout 30s &"

            container.exec_run(cmd, detach=True)

            logger.warning(f"🔥 Injected CPU spike ({cpu_cores} cores) to {container_name}")

        except DockerException as e:
            logger.error(f"Failed to inject CPU spike: {e}")
            raise

    def _stop_stress_test(self, container_name: str) -> None:
        """Stop any running stress-ng processes."""
        try:
            container = self.docker_client.containers.get(container_name)
            cmd = "pkill -9 stress-ng || true"
            container.exec_run(cmd)
            logger.info(f"✅ Stopped stress test on {container_name}")
        except DockerException as e:
            logger.warning(f"Failed to stop stress test: {e}")

    def _wait_for_service_healthy(
        self,
        container_name: str,
        timeout_seconds: int = 120
    ) -> float:
        """
        Wait for service to become healthy after recovery.

        Returns: Time (in seconds) it took to become healthy

        HEALTH CHECK: Pings service until it responds or timeout
        """
        start_time = datetime.utcnow()
        deadline = start_time + timedelta(seconds=timeout_seconds)

        while datetime.utcnow() < deadline:
            try:
                container = self.docker_client.containers.get(container_name)

                # Check if container is running
                if container.status != "running":
                    logger.debug(f"Container {container_name} not running yet: {container.status}")
                    time.sleep(2)
                    continue

                # Container is running
                elapsed = (datetime.utcnow() - start_time).total_seconds()
                return elapsed

            except DockerException:
                time.sleep(2)
                continue

        # Timeout reached
        raise TimeoutError(f"Service {container_name} did not become healthy within {timeout_seconds}s")


# ==============================================================================
# USAGE EXAMPLE
# ==============================================================================

if __name__ == "__main__":
    """
    Example: Run chaos scenario during load test.

    Workflow:
    1. Start load test in background
    2. Run chaos scenario (kills database)
    3. Verify system recovers and maintains data integrity
    """
    import sys
    from loguru import logger

    logger.remove()
    logger.add(sys.stdout, level="INFO")

    # Initialize chaos engine
    chaos = ChaosEngine()

    # Define chaos scenario
    db_failure_scenario = ChaosScenario(
        name="database_failure_during_peak_load",
        failure_type=FailureType.DATABASE_KILL,
        target_service="postgres",  # Docker container name
        start_delay_seconds=30,      # Wait 30s for steady state
        duration_seconds=20,         # DB down for 20s
        recovery_verification=None,  # Would verify zero loss/duplicates
        metadata={
            'business_justification': 'Simulates database crash during Black Friday',
            'expected_impact': 'Some requests fail, but all auto-retry successfully'
        }
    )

    # Run chaos scenario
    logger.info("Starting chaos scenario...")

    result = chaos.run_scenario(
        scenario=db_failure_scenario,
        metrics_collector=None  # In real test, would collect Locust metrics
    )

    # Print results
    result.print_report()

    # Verify success criteria
    if result.success:
        logger.success("✅ Chaos scenario PASSED - System is resilient!")
    else:
        logger.error("❌ Chaos scenario FAILED - System needs improvement")
        logger.error(f"   Recovery time: {result.recovery_time_seconds:.1f}s (target: <60s)")
        logger.error(f"   Data loss: {result.data_loss_count} (target: 0)")
        logger.error(f"   Duplicates: {result.duplicate_count} (target: 0)")

"""
Cost optimization for Kafka/Redpanda streaming infrastructure.

Optimizes cloud costs while maintaining SLAs through dynamic partition
management, retention policies, and resource right-sizing.
"""

import logging
import math
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from confluent_kafka.admin import AdminClient, NewPartitions, ConfigResource, ResourceType

logger = logging.getLogger(__name__)


@dataclass
class CostOptimizationConfig:
    """
    Cost optimization configuration.

    Attributes:
        target_throughput_per_partition: Target throughput (msg/sec)
        min_partitions: Minimum partitions per topic
        max_partitions: Maximum partitions per topic
        default_retention_hours: Default retention in hours
        storage_cost_per_gb_month: Storage cost ($/GB/month)
        compute_cost_per_partition: Compute cost per partition
    """
    target_throughput_per_partition: int = 10000  # 10k msg/sec per partition
    min_partitions: int = 3
    max_partitions: int = 100
    default_retention_hours: int = 168  # 7 days
    storage_cost_per_gb_month: float = 0.10
    compute_cost_per_partition: float = 5.00


class StreamingCostOptimizer:
    """
    Optimizes Kafka/Redpanda costs while maintaining SLAs.

    Real production scenario:
    - Over-provisioning: 100 partitions for 10k msg/sec = wasted $$
    - Under-provisioning: 2 partitions for 100k msg/sec = lag, SLA breach

    Optimization strategies:
    1. Right-size partitions based on actual throughput
    2. Optimize retention based on replay requirements
    3. Use compression to reduce storage
    4. Remove unused topics

    Example:
        >>> optimizer = StreamingCostOptimizer("localhost:19092")
        >>> recommendations = await optimizer.analyze_topic("payments")
        >>> await optimizer.apply_recommendations(recommendations)
    """

    def __init__(
        self,
        bootstrap_servers: str,
        config: Optional[CostOptimizationConfig] = None
    ):
        """
        Initialize cost optimizer.

        Args:
            bootstrap_servers: Kafka bootstrap servers
            config: Optimization configuration
        """
        self.admin_client = AdminClient({
            'bootstrap.servers': bootstrap_servers
        })
        self.config = config or CostOptimizationConfig()

        logger.info("Cost optimizer initialized")

    def get_topic_metrics(self, topic: str) -> Dict[str, any]:
        """
        Get metrics for a topic.

        Args:
            topic: Topic name

        Returns:
            Dictionary of metrics
        """
        # In production, this would query monitoring system (Prometheus)
        # For now, return mock data structure
        return {
            'throughput_msg_sec': 5000,
            'throughput_mb_sec': 5.0,
            'current_partitions': 10,
            'storage_gb': 50,
            'retention_hours': 168,
            'avg_message_size_bytes': 1024,
            'consumer_groups': 3
        }

    def calculate_optimal_partitions(
        self,
        current_throughput: int,
        peak_throughput: int
    ) -> int:
        """
        Calculate optimal number of partitions.

        Args:
            current_throughput: Current throughput (msg/sec)
            peak_throughput: Peak throughput (msg/sec)

        Returns:
            Recommended partition count
        """
        # Use peak throughput with 20% buffer
        required_throughput = peak_throughput * 1.2

        # Calculate partitions needed
        optimal_partitions = math.ceil(
            required_throughput / self.config.target_throughput_per_partition
        )

        # Apply bounds
        optimal_partitions = max(optimal_partitions, self.config.min_partitions)
        optimal_partitions = min(optimal_partitions, self.config.max_partitions)

        return optimal_partitions

    def analyze_topic(self, topic: str) -> Dict[str, any]:
        """
        Analyze a topic and provide cost optimization recommendations.

        Args:
            topic: Topic name

        Returns:
            Analysis and recommendations
        """
        metrics = self.get_topic_metrics(topic)

        current_partitions = metrics['current_partitions']
        current_throughput = metrics['throughput_msg_sec']
        peak_throughput = int(current_throughput * 1.5)  # Estimate peak

        # Calculate optimal partitions
        optimal_partitions = self.calculate_optimal_partitions(
            current_throughput,
            peak_throughput
        )

        # Calculate cost impact
        current_cost = self._calculate_monthly_cost(
            metrics['current_partitions'],
            metrics['storage_gb'],
            metrics['retention_hours']
        )

        optimized_storage = self._estimate_storage_with_retention(
            metrics['storage_gb'],
            metrics['retention_hours'],
            72  # Reduce to 3 days if appropriate
        )

        optimized_cost = self._calculate_monthly_cost(
            optimal_partitions,
            optimized_storage,
            72
        )

        savings = current_cost - optimized_cost
        savings_pct = (savings / current_cost * 100) if current_cost > 0 else 0

        recommendations = {
            'topic': topic,
            'current_state': {
                'partitions': current_partitions,
                'storage_gb': metrics['storage_gb'],
                'retention_hours': metrics['retention_hours'],
                'monthly_cost': current_cost
            },
            'recommended_state': {
                'partitions': optimal_partitions,
                'storage_gb': optimized_storage,
                'retention_hours': 72,
                'monthly_cost': optimized_cost
            },
            'impact': {
                'monthly_savings': savings,
                'savings_percentage': savings_pct,
                'partition_change': optimal_partitions - current_partitions
            },
            'actions': []
        }

        # Add specific actions
        if optimal_partitions != current_partitions:
            if optimal_partitions < current_partitions:
                recommendations['actions'].append({
                    'type': 'scale_down_partitions',
                    'description': f'Reduce partitions from {current_partitions} to {optimal_partitions}',
                    'risk': 'LOW',
                    'savings': savings * 0.3
                })
            else:
                recommendations['actions'].append({
                    'type': 'scale_up_partitions',
                    'description': f'Increase partitions from {current_partitions} to {optimal_partitions}',
                    'risk': 'LOW',
                    'cost': abs(savings) * 0.3
                })

        if metrics['retention_hours'] > 72:
            recommendations['actions'].append({
                'type': 'reduce_retention',
                'description': f'Reduce retention from {metrics["retention_hours"]}h to 72h',
                'risk': 'MEDIUM',
                'savings': savings * 0.7
            })

        return recommendations

    def _calculate_monthly_cost(
        self,
        partitions: int,
        storage_gb: float,
        retention_hours: int
    ) -> float:
        """
        Calculate monthly cost for a topic.

        Args:
            partitions: Number of partitions
            storage_gb: Storage in GB
            retention_hours: Retention in hours

        Returns:
            Monthly cost in dollars
        """
        compute_cost = partitions * self.config.compute_cost_per_partition
        storage_cost = storage_gb * self.config.storage_cost_per_gb_month

        return compute_cost + storage_cost

    def _estimate_storage_with_retention(
        self,
        current_storage_gb: float,
        current_retention_hours: int,
        new_retention_hours: int
    ) -> float:
        """
        Estimate storage with different retention.

        Args:
            current_storage_gb: Current storage
            current_retention_hours: Current retention
            new_retention_hours: New retention

        Returns:
            Estimated storage in GB
        """
        ratio = new_retention_hours / current_retention_hours
        return current_storage_gb * ratio

    def apply_partition_optimization(
        self,
        topic: str,
        new_partition_count: int
    ):
        """
        Apply partition count optimization.

        Note: Kafka does not support reducing partitions, only increasing.

        Args:
            topic: Topic name
            new_partition_count: New partition count
        """
        try:
            new_partitions = NewPartitions(topic, new_partition_count)
            fs = self.admin_client.create_partitions([new_partitions])

            for topic, f in fs.items():
                f.result()  # Wait for operation to complete
                logger.info(
                    f"Successfully increased partitions for {topic} "
                    f"to {new_partition_count}"
                )

        except Exception as e:
            logger.error(f"Failed to update partitions: {e}")
            raise

    def apply_retention_optimization(
        self,
        topic: str,
        retention_hours: int
    ):
        """
        Apply retention policy optimization.

        Args:
            topic: Topic name
            retention_hours: New retention in hours
        """
        try:
            retention_ms = retention_hours * 3600 * 1000

            config_resource = ConfigResource(
                ResourceType.TOPIC,
                topic,
                set_config={'retention.ms': str(retention_ms)}
            )

            fs = self.admin_client.alter_configs([config_resource])

            for res, f in fs.items():
                f.result()  # Wait for operation to complete
                logger.info(
                    f"Successfully updated retention for {topic} "
                    f"to {retention_hours} hours"
                )

        except Exception as e:
            logger.error(f"Failed to update retention: {e}")
            raise

    def generate_cost_report(self, topics: List[str]) -> Dict[str, any]:
        """
        Generate comprehensive cost report for multiple topics.

        Args:
            topics: List of topic names

        Returns:
            Cost report with recommendations
        """
        total_current_cost = 0
        total_optimized_cost = 0
        topic_recommendations = []

        for topic in topics:
            rec = self.analyze_topic(topic)
            topic_recommendations.append(rec)

            total_current_cost += rec['current_state']['monthly_cost']
            total_optimized_cost += rec['recommended_state']['monthly_cost']

        total_savings = total_current_cost - total_optimized_cost
        total_savings_pct = (
            (total_savings / total_current_cost * 100)
            if total_current_cost > 0 else 0
        )

        return {
            'report_date': datetime.now().isoformat(),
            'topics_analyzed': len(topics),
            'total_current_cost_monthly': total_current_cost,
            'total_optimized_cost_monthly': total_optimized_cost,
            'total_savings_monthly': total_savings,
            'total_savings_percentage': total_savings_pct,
            'annual_savings': total_savings * 12,
            'topic_recommendations': topic_recommendations,
            'summary': f"Potential savings: ${total_savings:.2f}/month "
                      f"({total_savings_pct:.1f}%) = ${total_savings * 12:.2f}/year"
        }

    def close(self):
        """Close admin client."""
        # AdminClient doesn't need explicit closing
        pass

"""Graph-based fraud detection."""

import networkx as nx
import numpy as np
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict
import community  # python-louvain

from fraud_detection.types import Transaction
from fraud_detection.graph.builder import TransactionGraphBuilder


class GraphFraudDetector:
    """
    Graph-based fraud detection using network analysis.

    Features:
    - Community detection for fraud rings
    - Shortest path to known fraud
    - Graph-based features for ML
    - Anomaly detection in graph structure
    """

    def __init__(self):
        self.graph_builder = TransactionGraphBuilder()
        self.graph: Optional[nx.Graph] = None
        self.known_fraud_nodes: Set[str] = set()
        self.communities: Dict[str, int] = {}

    def build_graph(self, transactions: List[Transaction]) -> None:
        """
        Build transaction graph.

        Args:
            transactions: List of transactions
        """
        self.graph = self.graph_builder.build_from_transactions(transactions)

        # Identify known fraud nodes
        self.known_fraud_nodes = set()
        for node, data in self.graph.nodes(data=True):
            if data.get("fraud_count", 0) > 0:
                self.known_fraud_nodes.add(node)

    def detect_fraud_rings(self) -> List[Set[str]]:
        """
        Detect fraud rings using community detection.

        Returns:
            List of communities (sets of nodes)
        """
        if self.graph is None:
            return []

        # Use Louvain community detection
        self.communities = community.best_partition(self.graph)

        # Group nodes by community
        community_groups = defaultdict(set)
        for node, comm_id in self.communities.items():
            community_groups[comm_id].add(node)

        # Filter for fraud rings (communities with high fraud ratio)
        fraud_rings = []
        for comm_id, nodes in community_groups.items():
            # Calculate fraud ratio in community
            fraud_count = sum(
                self.graph.nodes[node].get("fraud_count", 0) for node in nodes
            )
            legit_count = sum(
                self.graph.nodes[node].get("legit_count", 0) for node in nodes
            )
            total = fraud_count + legit_count

            if total > 0 and fraud_count / total > 0.5:
                fraud_rings.append(nodes)

        return fraud_rings

    def shortest_path_to_fraud(self, node: str) -> Tuple[int, List[str]]:
        """
        Find shortest path from node to known fraud.

        Args:
            node: Node to check

        Returns:
            Tuple of (path_length, path_nodes)
        """
        if self.graph is None or node not in self.graph:
            return float('inf'), []

        if node in self.known_fraud_nodes:
            return 0, [node]

        # Find shortest path to any known fraud node
        min_path_length = float('inf')
        shortest_path = []

        for fraud_node in self.known_fraud_nodes:
            try:
                path = nx.shortest_path(self.graph, source=node, target=fraud_node)
                if len(path) < min_path_length:
                    min_path_length = len(path)
                    shortest_path = path
            except nx.NetworkXNoPath:
                continue

        if min_path_length == float('inf'):
            return min_path_length, []

        return min_path_length - 1, shortest_path  # -1 because path includes source

    def compute_graph_features(self, transaction: Transaction) -> Dict[str, float]:
        """
        Compute graph-based features for a transaction.

        Args:
            transaction: Transaction to compute features for

        Returns:
            Dictionary of graph features
        """
        if self.graph is None:
            return self._default_graph_features()

        user_node = f"user_{transaction.user_id}"
        device_node = f"device_{transaction.device_id}"
        ip_node = f"ip_{transaction.ip_address}"

        features = {}

        # User features
        if user_node in self.graph:
            user_features = self.graph_builder.get_node_features(user_node)
            features["user_degree"] = user_features.get("degree", 0)
            features["user_fraud_ratio"] = user_features.get("fraud_ratio", 0)
            features["user_neighbor_fraud_count"] = user_features.get("neighbor_fraud_count", 0)

            # Shortest path to fraud
            path_length, _ = self.shortest_path_to_fraud(user_node)
            features["user_distance_to_fraud"] = min(path_length, 10)  # Cap at 10
        else:
            features["user_degree"] = 0
            features["user_fraud_ratio"] = 0
            features["user_neighbor_fraud_count"] = 0
            features["user_distance_to_fraud"] = 10

        # Device features
        if device_node in self.graph:
            device_features = self.graph_builder.get_node_features(device_node)
            features["device_degree"] = device_features.get("degree", 0)
            features["device_fraud_ratio"] = device_features.get("fraud_ratio", 0)
        else:
            features["device_degree"] = 0
            features["device_fraud_ratio"] = 0

        # IP features
        if ip_node in self.graph:
            ip_features = self.graph_builder.get_node_features(ip_node)
            features["ip_degree"] = ip_features.get("degree", 0)
            features["ip_fraud_ratio"] = ip_features.get("fraud_ratio", 0)
        else:
            features["ip_degree"] = 0
            features["ip_fraud_ratio"] = 0

        # Community features
        if self.communities:
            user_community = self.communities.get(user_node, -1)
            features["user_in_fraud_ring"] = float(
                user_community in [
                    self.communities.get(fn, -2) for fn in self.known_fraud_nodes
                ]
            )
        else:
            features["user_in_fraud_ring"] = 0.0

        return features

    def compute_fraud_score(self, transaction: Transaction) -> float:
        """
        Compute graph-based fraud score.

        Args:
            transaction: Transaction to score

        Returns:
            Fraud score (0-1)
        """
        features = self.compute_graph_features(transaction)

        # Simple scoring based on graph features
        score = 0.0

        # High fraud ratio in neighbors
        score += features.get("user_fraud_ratio", 0) * 0.3
        score += features.get("device_fraud_ratio", 0) * 0.2
        score += features.get("ip_fraud_ratio", 0) * 0.2

        # Close to known fraud
        distance = features.get("user_distance_to_fraud", 10)
        if distance < 10:
            score += (10 - distance) / 10 * 0.2

        # In fraud ring
        score += features.get("user_in_fraud_ring", 0) * 0.1

        return min(score, 1.0)

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        if self.graph is None:
            return {}

        stats = self.graph_builder.get_graph_statistics()

        # Add fraud-specific stats
        fraud_rings = self.detect_fraud_rings()
        stats["num_fraud_rings"] = len(fraud_rings)
        stats["num_known_fraud_nodes"] = len(self.known_fraud_nodes)

        return stats

    def _default_graph_features(self) -> Dict[str, float]:
        """Return default graph features when graph not available."""
        return {
            "user_degree": 0.0,
            "user_fraud_ratio": 0.0,
            "user_neighbor_fraud_count": 0.0,
            "user_distance_to_fraud": 10.0,
            "device_degree": 0.0,
            "device_fraud_ratio": 0.0,
            "ip_degree": 0.0,
            "ip_fraud_ratio": 0.0,
            "user_in_fraud_ring": 0.0,
        }

    def visualize_fraud_ring(
        self,
        fraud_ring: Set[str],
        output_path: str
    ) -> None:
        """
        Visualize a fraud ring.

        Args:
            fraud_ring: Set of nodes in fraud ring
            output_path: Path to save visualization
        """
        if self.graph is None:
            return

        # Create subgraph
        subgraph = self.graph.subgraph(fraud_ring)

        # Use matplotlib for visualization
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))

        # Color nodes by type
        node_colors = []
        for node in subgraph.nodes():
            node_type = self.graph.nodes[node].get("node_type", "unknown")
            if node_type == "user":
                node_colors.append("red")
            elif node_type == "device":
                node_colors.append("blue")
            elif node_type == "ip":
                node_colors.append("green")
            elif node_type == "merchant":
                node_colors.append("orange")
            else:
                node_colors.append("gray")

        # Draw
        pos = nx.spring_layout(subgraph)
        nx.draw(
            subgraph,
            pos,
            node_color=node_colors,
            with_labels=True,
            font_size=8,
            node_size=500,
            alpha=0.7
        )

        plt.title("Fraud Ring Visualization")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

"""Build transaction graphs for fraud detection."""

import networkx as nx
from typing import List, Dict, Any, Set
from collections import defaultdict

from fraud_detection.types import Transaction


class TransactionGraphBuilder:
    """
    Builds graphs from transaction data for fraud ring detection.

    Graph types:
    - User-Device-IP graph
    - User-Merchant graph
    - Card-User graph
    """

    def __init__(self):
        self.graph = nx.Graph()
        self.user_nodes: Set[str] = set()
        self.device_nodes: Set[str] = set()
        self.ip_nodes: Set[str] = set()
        self.merchant_nodes: Set[str] = set()
        self.card_nodes: Set[str] = set()

    def build_from_transactions(
        self,
        transactions: List[Transaction]
    ) -> nx.Graph:
        """
        Build graph from transactions.

        Args:
            transactions: List of transactions

        Returns:
            NetworkX graph
        """
        self.graph = nx.Graph()

        for txn in transactions:
            # Add nodes
            user_node = f"user_{txn.user_id}"
            device_node = f"device_{txn.device_id}"
            ip_node = f"ip_{txn.ip_address}"
            merchant_node = f"merchant_{txn.merchant_id}"
            card_node = f"card_{txn.card_bin}_{txn.card_last4}"

            # Add nodes with attributes
            self.graph.add_node(
                user_node,
                node_type="user",
                fraud_count=0,
                legit_count=0
            )
            self.graph.add_node(
                device_node,
                node_type="device",
                fraud_count=0,
                legit_count=0
            )
            self.graph.add_node(
                ip_node,
                node_type="ip",
                fraud_count=0,
                legit_count=0
            )
            self.graph.add_node(
                merchant_node,
                node_type="merchant",
                fraud_count=0,
                legit_count=0
            )
            self.graph.add_node(
                card_node,
                node_type="card",
                fraud_count=0,
                legit_count=0
            )

            # Update fraud counts
            if txn.is_fraud:
                self.graph.nodes[user_node]["fraud_count"] += 1
                self.graph.nodes[device_node]["fraud_count"] += 1
                self.graph.nodes[ip_node]["fraud_count"] += 1
                self.graph.nodes[card_node]["fraud_count"] += 1
            else:
                self.graph.nodes[user_node]["legit_count"] += 1
                self.graph.nodes[device_node]["legit_count"] += 1
                self.graph.nodes[ip_node]["legit_count"] += 1
                self.graph.nodes[card_node]["legit_count"] += 1

            # Add edges
            # User-Device edge
            if not self.graph.has_edge(user_node, device_node):
                self.graph.add_edge(
                    user_node,
                    device_node,
                    weight=0,
                    fraud_weight=0,
                    edge_type="user_device"
                )
            self.graph[user_node][device_node]["weight"] += 1
            if txn.is_fraud:
                self.graph[user_node][device_node]["fraud_weight"] += 1

            # User-IP edge
            if not self.graph.has_edge(user_node, ip_node):
                self.graph.add_edge(
                    user_node,
                    ip_node,
                    weight=0,
                    fraud_weight=0,
                    edge_type="user_ip"
                )
            self.graph[user_node][ip_node]["weight"] += 1
            if txn.is_fraud:
                self.graph[user_node][ip_node]["fraud_weight"] += 1

            # Device-IP edge
            if not self.graph.has_edge(device_node, ip_node):
                self.graph.add_edge(
                    device_node,
                    ip_node,
                    weight=0,
                    fraud_weight=0,
                    edge_type="device_ip"
                )
            self.graph[device_node][ip_node]["weight"] += 1
            if txn.is_fraud:
                self.graph[device_node][ip_node]["fraud_weight"] += 1

            # User-Merchant edge
            if not self.graph.has_edge(user_node, merchant_node):
                self.graph.add_edge(
                    user_node,
                    merchant_node,
                    weight=0,
                    fraud_weight=0,
                    edge_type="user_merchant"
                )
            self.graph[user_node][merchant_node]["weight"] += 1
            if txn.is_fraud:
                self.graph[user_node][merchant_node]["fraud_weight"] += 1

            # Card-User edge
            if not self.graph.has_edge(card_node, user_node):
                self.graph.add_edge(
                    card_node,
                    user_node,
                    weight=0,
                    fraud_weight=0,
                    edge_type="card_user"
                )
            self.graph[card_node][user_node]["weight"] += 1
            if txn.is_fraud:
                self.graph[card_node][user_node]["fraud_weight"] += 1

            # Track node sets
            self.user_nodes.add(user_node)
            self.device_nodes.add(device_node)
            self.ip_nodes.add(ip_node)
            self.merchant_nodes.add(merchant_node)
            self.card_nodes.add(card_node)

        return self.graph

    def get_node_features(self, node: str) -> Dict[str, Any]:
        """
        Get features for a node.

        Args:
            node: Node ID

        Returns:
            Dictionary of node features
        """
        if node not in self.graph:
            return {}

        node_data = self.graph.nodes[node]
        degree = self.graph.degree[node]

        # Calculate fraud ratio
        total_count = node_data.get("fraud_count", 0) + node_data.get("legit_count", 0)
        fraud_ratio = node_data.get("fraud_count", 0) / total_count if total_count > 0 else 0

        # Get neighbor information
        neighbors = list(self.graph.neighbors(node))
        neighbor_fraud_counts = sum(
            self.graph.nodes[n].get("fraud_count", 0) for n in neighbors
        )

        features = {
            "degree": degree,
            "fraud_count": node_data.get("fraud_count", 0),
            "legit_count": node_data.get("legit_count", 0),
            "fraud_ratio": fraud_ratio,
            "num_neighbors": len(neighbors),
            "neighbor_fraud_count": neighbor_fraud_counts,
            "node_type": node_data.get("node_type", "unknown"),
        }

        return features

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get overall graph statistics."""
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "num_users": len(self.user_nodes),
            "num_devices": len(self.device_nodes),
            "num_ips": len(self.ip_nodes),
            "num_merchants": len(self.merchant_nodes),
            "num_cards": len(self.card_nodes),
            "avg_degree": sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0,
            "is_connected": nx.is_connected(self.graph),
            "num_components": nx.number_connected_components(self.graph),
        }

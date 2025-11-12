"""Feature engineering for fraud detection."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any
from fraud_detection.types import Transaction


class FeatureEngineer:
    """
    Feature engineering for fraud detection with adversarially robust features.

    Focuses on features that are difficult for attackers to game.
    """

    def __init__(self):
        self.user_history: Dict[str, List[Transaction]] = {}
        self.device_history: Dict[str, List[Transaction]] = {}
        self.ip_history: Dict[str, List[Transaction]] = {}

    def extract_features(self, transaction: Transaction) -> Dict[str, Any]:
        """
        Extract features from a transaction.

        Args:
            transaction: Transaction to extract features from

        Returns:
            Dictionary of features
        """
        features = {}

        # Basic transaction features
        features['amount'] = transaction.amount
        features['amount_log'] = np.log1p(transaction.amount)
        features['hour_of_day'] = transaction.timestamp.hour
        features['day_of_week'] = transaction.timestamp.weekday()
        features['is_weekend'] = int(transaction.timestamp.weekday() >= 5)

        # Time-based features
        features['time_since_account_creation_hours'] = transaction.time_since_account_creation_hours
        features['is_first_transaction'] = int(transaction.is_first_transaction)

        # Velocity features (hard to game without detection)
        features.update(self._compute_velocity_features(transaction))

        # Device and IP diversity (hard to fake consistently)
        features.update(self._compute_diversity_features(transaction))

        # Behavioral consistency features
        features.update(self._compute_behavioral_features(transaction))

        # Network features
        features.update(self._compute_network_features(transaction))

        return features

    def _compute_velocity_features(self, transaction: Transaction) -> Dict[str, Any]:
        """Compute velocity-based features."""
        features = {}

        # User velocity
        user_txns = self.user_history.get(transaction.user_id, [])
        features['user_txn_count_24h'] = sum(
            1 for t in user_txns
            if (transaction.timestamp - t.timestamp).total_seconds() < 86400
        )
        features['user_txn_count_1h'] = sum(
            1 for t in user_txns
            if (transaction.timestamp - t.timestamp).total_seconds() < 3600
        )

        # Amount velocity
        features['user_amount_24h'] = sum(
            t.amount for t in user_txns
            if (transaction.timestamp - t.timestamp).total_seconds() < 86400
        )

        # Device velocity
        device_txns = self.device_history.get(transaction.device_id, [])
        features['device_txn_count_24h'] = sum(
            1 for t in device_txns
            if (transaction.timestamp - t.timestamp).total_seconds() < 86400
        )

        # Time since last transaction
        if user_txns:
            last_txn = max(user_txns, key=lambda t: t.timestamp)
            features['seconds_since_last_txn'] = (
                transaction.timestamp - last_txn.timestamp
            ).total_seconds()
        else:
            features['seconds_since_last_txn'] = float('inf')

        return features

    def _compute_diversity_features(self, transaction: Transaction) -> Dict[str, Any]:
        """Compute diversity-based features (hard to game)."""
        features = {}

        user_txns = self.user_history.get(transaction.user_id, [])

        if user_txns:
            # Device diversity for this user
            unique_devices = len(set(t.device_id for t in user_txns))
            features['user_device_diversity'] = unique_devices

            # IP diversity
            unique_ips = len(set(t.ip_address for t in user_txns))
            features['user_ip_diversity'] = unique_ips

            # Merchant diversity
            unique_merchants = len(set(t.merchant_id for t in user_txns))
            features['user_merchant_diversity'] = unique_merchants

            # Device-User ratio (one device, many users = suspicious)
            device_txns = self.device_history.get(transaction.device_id, [])
            if device_txns:
                unique_users_on_device = len(set(t.user_id for t in device_txns))
                features['device_user_ratio'] = unique_users_on_device
            else:
                features['device_user_ratio'] = 1
        else:
            features['user_device_diversity'] = 0
            features['user_ip_diversity'] = 0
            features['user_merchant_diversity'] = 0
            features['device_user_ratio'] = 1

        return features

    def _compute_behavioral_features(self, transaction: Transaction) -> Dict[str, Any]:
        """Compute behavioral consistency features."""
        features = {}

        user_txns = self.user_history.get(transaction.user_id, [])

        if len(user_txns) >= 5:
            # Amount patterns
            amounts = [t.amount for t in user_txns]
            features['user_avg_amount'] = np.mean(amounts)
            features['user_std_amount'] = np.std(amounts)
            features['amount_deviation_from_avg'] = abs(
                transaction.amount - features['user_avg_amount']
            )

            # Time of day consistency
            hours = [t.timestamp.hour for t in user_txns]
            features['user_avg_hour'] = np.mean(hours)
            features['hour_deviation'] = abs(transaction.timestamp.hour - features['user_avg_hour'])

            # Merchant consistency
            merchant_counts = {}
            for t in user_txns:
                merchant_counts[t.merchant_id] = merchant_counts.get(t.merchant_id, 0) + 1
            features['is_new_merchant'] = int(
                transaction.merchant_id not in merchant_counts
            )
        else:
            features['user_avg_amount'] = transaction.amount
            features['user_std_amount'] = 0
            features['amount_deviation_from_avg'] = 0
            features['user_avg_hour'] = transaction.timestamp.hour
            features['hour_deviation'] = 0
            features['is_new_merchant'] = 1

        return features

    def _compute_network_features(self, transaction: Transaction) -> Dict[str, Any]:
        """Compute network-based features."""
        features = {}

        # Card BIN features
        features['card_bin'] = int(transaction.card_bin) if transaction.card_bin else 0

        # Country features
        user_txns = self.user_history.get(transaction.user_id, [])
        if user_txns:
            countries = [t.country for t in user_txns if t.country]
            if countries:
                most_common_country = max(set(countries), key=countries.count)
                features['is_different_country'] = int(
                    transaction.country != most_common_country
                )
            else:
                features['is_different_country'] = 0
        else:
            features['is_different_country'] = 0

        return features

    def update_history(self, transaction: Transaction) -> None:
        """Update historical data with new transaction."""
        # Keep only recent history (last 30 days)
        cutoff = transaction.timestamp - timedelta(days=30)

        # Update user history
        if transaction.user_id not in self.user_history:
            self.user_history[transaction.user_id] = []
        self.user_history[transaction.user_id].append(transaction)
        self.user_history[transaction.user_id] = [
            t for t in self.user_history[transaction.user_id]
            if t.timestamp > cutoff
        ]

        # Update device history
        if transaction.device_id not in self.device_history:
            self.device_history[transaction.device_id] = []
        self.device_history[transaction.device_id].append(transaction)
        self.device_history[transaction.device_id] = [
            t for t in self.device_history[transaction.device_id]
            if t.timestamp > cutoff
        ]

        # Update IP history
        if transaction.ip_address not in self.ip_history:
            self.ip_history[transaction.ip_address] = []
        self.ip_history[transaction.ip_address].append(transaction)
        self.ip_history[transaction.ip_address] = [
            t for t in self.ip_history[transaction.ip_address]
            if t.timestamp > cutoff
        ]

    def extract_batch_features(self, transactions: List[Transaction]) -> pd.DataFrame:
        """
        Extract features from a batch of transactions.

        Args:
            transactions: List of transactions

        Returns:
            DataFrame with extracted features
        """
        feature_list = []

        for txn in sorted(transactions, key=lambda t: t.timestamp):
            features = self.extract_features(txn)
            features['transaction_id'] = txn.transaction_id
            features['is_fraud'] = txn.is_fraud
            feature_list.append(features)
            self.update_history(txn)

        return pd.DataFrame(feature_list)

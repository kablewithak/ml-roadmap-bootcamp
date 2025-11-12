"""Detection models for fraud defense."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

from fraud_detection.types import Transaction, FraudLabel
from fraud_detection.utils.features import FeatureEngineer


class RuleBasedDetector:
    """
    Rule-based fraud detection.

    Uses hand-crafted rules that are hard to game.
    """

    def __init__(self):
        self.rules = []
        self._setup_rules()

    def _setup_rules(self):
        """Define fraud detection rules."""

        # High velocity rules
        def high_velocity_rule(features: Dict) -> Tuple[bool, float, str]:
            if features.get('user_txn_count_1h', 0) > 10:
                return True, 0.9, "High velocity: >10 transactions in 1 hour"
            if features.get('user_txn_count_24h', 0) > 50:
                return True, 0.85, "High velocity: >50 transactions in 24 hours"
            return False, 0.0, ""

        # New account rules
        def new_account_rule(features: Dict) -> Tuple[bool, float, str]:
            if (features.get('time_since_account_creation_hours', 1000) < 1.0 and
                features.get('amount', 0) > 500):
                return True, 0.95, "New account with high amount"
            return False, 0.0, ""

        # Device-user ratio rule
        def device_sharing_rule(features: Dict) -> Tuple[bool, float, str]:
            if features.get('device_user_ratio', 1) > 10:
                return True, 0.8, "Device shared by >10 users"
            return False, 0.0, ""

        # Amount deviation rule
        def amount_anomaly_rule(features: Dict) -> Tuple[bool, float, str]:
            avg = features.get('user_avg_amount', 0)
            std = features.get('user_std_amount', 0)
            current = features.get('amount', 0)

            if avg > 0 and std > 0:
                deviation = abs(current - avg) / (std + 1e-6)
                if deviation > 3.0:  # 3 standard deviations
                    return True, 0.7, f"Amount anomaly: {deviation:.1f} std devs"
            return False, 0.0, ""

        # Time-based rules
        def unusual_time_rule(features: Dict) -> Tuple[bool, float, str]:
            hour = features.get('hour_of_day', 12)
            if 2 <= hour <= 5:  # Late night transactions
                if features.get('amount', 0) > 1000:
                    return True, 0.65, "Large transaction at unusual hour"
            return False, 0.0, ""

        # Card testing rule
        def card_testing_rule(features: Dict) -> Tuple[bool, float, str]:
            if (features.get('amount', 0) == 1.0 and
                features.get('user_txn_count_1h', 0) > 5):
                return True, 0.95, "Card testing pattern detected"
            return False, 0.0, ""

        # Country mismatch rule
        def country_mismatch_rule(features: Dict) -> Tuple[bool, float, str]:
            if (features.get('is_different_country', 0) == 1 and
                features.get('seconds_since_last_txn', float('inf')) < 3600):
                return True, 0.75, "Country change within 1 hour"
            return False, 0.0, ""

        self.rules = [
            high_velocity_rule,
            new_account_rule,
            device_sharing_rule,
            amount_anomaly_rule,
            unusual_time_rule,
            card_testing_rule,
            country_mismatch_rule,
        ]

    def predict(self, features: Dict) -> Tuple[bool, float, List[str]]:
        """
        Predict fraud using rules.

        Args:
            features: Transaction features

        Returns:
            Tuple of (is_fraud, max_score, triggered_rules)
        """
        max_score = 0.0
        triggered_rules = []

        for rule in self.rules:
            triggered, score, reason = rule(features)
            if triggered:
                max_score = max(max_score, score)
                triggered_rules.append(reason)

        is_fraud = max_score > 0.6  # Threshold for blocking

        return is_fraud, max_score, triggered_rules

    def predict_batch(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Predict fraud for batch of transactions.

        Args:
            features_df: DataFrame with features

        Returns:
            Array of fraud scores
        """
        scores = []
        for idx, row in features_df.iterrows():
            features = row.to_dict()
            _, score, _ = self.predict(features)
            scores.append(score)

        return np.array(scores)


class MLDetector:
    """
    Machine learning fraud detector.

    Uses ensemble of models for robust detection.
    """

    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False

    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the model.

        Args:
            X: Features
            y: Labels (0 = legitimate, 1 = fraud)
            validation_split: Fraction for validation

        Returns:
            Training metrics
        """
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

        # Store feature names
        self.feature_names = X.columns.tolist()

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Train model
        if self.model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='auc'
            )
        elif self.model_type == "lightgbm":
            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif self.model_type == "logistic":
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        # Evaluate
        y_pred = self.model.predict(X_val_scaled)
        y_pred_proba = self.model.predict_proba(X_val_scaled)[:, 1]

        metrics = {
            "roc_auc": roc_auc_score(y_val, y_pred_proba),
            "precision": precision_score(y_val, y_pred),
            "recall": recall_score(y_val, y_pred),
            "f1_score": f1_score(y_val, y_pred),
        }

        return metrics

    def predict(self, features: Dict) -> float:
        """
        Predict fraud score for single transaction.

        Args:
            features: Transaction features

        Returns:
            Fraud score (0-1)
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        # Convert to DataFrame
        X = pd.DataFrame([features])[self.feature_names]
        X_scaled = self.scaler.transform(X)

        # Predict
        score = self.model.predict_proba(X_scaled)[0, 1]
        return score

    def predict_batch(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict fraud scores for batch.

        Args:
            X: Features DataFrame

        Returns:
            Array of fraud scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        X_scaled = self.scaler.transform(X[self.feature_names])
        scores = self.model.predict_proba(X_scaled)[:, 1]
        return scores

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            return pd.DataFrame()

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return importance_df


class EnsembleDetector:
    """
    Ensemble detector combining rules, ML, and graph features.

    Provides robust defense against adversarial attacks.
    """

    def __init__(self):
        self.rule_detector = RuleBasedDetector()
        self.ml_detector = MLDetector(model_type="xgboost")
        self.graph_detector = None  # Set by DefenseSystem

        self.weights = {
            "rules": 0.3,
            "ml": 0.5,
            "graph": 0.2
        }

        self.threshold = 0.7  # Combined score threshold

    def set_graph_detector(self, graph_detector):
        """Set the graph-based detector."""
        self.graph_detector = graph_detector

    def train(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """Train the ML component."""
        return self.ml_detector.train(X, y)

    def predict(
        self,
        features: Dict,
        graph_score: Optional[float] = None
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Predict using ensemble.

        Args:
            features: Transaction features
            graph_score: Optional graph-based fraud score

        Returns:
            Tuple of (is_fraud, combined_score, details)
        """
        # Get rule-based prediction
        rule_fraud, rule_score, triggered_rules = self.rule_detector.predict(features)

        # Get ML prediction
        ml_score = 0.0
        if self.ml_detector.is_trained:
            try:
                ml_score = self.ml_detector.predict(features)
            except:
                ml_score = 0.5  # Default if prediction fails

        # Get graph score
        if graph_score is None:
            graph_score = 0.0

        # Combine scores
        combined_score = (
            self.weights["rules"] * rule_score +
            self.weights["ml"] * ml_score +
            self.weights["graph"] * graph_score
        )

        is_fraud = combined_score > self.threshold

        details = {
            "rule_score": rule_score,
            "ml_score": ml_score,
            "graph_score": graph_score,
            "combined_score": combined_score,
            "triggered_rules": triggered_rules,
            "threshold": self.threshold,
        }

        return is_fraud, combined_score, details

    def predict_batch(
        self,
        features_df: pd.DataFrame,
        graph_scores: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict for batch of transactions.

        Args:
            features_df: Features DataFrame
            graph_scores: Optional array of graph scores

        Returns:
            Array of combined fraud scores
        """
        # Get rule scores
        rule_scores = self.rule_detector.predict_batch(features_df)

        # Get ML scores
        ml_scores = np.zeros(len(features_df))
        if self.ml_detector.is_trained:
            try:
                ml_scores = self.ml_detector.predict_batch(features_df)
            except:
                ml_scores = np.full(len(features_df), 0.5)

        # Get graph scores
        if graph_scores is None:
            graph_scores = np.zeros(len(features_df))

        # Combine
        combined_scores = (
            self.weights["rules"] * rule_scores +
            self.weights["ml"] * ml_scores +
            self.weights["graph"] * graph_scores
        )

        return combined_scores

    def update_weights(self, new_weights: Dict[str, float]):
        """Update ensemble weights."""
        total = sum(new_weights.values())
        self.weights = {k: v / total for k, v in new_weights.items()}

    def update_threshold(self, new_threshold: float):
        """Update fraud threshold."""
        self.threshold = new_threshold

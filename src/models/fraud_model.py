"""
Fraud Detection Model (Demo)

This is a SIMPLIFIED fraud model for demonstration purposes.
In production, you'd use:
- XGBoost/LightGBM (better performance)
- Feature engineering pipelines (temporal features, aggregations)
- Online learning (model updates without retraining)

CRITICAL: Why we keep the model simple
---------------------------------------
The OBSERVABILITY patterns are the same whether your model is:
- Logistic Regression (this demo)
- Deep Neural Network (production)

The complexity that matters is:
1. How do you detect when the model degrades?
2. How do you measure latency under load?
3. How do you track business impact?

These questions are MODEL-AGNOSTIC.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class FraudDetectionModel:
    """
    Binary classifier for fraud detection.

    Features (in production, you'd have 50-100 features):
    1. transaction_amount: Payment amount in USD
    2. transaction_hour: Hour of day (0-23)
    3. days_since_signup: Account age
    4. transaction_count_24h: Recent transaction velocity
    5. avg_transaction_amount: Historical average
    6. is_international: Cross-border transaction (0/1)

    BUSINESS LOGIC:
    - threshold > 0.7: Block (high fraud risk)
    - threshold 0.3-0.7: Manual review
    - threshold < 0.3: Approve
    """

    def __init__(self, threshold: float = 0.7):
        """
        Args:
            threshold: Fraud probability threshold for blocking (0.0 to 1.0)

        TRADE-OFF: Threshold tuning
        ----------------------------
        High threshold (0.9): Low false positives, but miss some fraud
        Low threshold (0.3): Catch more fraud, but annoy legitimate users

        BUSINESS IMPACT:
        - False positive: Lost customer ($100 lifetime value)
        - False negative: Fraud loss ($500 average)

        Optimal threshold depends on these costs.
        """
        self.threshold = threshold
        self.model = None
        self.scaler = None

    def train(self, X: np.ndarray, y: np.ndarray) -> "FraudDetectionModel":
        """
        Train the fraud detection model.

        In production, you'd:
        1. Split train/validation/test
        2. Cross-validate hyperparameters
        3. Track training metrics in MLflow
        4. A/B test before deploying

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (0=legitimate, 1=fraud)

        Returns:
            self (for chaining)
        """
        # Standardize features (important for LogisticRegression)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model = LogisticRegression(
            penalty='l2',
            C=1.0,
            max_iter=1000,
            random_state=42
        )
        self.model.fit(X_scaled, y)

        # Log training metrics
        train_accuracy = self.model.score(X_scaled, y)
        logger.info(f"Model trained: accuracy={train_accuracy:.3f}")

        return self

    def predict_proba(self, features: Dict[str, float]) -> float:
        """
        Predict fraud probability for a single transaction.

        Args:
            features: Dict of feature values

        Returns:
            Fraud probability (0.0 to 1.0)

        CRITICAL: Feature order must match training!
        If you trained with [amount, hour, days_since_signup] but predict with
        [hour, amount, days_since_signup], your predictions will be garbage.

        SOLUTION: Use feature names (not positional arguments)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Extract features in correct order
        feature_names = [
            'transaction_amount',
            'transaction_hour',
            'days_since_signup',
            'transaction_count_24h',
            'avg_transaction_amount',
            'is_international'
        ]

        X = np.array([[features.get(name, 0.0) for name in feature_names]])
        X_scaled = self.scaler.transform(X)

        # Return probability of fraud (class 1)
        return self.model.predict_proba(X_scaled)[0, 1]

    def predict(self, features: Dict[str, float]) -> Dict[str, any]:
        """
        Make fraud prediction with business logic.

        Returns:
            {
                "fraud_probability": 0.85,
                "decision": "block",  # block | review | approve
                "risk_level": "high"   # high | medium | low
            }
        """
        fraud_prob = self.predict_proba(features)

        if fraud_prob >= self.threshold:
            decision = "block"
            risk_level = "high"
        elif fraud_prob >= 0.3:
            decision = "review"
            risk_level = "medium"
        else:
            decision = "approve"
            risk_level = "low"

        return {
            "fraud_probability": float(fraud_prob),
            "decision": decision,
            "risk_level": risk_level,
        }

    def save(self, path: str) -> None:
        """Save model to disk."""
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'threshold': self.threshold,
        }, path)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "FraudDetectionModel":
        """Load model from disk."""
        data = joblib.load(path)

        instance = cls(threshold=data['threshold'])
        instance.model = data['model']
        instance.scaler = data['scaler']

        logger.info(f"Model loaded from {path}")
        return instance


# ============================================================================
# DEMO: Train a simple model
# ============================================================================

def train_demo_model(output_path: str = "models/fraud_model.pkl") -> None:
    """
    Train a demo fraud detection model with synthetic data.

    In production, you'd:
    1. Load data from warehouse (BigQuery, Snowflake)
    2. Feature engineering pipeline
    3. Train/validation split
    4. Hyperparameter tuning
    5. Track in MLflow
    """
    logger.info("Training demo fraud model...")

    # Generate synthetic training data
    np.random.seed(42)
    n_samples = 10000

    # Legitimate transactions
    n_legit = int(n_samples * 0.95)  # 95% legitimate (class imbalance!)
    X_legit = np.random.normal(loc=[
        100,    # avg amount: $100
        12,     # avg hour: noon
        365,    # avg days since signup: 1 year
        5,      # avg transactions/24h
        95,     # avg historical amount
        0.1     # 10% international
    ], scale=[50, 6, 200, 3, 40, 0.3], size=(n_legit, 6))
    y_legit = np.zeros(n_legit)

    # Fraudulent transactions (different distribution)
    n_fraud = n_samples - n_legit
    X_fraud = np.random.normal(loc=[
        500,    # higher amounts
        3,      # late night transactions
        10,     # new accounts
        20,     # high velocity
        80,     # inconsistent with history
        0.8     # mostly international
    ], scale=[200, 4, 20, 10, 50, 0.2], size=(n_fraud, 6))
    y_fraud = np.ones(n_fraud)

    # Combine
    X = np.vstack([X_legit, X_fraud])
    y = np.hstack([y_legit, y_fraud])

    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]

    # Train model
    model = FraudDetectionModel(threshold=0.7)
    model.train(X, y)

    # Save
    model.save(output_path)

    logger.info(f"✓ Demo model trained and saved to {output_path}")


if __name__ == "__main__":
    # Setup basic logging for training script
    logging.basicConfig(level=logging.INFO)

    # Train demo model
    train_demo_model()

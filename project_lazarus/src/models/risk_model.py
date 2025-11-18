"""Risk scoring model for loan applications."""

import joblib
import numpy as np
import structlog
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from src.core.models import UserFeatures
from config.settings import get_settings

logger = structlog.get_logger(__name__)


class RiskModel:
    """
    Risk scoring model for loan applications.

    This model predicts the probability that a loan will default.
    Higher scores indicate higher risk.
    """

    def __init__(self, model_path: str | None = None):
        """
        Initialize the risk model.

        Args:
            model_path: Path to saved model file
        """
        settings = get_settings()
        self.model_path = Path(model_path or settings.model_path)
        self.model: GradientBoostingClassifier | None = None
        self.scaler: StandardScaler | None = None
        self.feature_names = [
            "age",
            "income",
            "debt",
            "credit_score",
            "employment_years",
            "num_credit_lines",
            "avg_txn_amt_30d",
            "credit_history_months",
            "debt_to_income",
        ]

    def load_or_create_default(self) -> None:
        """Load model from disk or create a default model."""
        if self.model_path.exists():
            self.load()
        else:
            logger.warning("model_not_found", path=str(self.model_path))
            self._create_default_model()

    def load(self) -> None:
        """Load model from disk."""
        try:
            data = joblib.load(self.model_path)
            self.model = data["model"]
            self.scaler = data["scaler"]
            logger.info("model_loaded", path=str(self.model_path))
        except Exception as e:
            logger.error("model_load_failed", error=str(e))
            raise

    def save(self) -> None:
        """Save model to disk."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "scaler": self.scaler},
            self.model_path
        )
        logger.info("model_saved", path=str(self.model_path))

    def _create_default_model(self) -> None:
        """Create a default model with reasonable weights."""
        # Create a simple model that uses basic heuristics
        # This will be replaced by the causal training pipeline
        self.scaler = StandardScaler()
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )

        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 1000

        X = np.random.randn(n_samples, len(self.feature_names))
        # Generate labels based on simple rules
        # Higher income, credit score = lower risk
        y = (
            (X[:, 1] < 0) &  # Low income
            (X[:, 3] < 0) |  # Low credit score
            (X[:, 2] > 1)    # High debt
        ).astype(int)

        self.scaler.fit(X)
        self.model.fit(X, y)

        self.save()
        logger.info("default_model_created")

    def _extract_features(self, user_features: UserFeatures) -> np.ndarray:
        """Extract feature vector from UserFeatures."""
        # Calculate derived features
        debt_to_income = (
            user_features.debt / user_features.income
            if user_features.income > 0 else 1.0
        )

        features = np.array([
            user_features.age,
            user_features.income,
            user_features.debt,
            user_features.credit_score,
            user_features.employment_years,
            user_features.num_credit_lines,
            user_features.avg_txn_amt_30d,
            user_features.credit_history_months,
            debt_to_income,
        ]).reshape(1, -1)

        return features

    def predict(self, user_features: UserFeatures) -> float:
        """
        Predict risk score for a user.

        Args:
            user_features: User features

        Returns:
            Risk score between 0.0 (safe) and 1.0 (risky)
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not loaded")

        features = self._extract_features(user_features)
        scaled_features = self.scaler.transform(features)

        # Get probability of default (positive class)
        risk_score = self.model.predict_proba(scaled_features)[0, 1]

        return float(risk_score)

    def predict_batch(self, features_list: list[UserFeatures]) -> list[float]:
        """
        Predict risk scores for multiple users.

        Args:
            features_list: List of user features

        Returns:
            List of risk scores
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not loaded")

        X = np.vstack([
            self._extract_features(f) for f in features_list
        ])
        scaled_X = self.scaler.transform(X)

        risk_scores = self.model.predict_proba(scaled_X)[:, 1]

        return risk_scores.tolist()

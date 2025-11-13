"""
Ensemble alpha generation system.

Combines multiple models (XGBoost, LightGBM, CatBoost, Neural Networks)
with weighted averaging for robust alpha signals.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

try:
    import xgboost as xgb
    import lightgbm as lgb
    import catboost as cb
except ImportError:
    xgb = lgb = cb = None

import torch
import torch.nn as nn

from alpha_platform.utils.config import get_config
from alpha_platform.utils.logger import get_logger

logger = get_logger(__name__)


class AlphaEnsemble:
    """
    Ensemble of models for alpha generation.

    Combines gradient boosting models and neural networks with
    weighted averaging based on information coefficient.
    """

    def __init__(
        self,
        model_configs: Optional[List[Dict[str, Any]]] = None,
        ensemble_method: str = "weighted_avg",
        min_ic: float = 0.05,
    ):
        """
        Initialize alpha ensemble.

        Args:
            model_configs: List of model configurations
            ensemble_method: Ensemble method ('weighted_avg', 'stacking')
            min_ic: Minimum information coefficient threshold
        """
        self.config = get_config()
        self.ensemble_method = ensemble_method
        self.min_ic = min_ic

        # Default model configs
        if model_configs is None:
            model_configs = self._get_default_configs()

        self.model_configs = model_configs
        self.models = []
        self.weights = []
        self.feature_names = None

        logger.info(f"Initialized alpha ensemble with {len(model_configs)} models")

    def _get_default_configs(self) -> List[Dict[str, Any]]:
        """Get default model configurations from config file."""
        alpha_config = self.config.alpha_generation

        return alpha_config.get("models", {}).get("ensemble", [])

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2,
    ) -> None:
        """
        Fit ensemble models.

        Args:
            X: Feature dataframe
            y: Target series (returns or labels)
            validation_split: Validation split ratio
        """
        logger.info(f"Training ensemble on {len(X)} samples, {len(X.columns)} features")

        self.feature_names = X.columns.tolist()

        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        # Train each model
        self.models = []
        model_ics = []

        for model_config in self.model_configs:
            model_type = model_config["type"]
            params = model_config.get("params", {})

            logger.info(f"Training {model_type}")

            model = self._train_model(model_type, params, X_train, y_train)
            self.models.append({"type": model_type, "model": model})

            # Calculate IC on validation set
            val_pred = self._predict_model(model, model_type, X_val)
            ic = np.corrcoef(val_pred, y_val)[0, 1]
            model_ics.append(max(ic, 0))  # Clip negative ICs to 0

            logger.info(f"{model_type} IC: {ic:.4f}")

        # Calculate ensemble weights based on IC
        total_ic = sum(model_ics)
        if total_ic > 0:
            self.weights = [ic / total_ic for ic in model_ics]
        else:
            # Equal weights if all ICs are low
            self.weights = [1.0 / len(self.models)] * len(self.models)

        logger.info(f"Ensemble weights: {self.weights}")

    def _train_model(
        self,
        model_type: str,
        params: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Any:
        """Train a single model."""
        if model_type == "xgboost":
            if xgb is None:
                raise ImportError("xgboost not installed")

            dtrain = xgb.DMatrix(X_train, label=y_train)
            model = xgb.train(params, dtrain, num_boost_round=params.get("n_estimators", 1000))
            return model

        elif model_type == "lightgbm":
            if lgb is None:
                raise ImportError("lightgbm not installed")

            train_data = lgb.Dataset(X_train, label=y_train)
            model = lgb.train(params, train_data, num_boost_round=params.get("n_estimators", 1000))
            return model

        elif model_type == "catboost":
            if cb is None:
                raise ImportError("catboost not installed")

            model = cb.CatBoostRegressor(**params, verbose=False)
            model.fit(X_train, y_train)
            return model

        elif model_type == "neural_network":
            model = SimpleNN(
                input_dim=len(X_train.columns),
                hidden_layers=params.get("hidden_layers", [256, 128, 64]),
                dropout=params.get("dropout", 0.3),
            )

            # Train neural network
            self._train_nn(model, X_train, y_train, epochs=100)
            return model

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _train_nn(
        self,
        model: nn.Module,
        X: pd.DataFrame,
        y: pd.Series,
        epochs: int = 100,
        batch_size: int = 256,
    ) -> None:
        """Train neural network."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Convert to tensors
        X_tensor = torch.FloatTensor(X.values).to(device)
        y_tensor = torch.FloatTensor(y.values).unsqueeze(1).to(device)

        model.train()
        for epoch in range(epochs):
            # Mini-batch training
            indices = torch.randperm(len(X_tensor))
            for i in range(0, len(X_tensor), batch_size):
                batch_indices = indices[i : i + batch_size]
                batch_X = X_tensor[batch_indices]
                batch_y = y_tensor[batch_indices]

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        model.eval()

    def _predict_model(
        self, model: Any, model_type: str, X: pd.DataFrame
    ) -> np.ndarray:
        """Get predictions from a single model."""
        if model_type == "xgboost":
            dtest = xgb.DMatrix(X)
            return model.predict(dtest)

        elif model_type == "lightgbm":
            return model.predict(X)

        elif model_type == "catboost":
            return model.predict(X)

        elif model_type == "neural_network":
            device = next(model.parameters()).device
            X_tensor = torch.FloatTensor(X.values).to(device)
            with torch.no_grad():
                predictions = model(X_tensor)
            return predictions.cpu().numpy().flatten()

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate ensemble predictions.

        Args:
            X: Feature dataframe

        Returns:
            Ensemble predictions
        """
        if not self.models:
            raise ValueError("Models not trained. Call fit() first.")

        # Get predictions from each model
        predictions = []
        for model_info, weight in zip(self.models, self.weights):
            model = model_info["model"]
            model_type = model_info["type"]

            pred = self._predict_model(model, model_type, X)
            predictions.append(pred * weight)

        # Weighted average
        ensemble_pred = np.sum(predictions, axis=0)

        return ensemble_pred

    def predict_proba(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get predictions from individual models.

        Returns:
            Dictionary with predictions from each model
        """
        predictions = {}

        for i, model_info in enumerate(self.models):
            model = model_info["model"]
            model_type = model_info["type"]

            pred = self._predict_model(model, model_type, X)
            predictions[f"{model_type}_{i}"] = pred

        return predictions

    def calculate_ic(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate information coefficient for ensemble and individual models.

        Args:
            X: Features
            y: Targets

        Returns:
            Dictionary of ICs
        """
        ics = {}

        # Ensemble IC
        ensemble_pred = self.predict(X)
        ics["ensemble"] = float(np.corrcoef(ensemble_pred, y)[0, 1])

        # Individual model ICs
        for i, model_info in enumerate(self.models):
            model = model_info["model"]
            model_type = model_info["type"]

            pred = self._predict_model(model, model_type, X)
            ic = np.corrcoef(pred, y)[0, 1]
            ics[f"{model_type}_{i}"] = float(ic)

        return ics


class SimpleNN(nn.Module):
    """Simple feedforward neural network for alpha generation."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int] = [256, 128, 64],
        dropout: float = 0.3,
    ):
        """Initialize neural network."""
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)

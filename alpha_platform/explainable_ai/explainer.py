"""
Model explainer using SHAP, LIME, and attention mechanisms.

Provides interpretable explanations for trading signals and predictions.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import shap
except ImportError:
    warnings.warn("shap not installed. Install with: pip install shap")
    shap = None

try:
    from lime import lime_tabular
except ImportError:
    warnings.warn("lime not installed. Install with: pip install lime")
    lime_tabular = None

import matplotlib.pyplot as plt

from alpha_platform.utils.config import get_config
from alpha_platform.utils.logger import get_logger

logger = get_logger(__name__)


class ModelExplainer:
    """
    Explain model predictions using various interpretability methods.

    Supports:
    - SHAP values for feature attribution
    - LIME for local explanations
    - Attention visualization for transformer models
    - Partial dependence plots
    """

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        model_type: str = "tree",
    ):
        """
        Initialize model explainer.

        Args:
            model: Trained model to explain
            feature_names: List of feature names
            model_type: Type of model ('tree', 'linear', 'deep')
        """
        self.config = get_config()
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type

        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None

        logger.info(f"Model explainer initialized for {model_type} model")

    def explain_prediction(
        self,
        X: np.ndarray,
        method: str = "shap",
        background_data: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Explain a single prediction or batch of predictions.

        Args:
            X: Input features [n_samples, n_features]
            method: Explanation method ('shap', 'lime', 'both')
            background_data: Background dataset for SHAP

        Returns:
            Dictionary with explanations
        """
        explanations = {}

        if method in ["shap", "both"]:
            if shap is None:
                logger.warning("SHAP not available, skipping")
            else:
                explanations["shap"] = self._explain_with_shap(X, background_data)

        if method in ["lime", "both"]:
            if lime_tabular is None:
                logger.warning("LIME not available, skipping")
            else:
                explanations["lime"] = self._explain_with_lime(X, background_data)

        return explanations

    def _explain_with_shap(
        self,
        X: np.ndarray,
        background_data: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Explain using SHAP values.

        Args:
            X: Input features
            background_data: Background dataset

        Returns:
            SHAP values and related information
        """
        # Initialize SHAP explainer if needed
        if self.shap_explainer is None:
            if self.model_type == "tree":
                self.shap_explainer = shap.TreeExplainer(self.model)
            else:
                # Use KernelExplainer for other models
                if background_data is None:
                    raise ValueError("background_data required for non-tree models")

                def model_predict(x):
                    return self.model.predict(x)

                self.shap_explainer = shap.KernelExplainer(
                    model_predict,
                    background_data,
                )

        # Calculate SHAP values
        shap_values = self.shap_explainer.shap_values(X)

        # Handle different output formats
        if isinstance(shap_values, list):
            # Multi-class or multi-output
            shap_values = shap_values[0]

        # Get feature importance
        feature_importance = np.abs(shap_values).mean(axis=0)

        # Create explanation
        explanation = {
            "shap_values": shap_values,
            "feature_importance": dict(zip(self.feature_names, feature_importance)),
            "base_value": self.shap_explainer.expected_value
            if hasattr(self.shap_explainer, "expected_value")
            else 0.0,
        }

        # Top features
        top_k = min(10, len(self.feature_names))
        top_indices = np.argsort(-feature_importance)[:top_k]
        explanation["top_features"] = [
            {
                "name": self.feature_names[idx],
                "importance": float(feature_importance[idx]),
            }
            for idx in top_indices
        ]

        return explanation

    def _explain_with_lime(
        self,
        X: np.ndarray,
        background_data: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Explain using LIME.

        Args:
            X: Input features
            background_data: Training data for LIME

        Returns:
            LIME explanations
        """
        if background_data is None:
            raise ValueError("background_data required for LIME")

        # Initialize LIME explainer if needed
        if self.lime_explainer is None:
            self.lime_explainer = lime_tabular.LimeTabularExplainer(
                background_data,
                feature_names=self.feature_names,
                mode="regression",
                verbose=False,
            )

        # Explain first instance (or can loop for batch)
        instance = X[0] if len(X.shape) > 1 else X

        # Get explanation
        exp = self.lime_explainer.explain_instance(
            instance,
            self.model.predict if hasattr(self.model, "predict") else self.model,
            num_features=min(10, len(self.feature_names)),
        )

        # Extract feature contributions
        feature_contributions = dict(exp.as_list())

        explanation = {
            "feature_contributions": feature_contributions,
            "score": exp.score,
            "local_prediction": exp.local_pred[0] if hasattr(exp, "local_pred") else None,
        }

        return explanation

    def get_feature_importance(
        self,
        X: np.ndarray,
        method: str = "shap",
    ) -> pd.DataFrame:
        """
        Get global feature importance.

        Args:
            X: Sample of data
            method: Method to use ('shap', 'permutation')

        Returns:
            DataFrame with feature importance
        """
        if method == "shap":
            explanation = self._explain_with_shap(X)
            importance = explanation["feature_importance"]

            df = pd.DataFrame(
                {
                    "feature": list(importance.keys()),
                    "importance": list(importance.values()),
                }
            )
            df = df.sort_values("importance", ascending=False)

            return df

        else:
            raise ValueError(f"Unknown method: {method}")

    def visualize_shap(
        self,
        X: np.ndarray,
        plot_type: str = "summary",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Create SHAP visualization.

        Args:
            X: Input data
            plot_type: Type of plot ('summary', 'waterfall', 'force')
            save_path: Optional path to save plot
        """
        if shap is None:
            logger.warning("SHAP not available")
            return

        explanation = self._explain_with_shap(X)
        shap_values = explanation["shap_values"]

        plt.figure(figsize=(12, 8))

        if plot_type == "summary":
            shap.summary_plot(
                shap_values, X, feature_names=self.feature_names, show=False
            )
        elif plot_type == "waterfall":
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=explanation["base_value"],
                    data=X[0],
                    feature_names=self.feature_names,
                ),
                show=False,
            )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved SHAP plot to {save_path}")
        else:
            plt.show()

        plt.close()

    def generate_prediction_report(
        self,
        X: np.ndarray,
        prediction: float,
        method: str = "shap",
    ) -> str:
        """
        Generate human-readable explanation for a prediction.

        Args:
            X: Input features
            prediction: Model prediction
            method: Explanation method

        Returns:
            Text explanation
        """
        explanation = self.explain_prediction(X, method=method)

        if method == "shap" and "shap" in explanation:
            shap_exp = explanation["shap"]
            top_features = shap_exp["top_features"]

            report = f"Prediction: {prediction:.4f}\n\n"
            report += "Top Contributing Features:\n"

            for i, feat in enumerate(top_features, 1):
                report += f"{i}. {feat['name']}: {feat['importance']:.4f}\n"

            return report

        return "No explanation available"

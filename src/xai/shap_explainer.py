"""
SHAP Explanation Module for Predictive Maintenance.
Provides local and global feature attribution using SHAP TreeExplainer on Random Forest
models, translating numerical SHAP values into ranked factors and qualitative impact tiers.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import shap

from src.preprocessing.preprocessor import CANONICAL_FEATURES


# Friendly feature display names
FEATURE_DISPLAY_NAMES = {
    "Type": "Product Variant Type",
    "Air temperature [K]": "Air Temperature",
    "Process temperature [K]": "Process Temperature",
    "Rotational speed [rpm]": "Rotational Speed",
    "Torque [Nm]": "Torque",
    "Tool wear [min]": "Tool Wear",
    "Machine failure": "Machine Failure State",
}


class ShapExplainer:
    """Computes and formats SHAP feature attributions for model predictions."""

    def __init__(self, model: Any, feature_names: Optional[List[str]] = None, class_names: Optional[List[str]] = None):
        self.model = model
        self.feature_names = feature_names or CANONICAL_FEATURES
        self.class_names = class_names or []
        self.explainer = shap.TreeExplainer(self.model)

    def _determine_impact_tier(self, abs_val: float, max_val: float) -> str:
        """Categorizes relative SHAP magnitude into 'High', 'Medium', or 'Low'."""
        if max_val <= 1e-6:
            return "Low"
        ratio = abs_val / max_val
        if ratio >= 0.65 or abs_val >= 0.15:
            return "High"
        elif ratio >= 0.30 or abs_val >= 0.05:
            return "Medium"
        return "Low"

    def explain_instance(
        self,
        instance_df: pd.DataFrame,
        target_class: Optional[Union[str, int]] = None,
        top_k: int = 4,
    ) -> Dict[str, Any]:
        """
        Computes SHAP feature contributions for a single instance.

        Parameters
        ----------
        instance_df : pd.DataFrame
            Single-row DataFrame matching the model's expected features.
        target_class : str or int, optional
            The specific class for which to compute explanations.
        top_k : int
            Number of top features to return.

        Returns
        -------
        dict containing 'top_factors', 'raw_shap_values', 'base_value', and 'prediction_class'.
        """
        # Ensure column subset matches model features
        X = instance_df[self.feature_names]

        # Compute SHAP values
        shap_res = self.explainer(X)
        values = shap_res.values  # Can be (1, n_features) or (1, n_features, n_classes)

        # Handle multiclass vs binary dimensions
        if values.ndim == 3:
            n_classes = values.shape[2]
            class_idx = 1 if n_classes == 2 else 0

            if target_class is not None:
                if isinstance(target_class, int) and target_class < n_classes:
                    class_idx = target_class
                elif isinstance(target_class, str) and target_class in self.class_names:
                    class_idx = self.class_names.index(target_class)

            sample_shap = values[0, :, class_idx]
            base_val = (
                shap_res.base_values[0, class_idx]
                if hasattr(shap_res.base_values, "__getitem__") and np.ndim(shap_res.base_values) > 1
                else float(shap_res.base_values[0])
            )
            selected_class = self.class_names[class_idx] if class_idx < len(self.class_names) else str(class_idx)
        else:
            sample_shap = values[0]
            base_val = float(shap_res.base_values[0]) if hasattr(shap_res.base_values, "__len__") else float(shap_res.base_values)
            selected_class = "Failure" if target_class is None else str(target_class)

        # Pair features with their SHAP values and original inputs
        factors = []
        max_abs = float(np.max(np.abs(sample_shap))) if len(sample_shap) > 0 else 1.0

        for feat_name, shap_val in zip(self.feature_names, sample_shap):
            raw_val = instance_df[feat_name].iloc[0] if feat_name in instance_df.columns else None
            abs_val = abs(float(shap_val))
            impact_tier = self._determine_impact_tier(abs_val, max_abs)
            display_name = FEATURE_DISPLAY_NAMES.get(feat_name, feat_name)

            factors.append({
                "feature": display_name,
                "raw_feature_name": feat_name,
                "feature_value": float(raw_val) if isinstance(raw_val, (int, float, np.number)) else str(raw_val),
                "shap_value": round(float(shap_val), 4),
                "impact": impact_tier,
                "direction": "Increases Risk" if shap_val > 0 else "Decreases Risk",
            })

        # Filter to real physical sensor factors
        sensor_factors = [
            f for f in factors if f["raw_feature_name"] not in ["Machine failure", "machine_failure"]
        ]

        # Sort factors by absolute SHAP contribution descending
        sensor_factors.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        top_factors = sensor_factors[:top_k]

        return {
            "selected_class": selected_class,
            "base_value": round(float(base_val), 4),
            "top_factors": top_factors,
            "all_factors": factors,
        }

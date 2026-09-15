"""
Global Feature Importance Analysis and Visualization Module.
Extracts model-level feature importance metrics (Mean Decrease in Impurity and global SHAP),
useful for executive reporting and model diagnostics.
"""

from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.xai.shap_explainer import FEATURE_DISPLAY_NAMES


class FeatureImportanceAnalyzer:
    """Analyzes and plots feature importance across models."""

    def __init__(self, model: Any, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names

    def get_tree_importance(self) -> List[Dict[str, Any]]:
        """Extracts tree-based feature importances from a fitted Random Forest."""
        if not hasattr(self.model, "feature_importances_"):
            raise ValueError("Model does not have feature_importances_ attribute.")

        importances = self.model.feature_importances_
        results = []
        for name, imp in zip(self.feature_names, importances):
            results.append({
                "feature": FEATURE_DISPLAY_NAMES.get(name, name),
                "raw_name": name,
                "importance": round(float(imp), 4),
            })
        results.sort(key=lambda x: x["importance"], reverse=True)
        return results

    def plot_feature_importance(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plots horizontal bar chart of feature importances."""
        data = self.get_tree_importance()
        features = [d["feature"] for d in reversed(data)]
        scores = [d["importance"] for d in reversed(data)]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(features, scores, color="#2563eb", edgecolor="#1d4ed8", alpha=0.85)
        ax.set_xlabel("Relative Importance (MDI)")
        ax.set_title("Random Forest Feature Importance", fontsize=13, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        # Value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}",
                ha="left",
                va="center",
                fontsize=9,
            )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        return fig

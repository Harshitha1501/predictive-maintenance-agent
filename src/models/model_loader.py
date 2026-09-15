"""
Unified Model Loader and Inference Engine.
Loads Jatin's Binary Model and Seenu's Multi-Class Model, providing coordinated
predictions and probability distributions.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd

from src.preprocessing.preprocessor import CANONICAL_FEATURES, TelemetryPreprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY_MODEL_PATH = PROJECT_ROOT / "models" / "binary_model.pkl"
DEFAULT_MULTICLASS_MODEL_PATH = PROJECT_ROOT / "models" / "multiclass_model.pkl"

FAILURE_LABEL_DESCRIPTIONS = {
    "No Failure": "No Failure",
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure",
    "TWF": "Tool Wear Failure",
    "RNF": "Random Failure",
}


class ModelLoader:
    """Loads, verifies, and executes binary and multi-class predictive maintenance models."""

    def __init__(
        self,
        binary_path: Optional[Union[str, Path]] = None,
        multiclass_path: Optional[Union[str, Path]] = None,
    ):
        self.binary_path = Path(binary_path) if binary_path else DEFAULT_BINARY_MODEL_PATH
        self.multiclass_path = Path(multiclass_path) if multiclass_path else DEFAULT_MULTICLASS_MODEL_PATH
        self.binary_bundle: Optional[Dict[str, Any]] = None
        self.multiclass_bundle: Optional[Dict[str, Any]] = None
        self.preprocessor = TelemetryPreprocessor()
        self._load_models()

    def _load_models(self) -> None:
        """Loads both model bundles from disk using explicit absolute path verification."""
        binary_p = Path(self.binary_path)
        if not binary_p.is_absolute():
            binary_p = PROJECT_ROOT / binary_p

        multiclass_p = Path(self.multiclass_path)
        if not multiclass_p.is_absolute():
            multiclass_p = PROJECT_ROOT / multiclass_p

        if binary_p.exists():
            self.binary_bundle = joblib.load(str(binary_p))
        else:
            print(f"Warning: Binary model not found at {binary_p}")

        if multiclass_p.exists():
            self.multiclass_bundle = joblib.load(str(multiclass_p))
            # Update preprocessor with Seenu's Type encoder if available
            if "feature_encoders" in self.multiclass_bundle and "Type" in self.multiclass_bundle["feature_encoders"]:
                self.preprocessor.type_encoder = self.multiclass_bundle["feature_encoders"]["Type"]
        else:
            print(f"Warning: Multiclass model not found at {multiclass_p}")

    @property
    def binary_model(self):
        return self.binary_bundle["model"] if self.binary_bundle else None

    @property
    def multiclass_model(self):
        return self.multiclass_bundle["model"] if self.multiclass_bundle else None

    def predict_binary(self, X: pd.DataFrame) -> Tuple[int, str, float]:
        """
        Executes binary failure prediction.

        Returns
        -------
        (prediction_code, prediction_label, failure_probability)
        """
        if self.binary_model is None:
            raise RuntimeError("Binary classification model is not loaded.")

        X_canonical = X[CANONICAL_FEATURES]
        pred = int(self.binary_model.predict(X_canonical)[0])
        probs = self.binary_model.predict_proba(X_canonical)[0]

        # Probability of class 1 (Failure)
        fail_prob = float(probs[1]) if len(probs) > 1 else float(pred)
        label = "Machine Failure" if pred == 1 else "No Failure"
        return pred, label, fail_prob

    def predict_multiclass(
        self, X: pd.DataFrame, binary_pred: Optional[int] = None
    ) -> Tuple[str, str, float, Dict[str, float]]:
        """
        Executes multi-class failure prediction.

        Parameters
        ----------
        X : pd.DataFrame
        binary_pred : Optional[int]
            The binary failure state (0 or 1). If None, binary model will be consulted if available.

        Returns
        -------
        (raw_type_code, full_description, top_type_probability, all_class_probabilities)
        """
        if self.multiclass_model is None:
            raise RuntimeError("Multiclass classification model is not loaded.")

        target_encoder = self.multiclass_bundle["target_encoder"]
        classes = list(target_encoder.classes_)

        # Prepare features for Seenu's model which expects Machine failure column
        X_multi = X.copy()
        if "Machine failure" not in X_multi.columns:
            if binary_pred is not None:
                X_multi["Machine failure"] = binary_pred
            elif self.binary_model is not None:
                b_pred, _, _ = self.predict_binary(X)
                X_multi["Machine failure"] = b_pred
            else:
                X_multi["Machine failure"] = 0

        feature_names = self.multiclass_bundle.get("feature_names", CANONICAL_FEATURES + ["Machine failure"])
        X_input = X_multi[feature_names]

        pred_idx = self.multiclass_model.predict(X_input)[0]
        # In case prediction is index or string
        if isinstance(pred_idx, (int, np.integer)):
            raw_code = target_encoder.inverse_transform([pred_idx])[0]
        else:
            raw_code = str(pred_idx)

        probs = self.multiclass_model.predict_proba(X_input)[0]
        prob_dict = {classes[i]: float(probs[i]) for i in range(len(classes))}
        
        # If binary model says Failure (or failure probability is high), ensure failure type is not 'No Failure'
        # if a strong candidate failure mode exists
        if binary_pred == 1 and raw_code == "No Failure":
            # Pick highest non-'No Failure' probability
            non_no_fail = {k: v for k, v in prob_dict.items() if k != "No Failure"}
            if non_no_fail:
                best_fail = max(non_no_fail.items(), key=lambda x: x[1])
                if best_fail[1] > 0.05:  # meaningful threshold
                    raw_code = best_fail[0]

        top_prob = prob_dict.get(raw_code, max(probs))
        full_desc = FAILURE_LABEL_DESCRIPTIONS.get(raw_code, raw_code)

        return raw_code, full_desc, float(top_prob), prob_dict

"""Unit tests for the SHAP Explainer module."""

import joblib
import pandas as pd
import pytest
from src.preprocessing.preprocessor import CANONICAL_FEATURES
from src.xai.shap_explainer import ShapExplainer


@pytest.fixture
def binary_explainer():
    bundle = joblib.load("models/binary_model.pkl")
    return ShapExplainer(
        model=bundle["model"],
        feature_names=CANONICAL_FEATURES,
        class_names=["No Failure", "Machine Failure"],
    )


def test_shap_explanation_structure(binary_explainer):
    sample = pd.DataFrame([{
        "Type": 1,
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 45.0,
        "Tool wear [min]": 120.0,
    }])
    explanation = binary_explainer.explain_instance(sample, target_class=1, top_k=3)

    assert "top_factors" in explanation
    assert len(explanation["top_factors"]) == 3
    assert "base_value" in explanation

    first_factor = explanation["top_factors"][0]
    assert "feature" in first_factor
    assert "impact" in first_factor
    assert first_factor["impact"] in ["High", "Medium", "Low"]
    assert "shap_value" in first_factor
    assert "direction" in first_factor

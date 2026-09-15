"""
Predictive Maintenance Reasoning Agent.
Orchestrates the complete intelligent maintenance lifecycle:
Sensor Data -> Dual ML Inference -> SHAP Feature Attribution -> Root Cause Engine -> Prescriptive Action.
Produces clean, API-ready structured data for backend and dashboard integration.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from src.models.model_loader import ModelLoader
from src.preprocessing.preprocessor import CANONICAL_FEATURES, TelemetryPreprocessor
from src.xai.root_cause_engine import RootCauseEngine
from src.xai.shap_explainer import ShapExplainer


class PredictiveMaintenanceAgent:
    """End-to-end intelligent agent explaining machine health and prescribing actions."""

    def __init__(
        self,
        binary_model_path: Optional[Any] = None,
        multiclass_model_path: Optional[Any] = None,
    ):
        self.model_loader = ModelLoader(
            binary_path=binary_model_path,
            multiclass_path=multiclass_model_path,
        )
        self.preprocessor = self.model_loader.preprocessor
        self.root_cause_engine = RootCauseEngine()

        # Initialize SHAP explainers
        self.binary_shap: Optional[ShapExplainer] = None
        self.multiclass_shap: Optional[ShapExplainer] = None

        if self.model_loader.binary_model is not None:
            self.binary_shap = ShapExplainer(
                model=self.model_loader.binary_model,
                feature_names=CANONICAL_FEATURES,
                class_names=["No Failure", "Machine Failure"],
            )

        if self.model_loader.multiclass_model is not None:
            classes = list(self.model_loader.multiclass_bundle["target_encoder"].classes_)
            features = self.model_loader.multiclass_bundle.get(
                "feature_names", CANONICAL_FEATURES + ["Machine failure"]
            )
            self.multiclass_shap = ShapExplainer(
                model=self.model_loader.multiclass_model,
                feature_names=features,
                class_names=classes,
            )

    def analyze(self, sensor_reading: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Executes the full predictive maintenance reasoning workflow on a single sensor reading.

        Parameters
        ----------
        sensor_reading : dict or single-row DataFrame
            Sensor measurements ('Type', 'Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]').

        Returns
        -------
        dict matching Trusha's API schema plus rich diagnostic telemetry.
        """
        # 1. Preprocess & Validate Input
        raw_dict = sensor_reading if isinstance(sensor_reading, dict) else sensor_reading.iloc[0].to_dict()
        normalized_dict = self.preprocessor.normalize_keys(raw_dict)
        warnings = self.preprocessor.validate_reading(normalized_dict)

        df_input = self.preprocessor.prepare_input(normalized_dict, include_machine_failure=False)

        # 2. Binary Failure Classification
        if self.model_loader.binary_model is not None:
            binary_code, binary_label, failure_prob = self.model_loader.predict_binary(df_input)
        else:
            # Fallback if binary model is unavailable
            binary_code, binary_label, failure_prob = 0, "No Failure", 0.05

        # 3. Multi-Class Failure Identification
        if self.model_loader.multiclass_model is not None:
            raw_type_code, failure_type, type_prob, all_type_probs = self.model_loader.predict_multiclass(
                df_input, binary_pred=binary_code
            )
        else:
            raw_type_code, failure_type, type_prob, all_type_probs = (
                "No Failure",
                "No Failure",
                1.0 - failure_prob,
                {"No Failure": 1.0 - failure_prob},
            )

        # Harmonize binary and multi-class predictions
        # If multi-class identifies a specific failure with high confidence, align binary
        if failure_type != "No Failure" and binary_code == 0 and failure_prob < 0.5:
            if type_prob > 0.40:
                binary_label = "Machine Failure"
                binary_code = 1
                failure_prob = max(failure_prob, type_prob)

        if binary_code == 0:
            failure_type = "No Failure"

        # 4. Compute SHAP Explanations
        top_factors = []
        shap_details = {}

        if binary_code == 1 and self.multiclass_shap is not None and raw_type_code != "No Failure":
            # Explain the specific failure mode in multiclass model
            df_with_fail = df_input.copy()
            df_with_fail["Machine failure"] = binary_code
            explanation = self.multiclass_shap.explain_instance(df_with_fail, target_class=raw_type_code, top_k=4)
            top_factors = explanation["top_factors"]
            shap_details = explanation
        elif self.binary_shap is not None:
            # Explain binary failure/healthy status
            explanation = self.binary_shap.explain_instance(df_input, target_class=1 if binary_code == 1 else 0, top_k=4)
            top_factors = explanation["top_factors"]
            shap_details = explanation

        # Simplify top_factors to format requested by Trusha: [{"feature": "Torque", "impact": "High"}, ...]
        formatted_factors = [
            {
                "feature": f["feature"],
                "impact": f["impact"],
                "value": f["feature_value"],
                "direction": f["direction"],
                "shap_value": f["shap_value"],
            }
            for f in top_factors
        ]

        # 5. Root-Cause Reasoning Engine
        diagnosis = self.root_cause_engine.diagnose(
            sensor_data=normalized_dict,
            failure_type=failure_type,
            failure_probability=failure_prob,
            top_factors=top_factors,
        )

        # 6. Assemble API-friendly response
        response = {
            "prediction": binary_label,
            "failure_probability": round(float(failure_prob), 4),
            "failure_type": failure_type,
            "root_cause": diagnosis["root_cause"],
            "risk_level": diagnosis["risk_level"],
            "top_factors": [
                {"feature": f["feature"], "impact": f["impact"]} for f in formatted_factors
            ],
            "recommendation": diagnosis["recommendation"],
            # Rich auxiliary telemetry for dashboard integration
            "detailed_factors": formatted_factors,
            "physical_telemetry": diagnosis["physical_telemetry"],
            "all_class_probabilities": {k: round(v, 4) for k, v in all_type_probs.items()},
            "alert": diagnosis["risk_level"] in ["High", "Critical"],
            "warnings": warnings,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        return response

    def format_text_summary(self, result: Dict[str, Any]) -> str:
        """Renders an executive summary card for display in logs or CLI."""
        bar = "=" * 48
        sub_bar = "-" * 48
        factors_text = "\n".join(
            [f"  {i+1}. {f['feature']:<20} -> {f['impact']} risk" for i, f in enumerate(result["top_factors"][:3])]
        )

        return (
            f"\n+{bar}+\n"
            f"|            MACHINE HEALTH ANALYSIS             |\n"
            f"+{sub_bar}+\n"
            f"| Prediction:          {result['prediction']:<26}|\n"
            f"| Probability:         {result['failure_probability'] * 100:.1f}%{'':<23}|\n"
            f"| Risk Level:          {result['risk_level']:<26}|\n"
            f"| Failure Mode:        {result['failure_type']:<26}|\n"
            f"+{sub_bar}+\n"
            f"| TOP CONTRIBUTING FACTORS (SHAP):               |\n"
            f"{factors_text}\n"
            f"+{sub_bar}+\n"
            f"| ROOT CAUSE:                                    |\n"
            f"| {result['root_cause'][:46]:<47}|\n"
            f"|                                                |\n"
            f"| MAINTENANCE RECOMMENDATION:                    |\n"
            f"| {result['recommendation'][:46]:<47}|\n"
            f"+{bar}+\n"
        )

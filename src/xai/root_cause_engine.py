"""
Root-Cause Reasoning Engine.
Synthesizes machine learning predictions, SHAP attribution weights, and physics-based
manufacturing failure rules to identify the exact root cause of anomalies and recommend
concrete maintenance actions.
"""

from typing import Any, Dict, List, Optional
from src.xai.maintenance_rules import evaluate_physical_signals, get_maintenance_rule


class RootCauseEngine:
    """Intelligent reasoning layer diagnosing physical root causes from ML & SHAP outputs."""

    def __init__(self):
        pass

    def diagnose(
        self,
        sensor_data: Dict[str, Any],
        failure_type: str,
        failure_probability: float,
        top_factors: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Diagnoses the failure root cause and selects prescriptive maintenance actions.

        Parameters
        ----------
        sensor_data : dict
            Original sensor readings
        failure_type : str
            Predicted failure mode ('Overstrain Failure', 'Power Failure', etc.)
        failure_probability : float
            Probability score of machine failure
        top_factors : list
            SHAP top contributing factors

        Returns
        -------
        dict with root_cause, recommendation, risk_level, physical_metrics, and diagnostics.
        """
        # 1. Evaluate physical calculations
        physical_signals = evaluate_physical_signals(sensor_data)

        # 2. Match maintenance rule
        rule_output = get_maintenance_rule(failure_type, physical_signals, top_factors)
        root_cause = rule_output["root_cause"]
        recommendation = rule_output["recommendation"]
        risk_level = rule_output["risk_level"]

        # 3. Dynamic risk escalation based on failure probability
        if failure_probability >= 0.85:
            if risk_level not in ["Critical"]:
                risk_level = "High"
        elif failure_probability >= 0.50:
            if risk_level == "Low":
                risk_level = "Medium"
        elif failure_probability < 0.20:
            if failure_type in ["No Failure", "NORMAL"]:
                risk_level = "Low"

        # 4. Synthesize diagnostic narrative highlighting the primary SHAP drivers
        driver_strings = [
            f"{f['feature']} ({f['impact']} impact, value: {f['feature_value']})"
            for f in top_factors[:3]
        ]
        drivers_summary = ", ".join(driver_strings) if driver_strings else "Normal operational variance"

        return {
            "root_cause": root_cause,
            "recommendation": recommendation,
            "risk_level": risk_level,
            "drivers_summary": drivers_summary,
            "physical_telemetry": {
                "temperature_difference_k": physical_signals["temp_diff"],
                "mechanical_power_watts": physical_signals["power_watts"],
                "overstrain_product_min_nm": physical_signals["strain_product"],
                "overstrain_threshold": physical_signals["osf_limit"],
            },
        }

"""
Predictive Maintenance Agent Demo & Interactive Validation Script.
Executes end-to-end inference and reasoning on representative AI4I 2020 machine states.
Run with: python demo.py
"""

import json
from src.xai.agent import PredictiveMaintenanceAgent


def run_demo():
    print("\n" + "=" * 60)
    print("      AI4I 2020 PREDICTIVE MAINTENANCE AGENT DEMO")
    print("=" * 60)

    agent = PredictiveMaintenanceAgent()

    test_scenarios = [
        {
            "scenario": "1. Nominal Machine Health (Healthy)",
            "telemetry": {
                "type": "L",
                "air_temperature": 298.1,
                "process_temperature": 308.6,
                "rotational_speed": 1551.0,
                "torque": 42.8,
                "tool_wear": 12.0,
            },
        },
        {
            "scenario": "2. Overstrain Failure (High Torque + Worn Tool)",
            "telemetry": {
                "type": "L",
                "air_temperature": 298.4,
                "process_temperature": 308.2,
                "rotational_speed": 1282.0,
                "torque": 60.7,
                "tool_wear": 216.0,
            },
        },
        {
            "scenario": "3. Heat Dissipation Failure (Cooling Defect)",
            "telemetry": {
                "type": "M",
                "air_temperature": 302.4,
                "process_temperature": 310.2,
                "rotational_speed": 1332.0,
                "torque": 52.3,
                "tool_wear": 142.0,
            },
        },
        {
            "scenario": "4. Power Failure (Drive Overload / Stall)",
            "telemetry": {
                "type": "L",
                "air_temperature": 298.9,
                "process_temperature": 309.1,
                "rotational_speed": 2861.0,
                "torque": 4.6,
                "tool_wear": 143.0,
            },
        },
        {
            "scenario": "5. Tool Wear Failure (Exhausted Cutting Insert)",
            "telemetry": {
                "type": "L",
                "air_temperature": 298.8,
                "process_temperature": 308.9,
                "rotational_speed": 1455.0,
                "torque": 41.3,
                "tool_wear": 208.0,
            },
        },
    ]

    for item in test_scenarios:
        print(f"\n>>> Running Scenario: {item['scenario']}")
        result = agent.analyze(item["telemetry"])
        print(agent.format_text_summary(result))

    print("\n" + "=" * 60)
    print("Example API Response Payload for Trusha's Backend:")
    print("=" * 60)
    sample_res = agent.analyze(test_scenarios[1]["telemetry"])
    api_payload = {
        "prediction": sample_res["prediction"],
        "failure_probability": sample_res["failure_probability"],
        "failure_type": sample_res["failure_type"],
        "root_cause": sample_res["root_cause"],
        "risk_level": sample_res["risk_level"],
        "top_factors": sample_res["top_factors"],
        "recommendation": sample_res["recommendation"],
    }
    print(json.dumps(api_payload, indent=2))
    print("\n[SUCCESS] All reasoning chains executed successfully.\n")


if __name__ == "__main__":
    run_demo()

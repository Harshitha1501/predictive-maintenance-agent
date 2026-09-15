"""
Frontend & Integration QA Test Suite for Neethu's Streamlit Dashboard.
Tests payload contract compatibility, preset cases, and session state data structures.
"""

import pytest
from src.xai.agent import PredictiveMaintenanceAgent
from src.api.schemas import SensorInput, MaintenanceResponse


@pytest.fixture(scope="module")
def agent():
    return PredictiveMaintenanceAgent()


def test_preset_scenarios_validity(agent):
    """Verifies that all standard 1-click test scenarios produce valid structured outputs."""
    preset_scenarios = [
        {
            "name": "Healthy Machine (Nominal)",
            "telemetry": {
                "type": "L",
                "air_temperature": 298.1,
                "process_temperature": 308.6,
                "rotational_speed": 1551.0,
                "torque": 42.8,
                "tool_wear": 12.0,
            },
            "expected_pred": "No Failure",
        },
        {
            "name": "Overstrain Failure (OSF)",
            "telemetry": {
                "type": "L",
                "air_temperature": 298.4,
                "process_temperature": 308.2,
                "rotational_speed": 1282.0,
                "torque": 60.7,
                "tool_wear": 216.0,
            },
            "expected_pred": "Machine Failure",
        },
        {
            "name": "Heat Dissipation Failure (HDF)",
            "telemetry": {
                "type": "M",
                "air_temperature": 302.4,
                "process_temperature": 310.2,
                "rotational_speed": 1332.0,
                "torque": 52.3,
                "tool_wear": 142.0,
            },
            "expected_pred": "Machine Failure",
        },
        {
            "name": "Power Failure (PWF)",
            "telemetry": {
                "type": "L",
                "air_temperature": 298.9,
                "process_temperature": 309.1,
                "rotational_speed": 2861.0,
                "torque": 4.6,
                "tool_wear": 143.0,
            },
            "expected_pred": "Machine Failure",
        },
        {
            "name": "Tool Wear Failure (TWF)",
            "telemetry": {
                "type": "L",
                "air_temperature": 298.8,
                "process_temperature": 308.9,
                "rotational_speed": 1455.0,
                "torque": 41.3,
                "tool_wear": 208.0,
            },
            "expected_pred": "Machine Failure",
        },
    ]

    for scenario in preset_scenarios:
        res = agent.analyze(scenario["telemetry"])
        assert res["prediction"] == scenario["expected_pred"], (
            f"Scenario '{scenario['name']}' predicted {res['prediction']} instead of {scenario['expected_pred']}"
        )
        assert "risk_level" in res
        assert "recommendation" in res
        assert "root_cause" in res
        assert isinstance(res["top_factors"], list)


def test_pydantic_schema_validation():
    """Ensures raw telemetry conforms to Trusha's API and Neethu's UI schema."""
    input_data = {
        "type": "M",
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500.0,
        "torque": 40.0,
        "tool_wear": 50.0,
    }
    validated = SensorInput(**input_data)
    assert validated.type == "M"
    assert validated.air_temperature == 300.0

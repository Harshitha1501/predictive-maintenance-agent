"""
Edge-Case & Outlier Robustness Test Suite for Predictive Maintenance Agent & API.
Verifies that extreme boundary values (e.g., 0 torque, 250 min tool wear, inverted thermal
gradients, extreme spindle speeds) are handled gracefully without unhandled exceptions.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.xai.agent import PredictiveMaintenanceAgent

client = TestClient(app)


@pytest.fixture(scope="module")
def agent():
    return PredictiveMaintenanceAgent()


def test_edge_case_zero_torque(agent):
    """Zero torque condition (e.g., motor completely unloaded or idle spinning)."""
    telemetry = {
        "type": "L",
        "air_temperature": 298.0,
        "process_temperature": 308.0,
        "rotational_speed": 1500.0,
        "torque": 0.0,
        "tool_wear": 50.0,
    }
    result = agent.analyze(telemetry)
    assert result is not None
    assert "prediction" in result
    assert "failure_probability" in result
    assert 0.0 <= result["failure_probability"] <= 1.0
    assert "root_cause" in result
    assert "recommendation" in result
    # Physical telemetry power should be exactly 0
    assert result["physical_telemetry"]["mechanical_power_watts"] == 0.0


def test_edge_case_maximum_tool_wear(agent):
    """Maximum tool wear of 250 min handled gracefully without throwing unhandled exceptions."""
    telemetry = {
        "type": "M",
        "air_temperature": 299.0,
        "process_temperature": 309.0,
        "rotational_speed": 1400.0,
        "torque": 45.0,
        "tool_wear": 250.0,
    }
    result = agent.analyze(telemetry)
    assert result is not None
    assert "prediction" in result
    assert "failure_probability" in result
    assert 0.0 <= result["failure_probability"] <= 1.0
    assert "risk_level" in result
    assert "root_cause" in result
    assert "recommendation" in result
    assert isinstance(result["top_factors"], list)


def test_edge_case_extreme_spindle_speed(agent):
    """Extreme spindle speed at upper operating envelope (3500 RPM)."""
    telemetry = {
        "type": "H",
        "air_temperature": 297.5,
        "process_temperature": 307.5,
        "rotational_speed": 3500.0,
        "torque": 15.0,
        "tool_wear": 10.0,
    }
    result = agent.analyze(telemetry)
    assert result is not None
    assert "failure_type" in result
    assert isinstance(result["top_factors"], list)


def test_edge_case_thermal_gradient_collapse(agent):
    """Process temperature equal to or less than air temperature (cooling failure or sensor defect)."""
    telemetry = {
        "type": "L",
        "air_temperature": 305.0,
        "process_temperature": 305.0,
        "rotational_speed": 1300.0,
        "torque": 55.0,
        "tool_wear": 100.0,
    }
    result = agent.analyze(telemetry)
    assert result is not None
    assert result["physical_telemetry"]["temperature_difference_k"] == 0.0


def test_api_predict_outlier_payload():
    """Verifies FastAPI /api/v1/predict handles boundary values without 500 error."""
    payload = {
        "type": "L",
        "air_temperature": 298.0,
        "process_temperature": 308.0,
        "rotational_speed": 1500.0,
        "torque": 0.0,
        "tool_wear": 250.0,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "failure_probability" in data
    assert "root_cause" in data
    assert "recommendation" in data

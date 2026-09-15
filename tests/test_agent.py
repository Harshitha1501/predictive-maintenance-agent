"""End-to-end integration tests for the PredictiveMaintenanceAgent."""

import pandas as pd
import pytest
from src.xai.agent import PredictiveMaintenanceAgent


@pytest.fixture
def agent():
    return PredictiveMaintenanceAgent()


def test_agent_normal_reading(agent):
    normal_reading = {
        "type": "L",
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1551.0,
        "torque": 42.8,
        "tool_wear": 12.0,
    }
    result = agent.analyze(normal_reading)

    assert result["prediction"] == "No Failure"
    assert result["risk_level"] == "Low"
    assert result["alert"] is False
    assert result["failure_probability"] < 0.20
    assert len(result["top_factors"]) > 0

    # Ensure keys match Trusha's API requirement
    required_keys = [
        "prediction",
        "failure_probability",
        "failure_type",
        "root_cause",
        "risk_level",
        "top_factors",
        "recommendation",
    ]
    for key in required_keys:
        assert key in result

    # Check top_factors format: [{"feature": ..., "impact": ...}]
    for factor in result["top_factors"]:
        assert "feature" in factor
        assert "impact" in factor
        assert factor["impact"] in ["High", "Medium", "Low"]


def test_agent_overstrain_failure(agent):
    osf_reading = {
        "type": "L",
        "air_temperature": 298.4,
        "process_temperature": 308.2,
        "rotational_speed": 1282.0,
        "torque": 60.7,
        "tool_wear": 216.0,
    }
    result = agent.analyze(osf_reading)

    assert result["prediction"] == "Machine Failure"
    assert result["risk_level"] in ["High", "Critical"]
    assert result["alert"] is True
    assert result["failure_probability"] > 0.50
    assert "inspect" in result["recommendation"].lower()
    assert "load" in result["recommendation"].lower() or "tool" in result["recommendation"].lower()


def test_text_summary_formatting(agent):
    reading = {
        "type": "L",
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1551.0,
        "torque": 42.8,
        "tool_wear": 12.0,
    }
    result = agent.analyze(reading)
    summary = agent.format_text_summary(result)

    assert "MACHINE HEALTH ANALYSIS" in summary
    assert "Prediction:" in summary
    assert "TOP CONTRIBUTING FACTORS" in summary

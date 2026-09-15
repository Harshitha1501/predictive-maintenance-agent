"""Integration tests for the FastAPI REST API service."""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "endpoints" in data


def test_health_check_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["binary_model_loaded"] is True
    assert data["multiclass_model_loaded"] is True
    assert data["shap_explainer_ready"] is True


def test_predict_endpoint_normal():
    payload = {
        "type": "L",
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1551.0,
        "torque": 42.8,
        "tool_wear": 12.0,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Verify Trusha's required fields
    assert "prediction" in data
    assert "failure_probability" in data
    assert "failure_type" in data
    assert "root_cause" in data
    assert "risk_level" in data
    assert "top_factors" in data
    assert "recommendation" in data

    assert data["prediction"] == "No Failure"
    assert data["risk_level"] == "Low"
    assert isinstance(data["top_factors"], list)


def test_predict_endpoint_failure_mode():
    payload = {
        "type": "L",
        "air_temperature": 298.4,
        "process_temperature": 308.2,
        "rotational_speed": 1282.0,
        "torque": 60.7,
        "tool_wear": 216.0,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["prediction"] == "Machine Failure"
    assert data["risk_level"] in ["High", "Critical"]
    assert data["alert"] is True
    assert len(data["top_factors"]) > 0


def test_sample_cases_endpoint():
    response = client.get("/api/v1/sample-cases")
    assert response.status_code == 200
    data = response.json()
    assert "normal_operating_condition" in data
    assert "overstrain_failure_osf" in data
    assert "heat_dissipation_failure_hdf" in data
    assert "power_failure_pwf" in data
    assert "tool_wear_failure_twf" in data

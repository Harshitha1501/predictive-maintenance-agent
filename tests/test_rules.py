"""Unit tests for the physical domain rules and calculations."""

import pytest
from src.xai.maintenance_rules import (
    compute_mechanical_power,
    evaluate_physical_signals,
    get_maintenance_rule,
)


def test_power_calculation():
    # Torque = 40 Nm, Speed = 1500 rpm -> Power = 40 * (1500 * 2 * pi / 60) = 40 * 157.0796 = ~6283.18 W
    power = compute_mechanical_power(40.0, 1500.0)
    assert 6280.0 < power < 6290.0


def test_osf_condition():
    # L variant threshold is 11,000 min*Nm
    sensor_data = {
        "Type": "L",
        "Air temperature [K]": 298.0,
        "Process temperature [K]": 308.0,
        "Rotational speed [rpm]": 1400.0,
        "Torque [Nm]": 65.0,
        "Tool wear [min]": 200.0,
    }
    signals = evaluate_physical_signals(sensor_data)
    assert signals["osf_triggered"] is True  # 65 * 200 = 13000 > 11000

    rule = get_maintenance_rule("Overstrain Failure", signals)
    assert "overstrain" in rule["root_cause"].lower()
    assert "reduce" in rule["recommendation"].lower()
    assert rule["risk_level"] in ["High", "Critical"]


def test_hdf_condition():
    # Process - Air < 8.6 K and Speed < 1380 rpm
    sensor_data = {
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 305.0,  # diff = 5.0 < 8.6
        "Rotational speed [rpm]": 1200.0,   # speed = 1200 < 1380
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 50.0,
    }
    signals = evaluate_physical_signals(sensor_data)
    assert signals["hdf_triggered"] is True

    rule = get_maintenance_rule("Heat Dissipation Failure", signals)
    assert "cooling" in rule["recommendation"].lower()


def test_pwf_condition():
    # Low speed and high torque or extreme speed
    sensor_data = {
        "Type": "L",
        "Air temperature [K]": 298.0,
        "Process temperature [K]": 308.0,
        "Rotational speed [rpm]": 1000.0,
        "Torque [Nm]": 20.0,  # Power = 20 * 104.7 = ~2094 W (< 3500 W safe boundary)
        "Tool wear [min]": 50.0,
    }
    signals = evaluate_physical_signals(sensor_data)
    assert signals["pwf_triggered"] is True

    rule = get_maintenance_rule("Power Failure", signals)
    assert "power" in rule["root_cause"].lower()
    assert "motor" in rule["recommendation"].lower()

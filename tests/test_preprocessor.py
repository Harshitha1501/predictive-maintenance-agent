"""Unit tests for the sensor data preprocessor."""

import pandas as pd
import pytest
from src.preprocessing.preprocessor import TelemetryPreprocessor


def test_key_normalization():
    pre = TelemetryPreprocessor()
    raw = {
        "air_temperature": 298.5,
        "process_temp": 308.2,
        "speed": 1500,
        "torque": 42.0,
        "tool_wear": 50,
        "type": "M",
    }
    normalized = pre.normalize_keys(raw)
    assert "Air temperature [K]" in normalized
    assert "Process temperature [K]" in normalized
    assert "Rotational speed [rpm]" in normalized
    assert "Torque [Nm]" in normalized
    assert "Tool wear [min]" in normalized
    assert "Type" in normalized


def test_type_encoding():
    pre = TelemetryPreprocessor()
    assert pre.encode_type("L") == 1
    assert pre.encode_type("M") == 2
    assert pre.encode_type("H") == 0


def test_prepare_input_dataframe():
    pre = TelemetryPreprocessor()
    raw = {
        "type": "L",
        "air_temperature": 298.0,
        "process_temperature": 308.0,
        "rotational_speed": 1400.0,
        "torque": 45.0,
        "tool_wear": 100.0,
    }
    df = pre.prepare_input(raw)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 6)
    assert df["Type"].iloc[0] == 1
    assert df["Torque [Nm]"].iloc[0] == 45.0


def test_out_of_bounds_validation():
    pre = TelemetryPreprocessor()
    raw = {
        "Air temperature [K]": 400.0,  # Physically impossible ambient
        "Process temperature [K]": 300.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 50.0,
    }
    warnings = pre.validate_reading(raw)
    assert len(warnings) > 0
    assert "outside standard operating range" in warnings[0]

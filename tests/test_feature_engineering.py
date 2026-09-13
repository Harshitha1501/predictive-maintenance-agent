import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "03_feature_engineering.py"

spec = importlib.util.spec_from_file_location(
    "feature_engineering",
    MODULE_PATH,
)
fe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fe)


def sample_df():
    return pd.DataFrame(
        {
            "udi": [1, 2, 3],
            "product_id": ["L1", "M1", "H1"],
            "type": ["L", "M", "H"],
            "air_temp_k": [300.0, 301.0, 302.0],
            "process_temp_k": [310.0, 309.0, 311.0],
            "rotational_speed_rpm": [1500.0, 1200.0, 1800.0],
            "torque_nm": [10.0, 8.0, 12.0],
            "tool_wear_min": [100.0, 210.0, 230.0],
            "machine_failure": [0, 1, 1],
            "twf": [0, 1, 0],
            "hdf": [0, 0, 1],
            "pwf": [0, 0, 0],
            "osf": [0, 0, 1],
            "rnf": [0, 0, 0],
            "temp_diff_k": [10.0, 8.0, 9.0],
            "power_w": [1570.8, 1005.3, 2261.9],
        }
    )


def test_row_count_and_existing_features_preserved():
    df = sample_df()
    out = fe.add_features(df)

    assert len(out) == len(df)
    assert np.allclose(out["temp_diff_k"], df["temp_diff_k"])
    assert np.allclose(out["power_w"], df["power_w"])


def test_interactions_are_correct():
    out = fe.add_features(sample_df())

    assert out.loc[0, "torque_speed_product"] == 15000
    assert out.loc[1, "tool_wear_torque"] == 1680
    assert out.loc[2, "torque_squared"] == 144


def test_one_hot_types_are_stable():
    out = fe.add_features(sample_df())

    assert out[["type_L", "type_M", "type_H"]].values.tolist() == [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]


def test_model_matrix_has_no_ids_or_targets():
    engineered = fe.add_features(sample_df())
    x = fe.create_model_matrix(engineered)

    assert list(x.columns) == fe.MODEL_FEATURE_COLUMNS

    forbidden = set(fe.ID_COLUMNS + fe.TARGET_COLUMNS)
    assert not forbidden.intersection(x.columns)


def test_no_nan_or_infinity_in_model_features():
    engineered = fe.add_features(sample_df())
    x = fe.create_model_matrix(engineered)

    assert np.isfinite(x.to_numpy(dtype=float)).all()


def test_rule_indicators_follow_hackathon_rules():
    rules = fe.create_rule_indicators(
        fe.add_features(sample_df())
    )

    assert rules.loc[1, "risk_hdf"] == 1
    assert rules.loc[1, "risk_twf"] == 1
    assert rules.loc[0, "risk_pwf"] == 1
    assert rules.loc[1, "risk_osf"] == 0


def test_missing_required_column_raises_error():
    df = sample_df().drop(columns=["power_w"])

    try:
        fe.add_features(df)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "Missing required columns" in str(exc)

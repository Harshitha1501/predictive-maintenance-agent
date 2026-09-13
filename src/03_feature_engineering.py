"""
Feature Engineering for Predictive Maintenance Agent

Role:
    Feature Engineering Specialist

Input:
    data/train.csv
    data/test.csv

Output:
    data/processed/engineered_train.csv
    data/processed/engineered_test.csv
    data/processed/X_train.csv
    data/processed/X_test.csv
    data/processed/rule_indicators_train.csv
    data/processed/rule_indicators_test.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# PROJECT PATHS
# -------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT_DIR / "data" / "train.csv"
TEST_PATH = ROOT_DIR / "data" / "test.csv"

OUTPUT_DIR = ROOT_DIR / "data" / "processed"


# -------------------------------------------------------------------
# REQUIRED COLUMNS
# -------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "udi",
    "product_id",
    "type",
    "air_temp_k",
    "process_temp_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "machine_failure",
    "twf",
    "hdf",
    "pwf",
    "osf",
    "rnf",
    "temp_diff_k",
    "power_w",
]


TARGET_COLUMNS = [
    "machine_failure",
    "twf",
    "hdf",
    "pwf",
    "osf",
    "rnf",
]


ID_COLUMNS = [
    "udi",
    "product_id",
]


# -------------------------------------------------------------------
# MODEL FEATURE COLUMNS
# -------------------------------------------------------------------

MODEL_FEATURE_COLUMNS = [
    "air_temp_k",
    "process_temp_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "temp_diff_k",
    "power_w",
    "temperature_ratio",
    "temp_diff_ratio",
    "torque_speed_product",
    "tool_wear_torque",
    "tool_wear_ratio",
    "torque_per_rpm",
    "rpm_per_torque",
    "torque_squared",
    "rpm_squared",
    "tool_wear_squared",
    "type_L",
    "type_M",
    "type_H",
]


# -------------------------------------------------------------------
# VALIDATION
# -------------------------------------------------------------------

def validate_input(df: pd.DataFrame) -> None:
    """Validate that the dataset contains the expected columns."""

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


# -------------------------------------------------------------------
# SAFE DIVISION
# -------------------------------------------------------------------

def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series
) -> pd.Series:
    """
    Divide two Series safely.

    Any division by zero or invalid result is replaced with 0.
    """

    result = numerator.div(
        denominator.replace(0, np.nan)
    )

    return result.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0.0)


# -------------------------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to the cleaned dataset.

    Existing features temp_diff_k and power_w are preserved because
    they were already created during data ingestion.
    """

    validate_input(df)

    result = df.copy()

    # ---------------------------------------------------------------
    # Temperature features
    # ---------------------------------------------------------------

    result["temperature_ratio"] = safe_divide(
        result["process_temp_k"],
        result["air_temp_k"]
    )

    result["temp_diff_ratio"] = safe_divide(
        result["temp_diff_k"],
        result["air_temp_k"]
    )

    # ---------------------------------------------------------------
    # Torque + rotational speed interaction
    # ---------------------------------------------------------------

    result["torque_speed_product"] = (
        result["torque_nm"]
        * result["rotational_speed_rpm"]
    )

    # ---------------------------------------------------------------
    # Tool wear + load interaction
    # ---------------------------------------------------------------

    result["tool_wear_torque"] = (
        result["tool_wear_min"]
        * result["torque_nm"]
    )

    # ---------------------------------------------------------------
    # Ratio features
    # ---------------------------------------------------------------

    result["tool_wear_ratio"] = safe_divide(
        result["tool_wear_min"],
        result["torque_nm"]
    )

    result["torque_per_rpm"] = safe_divide(
        result["torque_nm"],
        result["rotational_speed_rpm"]
    )

    result["rpm_per_torque"] = safe_divide(
        result["rotational_speed_rpm"],
        result["torque_nm"]
    )

    # ---------------------------------------------------------------
    # Non-linear features
    # ---------------------------------------------------------------

    result["torque_squared"] = (
        result["torque_nm"] ** 2
    )

    result["rpm_squared"] = (
        result["rotational_speed_rpm"] ** 2
    )

    result["tool_wear_squared"] = (
        result["tool_wear_min"] ** 2
    )

    # ---------------------------------------------------------------
    # One-hot encode machine type
    # ---------------------------------------------------------------

    type_dummies = pd.get_dummies(
        result["type"].astype(str),
        prefix="type",
        dtype=int
    )

    for column in ["type_L", "type_M", "type_H"]:
        if column not in type_dummies.columns:
            type_dummies[column] = 0

    type_dummies = type_dummies[
        ["type_L", "type_M", "type_H"]
    ]

    result = pd.concat(
        [result, type_dummies],
        axis=1
    )

    return result


# -------------------------------------------------------------------
# MODEL MATRIX
# -------------------------------------------------------------------

def create_model_matrix(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create the model-ready feature matrix.

    Failure labels and identifiers are intentionally excluded
    to prevent target leakage.
    """

    missing_features = [
        column
        for column in MODEL_FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing engineered features: {missing_features}"
        )

    X = df[MODEL_FEATURE_COLUMNS].copy()

    # Replace any unexpected infinite values.
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Fill any remaining numerical missing values.
    X = X.fillna(0.0)

    return X


# -------------------------------------------------------------------
# RULE INDICATORS FOR XAI / AGENT
# -------------------------------------------------------------------

def create_rule_indicators(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create transparent engineering-rule indicators.

    These indicators are intended for the XAI / Agent layer.
    They are NOT included in the model feature matrix because
    they are directly derived from the failure-rule definitions.
    """

    rules = pd.DataFrame(index=df.index)

    # Tool Wear Failure rule.
    rules["risk_twf"] = (
        df["tool_wear_min"].between(
            200,
            240,
            inclusive="both"
        )
    ).astype(int)

    # Heat Dissipation Failure rule.
    rules["risk_hdf"] = (
        (df["temp_diff_k"] < 8.6)
        & (df["rotational_speed_rpm"] < 1380)
    ).astype(int)

    # Power Failure rule from the hackathon specification.
    rules["risk_pwf"] = (
        (df["torque_speed_product"] < 3500)
        | (df["torque_speed_product"] > 9000)
    ).astype(int)

    # Overstrain Failure rule.
    thresholds = {
        "L": 11000,
        "M": 12000,
        "H": 13000,
    }

    osf_threshold = (
        df["type"]
        .astype(str)
        .map(thresholds)
    )

    rules["risk_osf"] = (
        df["tool_wear_torque"] > osf_threshold
    ).fillna(False).astype(int)

    # Total number of deterministic rule indicators triggered.
    rules["rule_trigger_count"] = (
        rules[
            [
                "risk_twf",
                "risk_hdf",
                "risk_pwf",
                "risk_osf",
            ]
        ].sum(axis=1)
    )

    return rules


# -------------------------------------------------------------------
# PROCESS ONE DATASET
# -------------------------------------------------------------------

def process_dataset(
    input_path: Path,
    engineered_output_path: Path,
    model_output_path: Path,
    rule_output_path: Path,
) -> None:
    """Process one train/test dataset."""

    df = pd.read_csv(input_path)

    # Add engineered features.
    engineered_df = add_features(df)

    # Create model-ready matrix.
    X = create_model_matrix(engineered_df)

    # Create transparent rule indicators for XAI.
    rules = create_rule_indicators(engineered_df)

    # Save outputs.
    engineered_df.to_csv(
        engineered_output_path,
        index=False
    )

    X.to_csv(
        model_output_path,
        index=False
    )

    rules.to_csv(
        rule_output_path,
        index=False
    )


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main() -> None:
    """Run feature engineering for train and test datasets."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    process_dataset(
        TRAIN_PATH,
        OUTPUT_DIR / "engineered_train.csv",
        OUTPUT_DIR / "X_train.csv",
        OUTPUT_DIR / "rule_indicators_train.csv",
    )

    process_dataset(
        TEST_PATH,
        OUTPUT_DIR / "engineered_test.csv",
        OUTPUT_DIR / "X_test.csv",
        OUTPUT_DIR / "rule_indicators_test.csv",
    )

    print("Feature engineering completed successfully.")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    print("Created files:")
    print("- engineered_train.csv")
    print("- engineered_test.csv")
    print("- X_train.csv")
    print("- X_test.csv")
    print("- rule_indicators_train.csv")
    print("- rule_indicators_test.csv")
    print()
    print("Model feature count:", len(MODEL_FEATURE_COLUMNS))


if __name__ == "__main__":
    main()

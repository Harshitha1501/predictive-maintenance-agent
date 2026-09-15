"""
Data Ingestion Script — Predictive Maintenance Agent (Cognizant Hackathon, AI4I 2020)
Role: Data Ingestion & EDA

What this does:
1. Loads the raw AI4I 2020 CSV
2. Validates schema and data types
3. Cleans / standardizes columns (renames long bracketed names to snake_case)
4. Flags data-quality issues in the target labels (documented, not silently fixed)
5. Produces a stratified train/test split
6. Saves clean_data.csv, train.csv, test.csv for the rest of the team

Run: python 01_data_ingestion.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "ai4i2020.csv"
OUT_DIR = "."

EXPECTED_COLUMNS = [
    "UDI", "Product ID", "Type",
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
    "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF",
]

RENAME_MAP = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}

FAILURE_FLAGS = ["twf", "hdf", "pwf", "osf", "rnf"]


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=RENAME_MAP).copy()

    # Type checks
    df["type"] = df["type"].astype("category")
    for flag in ["machine_failure"] + FAILURE_FLAGS:
        df[flag] = df[flag].astype(int)

    # Duplicates
    n_dupes = df.duplicated().sum()
    if n_dupes:
        df = df.drop_duplicates()

    # Missing values (dataset has none, but keep this defensive)
    n_missing = df.isnull().sum().sum()
    if n_missing:
        df = df.dropna()

    # Derived feature the modeling teammate will likely want:
    # temperature differential is a known signal for HDF (heat dissipation failure)
    df["temp_diff_k"] = df["process_temp_k"] - df["air_temp_k"]

    # power = torque * rotational speed (rad/s) — relevant to PWF (power failure)
    df["power_w"] = df["torque_nm"] * (df["rotational_speed_rpm"] * 2 * 3.141592653589793 / 60)

    return df


def audit_label_quality(df: pd.DataFrame) -> dict:
    """
    Known quirk in this dataset: `machine_failure` is meant to be 1 whenever
    any of the 5 failure-mode flags (twf/hdf/pwf/osf/rnf) is 1, but a handful
    of rows don't follow that rule exactly. Flag it for the team rather than
    silently 'fixing' it — the modeling teammate should decide how to handle it.
    """
    flag_sum = df[FAILURE_FLAGS].sum(axis=1)
    no_flag_but_failure = df[(df["machine_failure"] == 1) & (flag_sum == 0)]
    multi_flag_rows = df[flag_sum > 1]
    return {
        "n_failure_rows": int(df["machine_failure"].sum()),
        "failure_rate_pct": round(df["machine_failure"].mean() * 100, 2),
        "n_failure_no_flag_set": int(len(no_flag_but_failure)),
        "n_rows_multiple_flags": int(len(multi_flag_rows)),
        "failure_no_flag_udis": no_flag_but_failure["udi"].tolist(),
    }


def main():
    df_raw = load_raw(RAW_PATH)
    df_clean = clean(df_raw)

    audit = audit_label_quality(df_clean)
    print("=== Label quality audit ===")
    for k, v in audit.items():
        print(f"{k}: {v}")

    df_clean.to_csv(f"{OUT_DIR}/clean_data.csv", index=False)

    # Stratified split on machine_failure to preserve the ~3.4% failure rate
    # in both train and test sets — important given the class imbalance.
    train_df, test_df = train_test_split(
        df_clean,
        test_size=0.2,
        random_state=42,
        stratify=df_clean["machine_failure"],
    )
    train_df.to_csv(f"{OUT_DIR}/train.csv", index=False)
    test_df.to_csv(f"{OUT_DIR}/test.csv", index=False)

    print(f"\nSaved: clean_data.csv ({len(df_clean)} rows)")
    print(f"Saved: train.csv ({len(train_df)} rows), test.csv ({len(test_df)} rows)")
    print(f"Train failure rate: {train_df['machine_failure'].mean()*100:.2f}%")
    print(f"Test failure rate: {test_df['machine_failure'].mean()*100:.2f}%")


if __name__ == "__main__":
    main()

"""
Data Preprocessing and Telemetry Validation Module for AI4I 2020.

Normalizes sensor input, validates physiological boundaries, encodes categorical
types ('L', 'M', 'H'), and prepares aligned dataframes for model inference.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

# Canonical feature names expected by AI4I models
CANONICAL_FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

ALL_MODEL_FEATURES = CANONICAL_FEATURES + ["Machine failure"]

# Mapping dictionary for standard snake_case and common aliases to canonical names
ALIAS_MAP = {
    "type": "Type",
    "product_type": "Type",
    "product type": "Type",
    "variant": "Type",
    "air_temperature": "Air temperature [K]",
    "air_temp": "Air temperature [K]",
    "air_temp_k": "Air temperature [K]",
    "air temp k": "Air temperature [K]",
    "air temp [k]": "Air temperature [K]",
    "air temp (k)": "Air temperature [K]",
    "air temperature [k]": "Air temperature [K]",
    "air temperature (k)": "Air temperature [K]",
    "air_temperature_k": "Air temperature [K]",
    "airtemp": "Air temperature [K]",
    "air_temperature_[k]": "Air temperature [K]",
    "process_temperature": "Process temperature [K]",
    "process_temp": "Process temperature [K]",
    "process_temp_k": "Process temperature [K]",
    "process temp k": "Process temperature [K]",
    "process temp [k]": "Process temperature [K]",
    "process temp (k)": "Process temperature [K]",
    "process temperature [k]": "Process temperature [K]",
    "process temperature (k)": "Process temperature [K]",
    "process_temperature_k": "Process temperature [K]",
    "processtemp": "Process temperature [K]",
    "process_temperature_[k]": "Process temperature [K]",
    "rotational_speed": "Rotational speed [rpm]",
    "speed": "Rotational speed [rpm]",
    "rotational_speed_rpm": "Rotational speed [rpm]",
    "rotational speed rpm": "Rotational speed [rpm]",
    "rotational speed [rpm]": "Rotational speed [rpm]",
    "rotational speed (rpm)": "Rotational speed [rpm]",
    "rotational_speed_[rpm]": "Rotational speed [rpm]",
    "rotationspeed": "Rotational speed [rpm]",
    "speed_rpm": "Rotational speed [rpm]",
    "rpm": "Rotational speed [rpm]",
    "torque": "Torque [Nm]",
    "torque_nm": "Torque [Nm]",
    "torque nm": "Torque [Nm]",
    "torque [nm]": "Torque [Nm]",
    "torque (nm)": "Torque [Nm]",
    "torque_[nm]": "Torque [Nm]",
    "tool_wear": "Tool wear [min]",
    "tool_wear_min": "Tool wear [min]",
    "tool wear min": "Tool wear [min]",
    "tool wear [min]": "Tool wear [min]",
    "tool wear (min)": "Tool wear [min]",
    "tool_wear_[min]": "Tool wear [min]",
    "toolwear": "Tool wear [min]",
    "machine_failure": "Machine failure",
    "machine failure": "Machine failure",
}

# Physical bounds for validation & anomaly flags
SENSOR_BOUNDS = {
    "Air temperature [K]": (280.0, 320.0),
    "Process temperature [K]": (290.0, 330.0),
    "Rotational speed [rpm]": (500.0, 4000.0),
    "Torque [Nm]": (0.0, 150.0),
    "Tool wear [min]": (0.0, 400.0),
}

# Category encoding: L -> 1, M -> 2, H -> 0 (matches standard sklearn LabelEncoder on ['H', 'L', 'M'])
TYPE_ENCODING_MAP = {"H": 0, "L": 1, "M": 2}
TYPE_DECODING_MAP = {0: "H", 1: "L", 2: "M"}


class TelemetryPreprocessor:
    """Preprocesses raw sensor telemetry for binary and multi-class failure models."""

    def __init__(self, type_encoder: Optional[Any] = None):
        self.type_encoder = type_encoder

    def normalize_keys(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Maps any alternative key aliases to the canonical dataset column names."""
        normalized = {}
        for key, value in data.items():
            cleaned_key = str(key).strip().lower()
            canonical = ALIAS_MAP.get(cleaned_key)
            if not canonical:
                # Fuzzy fallback matching
                if "air" in cleaned_key and ("temp" in cleaned_key or "k" in cleaned_key):
                    canonical = "Air temperature [K]"
                elif ("proc" in cleaned_key or "process" in cleaned_key) and ("temp" in cleaned_key or "k" in cleaned_key):
                    canonical = "Process temperature [K]"
                elif "rotat" in cleaned_key or "rpm" in cleaned_key or "speed" in cleaned_key:
                    canonical = "Rotational speed [rpm]"
                elif "torq" in cleaned_key or "nm" in cleaned_key:
                    canonical = "Torque [Nm]"
                elif "wear" in cleaned_key or "tool" in cleaned_key:
                    canonical = "Tool wear [min]"
                elif cleaned_key in ["type", "product_type", "variant"]:
                    canonical = "Type"
                else:
                    canonical = key
            normalized[canonical] = value
        return normalized

    def validate_reading(self, reading: Dict[str, Any]) -> List[str]:
        """Checks sensor readings against physical operating thresholds, returning warnings."""
        warnings = []
        for feature, (low, high) in SENSOR_BOUNDS.items():
            if feature in reading and reading[feature] is not None:
                val = float(reading[feature])
                if val < low or val > high:
                    warnings.append(
                        f"{feature} value {val} is outside standard operating range [{low}, {high}]."
                    )
        return warnings

    def encode_type(self, type_val: Union[str, int]) -> int:
        """Encodes 'L', 'M', 'H' into model-ready integer."""
        if isinstance(type_val, (int, np.integer)):
            return int(type_val)
        str_val = str(type_val).strip().upper()
        if self.type_encoder is not None:
            try:
                return int(self.type_encoder.transform([str_val])[0])
            except Exception:
                pass
        if str_val in TYPE_ENCODING_MAP:
            return TYPE_ENCODING_MAP[str_val]
        # Default fallback
        return TYPE_ENCODING_MAP["L"]

    def prepare_input(
        self,
        raw_data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
        include_machine_failure: bool = False,
        default_failure_val: int = 0,
    ) -> pd.DataFrame:
        """
        Converts input into a properly typed and ordered pandas DataFrame.

        Parameters
        ----------
        raw_data : dict, list of dicts, or pd.DataFrame
        include_machine_failure : bool
            Whether to include the 'Machine failure' column (needed by Seenu's multiclass RF).
        default_failure_val : int
            Value to populate for 'Machine failure' if not present in input.
        """
        if isinstance(raw_data, dict):
            raw_list = [self.normalize_keys(raw_data)]
            df = pd.DataFrame(raw_list)
        elif isinstance(raw_data, list):
            raw_list = [self.normalize_keys(item) for item in raw_data]
            df = pd.DataFrame(raw_list)
        elif isinstance(raw_data, pd.DataFrame):
            df = raw_data.copy()
            # Normalize dataframe columns using full alias and fuzzy mapping
            renames = {}
            for col in df.columns:
                normalized_dummy = self.normalize_keys({col: None})
                mapped_col = list(normalized_dummy.keys())[0]
                if mapped_col != col:
                    renames[col] = mapped_col
            df.rename(columns=renames, inplace=True)
        else:
            raise TypeError(f"Unsupported data type for sensor input: {type(raw_data)}")

        # Ensure all canonical features exist, filling with nominal defaults if absent
        default_nominal_values = {
            "Type": "L",
            "Air temperature [K]": 298.2,
            "Process temperature [K]": 308.6,
            "Rotational speed [rpm]": 1500.0,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 0.0,
        }
        for col in CANONICAL_FEATURES:
            if col not in df.columns:
                df[col] = default_nominal_values.get(col, 0.0)

        # Encode Type column
        df["Type"] = df["Type"].apply(self.encode_type)

        # Ensure numeric casting
        numeric_cols = [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if include_machine_failure:
            if "Machine failure" not in df.columns:
                df["Machine failure"] = default_failure_val
            else:
                df["Machine failure"] = pd.to_numeric(df["Machine failure"], errors="coerce").fillna(
                    default_failure_val
                ).astype(int)
            return df[ALL_MODEL_FEATURES]

        return df[CANONICAL_FEATURES]

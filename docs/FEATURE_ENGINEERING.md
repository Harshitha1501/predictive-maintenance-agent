# Feature Engineering

## Purpose

This module prepares the predictive maintenance dataset for machine learning models and the XAI/agent layer.

Implementation:

`src/03_feature_engineering.py`

## Input

The pipeline uses:

- `data/train.csv`
- `data/test.csv`

The existing features from the data ingestion stage are preserved:

- `temp_diff_k`
- `power_w`

## Engineered Features

### Temperature Features

- `temperature_ratio = process_temp_k / air_temp_k`
- `temp_diff_ratio = temp_diff_k / air_temp_k`

### Interaction Features

- `torque_speed_product = torque_nm × rotational_speed_rpm`
- `tool_wear_torque = tool_wear_min × torque_nm`

### Ratio Features

- `tool_wear_ratio = tool_wear_min / torque_nm`
- `torque_per_rpm = torque_nm / rotational_speed_rpm`
- `rpm_per_torque = rotational_speed_rpm / torque_nm`

### Nonlinear Features

- `torque_squared`
- `rpm_squared`
- `tool_wear_squared`

### Product Type Encoding

The `type` column is converted into:

- `type_L`
- `type_M`
- `type_H`

## Model Features

The model-ready feature matrix contains 21 predictor features.

It excludes:

- `udi`
- `product_id`
- `machine_failure`
- `twf`
- `hdf`
- `pwf`
- `osf`
- `rnf`

This prevents identifier and target leakage.

## Rule Indicators

Separate rule indicators are generated for the XAI and agent layer.

### TWF

Tool wear between 200 and 240 minutes.

`risk_twf`

### HDF

Temperature difference below 8.6 K and rotational speed below 1380 RPM.

`risk_hdf`

### PWF

Torque × rotational speed is below 3500 or above 9000.

`risk_pwf`

### OSF

Tool wear × torque exceeds the threshold for the product type:

- L → 11000
- M → 12000
- H → 13000

`risk_osf`

A `rule_trigger_count` feature records how many deterministic rules are triggered.

These rule indicators are for XAI/agent reasoning and are not used as ML model predictors.

## Output Files

The pipeline generates:

- `data/processed/engineered_train.csv`
- `data/processed/engineered_test.csv`
- `data/processed/X_train.csv`
- `data/processed/X_test.csv`
- `data/processed/rule_indicators_train.csv`
- `data/processed/rule_indicators_test.csv`

## Handoff

### Role 4 — Binary Classification

Use:

`X_train.csv` and `X_test.csv`

Target:

`machine_failure`

### Role 5 — Multi-Class Classification

Use:

`X_train.csv` and `X_test.csv`

Targets:

`twf`, `hdf`, `pwf`, `osf`, `rnf`

### Role 6 — XAI & Agent Logic

Use:

`rule_indicators_train.csv` and `rule_indicators_test.csv`

These provide transparent rule-based evidence for the agent layer.

## Testing

Unit tests are available in:

`tests/test_feature_engineering.py`

The tests check feature calculations, type encoding, leakage prevention, valid numeric values, rule indicators, and input validation.

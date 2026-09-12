# Data Ingestion & EDA — Handoff Report
**Project:** Predictive Maintenance Agent for Manufacturing (Cognizant Hackathon)
**Dataset:** AI4I 2020 Predictive Maintenance Dataset (Kaggle)
**Prepared by:** Data Ingestion & EDA role

---

## 1. What's in this handoff

| File | Purpose |
|---|---|
| `01_data_ingestion.py` | Loads raw CSV, cleans, validates, engineers 2 features, splits train/test |
| `02_eda.py` | Generates all plots and summary stats below |
| `clean_data.csv` | Full cleaned dataset (10,000 rows), ready to use |
| `train.csv` / `test.csv` | 80/20 stratified split (preserves failure rate in both) |
| `plots/` | 6 PNG charts referenced below |

Re-run order if the raw CSV changes: `01_data_ingestion.py` then `02_eda.py`.

## 2. Dataset overview

- **10,000 rows, 14 raw columns**, no missing values, no duplicate rows.
- Renamed columns to snake_case for easier coding downstream (e.g. `Air temperature [K]` → `air_temp_k`). Full mapping is in `01_data_ingestion.py`.
- Added two engineered features that are physically meaningful for this problem:
  - `temp_diff_k` = process_temp − air_temp (relevant to **HDF**, heat dissipation failure)
  - `power_w` = torque × rotational speed (relevant to **PWF**, power failure)

## 3. ⚠️ Data quality issue the modeling teammate needs to know about

`machine_failure` is *supposed* to be 1 exactly when any of the 5 failure-mode flags (TWF, HDF, PWF, OSF, RNF) is 1 — but the raw data doesn't fully follow that rule:

- **9 rows** have `machine_failure = 1` but **no flag set** (UDIs: 1438, 2750, 4045, 4685, 5537, 5942, 6479, 8507, 9016).
- **24 rows** have **more than one flag** set simultaneously.

I did **not** silently fix this — it's a genuine ambiguity in the source data (this is a known quirk of the AI4I dataset). Whoever builds the classifier should decide: treat `machine_failure` as the single ground-truth label, or reconstruct it from the flags. I'd lean toward keeping `machine_failure` as-is and treating the flags as auxiliary info for the root-cause explanation step, but that's a modeling call, not mine.

## 4. Class balance

![Class balance](plots/01_class_balance.png)

Only **3.39% of rows are failures** (339 of 10,000). This is a significantly imbalanced problem — whoever builds the predictive model will need to account for this (class weighting, SMOTE, focal loss, or a threshold tuned on recall rather than accuracy). I stratified the train/test split so both sets keep this ~3.4% rate.

## 5. Which failure modes are most common

![Failure modes](plots/02_failure_modes.png)

Ranked by frequency: **HDF (115) > OSF (98) > PWF (95) > TWF (46) > RNF (19)**. Heat dissipation failure is the dominant mode — useful context for whoever builds the root-cause/explainability piece.

## 6. Feature distributions: failure vs no-failure

![Feature distributions](plots/03_feature_distributions.png)

Failures cluster at higher torque and higher power, and at lower temperature differentials — consistent with the physical failure modes in the dataset (heat buildup, power overload).

## 7. Correlation with the target

![Correlation heatmap](plots/04_correlation_heatmap.png)

Ranked correlation with `machine_failure`:

| Feature | Correlation |
|---|---|
| torque_nm | 0.191 |
| power_w | 0.176 |
| tool_wear_min | 0.105 |
| air_temp_k | 0.083 |
| process_temp_k | 0.036 |
| rotational_speed_rpm | −0.044 |
| temp_diff_k | −0.112 |

No feature is strongly linearly correlated on its own — expect the real signal to be in **interactions** (e.g. high torque + low rotational speed together), which fits with this being a multi-mode failure problem rather than a single-cause one. Worth flagging to whoever picks the model type — tree-based models (Random Forest / XGBoost) will likely pick up these interactions better than a linear model.

## 8. Outlier check

![Boxplots](plots/05_boxplots_outliers.png)

Using the IQR method: `rotational_speed_rpm` (418 outliers) and `power_w` (60) and `torque_nm` (69) have the most extreme values. These aren't data errors — they're legitimate operating extremes and likely informative for failure prediction, so I did **not** remove them. Flagging for the modeling teammate in case scaling/robust-scaling decisions are needed.

## 9. Failure rate by product quality tier

![Failure by type](plots/06_failure_by_type.png)

| Type | Failure rate |
|---|---|
| L (Low) | 3.92% |
| M (Medium) | 2.77% |
| H (High) | 2.09% |

Lower-quality-tier machines (`Type = L`) fail nearly twice as often as high-quality ones. `type` is a useful feature and also a natural one to highlight in the agent's "recommended maintenance actions" (e.g. prioritize L-tier equipment for inspection).

## 10. Suggested next steps for the team

- **Modeling teammate:** `train.csv`/`test.csv` are ready. Given the 3.4% imbalance, start with class-weighted tree models; track recall/F1 on the failure class, not just accuracy.
- **Root-cause / explainability teammate:** the 5 flag columns (`twf`, `hdf`, `pwf`, `osf`, `rnf`) are preserved in `clean_data.csv` for building the "identifies root causes" part of the agent.
- **Anyone building the agent's recommendation logic:** `type` and `tool_wear_min` are strong candidates for actionable, explainable recommendations (e.g. "L-tier units with >200 min tool wear are highest risk").

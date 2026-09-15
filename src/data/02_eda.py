"""
Exploratory Data Analysis — Predictive Maintenance Agent (Cognizant Hackathon, AI4I 2020)
Role: Data Ingestion & EDA

Reads clean_data.csv (produced by 01_data_ingestion.py) and generates:
  plots/01_class_balance.png
  plots/02_failure_modes.png
  plots/03_feature_distributions.png
  plots/04_correlation_heatmap.png
  plots/05_boxplots_outliers.png
  plots/06_failure_by_type.png

Run: python 02_eda.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

df = pd.read_csv("clean_data.csv")

NUMERIC_FEATURES = [
    "air_temp_k", "process_temp_k", "rotational_speed_rpm",
    "torque_nm", "tool_wear_min", "temp_diff_k", "power_w",
]
FAILURE_FLAGS = ["twf", "hdf", "pwf", "osf", "rnf"]

# 1. Class balance
fig, ax = plt.subplots(figsize=(5, 4))
counts = df["machine_failure"].value_counts().sort_index()
ax.bar(["No Failure", "Failure"], counts.values, color=["#4C72B0", "#C44E52"])
for i, v in enumerate(counts.values):
    ax.text(i, v + 50, f"{v} ({v/len(df)*100:.1f}%)", ha="center")
ax.set_title("Machine Failure Class Balance")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("plots/01_class_balance.png", dpi=150)
plt.close()

# 2. Failure mode breakdown
fig, ax = plt.subplots(figsize=(6, 4))
flag_counts = df[FAILURE_FLAGS].sum().sort_values(ascending=False)
ax.bar(flag_counts.index.str.upper(), flag_counts.values, color="#DD8452")
ax.set_title("Failure Mode Frequency (among all rows)")
ax.set_ylabel("Count")
for i, v in enumerate(flag_counts.values):
    ax.text(i, v + 2, str(v), ha="center")
plt.tight_layout()
plt.savefig("plots/02_failure_modes.png", dpi=150)
plt.close()

# 3. Feature distributions by failure status
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
axes = axes.flatten()
for i, feat in enumerate(NUMERIC_FEATURES):
    sns.kdeplot(data=df, x=feat, hue="machine_failure", ax=axes[i], fill=True, common_norm=False, alpha=0.4)
    axes[i].set_title(feat)
for j in range(len(NUMERIC_FEATURES), len(axes)):
    fig.delaxes(axes[j])
fig.suptitle("Feature Distributions: Failure vs No Failure", y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig("plots/03_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()

# 4. Correlation heatmap
fig, ax = plt.subplots(figsize=(9, 7))
corr = df[NUMERIC_FEATURES + ["machine_failure"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/04_correlation_heatmap.png", dpi=150)
plt.close()

# 5. Boxplots for outlier detection
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
axes = axes.flatten()
for i, feat in enumerate(NUMERIC_FEATURES):
    sns.boxplot(data=df, y=feat, ax=axes[i], color="#55A868")
    axes[i].set_title(feat)
for j in range(len(NUMERIC_FEATURES), len(axes)):
    fig.delaxes(axes[j])
fig.suptitle("Outlier Check (Boxplots)", y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig("plots/05_boxplots_outliers.png", dpi=150, bbox_inches="tight")
plt.close()

# 6. Failure rate by machine Type (L/M/H quality tiers)
fig, ax = plt.subplots(figsize=(6, 4))
failure_by_type = df.groupby("type", observed=True)["machine_failure"].mean() * 100
failure_by_type = failure_by_type.reindex(["L", "M", "H"])
ax.bar(failure_by_type.index, failure_by_type.values, color="#8172B2")
ax.set_title("Failure Rate by Product Quality Tier")
ax.set_ylabel("Failure rate (%)")
for i, v in enumerate(failure_by_type.values):
    ax.text(i, v + 0.05, f"{v:.2f}%", ha="center")
plt.tight_layout()
plt.savefig("plots/06_failure_by_type.png", dpi=150)
plt.close()

# --- Print summary stats used in the handoff report ---
print("=== Outlier bounds (IQR method) ===")
for feat in NUMERIC_FEATURES:
    q1, q3 = df[feat].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_outliers = ((df[feat] < lower) | (df[feat] > upper)).sum()
    print(f"{feat}: bounds=({lower:.1f}, {upper:.1f}), outliers={n_outliers}")

print("\n=== Correlation with machine_failure ===")
print(corr["machine_failure"].sort_values(ascending=False))

print("\n=== Failure rate by Type ===")
print(failure_by_type)

print("\nAll plots saved to plots/")

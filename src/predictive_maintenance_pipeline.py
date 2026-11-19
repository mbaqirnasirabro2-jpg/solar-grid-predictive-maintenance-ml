import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import shap

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

os.makedirs("data", exist_ok=True)
os.makedirs("visuals", exist_ok=True)

print("=================================================================")
print("  SOLAR & INDUSTRIAL INVERTER PREDICTIVE MAINTENANCE PIPELINE    ")
print("=================================================================\n")

# --- 1. REALISTIC SOLAR INVERTER & POWER TELEMETRY GENERATOR ---
def generate_inverter_telemetry(n_records: int = 4000) -> pd.DataFrame:
    np.random.seed(42)
    inverter_ids = [f"INV-{101 + (i % 15)}" for i in range(n_records)]
    ambient_temp_c = np.round(np.random.uniform(22.0, 46.0, n_records), 2)
    irradiance_wm2 = np.round(np.random.uniform(200.0, 1050.0, n_records), 1)
    dc_voltage_v = np.round(np.random.normal(680.0, 28.0, n_records), 2)
    ac_frequency_hz = np.round(np.random.normal(50.0, 0.18, n_records), 3)
    operating_hours = np.random.randint(500, 28000, n_records)
    harmonic_distortion_thd = np.round(np.random.exponential(1.8, n_records) + 1.2, 2)

    # Internal IGBT module temperature rises with ambient temp, load, and age
    igbt_temp_c = np.round(
        ambient_temp_c + 0.025 * irradiance_wm2 + 0.0003 * operating_hours + np.random.normal(0, 3.5, n_records), 2
    )

    # Conversion efficiency drops with extreme heat and harmonic distortion
    efficiency_pct = np.round(
        98.4 - 0.035 * np.maximum(0, igbt_temp_c - 55) - 0.28 * harmonic_distortion_thd + np.random.normal(0, 0.3, n_records), 2
    )

    # Realistic hardware fault risk within next 7 days
    fault_logit = (
        -6.2
        + 0.11 * np.maximum(0, igbt_temp_c - 62)
        + 0.45 * harmonic_distortion_thd
        + 0.00009 * operating_hours
        + 0.85 * (np.abs(ac_frequency_hz - 50.0) > 0.35)
        - 0.40 * (efficiency_pct - 96.0)
    )
    fault_prob = 1 / (1 + np.exp(-fault_logit))
    fault_within_7d = (np.random.rand(n_records) < fault_prob).astype(int)

    df = pd.DataFrame({
        "inverter_id": inverter_ids,
        "ambient_temp_c": ambient_temp_c,
        "igbt_temp_c": igbt_temp_c,
        "irradiance_wm2": irradiance_wm2,
        "dc_voltage_v": dc_voltage_v,
        "ac_frequency_hz": ac_frequency_hz,
        "harmonic_distortion_thd": harmonic_distortion_thd,
        "operating_hours": operating_hours,
        "efficiency_pct": efficiency_pct,
        "fault_within_7d": fault_within_7d
    })
    return df

df = generate_inverter_telemetry(4000)
df.to_csv("data/inverter_telemetry_data.csv", index=False)
print(f"[INFO] Generated telemetry logs: {df.shape[0]} rows | Fault Rate: {df['fault_within_7d'].mean()*100:.2f}%\n")

# --- 2. ELECTRICAL & THERMAL FEATURE ENGINEERING ---
df["thermal_gradient_c"] = df["igbt_temp_c"] - df["ambient_temp_c"]
df["freq_deviation_hz"] = np.abs(df["ac_frequency_hz"] - 50.0)
df["voltage_stress_index"] = df["dc_voltage_v"] / (df["efficiency_pct"] + 1e-5)
df["thermal_aging_load"] = df["igbt_temp_c"] * np.log1p(df["operating_hours"])

# Unsupervised Anomaly Score via Isolation Forest
feature_cols_raw = [
    "ambient_temp_c", "igbt_temp_c", "irradiance_wm2", "dc_voltage_v",
    "ac_frequency_hz", "harmonic_distortion_thd", "operating_hours",
    "efficiency_pct", "thermal_gradient_c", "freq_deviation_hz",
    "voltage_stress_index", "thermal_aging_load"
]
iso = IsolationForest(contamination=0.12, random_state=42)
df["unsupervised_anomaly_score"] = -iso.fit(df[feature_cols_raw]).decision_function(df[feature_cols_raw])

# --- 3. TELEMETRY EDA VISUALIZATION ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(data=df, x="fault_within_7d", y="igbt_temp_c", palette="coolwarm", ax=axes[0])
axes[0].set_title("IGBT Module Temperature vs. 7-Day Fault Risk", fontweight="bold")
axes[0].set_xticklabels(["Healthy (0)", "Fault Risk (1)"])
axes[0].set_ylabel("IGBT Temp (°C)")

sns.scatterplot(
    data=df.sample(600, random_state=42),
    x="harmonic_distortion_thd", y="efficiency_pct", hue="fault_within_7d",
    palette="Set1", alpha=0.75, ax=axes[1]
)
axes[1].set_title("Harmonic Distortion (THD) vs. Inverter Efficiency", fontweight="bold")
axes[1].set_xlabel("Total Harmonic Distortion (%)")
axes[1].set_ylabel("Conversion Efficiency (%)")

plt.tight_layout()
plt.savefig("visuals/telemetry_anomaly_eda.png", dpi=300)
plt.close()

# --- 4. MODEL TRAINING & IMBALANCED EVALUATION (ROC-AUC & PR-AUC) ---
X = df[feature_cols_raw + ["unsupervised_anomaly_score"]]
y = df["fault_within_7d"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

models = {
    "Random Forest (Balanced)": RandomForestClassifier(
        n_estimators=250, max_depth=9, class_weight="balanced", random_state=42
    ),
    "LightGBM (Predictive Maint.)": lgb.LGBMClassifier(
        n_estimators=350, learning_rate=0.04, num_leaves=31,
        class_weight="balanced", random_state=42, verbose=-1
    )
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
best_model = None
best_pr_auc = 0.0

print("[MODEL EVALUATION RESULTS]")
for name, clf in models.items():
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, probs)
    pr = average_precision_score(y_test, probs)
    print(f"  -> {name:<30} | ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f}")

    fpr, tpr, _ = roc_curve(y_test, probs)
    prec, rec, _ = precision_recall_curve(y_test, probs)
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc:.3f})")
    axes[1].plot(rec, prec, label=f"{name} (AP={pr:.3f})")

    if pr > best_pr_auc:
        best_pr_auc = pr
        best_model = clf

axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
axes[0].set_title("ROC-AUC Curve", fontweight="bold")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend(loc="lower right")

axes[1].set_title("Precision-Recall Curve (Fault Detection)", fontweight="bold")
axes[1].set_xlabel("Recall (Faults Caught)")
axes[1].set_ylabel("Precision")
axes[1].legend(loc="upper right")

plt.tight_layout()
plt.savefig("visuals/fault_model_roc_pr.png", dpi=300)
plt.close()

# --- 5. SHAP ROOT-CAUSE EXPLAINABILITY ---
explainer = shap.TreeExplainer(best_model)
shap_vals = explainer(X_test)
if len(shap_vals.shape) == 3:
    shap_vals = shap_vals[:, :, 1]

plt.figure(figsize=(10, 6))
shap.plots.bar(shap_vals, show=False)
plt.title("SHAP Root-Cause Drivers of Inverter Hardware Failure", fontweight="bold")
plt.tight_layout()
plt.savefig("visuals/shap_fault_drivers.png", dpi=300)
plt.close()

# --- 6. EXPORT PRIORITY MAINTENANCE SCHEDULE ---
alerts = df.loc[X_test.index, ["inverter_id", "igbt_temp_c", "harmonic_distortion_thd", "efficiency_pct"]].copy()
alerts["failure_risk_score"] = np.round(best_model.predict_proba(X_test)[:, 1], 4)
critical_alerts = alerts[alerts["failure_risk_score"] >= 0.75].sort_values(by="failure_risk_score", ascending=False)
critical_alerts.to_csv("data/maintenance_alert_schedule.csv", index=False)

print(f"\n[OPERATIONAL IMPACT] Flagged {len(critical_alerts)} high-risk inverter events (>=75% failure risk).")
print("[SUCCESS] All datasets, metrics, and 3 diagnostic charts generated!")

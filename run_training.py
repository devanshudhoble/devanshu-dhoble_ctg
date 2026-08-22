#!/usr/bin/env python3
"""
run_training.py — Execute the full training pipeline from the command line.

This is the script equivalent of workflow.ipynb.  It loads the CTU-CHB data,
extracts features, trains the model, evaluates on a held-out test set, and
saves all artifacts.
"""

import os
import sys
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")       # headless backend — no GUI required
import matplotlib.pyplot as plt
import seaborn as sns
import wfdb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, accuracy_score,
    precision_score, recall_score, f1_score,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
np.random.seed(42)

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import FetalDistressModel, extract_features, get_feature_names

# ========================================================================
# Configuration
# ========================================================================
DATA_DIR = r"C:\Users\devan\Downloads\ctu-chb-intrapartum-cardiotocography-database-1.0.0\ctu-chb-intrapartum-cardiotocography-database-1.0.0"
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# ========================================================================
# 1. Parse headers → outcomes DataFrame
# ========================================================================
def parse_hea_file(filepath):
    outcomes = {"record_id": os.path.basename(filepath).replace(".hea", "")}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("#"):
                continue
            content = line.lstrip("#").strip()
            if content.startswith("-") or content.startswith("!"):
                continue
            parts = content.split()
            if len(parts) < 2:
                continue
            key, val = parts[0], parts[-1]
            if "!NotReadyYet!" in val:
                continue
            try:
                outcomes[key] = float(val) if "." in val else int(val)
            except ValueError:
                outcomes[key] = val
    return outcomes


print("=" * 60)
print("  CTG FETAL DISTRESS DETECTION — TRAINING PIPELINE")
print("=" * 60)
print(f"\nData directory : {DATA_DIR}")
print(f"Artifacts dir  : {ARTIFACTS_DIR}\n")

hea_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.hea")))
data_list = [parse_hea_file(f) for f in hea_files]
df = pd.DataFrame(data_list)
print(f"Loaded {len(df)} clinical records.")

# ========================================================================
# 2. Define labels
# ========================================================================
df_clean = df.dropna(subset=["pH", "Apgar5"], how="all").copy()
ph_flag = df_clean["pH"].fillna(999) < 7.20
apgar_flag = df_clean["Apgar5"].fillna(999) < 7
df_clean["distressed"] = (ph_flag | apgar_flag).astype(int)

counts = df_clean["distressed"].value_counts()
print(f"Usable records: {len(df_clean)}")
print(f"  Normal (0):     {counts.get(0, 0)}")
print(f"  Distressed (1): {counts.get(1, 0)}")

# ========================================================================
# 3. Feature extraction
# ========================================================================
print("\nExtracting features...")
features_list, valid_labels, valid_records, failed = [], [], [], []

for i, (_, row) in enumerate(df_clean.iterrows()):
    rid = row["record_id"]
    try:
        rec = wfdb.rdrecord(os.path.join(DATA_DIR, rid))
        fhr = rec.p_signal[:, 0]
        uc = rec.p_signal[:, 1]
        feats = extract_features(fhr, uc)
        assert len(feats) == 29
        features_list.append(feats)
        valid_labels.append(row["distressed"])
        valid_records.append(rid)
    except Exception as e:
        failed.append((rid, str(e)))
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(df_clean)} done")

X = np.array(features_list)
y = np.array(valid_labels)
print(f"Features extracted: {X.shape[0]} records, {X.shape[1]} features each.")
if failed:
    print(f"Failed records: {len(failed)}")

# ========================================================================
# 4. Train / test split
# ========================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\nTrain: {X_train.shape[0]}  (class 0: {(y_train==0).sum()}, class 1: {(y_train==1).sum()})")
print(f"Test:  {X_test.shape[0]}  (class 0: {(y_test==0).sum()}, class 1: {(y_test==1).sum()})")

# Save processed data
np.savez(os.path.join(ARTIFACTS_DIR, "train_data.npz"), X=X_train, y=y_train)
np.savez(os.path.join(ARTIFACTS_DIR, "test_data.npz"), X=X_test, y=y_test)
np.savez(os.path.join(ARTIFACTS_DIR, "sample_inference.npz"), X=X_test[:5], y=y_test[:5])
print("Processed datasets saved to artifacts/.")

# ========================================================================
# 5. Train model
# ========================================================================
print("\nTraining RandomForest model...")
mdl = FetalDistressModel(
    n_estimators=300, max_depth=12, class_weight="balanced", random_state=42
)
mdl.fit(X_train, y_train)
mdl.save(ARTIFACTS_DIR)
print("Model saved.")

# ========================================================================
# 6. Evaluate
# ========================================================================
y_pred = mdl.predict(X_test)
y_proba = mdl.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
spec = tn / (tn + fp) if (tn + fp) > 0 else 0

print("\n" + "=" * 40)
print("    EVALUATION RESULTS")
print("=" * 40)
print(f"  Accuracy:    {acc:.4f}")
print(f"  ROC-AUC:     {auc:.4f}")
print(f"  Precision:   {prec:.4f}")
print(f"  Recall:      {rec:.4f}")
print(f"  F1 Score:    {f1:.4f}")
print(f"  Specificity: {spec:.4f}")
print("=" * 40)
print(f"\n  TP={tp}  FP={fp}  FN={fn}  TN={tn}\n")
print(classification_report(y_test, y_pred, target_names=["Normal", "Distressed"], zero_division=0))

# ========================================================================
# 7. Generate plots
# ========================================================================
feature_names = get_feature_names()

# Confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Distressed"],
            yticklabels=["Normal", "Distressed"], ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(ARTIFACTS_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(ARTIFACTS_DIR, "roc_curve.png"), dpi=150)
plt.close()

# Precision-Recall curve
prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_proba)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(rec_vals, prec_vals, color="purple", lw=2)
ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall Curve")
plt.tight_layout()
plt.savefig(os.path.join(ARTIFACTS_DIR, "pr_curve.png"), dpi=150)
plt.close()

# Feature importances
importances = mdl.feature_importances
idx = np.argsort(importances)[-15:]
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(range(len(idx)), importances[idx], color="steelblue")
ax.set_yticks(range(len(idx)))
ax.set_yticklabels([feature_names[i] for i in idx])
ax.set_xlabel("Importance"); ax.set_title("Top 15 Feature Importances")
plt.tight_layout()
plt.savefig(os.path.join(ARTIFACTS_DIR, "feature_importances.png"), dpi=150)
plt.close()

# Clinical distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(df["pH"].dropna(), kde=True, bins=25, ax=axes[0], color="steelblue")
axes[0].axvline(7.20, color="red", ls="--", lw=2); axes[0].set_title("Umbilical Artery pH")
sns.histplot(df["Apgar1"].dropna(), kde=False, bins=10, ax=axes[1], color="teal")
axes[1].set_title("Apgar at 1 min")
sns.histplot(df["Apgar5"].dropna(), kde=False, bins=10, ax=axes[2], color="coral")
axes[2].axvline(7, color="red", ls="--", lw=2); axes[2].set_title("Apgar at 5 min")
plt.tight_layout()
plt.savefig(os.path.join(ARTIFACTS_DIR, "clinical_distributions.png"), dpi=150)
plt.close()

print("\nAll plots saved to artifacts/.")
print("Training pipeline complete!")

# CTG Fetal Distress Detection -- Prototype

> **Author:** Devanshu Dhoble  
> **Assignment:** Janitri -- Intrapartum CTG Analysis  
> **Dataset:** [CTU-CHB Intrapartum Cardiotocography Database (PhysioNet)](https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/)  
> **Environment:** scikit-learn==1.6.1, scipy==1.15.2, numpy==2.2.1 (pinned for reproducibility)

---

## Overview

A working prototype for detecting **fetal distress / hypoxia** during labour using
cardiotocography (CTG) recordings. The system extracts 30 hand-crafted clinical
features from the fetal heart rate (FHR) and uterine contraction (UC) signals and
classifies each recording as *distressed* or *not distressed* using a **Random Forest**
classifier with balanced class weights.

### Label Definition

A recording is labelled **distressed (1)** when either:
- Umbilical-cord **pH < 7.20**, *or*
- **5-minute Apgar score < 7**.

Otherwise the label is **not distressed (0)**.

**Note:** The pH arm dominates (163 of 182 positive labels). The 7.20 threshold
represents mild acidemia (not severe acidosis at 7.05-7.10) and was chosen to
produce a workable 33% positive class on 552 records. BDecf was available as
an alternative label source but was not used. See report.pdf Section 1.

---

## Repository Structure

```
devanshu-dhoble-ctg/
├── README.md               # This file
├── workflow.ipynb          # End-to-end runnable notebook
├── model.py                # Importable model module (30 features + RF wrapper)
├── generate_outcomes.py    # Standalone inference script
├── artifacts/              # Saved model weights, scaler, and processed datasets
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── train_data.npz
│   ├── test_data.npz
│   └── sample_inference.npz
├── report.pdf              # 3-page technical write-up (4 sections)
└── requirements.txt        # Pinned Python dependencies
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/devanshudhoble/devanshu-dhoble_ctg.git
cd devanshu-dhoble_ctg
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
# Windows PowerShell
.\venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies (pinned versions)

```bash
pip install -r requirements.txt
```

### 4. Dataset

Download the CTU-CHB database from [PhysioNet](https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/)
and extract it. Either:
- Set the `CTG_DATA_DIR` environment variable to the extracted folder, or
- Edit `DATA_DIR` in `workflow.ipynb` directly.

---

## Running the Workflow

```bash
jupyter notebook workflow.ipynb
```

The notebook will:
1. Load and parse all 552 recordings and their clinical outcomes.
2. Define the distress label and decompose it by source (pH vs Apgar).
3. Extract 30 hand-crafted features per recording from the last 30 minutes.
4. **Compare against baselines** (DummyClassifier, single-feature LogReg) via cross-validation.
5. Split into 80/20 train/test sets (stratified).
6. Train a `RandomForestClassifier` (300 trees, depth 4, balanced class weights).
7. **Select the decision threshold** on training folds for ~80% recall.
8. Evaluate on the held-out test set.
9. Save the trained model and all artifacts.

---

## Inference (`generate_outcomes.py`)

### Input Specification

**Mode 1 -- Raw WFDB record:**
```bash
python generate_outcomes.py --record path/to/1001
```
Input: path to a `.dat`/`.hea` record pair (base name, no extension).

**Mode 2 -- Pre-extracted features:**
```bash
python generate_outcomes.py --features artifacts/sample_inference.npz
```
Input: `.npz` file with key `X` of shape `(n_samples, 30)`.

### Output Specification

| Output Field | Type | Description |
|---|---|---|
| Distress Probability | float (0-1) | Model's estimated likelihood of fetal distress |
| Predicted Label | int (0 or 1) | 0 = Normal, 1 = Distressed |

**Decision threshold:** 0.28 (selected on training folds for ~80% recall).
Override with `--threshold`.

### Quick Demo

```bash
python generate_outcomes.py --features artifacts/sample_inference.npz
```

---

## Evaluation Results

### Cross-Validated Performance (authoritative)

| Metric | Value |
|---|---|
| **CV ROC-AUC (5-fold x 6 repeats)** | **0.74 +/- 0.05** |
| DummyClassifier baseline | 0.50 +/- 0.04 |
| LogReg on fhr_iqr alone | 0.73 +/- 0.05 |

### Test Set Performance (single 80/20 split, threshold = 0.28)

| Metric | Value |
|---|---|
| Test ROC-AUC | ~0.73 |
| Recall | ~0.80 |
| Specificity | ~0.67 |

*Note: exact test-set metrics depend on the random split and are reported to
two decimal places to reflect the uncertainty inherent in 111 test samples.*

### Performance Ceiling Finding

Summary statistics over a 30-minute window saturate near 0.73-0.75 ROC-AUC on
this dataset. The full 30-feature model does not dramatically outperform a single
feature (`fhr_iqr`). Beating this ceiling likely requires temporal models on the
raw signal (1D-CNN, LSTM), not more feature engineering.

---

## Google Drive -- Shared Data

**Link:** [https://drive.google.com/drive/folders/1RQgg69oA3lRZnJsTvtDJX_RQTleik1tt?usp=drive_link](https://drive.google.com/drive/folders/1RQgg69oA3lRZnJsTvtDJX_RQTleik1tt?usp=drive_link)

The shared folder contains:
- `train_data.npz` -- Preprocessed training set (features + labels)
- `test_data.npz` -- Preprocessed test set (features + labels)
- `sample_inference.npz` -- 5-sample example input for `generate_outcomes.py`
- `model.pkl` -- Trained Random Forest model weights

*View access granted to:*
- `shrut.dalwadi@janitri.in`
- `ganesh.kavi@janitri.in`

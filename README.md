# CTG Fetal Distress Detection — Prototype

> **Author:** Devanshu Dhoble  
> **Assignment:** Janitri — Intrapartum CTG Analysis  
> **Dataset:** [CTU-CHB Intrapartum Cardiotocography Database (PhysioNet)](https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/)

---

## Overview

This repository contains a working prototype for detecting **fetal distress / hypoxia** during labour using cardiotocography (CTG) recordings. The system extracts hand-crafted clinical features from the fetal heart rate (FHR) and uterine contraction (UC) signals and classifies each recording as *distressed* or *not distressed* using a **Random Forest** classifier with balanced class weights.

### Label Definition

A recording is labelled **distressed (1)** when either:
- Umbilical-cord **pH < 7.20**, *or*
- **5-minute Apgar score < 7**.

Otherwise the label is **not distressed (0)**.

---

## Repository Structure

```
devanshu-dhoble-ctg/
├── README.md               # Setup, instructions, and documentation
├── workflow.ipynb          # End-to-end runnable notebook (data -> train -> evaluate)
├── model.py                # Importable model module (feature extraction + RF wrapper)
├── generate_outcomes.py    # Standalone inference script
├── artifacts/              # Saved model weights, scaler, and processed datasets
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── train_data.npz
│   ├── test_data.npz
│   └── sample_inference.npz
├── report.pdf              # 2-3 page technical write-up
└── requirements.txt        # Python dependencies
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/devanshudhoble/CTG_REPO.git
cd CTG_REPO
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
# Windows PowerShell
.\venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Dataset

Download the CTU-CHB database from [PhysioNet](https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/) and extract it. Set `DATA_DIR` in `workflow.ipynb` or define the `CTG_DATA_DIR` environment variable to point to the extracted folder containing the `.dat` and `.hea` files.

---

## Running the Workflow

Open the notebook and run all cells:

```bash
jupyter notebook workflow.ipynb
```

The notebook will:
1. Load and parse all 552 recordings and their clinical outcomes.
2. Define the distress label based on pH < 7.20 and 5-min Apgar < 7.
3. Extract 29 hand-crafted time-domain features per recording from the last 30 minutes.
4. Split into 80 / 20 train / test sets (stratified).
5. Train a `RandomForestClassifier` (300 trees, depth 12, balanced class weights).
6. Evaluate on the held-out test set (ROC-AUC: 0.733, Specificity: 0.878, Accuracy: 0.703).
7. Save the trained model and processed data to `artifacts/`.

---

## Inference (`generate_outcomes.py`)

### Input Specification

The inference script accepts two input modes:

**Mode 1 — Raw WFDB record:**
```bash
python generate_outcomes.py --record path/to/1001
```
- **Input**: Path to a `.dat`/`.hea` record pair without file extension (e.g. `data/1001`).
- **Shape**: Continuous 4 Hz time-series signal arrays for FHR (bpm) and UC (amplitude).

**Mode 2 — Pre-extracted features:**
```bash
python generate_outcomes.py --features artifacts/sample_inference.npz
```
- **Input**: Path to a `.npz` file containing a key `X`.
- **Shape**: 2D NumPy array of shape `(n_samples, 29)` representing the 29 engineered clinical features.

### Output Specification

For each recording, the script produces:

| Output Field | Type | Description |
|---|---|---|
| **Distress Probability** | `float` (0.00 – 1.00) | Model's estimated likelihood of fetal distress / hypoxia |
| **Predicted Label** | `int` (`0` or `1`) | `0` = Not Distressed (Normal), `1` = Distressed |

### Quick Demo Run

```bash
python generate_outcomes.py --features artifacts/sample_inference.npz
```

---

## Evaluation Results (Held-Out Test Set: 111 Recordings)

| Metric | Score | Clinical Interpretation |
|---|---|---|
| **ROC-AUC** | **0.7330** | Threshold-independent discrimination ability |
| **Accuracy** | **0.7027** | Overall correct classification rate |
| **Specificity** | **0.8784** | High rate of correctly identifying healthy fetuses (minimizes false alarms) |
| **Precision** | **0.5909** | 59% of positive alerts are true distressed cases |
| **Recall (Sensitivity)**| **0.3514** | Primary area for improvement via threshold tuning |
| **F1-Score** | **0.4407** | Harmonic mean of precision and recall |

---

## Google Drive — Shared Data

**Link:** [https://drive.google.com/drive/folders/1RQgg69oA3lRZnJsTvtDJX_RQTleik1tt?usp=drive_link](https://drive.google.com/drive/folders/1RQgg69oA3lRZnJsTvtDJX_RQTleik1tt?usp=drive_link)

The shared Google Drive folder contains:
- `train_data.npz` — Preprocessed training set ready for modeling.
- `test_data.npz` — Preprocessed test set ready for evaluation.
- `sample_inference.npz` — 5-sample example input for `generate_outcomes.py`.
- `model.pkl` — Trained Random Forest model weights.
- `scaler.pkl` — Fitted StandardScaler normalization statistics.

*View access has been granted to:*
- `shrut.dalwadi@janitri.in`
- `ganesh.kavi@janitri.in`

---

## Report

See [`report.pdf`](report.pdf) for the comprehensive 3-page write-up covering:
1. **How I framed it** — clinical problem formulation and label definition rationale.
2. **What I built and how I checked it** — data cleaning, feature engineering, model architecture, and metric analysis.
3. **Clinical utility and inference** — bedside deployment scenario, alarm policies, and workflow design.
4. **Limits and next steps** — honest appraisal of failure modes, low recall risks, and concrete engineering roadmap.

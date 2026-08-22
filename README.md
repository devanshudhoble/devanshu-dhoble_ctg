# CTG Fetal Distress Detection — Prototype

> **Author:** Devanshu Dhoble  
> **Assignment:** Janitri — Intrapartum CTG Analysis  
> **Dataset:** [CTU-CHB Intrapartum Cardiotocography Database (PhysioNet)](https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/)

---

## Overview

This repository contains a working prototype for detecting **fetal distress / hypoxia** during labour using cardiotocography (CTG) recordings.  The system extracts hand-crafted features from the fetal heart rate (FHR) and uterine contraction (UC) signals and classifies each recording as *distressed* or *not distressed* using a **Random Forest** classifier.

### Label Definition

A recording is labelled **distressed (1)** when either:
- Umbilical-cord **pH < 7.20**, *or*
- **5-minute Apgar score < 7**.

Otherwise the label is **not distressed (0)**.

---

## Repository Structure

```
CTG_REPO/
├── README.md                ← you are here
├── workflow.ipynb            ← end-to-end notebook (data → train → evaluate)
├── model.py                 ← importable model module (feature extraction + RF wrapper)
├── generate_outcomes.py     ← standalone inference script
├── requirements.txt         ← Python dependencies
├── artifacts/               ← saved model weights, scaler, processed data
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── train_data.npz
│   ├── test_data.npz
│   └── sample_inference.npz
└── report.pdf               ← 2–3 page write-up
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/devanshudhoble/CTG_REPO.git
cd CTG_REPO
```

### 2. Create a virtual environment (recommended)

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

### 4. Download the dataset

Download the CTU-CHB database from [PhysioNet](https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/) and extract it.  Update the `DATA_DIR` variable in `workflow.ipynb` to point to the folder containing the `.dat` and `.hea` files.

---

## Running the Workflow

Open the notebook and run all cells:

```bash
jupyter notebook workflow.ipynb
```

The notebook will:
1. Load and parse all 552 recordings and their outcomes.
2. Define the distress label.
3. Extract 29 hand-crafted features per recording.
4. Split into 80 / 20 train / test sets (stratified).
5. Train a `RandomForestClassifier` (300 trees, depth 12, balanced class weights).
6. Evaluate on the held-out test set (ROC-AUC, Recall, Precision, F1, confusion matrix).
7. Save the trained model and processed data to `artifacts/`.

---

## Inference (generate_outcomes.py)

### Input

The script accepts two input modes:

**Mode 1 — Raw WFDB record:**
```bash
python generate_outcomes.py --record path/to/1001
```
Pass the path to a `.dat`/`.hea` record pair (base name, no extension).

**Mode 2 — Pre-extracted features:**
```bash
python generate_outcomes.py --features artifacts/sample_inference.npz
```
Pass a `.npz` file containing a key `X` of shape `(n_samples, 29)`.

### Output

For each input recording the script prints:

| Field              | Type  | Meaning                                             |
|--------------------|-------|-----------------------------------------------------|
| Distress Probability | float (0–1) | Model's estimated likelihood of fetal distress |
| Predicted Label    | int (0 or 1) | 0 = not distressed, 1 = distressed               |

### Quick demo

```bash
python generate_outcomes.py --features artifacts/sample_inference.npz
```

---

## Model Details

| Component           | Choice                                      |
|---------------------|---------------------------------------------|
| **Algorithm**       | Random Forest (scikit-learn)                 |
| **Features**        | 29 hand-crafted time-domain features         |
| **Input window**    | Last 30 minutes of the CTG recording         |
| **Class balancing**  | `class_weight='balanced'` in RF              |
| **Primary metric**  | ROC-AUC                                      |
| **Secondary metric**| Recall (sensitivity)                         |

### Feature Groups

| Group             | Features                                                              |
|-------------------|-----------------------------------------------------------------------|
| FHR statistics    | Mean, std, median, min, max, range, IQR, skewness, kurtosis          |
| HRV metrics       | RMSSD, SDNN, mean/median absolute successive differences              |
| Variability       | Short-term variability (STV), long-term variability (LTV)            |
| Baseline          | Estimated FHR baseline                                                |
| Events            | Count of decelerations, count of accelerations                       |
| Signal quality     | FHR missing ratio (artefact fraction)                                |
| UC statistics     | Mean, std, max of uterine contraction signal                         |
| Contraction timing | Number of contractions, frequency, mean/std interval                 |
| FHR–UC coupling   | Pearson correlation, FHR during vs. between contractions             |

---

## Google Drive — Shared Data

**Link:** *(Add your Google Drive link here after uploading)*

The folder contains:
- `train_data.npz` — processed training set (features + labels)
- `test_data.npz` — processed test set (features + labels)
- `sample_inference.npz` — 5-sample example input for `generate_outcomes.py`
- `model.pkl` / `scaler.pkl` — trained model weights (if >25 MB)

Access granted to:
- shrut.dalwadi@janitri.in
- ganesh.kavi@janitri.in

---

## Report

See [`report.pdf`](report.pdf) for the full write-up covering:
1. **How I framed it** — problem formulation and label definition
2. **What I built and how I checked it** — data processing, model, evaluation metrics
3. **Clinical utility and inference** — deployment scenario, who uses it, when
4. **Limits and next steps** — honest assessment of weaknesses and future work

---

## License

This project uses the CTU-CHB database, which is available under the [PhysioNet Restricted Health Data License 1.5.0](https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/).

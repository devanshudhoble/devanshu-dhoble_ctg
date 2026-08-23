#!/usr/bin/env python3
"""
generate_outcomes.py -- Standalone inference script for fetal distress prediction
================================================================================

This script loads a trained FetalDistressModel from the ``artifacts/`` folder
and produces predictions for one or more CTG recordings.

INPUT
-----
One of two modes:

  1. **WFDB record** (raw .dat/.hea files):
       python generate_outcomes.py --record path/to/1001
     where ``path/to/1001`` is the base name (without extension) of a WFDB
     record.  The script will read FHR and UC from the .dat file using the
     companion .hea header, extract features, and predict.

  2. **Pre-processed .npz feature file**:
       python generate_outcomes.py --features path/to/features.npz
     The .npz file must contain a key ``X`` of shape ``(n_samples, 30)``
     (the 30-dimensional feature vector produced by ``model.extract_features``).

OUTPUT
------
For each input recording the script prints:

    Record ID | Distress Probability | Predicted Label
    --------- | -------------------- | ---------------
    1001      | 0.73                 | 1 (distressed)

- **Distress Probability** (float, 0-1): the model's estimated likelihood
  that the fetus is experiencing distress / hypoxia.
- **Predicted Label** (int, 0 or 1):
    - ``0`` = not distressed (pH >= 7.20 and Apgar5 >= 7 expected)
    - ``1`` = distressed     (pH < 7.20 or Apgar5 < 7 expected)

Decision threshold: 0.28 (selected on training folds for ~80% recall).
Can be overridden via --threshold.

ARTIFACTS REQUIRED
------------------
  artifacts/model.pkl   -- trained RandomForestClassifier
  artifacts/scaler.pkl  -- fitted StandardScaler
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import FetalDistressModel, extract_features, DEFAULT_THRESHOLD


def load_wfdb_record(record_path: str):
    """
    Load a single WFDB record and return (fhr, uc, record_name).

    Parameters
    ----------
    record_path : str
        Path to the record WITHOUT extension, e.g. ``data/1001``.
    """
    import wfdb

    rec = wfdb.rdrecord(record_path)
    fhr = rec.p_signal[:, 0]
    uc = rec.p_signal[:, 1]
    rec_name = os.path.basename(record_path)
    return fhr, uc, rec_name


def main():
    parser = argparse.ArgumentParser(
        description="Generate fetal distress predictions from a trained model."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--record",
        type=str,
        nargs="+",
        help="Path(s) to WFDB record(s) (base name, no extension).",
    )
    group.add_argument(
        "--features",
        type=str,
        help="Path to a .npz file containing pre-extracted features (key 'X').",
    )
    parser.add_argument(
        "--artifacts",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts"),
        help="Path to the artifacts directory (default: ./artifacts).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Probability threshold for the binary decision (default: {DEFAULT_THRESHOLD}).",
    )
    args = parser.parse_args()

    # ---- Load model ----
    print(f"Loading model from: {args.artifacts}")
    mdl = FetalDistressModel.load(args.artifacts)
    print(f"Model loaded successfully.  Decision threshold: {args.threshold}\n")

    # ---- Predict ----
    if args.features:
        data = np.load(args.features)
        X = data["X"]
        probs = mdl.predict_proba(X)
        labels = (probs >= args.threshold).astype(int)
        print(f"{'Index':<8} | {'Distress Prob':>14} | {'Predicted Label':>16}")
        print("-" * 46)
        for i, (p, l) in enumerate(zip(probs, labels)):
            tag = "distressed" if l == 1 else "not distressed"
            print(f"{i:<8} | {p:>14.4f} | {l:>2}  ({tag})")
    else:
        print(f"{'Record':<12} | {'Distress Prob':>14} | {'Predicted Label':>16}")
        print("-" * 50)
        for rec_path in args.record:
            fhr, uc, rec_name = load_wfdb_record(rec_path)
            feat = extract_features(fhr, uc).reshape(1, -1)
            prob = mdl.predict_proba(feat)[0]
            label = int(prob >= args.threshold)
            tag = "distressed" if label == 1 else "not distressed"
            print(f"{rec_name:<12} | {prob:>14.4f} | {label:>2}  ({tag})")

    print("\nDone.")


if __name__ == "__main__":
    main()

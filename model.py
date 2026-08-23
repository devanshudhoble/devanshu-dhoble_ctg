"""
model.py -- Fetal Distress Detection Model
==========================================

This module defines a reusable, importable model for predicting fetal distress
from cardiotocography (CTG) recordings.  It wraps a scikit-learn
RandomForestClassifier with standardised preprocessing (StandardScaler).

Usage
-----
    from model import FetalDistressModel

    # Training
    mdl = FetalDistressModel()
    mdl.fit(X_train, y_train)
    mdl.save("artifacts/")

    # Inference
    mdl = FetalDistressModel.load("artifacts/")
    probs = mdl.predict_proba(X_new)       # float array, shape (n,)
    labels = mdl.predict(X_new)            # int array,   shape (n,)

Feature vector
--------------
The model expects a 2-D NumPy array of shape ``(n_samples, n_features)`` where
each row contains the hand-crafted features extracted from one CTG recording.
See ``extract_features()`` in this module for the canonical feature list.

Label definition
----------------
A recording is labelled **distressed (1)** when:
    umbilical-cord pH < 7.20  OR  5-minute Apgar < 7
Otherwise the label is **not distressed (0)**.

Note on label composition (552-record CTU-CHB dataset):
    163 records flagged by pH < 7.20 only
     14 records flagged by both criteria
      5 records flagged by Apgar5 < 7 only
    ---
    182 total distressed  (33.0%)
    370 total normal      (67.0%)

The pH arm dominates.  The lenient 7.20 threshold (vs. the stricter 7.05-7.10
for severe acidosis) was chosen to produce a workable positive class on a
modest dataset.  See report.pdf Section 1 for the full rationale.
"""

from __future__ import annotations

import os
import warnings
from typing import Optional

import joblib
import numpy as np
from scipy import signal as scipy_signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLING_RATE = 4             # Hz  (CTU-CHB dataset)
PH_THRESHOLD = 7.20          # umbilical-cord pH below this -> distressed
APGAR5_THRESHOLD = 7         # 5-min Apgar below this -> distressed
FHR_VALID_RANGE = (50, 250)  # bpm -- values outside are artefact
WINDOW_SECONDS = 30 * 60     # last 30 minutes of the recording

# Default decision threshold, selected on training folds to target ~80% recall.
# At this threshold: Recall ~ 0.80, Precision ~ 0.48 on test set.
DEFAULT_THRESHOLD = 0.28

# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _clean_fhr(fhr: np.ndarray) -> np.ndarray:
    """Replace out-of-range and zero FHR values with NaN."""
    fhr = fhr.astype(float).copy()
    fhr[(fhr <= FHR_VALID_RANGE[0]) | (fhr >= FHR_VALID_RANGE[1])] = np.nan
    fhr[fhr == 0] = np.nan
    return fhr


def _clean_uc(uc: np.ndarray) -> np.ndarray:
    """Replace negative UC values with NaN."""
    uc = uc.astype(float).copy()
    uc[uc < 0] = np.nan
    return uc


def _successive_diffs_gap_aware(arr: np.ndarray,
                                max_gap_samples: int = 4) -> np.ndarray:
    """
    Return absolute successive differences, masking gaps.

    Unlike a naive approach that compacts valid values first (which would
    treat a 10-minute dropout as a single "successive difference"), this
    function differences in-place and only keeps differences where the
    two samples are separated by at most ``max_gap_samples`` NaN positions.

    Parameters
    ----------
    arr : np.ndarray
        Signal array (may contain NaN).
    max_gap_samples : int
        Maximum number of consecutive NaN samples allowed between two valid
        values for their difference to count as a genuine beat-to-beat change.
        Default 4 (= 1 second at 4 Hz).
    """
    n = len(arr)
    diffs = []
    last_valid_idx = -1
    last_valid_val = np.nan
    for i in range(n):
        if not np.isnan(arr[i]):
            if last_valid_idx >= 0:
                gap = i - last_valid_idx - 1  # number of NaN samples between
                if gap <= max_gap_samples:
                    diffs.append(abs(arr[i] - last_valid_val))
            last_valid_idx = i
            last_valid_val = arr[i]
    if len(diffs) == 0:
        return np.array([0.0])
    return np.array(diffs)


def _count_decelerations(fhr: np.ndarray, baseline: float,
                         threshold: float = 15.0,
                         min_duration_s: float = 15.0) -> int:
    """
    Count the number of FHR decelerations.

    A deceleration is defined as a drop of >= ``threshold`` bpm below the
    baseline that lasts at least ``min_duration_s`` seconds.
    """
    if np.isnan(baseline):
        return 0
    below = fhr < (baseline - threshold)
    # Forward-fill NaN positions as False
    below = np.where(np.isnan(fhr), False, below)
    min_samples = int(min_duration_s * SAMPLING_RATE)
    count = 0
    run = 0
    for b in below:
        if b:
            run += 1
        else:
            if run >= min_samples:
                count += 1
            run = 0
    if run >= min_samples:
        count += 1
    return count


def _count_accelerations(fhr: np.ndarray, baseline: float,
                         threshold: float = 15.0,
                         min_duration_s: float = 15.0) -> int:
    """
    Count the number of FHR accelerations.

    An acceleration is defined as a rise of >= ``threshold`` bpm above the
    baseline that lasts at least ``min_duration_s`` seconds.
    """
    if np.isnan(baseline):
        return 0
    above = fhr > (baseline + threshold)
    above = np.where(np.isnan(fhr), False, above)
    min_samples = int(min_duration_s * SAMPLING_RATE)
    count = 0
    run = 0
    for b in above:
        if b:
            run += 1
        else:
            if run >= min_samples:
                count += 1
            run = 0
    if run >= min_samples:
        count += 1
    return count


def _estimate_baseline(fhr: np.ndarray, window_min: float = 10.0) -> float:
    """
    Estimate the FHR baseline using a rolling median over a wide window.

    Computes a rolling median with a window of ``window_min`` minutes, then
    returns the median of that rolling series.  This filters out transient
    accelerations and decelerations to approximate the stable resting
    heart rate.

    Falls back to the global median if the signal is shorter than one window.
    """
    valid = fhr[~np.isnan(fhr)]
    if len(valid) < 2:
        return np.nan
    win = int(window_min * 60 * SAMPLING_RATE)
    if win > len(valid):
        return float(np.median(valid))
    # Compute rolling median over the valid (compacted) signal
    n = len(valid)
    half = win // 2
    rolling_medians = []
    for i in range(half, n - half, win // 4):  # step by quarter-window
        lo = max(0, i - half)
        hi = min(n, i + half)
        rolling_medians.append(np.median(valid[lo:hi]))
    if len(rolling_medians) == 0:
        return float(np.median(valid))
    return float(np.median(rolling_medians))


def _detect_contractions(uc: np.ndarray, min_prominence: float = 5.0,
                         min_distance_s: float = 30.0) -> np.ndarray:
    """
    Detect uterine contraction peaks from the UC signal.

    Returns an array of peak indices.
    """
    uc_clean = uc.copy()
    uc_clean[np.isnan(uc_clean)] = 0.0
    min_distance = int(min_distance_s * SAMPLING_RATE)
    try:
        peaks, _ = scipy_signal.find_peaks(
            uc_clean, prominence=min_prominence, distance=min_distance
        )
    except Exception:
        peaks = np.array([], dtype=int)
    return peaks


def extract_features(fhr_raw: np.ndarray, uc_raw: np.ndarray) -> np.ndarray:
    """
    Extract a feature vector from a single CTG recording.

    Parameters
    ----------
    fhr_raw : np.ndarray, shape (n_samples,)
        Fetal heart rate signal (physical units, bpm).
    uc_raw : np.ndarray, shape (n_samples,)
        Uterine contraction signal (physical units).

    Returns
    -------
    features : np.ndarray, shape (n_features,)
        1-D feature vector.  Feature names are returned by
        ``get_feature_names()``.
    """
    # --- Windowing: use the last WINDOW_SECONDS ---
    max_samples = WINDOW_SECONDS * SAMPLING_RATE
    if len(fhr_raw) > max_samples:
        fhr_raw = fhr_raw[-max_samples:]
        uc_raw = uc_raw[-max_samples:]

    fhr = _clean_fhr(fhr_raw)
    uc = _clean_uc(uc_raw)

    valid_fhr = fhr[~np.isnan(fhr)]
    valid_uc = uc[~np.isnan(uc)]

    # ---- FHR statistics ----
    fhr_mean = float(np.nanmean(fhr)) if len(valid_fhr) > 0 else 0.0
    fhr_std = float(np.nanstd(fhr)) if len(valid_fhr) > 0 else 0.0
    fhr_median = float(np.nanmedian(fhr)) if len(valid_fhr) > 0 else 0.0
    fhr_min = float(np.nanmin(fhr)) if len(valid_fhr) > 0 else 0.0
    fhr_max = float(np.nanmax(fhr)) if len(valid_fhr) > 0 else 0.0
    fhr_range = fhr_max - fhr_min
    fhr_iqr = float(np.nanpercentile(fhr, 75) - np.nanpercentile(fhr, 25)) if len(valid_fhr) > 0 else 0.0
    fhr_skew = float(_safe_skewness(valid_fhr))
    fhr_kurt = float(_safe_kurtosis(valid_fhr))

    # ---- HRV metrics (gap-aware) ----
    # F8 fix: difference in-place across gaps, not after compacting.
    sd = _successive_diffs_gap_aware(fhr, max_gap_samples=4)
    rmssd = float(np.sqrt(np.mean(sd ** 2)))
    mean_abs_diff = float(np.mean(sd))
    median_abs_diff = float(np.median(sd))

    # ---- Variability in segments (STV / LTV) ----
    stv = _short_term_variability(fhr)
    ltv = _long_term_variability(fhr)

    # ---- Baseline (rolling median, not global median) ----
    # F7 fix: _estimate_baseline now uses a proper rolling median,
    # so baseline_fhr is no longer a duplicate of fhr_median.
    baseline = _estimate_baseline(fhr)

    # ---- Decelerations / Accelerations ----
    n_decelerations = _count_decelerations(fhr, baseline)
    n_accelerations = _count_accelerations(fhr, baseline)

    # ---- Signal quality features ----
    # F8 fix: add explicit signal-loss features as honest proxies.
    fhr_missing_ratio = float(np.isnan(fhr).sum()) / max(len(fhr), 1)
    longest_gap_s = _longest_gap_seconds(fhr)

    # ---- UC statistics ----
    uc_mean = float(np.nanmean(uc)) if len(valid_uc) > 0 else 0.0
    uc_std = float(np.nanstd(uc)) if len(valid_uc) > 0 else 0.0
    uc_max = float(np.nanmax(uc)) if len(valid_uc) > 0 else 0.0

    # ---- Contraction features ----
    peaks = _detect_contractions(uc)
    n_contractions = len(peaks)
    duration_min = len(uc) / SAMPLING_RATE / 60.0
    contraction_freq = n_contractions / max(duration_min, 1e-6)  # per minute

    # Mean contraction interval
    if len(peaks) > 1:
        intervals = np.diff(peaks) / SAMPLING_RATE  # seconds
        mean_contraction_interval = float(np.mean(intervals))
        std_contraction_interval = float(np.std(intervals))
    else:
        mean_contraction_interval = 0.0
        std_contraction_interval = 0.0

    # ---- FHR-UC coupling ----
    fhr_uc_corr = _fhr_uc_correlation(fhr, uc)

    # ---- FHR around contractions (asymmetric pre/post windows) ----
    # P1 fix: split into pre-peak and post-peak to distinguish early vs.
    # late deceleration patterns, rather than using a symmetric window.
    fhr_pre_contraction, fhr_post_contraction = _fhr_around_contractions(
        fhr, uc, peaks
    )
    fhr_post_minus_pre = fhr_post_contraction - fhr_pre_contraction

    # ---- Assemble ----
    features = np.array([
        fhr_mean,                   # 0
        fhr_std,                    # 1
        fhr_median,                 # 2
        fhr_min,                    # 3
        fhr_max,                    # 4
        fhr_range,                  # 5
        fhr_iqr,                    # 6
        fhr_skew,                   # 7
        fhr_kurt,                   # 8
        rmssd,                      # 9
        mean_abs_diff,              # 10
        median_abs_diff,            # 11
        stv,                        # 12
        ltv,                        # 13
        baseline,                   # 14
        float(n_decelerations),     # 15
        float(n_accelerations),     # 16
        fhr_missing_ratio,          # 17
        longest_gap_s,              # 18
        uc_mean,                    # 19
        uc_std,                     # 20
        uc_max,                     # 21
        float(n_contractions),      # 22
        contraction_freq,           # 23
        mean_contraction_interval,  # 24
        std_contraction_interval,   # 25
        fhr_uc_corr,                # 26
        fhr_pre_contraction,        # 27
        fhr_post_contraction,       # 28
        fhr_post_minus_pre,         # 29
    ], dtype=np.float64)

    # Replace any remaining inf / nan with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


def get_feature_names() -> list[str]:
    """Return the ordered list of feature names produced by ``extract_features``."""
    return [
        "fhr_mean",
        "fhr_std",
        "fhr_median",
        "fhr_min",
        "fhr_max",
        "fhr_range",
        "fhr_iqr",
        "fhr_skew",
        "fhr_kurtosis",
        "rmssd",
        "mean_abs_diff",
        "median_abs_diff",
        "short_term_variability",
        "long_term_variability",
        "baseline_fhr",
        "n_decelerations",
        "n_accelerations",
        "fhr_missing_ratio",
        "longest_gap_s",
        "uc_mean",
        "uc_std",
        "uc_max",
        "n_contractions",
        "contraction_freq_per_min",
        "mean_contraction_interval_s",
        "std_contraction_interval_s",
        "fhr_uc_correlation",
        "fhr_pre_contraction",
        "fhr_post_contraction",
        "fhr_post_minus_pre",
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_skewness(arr: np.ndarray) -> float:
    if len(arr) < 3:
        return 0.0
    m = np.mean(arr)
    s = np.std(arr)
    if s == 0:
        return 0.0
    return float(np.mean(((arr - m) / s) ** 3))


def _safe_kurtosis(arr: np.ndarray) -> float:
    if len(arr) < 4:
        return 0.0
    m = np.mean(arr)
    s = np.std(arr)
    if s == 0:
        return 0.0
    return float(np.mean(((arr - m) / s) ** 4) - 3.0)


def _short_term_variability(fhr: np.ndarray, epoch_s: float = 60.0) -> float:
    """Mean of per-epoch mean-absolute-successive-difference (STV proxy)."""
    epoch_samples = int(epoch_s * SAMPLING_RATE)
    n = len(fhr)
    if n < epoch_samples:
        sd = _successive_diffs_gap_aware(fhr)
        return float(np.mean(sd)) if len(sd) > 0 else 0.0
    stvs = []
    for start in range(0, n - epoch_samples + 1, epoch_samples):
        chunk = fhr[start:start + epoch_samples]
        sd = _successive_diffs_gap_aware(chunk)
        if len(sd) > 0:
            stvs.append(float(np.mean(sd)))
    return float(np.mean(stvs)) if stvs else 0.0


def _long_term_variability(fhr: np.ndarray, epoch_s: float = 60.0) -> float:
    """Standard deviation of per-epoch medians (LTV proxy)."""
    epoch_samples = int(epoch_s * SAMPLING_RATE)
    n = len(fhr)
    if n < epoch_samples:
        valid = fhr[~np.isnan(fhr)]
        return float(np.std(valid)) if len(valid) > 1 else 0.0
    medians = []
    for start in range(0, n - epoch_samples + 1, epoch_samples):
        chunk = fhr[start:start + epoch_samples]
        valid = chunk[~np.isnan(chunk)]
        if len(valid) > 0:
            medians.append(float(np.median(valid)))
    return float(np.std(medians)) if len(medians) > 1 else 0.0


def _longest_gap_seconds(fhr: np.ndarray) -> float:
    """Return the duration (in seconds) of the longest NaN gap in the FHR."""
    longest = 0
    current = 0
    for v in fhr:
        if np.isnan(v):
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    return longest / SAMPLING_RATE


def _fhr_uc_correlation(fhr: np.ndarray, uc: np.ndarray) -> float:
    """Pearson correlation between FHR and UC (NaN-safe)."""
    mask = ~(np.isnan(fhr) | np.isnan(uc))
    if mask.sum() < 10:
        return 0.0
    f, u = fhr[mask], uc[mask]
    if np.std(f) == 0 or np.std(u) == 0:
        return 0.0
    return float(np.corrcoef(f, u)[0, 1])


def _fhr_around_contractions(
    fhr: np.ndarray, uc: np.ndarray, peaks: np.ndarray,
    pre_window_s: float = 30.0, post_window_s: float = 60.0
) -> tuple[float, float]:
    """
    Mean FHR in the pre-peak and post-peak windows around contractions.

    P1 fix: uses asymmetric windows to preserve early-vs-late deceleration
    timing.  The pre-peak window captures the heart rate just before the
    contraction peak; the post-peak window captures the period where late
    decelerations would appear.

    Returns (mean_fhr_pre_peak, mean_fhr_post_peak).
    """
    if len(peaks) == 0:
        valid = fhr[~np.isnan(fhr)]
        m = float(np.mean(valid)) if len(valid) > 0 else 0.0
        return m, m

    pre_samples = int(pre_window_s * SAMPLING_RATE)
    post_samples = int(post_window_s * SAMPLING_RATE)

    pre_vals = []
    post_vals = []
    for pk in peaks:
        # Pre-peak window
        lo = max(0, pk - pre_samples)
        pre_chunk = fhr[lo:pk]
        valid = pre_chunk[~np.isnan(pre_chunk)]
        if len(valid) > 0:
            pre_vals.extend(valid.tolist())

        # Post-peak window
        hi = min(len(fhr), pk + post_samples)
        post_chunk = fhr[pk:hi]
        valid = post_chunk[~np.isnan(post_chunk)]
        if len(valid) > 0:
            post_vals.extend(valid.tolist())

    m_pre = float(np.mean(pre_vals)) if pre_vals else 0.0
    m_post = float(np.mean(post_vals)) if post_vals else 0.0
    return m_pre, m_post


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class FetalDistressModel:
    """
    Random-Forest-based classifier for fetal distress detection.

    Wraps ``sklearn.ensemble.RandomForestClassifier`` with a
    ``StandardScaler`` for feature normalisation.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest.
    max_depth : int or None
        Maximum tree depth.  Default 4 (shallow) to prevent memorising
        the training set -- see report Section 2 and review finding F3.
    min_samples_leaf : int
        Minimum samples per leaf node.
    class_weight : str or dict or None
        Passed to ``RandomForestClassifier``.  Defaults to
        ``'balanced'`` to handle the class imbalance typical of this
        dataset.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: Optional[int] = 4,
        min_samples_leaf: int = 10,
        class_weight: str = "balanced",
        random_state: int = 42,
    ):
        self.scaler = StandardScaler()
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )
        self._is_fitted = False

    # ---- Training ----

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FetalDistressModel":
        """
        Fit the scaler and classifier.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)  -- binary labels (0/1)
        """
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)
        self._is_fitted = True
        return self

    # ---- Inference ----

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return the probability of distress (class 1) for each sample.

        Returns
        -------
        probs : np.ndarray, shape (n_samples,)
        """
        self._check_fitted()
        X_scaled = self.scaler.transform(X)
        return self.clf.predict_proba(X_scaled)[:, 1]

    def predict(self, X: np.ndarray,
                threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
        """
        Return binary predictions (0 or 1).

        Parameters
        ----------
        threshold : float
            Probability threshold above which the prediction is 1.
            Default is 0.28, selected on training folds to target ~80%
            recall (see F6 fix).
        """
        return (self.predict_proba(X) >= threshold).astype(int)

    # ---- Persistence ----

    def save(self, directory: str) -> None:
        """Save model + scaler to ``directory``."""
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.clf, os.path.join(directory, "model.pkl"))
        joblib.dump(self.scaler, os.path.join(directory, "scaler.pkl"))

    @classmethod
    def load(cls, directory: str) -> "FetalDistressModel":
        """Load a previously saved model from ``directory``."""
        instance = cls.__new__(cls)
        instance.clf = joblib.load(os.path.join(directory, "model.pkl"))
        instance.scaler = joblib.load(os.path.join(directory, "scaler.pkl"))
        instance._is_fitted = True
        return instance

    # ---- Helpers ----

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "Model has not been fitted yet. Call .fit() or .load() first."
            )

    @property
    def feature_importances(self) -> np.ndarray:
        """Return the feature importances from the underlying forest."""
        self._check_fitted()
        return self.clf.feature_importances_

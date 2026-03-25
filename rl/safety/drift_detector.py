"""
drift_detector.py
─────────────────
Distribution Drift Detector — detects when live market features have
drifted significantly from the training distribution.

When drift is detected, the agent's signals are blocked until the
operator acknowledges or the distribution normalises.

Method: Population Stability Index (PSI) per feature.
  PSI < 0.10  → no drift     (green)
  PSI < 0.25  → minor drift  (yellow — warn but allow)
  PSI ≥ 0.25  → major drift  (red — block trading)

PSI is computed over a rolling window of live observations vs the
reference distribution recorded during training.

Reference distribution is saved alongside the model as a .npz file.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from collections import deque
from pathlib import Path


# ── Thresholds ────────────────────────────────────────────────────────────────

PSI_WARN  = 0.10   # minor drift — log warning
PSI_BLOCK = 0.25   # major drift — block trading
N_BINS    = 10     # bins for PSI calculation

# How many live observations to accumulate before checking drift
MIN_LIVE_OBS = 100
WINDOW_SIZE  = 500   # rolling window of live obs


@dataclass
class DriftStatus:
    feature_psi:   np.ndarray   # PSI per feature
    max_psi:       float
    mean_psi:      float
    drifted_features: List[str]
    level:         str          # "ok", "warn", "block"
    n_live_obs:    int


class DriftDetector:
    """
    Maintains a rolling buffer of live observations and computes PSI
    against the training reference distribution.

    Usage:
        # At training time — call once after featurising training data:
        detector = DriftDetector.from_training_data(df_feat[FEATURE_COLS].values)
        detector.save("models/BTC_single/drift_ref.npz")

        # At inference time:
        detector = DriftDetector.load("models/BTC_single/drift_ref.npz")
        detector.add_observation(obs[:38])    # market features only (no pos-state)
        status = detector.check()
        if status.level == "block":
            # don't trade
    """

    def __init__(
        self,
        ref_percentiles: np.ndarray,   # (N_BINS+1, n_features) — bin edges
        ref_counts:      np.ndarray,   # (N_BINS, n_features) — expected frequencies
        feature_names:   Optional[List[str]] = None,
        window_size:     int = WINDOW_SIZE,
    ):
        self._ref_p    = ref_percentiles   # (N_BINS+1, n_features)
        self._ref_c    = ref_counts        # (N_BINS, n_features)  (normalised to sum=1)
        self._n_feat   = ref_counts.shape[1]
        self._names    = feature_names or [f"f{i}" for i in range(self._n_feat)]
        self._buffer   = deque(maxlen=window_size)

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def from_training_data(
        cls,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        n_bins: int = N_BINS,
        window_size: int = WINDOW_SIZE,
    ) -> "DriftDetector":
        """
        Build reference distribution from training feature matrix.
        X: (n_samples, n_features) float32/64
        """
        n_feat = X.shape[1]
        percentiles = np.zeros((n_bins + 1, n_feat))
        counts      = np.zeros((n_bins, n_feat))

        for i in range(n_feat):
            col = X[:, i]
            edges = np.percentile(col, np.linspace(0, 100, n_bins + 1))
            # Ensure strictly increasing edges (handle constant features)
            for j in range(1, len(edges)):
                if edges[j] <= edges[j - 1]:
                    edges[j] = edges[j - 1] + 1e-8
            percentiles[:, i] = edges
            hist, _ = np.histogram(col, bins=edges)
            counts[:, i] = (hist / max(hist.sum(), 1e-8))   # normalise to frequencies

        return cls(percentiles, counts, feature_names, window_size)

    @classmethod
    def load(cls, path: str) -> "DriftDetector":
        d = np.load(path, allow_pickle=True)
        names = list(d["feature_names"]) if "feature_names" in d else None
        return cls(
            ref_percentiles = d["ref_percentiles"],
            ref_counts      = d["ref_counts"],
            feature_names   = names,
            window_size     = int(d.get("window_size", WINDOW_SIZE)),
        )

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            ref_percentiles = self._ref_p,
            ref_counts      = self._ref_c,
            feature_names   = np.array(self._names),
            window_size     = np.array(self._buffer.maxlen),
        )

    # ── Live observation feed ─────────────────────────────────────────────────

    def add_observation(self, obs: np.ndarray):
        """
        Feed one market observation (38-dim, no position-state features).
        Call every step even when flat.
        """
        self._buffer.append(obs.astype(np.float32))

    # ── PSI calculation ───────────────────────────────────────────────────────

    def _psi_for_feature(self, feature_idx: int, live_col: np.ndarray) -> float:
        edges    = self._ref_p[:, feature_idx]
        ref_freq = self._ref_c[:, feature_idx]

        hist, _  = np.histogram(live_col, bins=edges)
        live_freq = hist / max(hist.sum(), 1e-8)

        # PSI = sum((live - ref) * ln(live / ref))
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        psi = float(np.sum(
            (live_freq - ref_freq) * np.log((live_freq + eps) / (ref_freq + eps))
        ))
        return max(psi, 0.0)

    def check(self) -> DriftStatus:
        """
        Compute PSI for all features against the live buffer.
        Returns DriftStatus with per-feature PSI and an overall level.
        """
        n_live = len(self._buffer)

        if n_live < MIN_LIVE_OBS:
            # Not enough data yet — return "ok" to not block early
            return DriftStatus(
                feature_psi      = np.zeros(self._n_feat),
                max_psi          = 0.0,
                mean_psi         = 0.0,
                drifted_features = [],
                level            = "ok",
                n_live_obs       = n_live,
            )

        live_matrix = np.stack(list(self._buffer))   # (window, n_feat)
        psi_vals    = np.array([
            self._psi_for_feature(i, live_matrix[:, i])
            for i in range(self._n_feat)
        ])

        max_psi  = float(psi_vals.max())
        mean_psi = float(psi_vals.mean())

        drifted = [self._names[i] for i, p in enumerate(psi_vals) if p >= PSI_BLOCK]

        if max_psi >= PSI_BLOCK:
            level = "block"
        elif max_psi >= PSI_WARN:
            level = "warn"
        else:
            level = "ok"

        return DriftStatus(
            feature_psi      = psi_vals,
            max_psi          = max_psi,
            mean_psi         = mean_psi,
            drifted_features = drifted,
            level            = level,
            n_live_obs       = n_live,
        )

    def summary(self) -> str:
        status = self.check()
        lines  = [
            f"DriftDetector  [{status.level.upper()}]  "
            f"n={status.n_live_obs}  "
            f"max_PSI={status.max_psi:.3f}  mean_PSI={status.mean_psi:.3f}"
        ]
        if status.drifted_features:
            lines.append(f"  Drifted: {', '.join(status.drifted_features)}")
        return "\n".join(lines)

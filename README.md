# xdsp_filters_hpeq_fit
```python
#!/usr/bin/env python3
"""
xdsp_hpeq_fit.py
----------------
Hybrid Orfanidis→RBJ fitter.

Goal:
- Use Orfanidis high-order Butterworth HPEQ as target.
- Fit a cascade of RBJ peak filters that matches its magnitude response.
- Avoid the comb / notch artifacts of the direct high-order realization.

Requirements:
- numpy
- scipy (for optimize.least_squares or differential_evolution)
- your xdsp_hpeq_butterworth.py providing:
    - Biquad
    - cascade_freq_response
    - design_hpeq_butterworth
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from math import pi, sin, cos
from typing import List
from scipy.optimize import least_squares  # or differential_evolution

from xdsp_hpeq_butterworth import (
    Biquad,
    cascade_freq_response,
    design_hpeq_butterworth,
)

# =========================================================
# RBJ peaking biquad
# =========================================================

def rbj_peak(f0: float, fs: float, Q: float, gain_db: float) -> Biquad:
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * pi * f0 / fs
    alpha = sin(w0) / (2.0 * Q)
    cw = cos(w0)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * cw
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cw
    a2 = 1.0 - alpha / A

    b0 /= a0; b1 /= a0; b2 /= a0; a1 /= a0; a2 /= a0
    return Biquad(b0, b1, b2, a1, a2)


def cascade_biquads(biquads: List[Biquad], fs: float, freqs: np.ndarray) -> np.ndarray:
    """
    Evaluate cascade frequency response on given freq grid (Hz).
    """
    w = 2.0 * np.pi * freqs / fs
    z = np.exp(1j * w)
    H = np.ones_like(z, dtype=complex)
    for bq in biquads:
        num = bq.b0 + bq.b1 / z + bq.b2 / (z**2)
        den = 1.0 + bq.a1 / z + bq.a2 / (z**2)
        H *= num / den
    return H


# =========================================================
# Build Orfanidis target response
# =========================================================

def orfanidis_target_mag(
    fs: float,
    f0: float,
    bw_hz: float,
    gain_db: float,
    order: int,
    freqs: np.ndarray,
) -> np.ndarray:
    """
    Magnitude (in dB) of the Orfanidis Butterworth HPEQ on given freq grid.
    """
    bqs = design_hpeq_butterworth(fs, f0, bw_hz, gain_db, order)
    _, H = cascade_freq_response(bqs, fs, n_fft=len(freqs) * 4)
    # Re-evaluate precisely on freqs to avoid grid mismatch:
    H_exact = cascade_biquads(bqs, fs, freqs)
    mag_db = 20.0 * np.log10(np.maximum(np.abs(H_exact), 1e-16))
    return mag_db


# =========================================================
# RBJ cascade parameterization
# =========================================================

@dataclass
class RBJStage:
    Q: float
    gain_db: float


def rbj_cascade_from_params(
    fs: float,
    f0: float,
    stages: List[RBJStage],
) -> List[Biquad]:
    return [rbj_peak(f0, fs, st.Q, st.gain_db) for st in stages]


def pack_params(stages: List[RBJStage]) -> np.ndarray:
    # [g1_db, logQ1, g2_db, logQ2, ...]
    p = []
    for st in stages:
        p.append(st.gain_db)
        p.append(np.log(st.Q))
    return np.array(p, dtype=float)


def unpack_params(p: np.ndarray) -> List[RBJStage]:
    stages = []
    for i in range(0, len(p), 2):
        g_db = p[i]
        Q = float(np.exp(p[i+1]))
        stages.append(RBJStage(Q=Q, gain_db=g_db))
    return stages


# =========================================================
# Fitting routine
# =========================================================

def fit_rbj_to_orfanidis(
    fs: float,
    f0: float,
    bw_hz: float,
    gain_db: float,
    order: int,
    n_stages: int | None = None,
    n_freqs: int = 512,
    weight_midband: float = 4.0,
) -> List[Biquad]:
    """
    Fit an RBJ peaking cascade to the Orfanidis HPEQ magnitude.

    Parameters
    ----------
    fs, f0, bw_hz, gain_db, order : Orfanidis design params
    n_stages : number of RBJ peaking sections.
               Default: order // 2 (reasonable starting point).
    n_freqs  : points in fitting grid.
    weight_midband : extra weight near the boosted region.

    Returns
    -------
    list[Biquad] : fitted RBJ cascade.
    """
    if n_stages is None:
        n_stages = max(1, order // 2)

    # Frequency grid: focus around f0 +/- some multiples of bw
    f_min = max(20.0, f0 / 8.0)
    f_max = min(fs / 2.0 * 0.999, f0 * 8.0)
    freqs = np.geomspace(f_min, f_max, n_freqs)

    # Target
    target_db = orfanidis_target_mag(fs, f0, bw_hz, gain_db, order, freqs)

    # Initial guess:
    # - split gain evenly
    # - Qs spaced between "broad" and "tighter"
    stages0 = []
    per_gain = gain_db / n_stages
    for k in range(n_stages):
        Q_init = 0.6 + (k / max(1, n_stages-1)) * 1.2  # ~0.6 .. 1.8
        stages0.append(RBJStage(Q=Q_init, gain_db=per_gain))
    p0 = pack_params(stages0)

    # Optional: construct weights emphasizing region near f0
    w = np.ones_like(freqs)
    mid_mask = (freqs > f0 / 2.0) & (freqs < f0 * 2.0)
    w[mid_mask] *= weight_midband

    # Constrain total gain roughly (softly) via regularization in residual.

    def residual(p: np.ndarray) -> np.ndarray:
        stages = unpack_params(p)
        # soft constraint: sum gains ~ desired
        total_g = sum(st.gain_db for st in stages)
        penalty = (total_g - gain_db) * 0.2  # gentle

        bqs = rbj_cascade_from_params(fs, f0, stages)
        H = cascade_biquads(bqs, fs, freqs)
        mag_db = 20.0 * np.log10(np.maximum(np.abs(H), 1e-16))

        err = (mag_db - target_db)
        return w * err + penalty

    # Use robust least squares
    res = least_squares(
        residual,
        p0,
        method="trf",
        max_nfev=400,
    )

    stages_opt = unpack_params(res.x)
    return rbj_cascade_from_params(fs, f0, stages_opt)


# =========================================================
# Quick demo / sanity check
# =========================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fs = 48000.0
    f0 = 2000.0
    bw_hz = 400.0
    gain_db = 6.0
    order = 8

    # Build target (Orfanidis)
    n_plot = 1024
    freqs = np.geomspace(20.0, fs/2.0*0.999, n_plot)
    target_db = orfanidis_target_mag(fs, f0, bw_hz, gain_db, order, freqs)

    # Fit RBJ cascade
    rbj_biquads = fit_rbj_to_orfanidis(fs, f0, bw_hz, gain_db, order, n_stages=order//2)
    H_rbj = cascade_biquads(rbj_biquads, fs, freqs)
    rbj_db = 20.0 * np.log10(np.maximum(np.abs(H_rbj), 1e-16))

    # Plot comparison
    plt.figure(figsize=(9,5))
    plt.semilogx(freqs, target_db, label="Orfanidis HPEQ (target)")
    plt.semilogx(freqs, rbj_db, "--", label="Fitted RBJ cascade")
    plt.grid(True, which="both", alpha=0.3)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Error plot
    diff = rbj_db - target_db
    print("Max abs error (dB):", np.max(np.abs(diff)))
    plt.figure(figsize=(9,3))
    plt.semilogx(freqs, diff)
    plt.grid(True, which="both", alpha=0.3)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("RBJ - Target (dB)")
    plt.tight_layout()
    plt.show()

```

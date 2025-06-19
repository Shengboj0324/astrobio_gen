"""Step 5a – simple SNR metric."""
from typing import Dict, Tuple
import numpy as np

_BANDS = {"O2": 0.76, "CH4": 1.65}

def _snr(w, f, centre):
    idx = (np.abs(w - centre)).argmin()
    return (1 - f[idx]) / 0.01        # 1 % noise floor

def score_spectrum(wave, flux) -> Tuple[Dict[str, float], float]:
    per = {g: _snr(wave, flux, c) for g, c in _BANDS.items()}
    combined = min(sum(per.values()) / (len(per) * 5), 1.0)
    return per, combined              # dict, D
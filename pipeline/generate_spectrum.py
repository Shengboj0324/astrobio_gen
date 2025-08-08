"""
Dispatch → fast Gaussian dip *or* high-fidelity PSG.

Env-var switch
--------------
FAST_MODE = 1   → cheap Nosci stub
FAST_MODE = 0   → live PSG call (needs PSG_KEY set)
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np

if os.getenv("FAST_MODE", "1") == "1":
    # ------- Lightweight stub --------
    PIVOT = {"O2": 0.76, "CH4": 1.65, "CO2": 1.6}

    def generate_spectrum(gases: Dict[str, float], planet=None) -> Tuple[np.ndarray, np.ndarray]:
        wave = np.linspace(0.5, 2.5, 100)
        flux = np.ones_like(wave)
        for gas, mix in gases.items():
            if gas in PIVOT:
                depth = 0.1 * mix / 0.21
                flux -= depth * np.exp(-((wave - PIVOT[gas]) ** 2) / (2 * 0.02**2))
        return wave, flux

else:
    # ------- High-fidelity mode --------
    from pipeline.generate_spectrum_psg import get_spectrum as _psg

    def generate_spectrum(gases: Dict[str, float], planet: dict) -> Tuple[np.ndarray, np.ndarray]:
        return _psg(gases, planet, instrument="JWST-NIRSpec", R=1000)

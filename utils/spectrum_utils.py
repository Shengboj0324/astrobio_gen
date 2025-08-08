"""
spectrum_utils.py
=================

Convenience functions for plotting and quick spectral operations.
Keeps heavy libs (specutils, astropy) out of the critical path.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_spectrum", "equivalent_width"]


def plot_spectrum(
    wave: np.ndarray,
    flux: np.ndarray,
    *,
    ax=None,
    title: str = "Spectrum",
    ylabel: str = "Relative flux",
    ylim: tuple[float, float] | None = None,
):
    """One-line Matplotlib wrapper."""
    ax = ax or plt.figure(figsize=(6, 3)).gca()
    ax.plot(wave, flux, lw=1.2)
    ax.set_xlabel("Wavelength [µm]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    plt.tight_layout()
    if ax is None:
        plt.show()


def equivalent_width(wave: np.ndarray, flux: np.ndarray, band: tuple[float, float]) -> float:
    """
    Crude equivalent-width (area under 1-flux) for a given line band.

    Parameters
    ----------
    band : (λ_min, λ_max) in same units as wave

    Returns
    -------
    float  EW in λ units
    """
    m = (wave >= band[0]) & (wave <= band[1])
    return np.trapz(1 - flux[m], wave[m])

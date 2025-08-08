import json
import pathlib

import numpy as np
import torch

from pipeline.generate_metabolism import generate_metabolism
from pipeline.generate_spectrum import generate_spectrum
from pipeline.simulate_atmosphere import simulate_atmosphere

N = 100_000
out = pathlib.Path("data/synth").mkdir(exist_ok=True)
BASE = {"N2": 0.78, "O2": 0.21, "CO2": 0.01}
for i in range(N):
    env = np.random.rand(4)
    _, flux = generate_metabolism(env)
    atm = simulate_atmosphere(BASE, flux)
    w, f = generate_spectrum(atm)
    np.savez(out / f"{i:06}.npz", env=env, flux=flux, atm=atm, wave=w, spec=f)

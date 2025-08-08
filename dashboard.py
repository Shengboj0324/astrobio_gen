import json
import pathlib

import numpy as np
import streamlit as st

from utils.spectrum_utils import plot_spectrum

files = sorted(pathlib.Path("data/synth").glob("*.npz"))[:500]
sel = st.slider("Index", 0, len(files) - 1, 0)
data = np.load(files[sel])
st.write("Env vector", data["env"])
plot_spectrum(data["wave"], data["spec"], title=f"Spectrum {sel}")

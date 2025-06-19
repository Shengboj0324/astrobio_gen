import streamlit as st, json, pathlib, numpy as np
from utils.spectrum_utils import plot_spectrum

files = sorted(pathlib.Path("data/synth").glob("*.npz"))[:500]
sel = st.slider("Index",0,len(files)-1,0)
data=np.load(files[sel])
st.write("Env vector", data["env"])
plot_spectrum(data["wave"], data["spec"], title=f"Spectrum {sel}")
import glob

import pandas as pd
import streamlit as st

csv = sorted(glob.glob("lightning_logs/**/metrics.csv"))[-1]
df = pd.read_csv(csv)
st.line_chart(df["train_loss"])
st.metric("Final loss", round(df["train_loss"].iloc[-1], 3))

#!/usr/bin/env python

from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
INT = Path("data/interim"); INT.mkdir(parents=True, exist_ok=True)

# ---------- load source tables ----------
pathways = pd.read_csv(RAW / "kegg_pathways.csv",
                       sep="\t", names=["pathway", "desc"])

tags     = pd.read_csv(INT / "pathway_env_tag.csv",
                       sep=",")                          # ← this one IS comma

# after reading the hsa file
hsa = pd.read_csv(
    RAW / "kegg_hsa_genes.csv",
    sep="\t",
    names=["gene_id", "description"],
    dtype=str,           # ← force everything to str
).dropna(subset=["description"])

hsa["pathway"] = hsa["description"].str.extract(r"(map\d{5})")
human_path_df = pd.read_csv(RAW / "kegg_hsa_pathways.csv", sep=",")
human_set = set(human_path_df["pathway"])

# ---------- default env values ----------
def default_row(pid):
    return dict(pathway=pid, pH=7.0, temp=298, O2=0.21, redox=0.0)

env_rows = { pid: default_row(pid) for pid in pathways["pathway"] }

# ---------- rule set derived from br08606 tags ----------
TAG_RULES = {
    "Aerobic"    : dict(O2=0.21, redox=0.0),
    "Anaerobic"  : dict(O2=0.00, redox=-0.1),
    "Methanogen" : dict(O2=0.00, redox=-0.3),
    "Thermophile": dict(temp=330),
    "Acidophile" : dict(pH=5.5),
    "Alkaliphile": dict(pH=9.0),
}

for _, row in tags.iterrows():
    pid, tag = row.pathway, row.tag
    if pid not in env_rows:
        env_rows[pid] = default_row(pid)
    if tag in TAG_RULES:
        env_rows[pid].update(TAG_RULES[tag])

# ---------- human-specific override ----------
for pid in human_set:
    if pid in env_rows:
        env_rows[pid].update(dict(temp=310, O2=0.21))

# ---------- emit CSV ----------
out = INT / "env_vectors.csv"
pd.DataFrame(env_rows.values()).to_csv(out, index=False)
print("✔ wrote", out, len(env_rows), "rows")
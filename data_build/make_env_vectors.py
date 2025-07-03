import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
INT = Path("data/interim"); INT.mkdir(exist_ok=True, parents=True)

# -------- read KEGG tables (TAB-separated) ----------
pathways = pd.read_csv(RAW / "kegg_pathways.csv", sep="\t",
                       names=["pathway","desc"], dtype=str)

tags = pd.read_csv(INT / "pathway_env_tag.csv")   # comma-sep

hsa = (pd.read_csv(RAW / "kegg_hsa_genes.csv", sep="\t",
                   names=["gene_id","description"], dtype=str)
         .dropna(subset=["description"]))
hsa["pathway"] = hsa["description"].str.extract(r"(map\\d{5})")

# -------- human pathway set ----------
human_set = set(hsa["pathway"].dropna())

# -------- tag → numeric env lookup ----------
TAG2ENV = {
    "Aerobic":     [7.0, 298, 0.21,  0.0],
    "Anaerobic":   [7.0, 298, 0.00, -0.1],
    "Methanogen":  [6.8, 330, 0.00, -0.3],
    "Thermophile": [7.0, 330, 0.05, -0.1],
}
DEFAULT = [7.0, 298, 0.21, 0.0]

env_rows = {}
for pid in pathways["pathway"]:
    vec = TAG2ENV.get(tags.set_index("pathway").get(pid, "Unknown"), DEFAULT).copy()
    if pid in human_set:
        vec[1] = 310  # temp
        vec[2] = 0.21 # O2
    env_rows[pid] = [pid] + vec

pd.DataFrame(env_rows.values(),
             columns=["pathway","pH","temp","O2","redox"]
            ).to_csv(INT / "env_vectors.csv", index=False)

print("env_vectors.csv written OK")
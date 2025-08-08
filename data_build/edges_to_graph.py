import csv
import json
from pathlib import Path

import networkx as nx
import numpy as np

EDGES = Path("data/interim/kegg_edges.csv")
ENV = Path("data/interim/env_vectors.csv")
OUT = Path("data/kegg_graphs")
OUT.mkdir(exist_ok=True)

env_map = {
    r["pathway"]: [float(r["pH"]), float(r["temp"]), float(r["O2"]), float(r["redox"])]
    for r in csv.DictReader(ENV.open())
}

edges = {}
for r in csv.DictReader(EDGES.open()):
    edges.setdefault(r["reaction"], []).append((r["substrate"], r["product"]))

for pid, e in edges.items():
    G = nx.DiGraph()
    G.add_edges_from(e)
    nodes = list(G)
    idx = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)), dtype=np.int8)
    for u, v in e:
        A[idx[u], idx[v]] = 1
    np.savez(
        OUT / f"{pid}.npz",
        adj=A,
        env=np.array(env_map.get(pid, [7, 298, 0.21, 0])),
        meta=json.dumps({"nodes": nodes}),
    )
print("NPZ graphs written →", OUT)

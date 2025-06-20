"""
Convert edges CSV + env_vectors.csv → .npz graphs
(each npz stores adj matrix, env_vec, pathway_id)
"""
import csv, numpy as np, networkx as nx, pathlib, json

EDGE_CSV = pathlib.Path("data/interim/kegg_edges.csv")
ENV_CSV  = pathlib.Path("data/interim/env_vectors.csv")
OUT_DIR  = pathlib.Path("data/kegg_graphs"); OUT_DIR.mkdir(exist_ok=True)

# load env vectors
env = {row["pathway"]: [float(row["pH"]), float(row["temp"]), float(row["O2"]), float(row["redox"])]
       for row in csv.DictReader(ENV_CSV.open())}

edges_by_path = {}
for row in csv.DictReader(EDGE_CSV.open()):
    edges_by_path.setdefault(row["reaction"], []).append((row["substrate"], row["product"]))

for pid, edges in edges_by_path.items():
    G = nx.DiGraph(); G.add_edges_from(edges)
    nodes = list(G.nodes()); idx = {n:i for i,n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)), dtype=int)
    for u,v in edges: A[idx[u], idx[v]] = 1
    f = OUT_DIR / f"{pid}.npz"
    np.savez(f, adj=A, env=np.array(env.get(pid, [7,298,0.2,0])), meta=json.dumps({"nodes":nodes}))
print("NPZ graphs saved to", OUT_DIR)
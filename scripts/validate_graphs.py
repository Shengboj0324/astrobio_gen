import numpy as np, random, pathlib, sys
gs = list(pathlib.Path("data/kegg_graphs").glob("*.npz"))
if not gs: sys.exit("No graphs found")
for p in random.sample(gs, min(5,len(gs))):
    n = np.load(p)
    assert n['adj'].shape[0]==n['adj'].shape[1] and len(n['env'])==4
print("Graph dataset looks good ✔")
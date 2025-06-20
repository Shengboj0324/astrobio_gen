"""
Quick sanity: load 10 random NPZ files, ensure adj is square and env_vec length == 4
"""
import random, numpy as np, pathlib
paths = random.sample(list(pathlib.Path("data/kegg_graphs").glob("*.npz")), 10)
for p in paths:
    dat = np.load(p)
    assert dat["adj"].shape[0] == dat["adj"].shape[1]
    assert len(dat["env"]) == 4
print("validation passed for", len(paths), "graphs")
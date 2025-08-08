"""Step 2 – generate a toy metabolic graph + gas-flux dict."""

from typing import Dict

import torch

from models.metabolism_model import MetabolismGenerator
from utils.graph_utils import adj_to_network


def generate_metabolism(env_vec) -> tuple[Dict, Dict[str, float]]:
    """Returns (network-dict, gas-flux-dict)."""
    model = MetabolismGenerator().to("cpu")  # CPU; .to("cuda") later
    adj = model.sample(torch.tensor(env_vec, dtype=torch.float32))
    network = adj_to_network(adj)
    # ↑ Dummy logic: each edge emits 0.1 CH4, plus 0.05 O2 “background”
    flux = {"CH4": 0.1 * len(network["edges"]), "O2": 0.05}
    return network, flux


if __name__ == "__main__":
    net, flx = generate_metabolism([0, 0, 0, 0])
    print(net)
    print(flx)

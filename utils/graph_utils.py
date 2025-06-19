"""
graph_utils.py
==============

Light-weight helpers for converting between adjacency matrices, edge lists
and NetworkX graphs.  Written to stay NumPy-only so it works on any machine.

Functions
---------
adj_to_network(adj) -> dict
    2-D numpy / torch / list matrix  →  {"nodes":[...], "edges":[(u,v),…]}.

network_to_adj(network, n) -> np.ndarray
    Edge-list dict → binary adjacency (n×n).

visualise(G)
    Quick Matplotlib figure (works for ≤30 nodes, debugging only).
"""
from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


__all__ = ["adj_to_network", "network_to_adj", "visualise"]


def _to_ndarray(a) -> np.ndarray:
    "Convert list / torch / ndarray to np.ndarray."
    if "torch" in str(type(a)):
        import torch

        return a.detach().cpu().numpy()
    return np.asarray(a)


# -------------------------------------------------------------------
#  public helpers
# -------------------------------------------------------------------
def adj_to_network(adj) -> Dict[str, List]:
    """
    Parameters
    ----------
    adj : (N,N) array-like  of 0/1

    Returns
    -------
    dict  { "nodes": [0..N-1], "edges": [(src,dst), …] }
    """
    a = _to_ndarray(adj)
    a = (a > 0.5).astype(int)
    G = nx.from_numpy_array(a, create_using=nx.DiGraph)
    edges = [(u, v) for u, v in G.edges() if a[u, v] == 1]
    return {"nodes": list(G.nodes()), "edges": edges}


def network_to_adj(net: Dict[str, List], n: int | None = None) -> np.ndarray:
    """
    Reverse of adj_to_network.

    Parameters
    ----------
    net : dict   as produced by adj_to_network
    n   : optional total node count (else inferred)

    Returns
    -------
    np.ndarray  shape (N,N)  binary adjacency
    """
    nodes = net["nodes"]
    edges = net["edges"]
    N = n or (max(nodes) + 1)
    A = np.zeros((N, N), dtype=int)
    for u, v in edges:
        A[u, v] = 1
    return A


def visualise(net_or_adj, title: str = "Metabolic graph") -> None:
    """
    Quick-and-dirty graph drawing for debugging; avoid for >30 nodes.
    """
    if isinstance(net_or_adj, dict):
        G = nx.DiGraph()
        G.add_nodes_from(net_or_adj["nodes"])
        G.add_edges_from(net_or_adj["edges"])
    else:
        G = nx.from_numpy_array(_to_ndarray(net_or_adj), create_using=nx.DiGraph)
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(4, 3))
    nx.draw_networkx(G, pos, node_size=200, arrowsize=12, with_labels=True)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
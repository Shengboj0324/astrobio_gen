"""Train the GraphVAE on random Erdos–Renyi graphs (n<=10).
Run *now* to verify gradient flow & GPU utilisation; later replace the
synthetic DataLoader with KEGG‑derived graphs.
"""
import torch, torch.nn.functional as F, networkx as nx
from torch_geometric.data import Data, DataLoader
from models.graph_vae import GVAE

# ---- hyper‑params ----
N_NODES = 10
BATCH = 32
EPOCHS = 300

# ---- synthetic dataset ----

def random_graph():
    g = nx.erdos_renyi_graph(N_NODES, p=0.3, directed=True)
    edge_index = torch.tensor(list(g.edges())).t().contiguous()
    x = torch.ones((N_NODES, 1))  # trivial node features
    return Data(x=x, edge_index=edge_index)

dataset = [random_graph() for _ in range(1000)]
loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)

# ---- model & optimiser ----
model = GVAE().to("cuda" if torch.cuda.is_available() else "cpu")
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, EPOCHS + 1):
    model.train(); total = 0
    for batch in loader:
        batch = batch.to(model.gc1.weight.device)
        adj_hat, mu, logvar = model(batch)
        # Binary cross‑entropy reconstruction (against batch adj matrix)
        # Use original edge_index → ground‑truth adj
        gt_adj = torch.zeros_like(adj_hat)
        for i,(u,v) in enumerate(batch.edge_index.t()):
            gt_adj[0,u,v] = 1
        recon = F.binary_cross_entropy(adj_hat, gt_adj)
        kld = -0.5 * torch.mean(1 + logvar - mu**2 - logvar.exp())
        loss = recon + 0.1 * kld
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: loss={total/len(loader):.4f}")

torch.save(model.state_dict(), "models/gvae_dummy.pt")
print("✔ saved models/gvae_dummy.pt")
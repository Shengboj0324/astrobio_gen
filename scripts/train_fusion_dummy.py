"""
Train the FusionModel on synthetic CSV rows so you can debug variable-schema
learning *today*.

Dataset: each row has random air_quality, rock_type, surface_vec → target=y
"""
import torch, pandas as pd, numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.fusion_transformer import FusionModel

# ---------- create synthetic table ----------
N = 800
df = pd.DataFrame({
    "air_quality": np.random.rand(N),
    "rock_type":   np.random.randint(0, 12, size=N),
    "surface_vec": list(np.random.randn(N, 64)),
})
y_reg = df["air_quality"] * 0.4 + (df["rock_type"]/11) * 0.6
y_cls = (y_reg > 0.5).astype(int) + (df["rock_type"]>6).astype(int)

# ---------- dataloader ----------
def to_tensor(col):
    return torch.tensor(np.stack(col.values if isinstance(col.iloc[0], np.ndarray) else col.values))

batch = {
    "air_quality": to_tensor(df["air_quality"]).float(),
    "rock_type":   to_tensor(df["rock_type"]).long(),
    "surface_vec": torch.tensor(np.stack(df["surface_vec"].values)).float(),
}
ds = TensorDataset(*batch.values(), torch.tensor(y_reg).float(), torch.tensor(y_cls).long())
dl = DataLoader(ds, batch_size=32, shuffle=True)

# ---------- model ----------
schema = {
    "air_quality": ("numeric", None),
    "rock_type":   ("categorical", 12),
    "surface_vec": ("vector", 64),
}
model = FusionModel(schema).train()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn_reg = torch.nn.MSELoss()
loss_fn_cls = torch.nn.CrossEntropyLoss()

for epoch in range(50):
    tot=0
    for batch_tup in dl:
        batch_dict = {k:b for k,b in zip(schema.keys(), batch_tup[:-2])}
        y_r, y_c = batch_tup[-2:]
        out = model(batch_dict)
        loss = loss_fn_reg(out["reg"], y_r) + loss_fn_cls(out["cls"], y_c)
        opt.zero_grad(); loss.backward(); opt.step()
        tot+=loss.item()
    if epoch%10==0: print(epoch, tot/len(dl))
torch.save(model.state_dict(), "models/fusion_dummy.pt")
print("✔ saved models/fusion_dummy.pt")
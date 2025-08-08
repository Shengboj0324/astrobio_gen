import time

import torch

from models.fusion_transformer import FusionModel
from utils.device import DEVICE


def test_inference_speed():
    m = FusionModel({"air_quality": ("numeric", None)}, latent_dim=64).to(DEVICE).eval()
    x = {"air_quality": torch.rand(1, 1, device=DEVICE)}
    t0 = time.time()
    m(x)
    assert (time.time() - t0) * 1000 < 5  # <5 ms

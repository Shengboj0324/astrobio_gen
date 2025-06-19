import time, torch
from models.metabolism_model import MetabolismGenerator
from utils.device import DEV
def test_forward_under5ms():
    m = MetabolismGenerator().to(DEV)
    x = torch.rand(1,4, device=DEV)
    t0=time.time(); m.sample(x)
    assert (time.time()-t0)*1000 < 5
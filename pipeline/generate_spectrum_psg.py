"""
Production-grade PSG interface
------------------------------
* Reads your PSG API key from `PSG_KEY` environment variable.
* Retries on 50x/timeout with exponential back-off.
* Caches identical requests on disk (see utils.cache).
* Returns numpy arrays (wave [µm], flux) already convolved to requested R.
"""
from __future__ import annotations
import os, time, logging, gzip, json, pathlib
import numpy as np
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from utils.cache import get as cache_get, put as cache_put

_LOG = logging.getLogger(__name__)
_API = "https://psg.gsfc.nasa.gov/api.php"

def _cfg_xml(atmos: dict[str, float], planet: dict, instrument: str, R: int) -> str:
    gases_xml = "\n".join(f"<{g}>{mix}</{g}>" for g, mix in atmos.items())
    return f"""
<OBJECT>
  <RADIUS>{planet['radius']:.3f}</RADIUS>
  <GRAVITY>{planet.get('gravity',9.8):.2f}</GRAVITY>
</OBJECT>
<ATMOSPHERE>{gases_xml}</ATMOSPHERE>
<GENERATOR>{instrument}</GENERATOR>
<RESOLUTION>{R}</RESOLUTION>
"""

@retry(wait=wait_exponential(multiplier=2), stop=stop_after_attempt(4))
def _call_psg(cfg: str) -> str:
    key = os.getenv("PSG_KEY")
    if not key:
        raise RuntimeError("Set environment variable PSG_KEY with your API token.")
    resp = requests.post(_API, files={"file": ("config.xml", cfg)}, data={"key": key}, timeout=120)
    resp.raise_for_status()
    return resp.text

def get_spectrum(atmos: dict[str,float], planet: dict,
                 instrument="JWST-NIRSpec", R=1000) -> tuple[np.ndarray,np.ndarray]:
    request_obj = {"atm": atmos, "pl": planet["pl_name"], "inst": instrument, "R": R}
    cached = cache_get(request_obj)
    if cached:
        wave, flux = np.loadtxt(cached, unpack=True, delimiter=",")
        return wave, flux
    cfg = _cfg_xml(atmos, planet, instrument, R)
    raw = _call_psg(cfg)
    # PSG returns gzip by default; auto-detect
    if raw[:2] == "PK":
        raw = gzip.decompress(raw).decode()
    wave, flux = np.loadtxt(raw.splitlines(), unpack=True)
    # cache
    fname = f"{planet['pl_name']}_{instrument}_{R}.csv".replace(" ", "_")
    path = pathlib.Path("data/psg_outputs") / fname
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack([wave, flux]), delimiter=",")
    cache_put(request_obj, str(path))
    _LOG.info("PSG spectrum cached ➜ %s", path)
    return wave, flux
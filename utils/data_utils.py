"""
data_utils.py
=============

CSV / JSON helpers plus a tiny `download()` utility for small HTTP assets.

All paths are *project-relative*; call from anywhere inside the repo.
"""
from __future__ import annotations

import csv
import json
import pathlib
import shutil
import urllib.request
from typing import Any, Dict, List


ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
#  dummy loaders (work immediately)
# ---------------------------------------------------------------------------
def load_dummy_planets(file: str | pathlib.Path | None = None) -> List[Dict[str, Any]]:
    """
    Returns list of dicts from dummy_planets.csv .
    """
    path = pathlib.Path(file) if file else DATA / "dummy_planets.csv"
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def load_dummy_metabolism(file: str | pathlib.Path | None = None) -> Dict[str, Any]:
    path = pathlib.Path(file) if file else DATA / "dummy_metabolism.json"
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
#  tiny HTTP helper for one-off downloads (e.g., PHOENIX FITS, small JSON)
# ---------------------------------------------------------------------------
def download(url: str, dest: str | pathlib.Path, overwrite=False, desc=None) -> pathlib.Path:
    """
    Parameters
    ----------
    url        : HTTP/HTTPS link
    dest       : filename or directory
    overwrite  : re-download even if dest exists
    desc       : optional print-friendly name

    Returns pathlib.Path to downloaded file.
    """
    dest = pathlib.Path(dest)
    if dest.is_dir():
        filename = url.split("/")[-1]
        dest = dest / filename
    if dest.exists() and not overwrite:
        print(f"[data_utils] using cached {dest.name}")
        return dest
    print(f"[data_utils] downloading {desc or dest.name} …")
    with urllib.request.urlopen(url) as response, open(dest, "wb") as out:
        shutil.copyfileobj(response, out)
    return dest
from __future__ import annotations

import csv
import json
import pathlib
import shutil
import urllib.request
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)


def load_real_planets(file: str | pathlib.Path | None = None) -> List[Dict[str, Any]]:
    """
    Load real exoplanet data from NASA Exoplanet Archive.
    DEPRECATED: load_dummy_planets() - Use load_real_planets() instead.
    """
    # Use real NASA exoplanet data
    path = pathlib.Path(file) if file else DATA / "planets" / "2025-06-exoplanets.csv"

    if not path.exists():
        logger.error(f"❌ CRITICAL: Real exoplanet data not found at {path}")
        logger.error("❌ Training cannot proceed without real data. Run data acquisition first.")
        raise FileNotFoundError(
            f"Real exoplanet data not found. Expected at: {path}\n"
            "Run: python -m data_build.comprehensive_13_sources_integration"
        )

    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def load_real_metabolism(file: str | pathlib.Path | None = None) -> Dict[str, Any]:
    """
    Load real metabolic pathway data from KEGG database.
    DEPRECATED: load_dummy_metabolism() - Use load_real_metabolism() instead.
    """
    # Use real KEGG pathway data
    path = pathlib.Path(file) if file else DATA / "processed" / "kegg" / "pathways_integrated.json"

    if not path.exists():
        logger.error(f"❌ CRITICAL: Real KEGG pathway data not found at {path}")
        logger.error("❌ Training cannot proceed without real data. Run data acquisition first.")
        raise FileNotFoundError(
            f"Real KEGG pathway data not found. Expected at: {path}\n"
            "Run: python -m data_build.kegg_real_data_integration"
        )

    return json.loads(path.read_text())


def download(url: str, dest: str | pathlib.Path, overwrite=False, desc=None) -> pathlib.Path:
    """Download real data from URL with proper error handling"""
    dest = pathlib.Path(dest)
    if dest.is_dir():
        filename = url.split("/")[-1]
        dest = dest / filename
    if dest.exists() and not overwrite:
        logger.info(f"[data_utils] using cached {dest.name}")
        return dest

    logger.info(f"[data_utils] downloading real data: {desc or dest.name} from {url}")

    try:
        with urllib.request.urlopen(url, timeout=30) as response, open(dest, "wb") as out:
            shutil.copyfileobj(response, out)
        logger.info(f"✅ Successfully downloaded {dest.name}")
        return dest
    except Exception as e:
        logger.error(f"❌ Failed to download {url}: {e}")
        raise RuntimeError(f"Data download failed: {e}. Training cannot proceed without real data.")

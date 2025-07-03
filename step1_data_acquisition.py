# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import pathlib
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import pandas as pd
import requests
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
import duckdb, pendulum
DB = duckdb.connect("data/astro.db")
# ---------------------------------------------------------------------------
# CONFIGURABLE CONSTANTS – tweak to raise / tighten planet selection criteria
# ---------------------------------------------------------------------------
MIN_RADIUS_RE = 0.5  # Earth radii
MAX_RADIUS_RE = 1.6
MIN_INSOL_SEFF = 0.2  # Insolation relative to Earth
MAX_INSOL_SEFF = 2.0
TAP_COLUMNS = [
    "pl_name",
    "hostname",
    "pl_orbper",
    "pl_rade",
    "pl_bmasse",
    "pl_insol",
    "st_teff",
    "st_logg",
    "st_met",
]
PHOENIX_BASE_URL = (
    "https://phoenix.astro.physik.uni-goettingen.de/2020/HiResFITS/PHOENIX-{{teff}}-5-0.fits"
)
THREADS = 4  # parallel SED downloads

DATA_DIR = pathlib.Path("data")
RAW_DIR = DATA_DIR / "raw"
PLANET_DIR = DATA_DIR / "planets"
SED_DIR = DATA_DIR / "stellar_seds"
LOG_FILE = DATA_DIR / "step1_download.log"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PLANET_DIR.mkdir(parents=True, exist_ok=True)
SED_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def last_update() -> str:
    try:
        return DB.execute("SELECT max(rowupdate) FROM exoplanets").fetchone()[0]
    except duckdb.CatalogException:
        return "1900-01-01"


def fetch_delta():
    since = last_update()
    query = f"""
        SELECT {', '.join(TAP_COLUMNS)}, rowupdate
        FROM ps WHERE rowupdate > '{since}'
    """
    tbl = NasaExoplanetArchive.query_tap(query, cache=False)
    df = tbl.to_pandas()
    if not df.empty:
        DB.execute("CREATE TABLE IF NOT EXISTS exoplanets AS SELECT * FROM df LIMIT 0")
        DB.execute("INSERT INTO exoplanets SELECT * FROM df")
    return DB.execute("SELECT * FROM exoplanets").fetch_df()


def habitable_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rocky, temperate exoplanets."""
    mask = (
        df["pl_rade"].between(MIN_RADIUS_RE, MAX_RADIUS_RE, inclusive="both")
        & df["pl_insol"].between(MIN_INSOL_SEFF, MAX_INSOL_SEFF, inclusive="both")
        & df["pl_orbper"].notna()
    )
    out = df.loc[mask].copy().reset_index(drop=True)
    logging.info("Filtered to %d rocky HZ candidates", len(out))
    return out


def _nearest_teff(teff: float) -> int:
    """Round Teff to nearest 100 K grid point available in PHOENIX set."""
    return int(round(teff / 100) * 100)


def _download_one_sed(teff: int) -> Tuple[int, pathlib.Path | None]:
    url = PHOENIX_BASE_URL.replace("{{teff}}", str(teff))
    target = SED_DIR / f"PHOENIX_{teff}.fits"
    if target.exists():
        return teff, target
    try:
        logging.debug("→ downloading %s", url)
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        target.write_bytes(resp.content)
        return teff, target
    except Exception as exc:
        logging.error("SED %s failed: %s", teff, exc)
        return teff, None


def fetch_seds(df: pd.DataFrame, max_workers: int = THREADS) -> Dict[int, pathlib.Path]:
    """Download required PHOENIX SEDs in parallel. Returns map Teff → path."""
    need = {_nearest_teff(t) for t in df["st_teff"].dropna()}
    logging.info("Need %d unique SED files", len(need))
    paths: Dict[int, pathlib.Path] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(_download_one_sed, t): t for t in need}
        for fut in as_completed(futures):
            teff, path = fut.result()
            if path:
                paths[teff] = path
    logging.info("SED download complete (%d/%d) files", len(paths), len(need))
    return paths


def build_manifest(df: pd.DataFrame) -> None:
    """Save planet records to newline‑delimited JSON for later stages."""
    import json

    manifest = PLANET_DIR / "planets.jsonl"
    with manifest.open("w", encoding="utf-8") as fh:
        for _, row in df.iterrows():
            teff_nearest = _nearest_teff(row["st_teff"])
            payload = {
                "pl_name": row["pl_name"],
                "host_star": row["hostname"],
                "radius_re": row["pl_rade"],
                "mass_me": row["pl_bmasse"],
                "orb_period_d": row["pl_orbper"],
                "insolation_seff": row["pl_insol"],
                "st_teff_K": row["st_teff"],
                "sed_file": str(SED_DIR / f"PHOENIX_{teff_nearest}.fits"),
            }
            fh.write(json.dumps(payload) + "\n")
    logging.info("Planet manifest ➜ %s (%d lines)", manifest, len(df))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 1: Exoplanet+SED data ingestion")
    p.add_argument("--min-radius", type=float, default=MIN_RADIUS_RE)
    p.add_argument("--max-radius", type=float, default=MAX_RADIUS_RE)
    p.add_argument("--min-insol", type=float, default=MIN_INSOL_SEFF)
    p.add_argument("--max-insol", type=float, default=MAX_INSOL_SEFF)
    p.add_argument("--threads", type=int, default=THREADS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    global MIN_RADIUS_RE, MAX_RADIUS_RE, MIN_INSOL_SEFF, MAX_INSOL_SEFF, THREADS
    MIN_RADIUS_RE, MAX_RADIUS_RE = args.min_radius, args.max_radius
    MIN_INSOL_SEFF, MAX_INSOL_SEFF = args.min_insol, args.max_insol
    THREADS = args.threads

    df_raw = fetch_delta()
    df_select = habitable_filter(df_raw)
    sed_map = fetch_seds(df_select)
    build_manifest(df_select)
    logging.info("Step 1 completed ✔ – ready for metabolic generation stage.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Interrupted by user – exiting.")

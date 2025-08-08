"""
fetch_kegg_lists.py
===================
Download two public KEGG lists:

1. https://rest.kegg.jp/list/pathway   →  data/raw/kegg_pathways.csv
2. https://rest.kegg.jp/list/hsa       →  data/raw/kegg_hsa_genes.csv

Both endpoints are tiny (plain text) and require no API key.
"""

import csv
from pathlib import Path

import requests

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def fetch(url: str, out_csv: Path, headers: list[str]):
    print(f"⇢ Fetching {url}")
    text = requests.get(url, timeout=30).text.strip().splitlines()
    rows = [line.split("\t") for line in text]
    with out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(headers)
        w.writerows(rows)
    print("  saved →", out_csv, f"({len(rows)} rows)")


if __name__ == "__main__":
    # 1) list of all KEGG pathways
    fetch(
        "https://rest.kegg.jp/list/pathway",
        RAW_DIR / "kegg_pathways.csv",
        ["pathway_id", "description"],
    )

    # 2) list of all H. sapiens genes with KEGG IDs
    fetch(
        "https://rest.kegg.jp/list/hsa",
        RAW_DIR / "kegg_hsa_genes.csv",
        ["gene_id", "description"],
    )

    print("Done.")

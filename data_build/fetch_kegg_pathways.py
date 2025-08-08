"""
Download N pathway KGML files listed in data/raw/kegg_pathways.csv
→ store in data/raw/kegg_xml/.
"""

import csv
import time
from pathlib import Path

import requests

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)
XML_DIR = RAW / "kegg_xml"
XML_DIR.mkdir(exist_ok=True)

import csv

with open(RAW / "kegg_pathways.csv") as fh:
    reader = csv.reader(fh)
    next(reader)  # skip header
    rows = list(reader)

# keep only metabolic pathways (mapXXXXX)
meta_rows = [r for r in rows if len(r) >= 2 and r[0].startswith("map")]

MAX = 200  # pull first 200 maps (enough to train)
for pid, name in meta_rows[:MAX]:
    map_id = pid.replace("path:", "")
    out = XML_DIR / f"{map_id}.xml"
    if out.exists():
        continue
    url = f"https://rest.kegg.jp/get/{map_id}/kgml"
    print("⇢", map_id, name)
    xml = requests.get(url, timeout=30).text
    out.write_text(xml)
    time.sleep(0.2)  # be polite
print("Done:", len(list(XML_DIR.glob("*.xml"))), "files")

"""
Parse KEGG KGML / SBML files in data/raw/kegg_xml/
→ emit data/interim/kegg_edges.csv
"""

import csv  # pip install beautifulsoup4
from pathlib import Path

import bs4

RAW = Path("data/raw/kegg_xml")
OUT = Path("data/interim/kegg_edges.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)


def parse_one(xml_path):
    soup = bs4.BeautifulSoup(xml_path.read_text(), "xml")
    for rxn in soup.find_all("reaction"):
        rid = rxn["id"]
        subs = [s["name"] for s in rxn.find_all("substrate")]
        prods = [p["name"] for p in rxn.find_all("product")]
        for s in subs:
            for p in prods:
                yield rid, s, p


with OUT.open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["reaction", "substrate", "product"])
    for xml in RAW.glob("*.xml"):
        w.writerows(parse_one(xml))
print("edges CSV written:", OUT)

import csv
import re
from pathlib import Path

from bs4 import BeautifulSoup

html = Path("data/raw/map01220.html").read_text(encoding="utf-8")
soup = BeautifulSoup(html, "lxml")

ids = []
for area in soup.find_all("area", href=True):
    m = re.search(r"/map/(map\d{5})", area["href"])
    if m:
        ids.append(m.group(1))

out = Path("data/interim/map01220_ids.csv")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as fh:
    csv.writer(fh).writerows([("pathway",)] + [(i,) for i in sorted(set(ids))])
print("Extracted", len(ids), "sub-pathway IDs →", out)

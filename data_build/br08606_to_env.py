"""
Parse KEGG   https://www.genome.jp/kegg/tables/br08606.html
→ data/interim/pathway_env_tag.csv
(pathway, tag)
"""
from pathlib import Path
import requests, csv, re
from bs4 import BeautifulSoup

RAW  = Path("data/raw"); RAW.mkdir(parents=True, exist_ok=True)
INT  = Path("data/interim"); INT.mkdir(parents=True,  exist_ok=True)

html_path = RAW / "br08606.html"

# ---------- download if absent ----------
if not html_path.exists():
    print("⇢ downloading br08606.html …")
    html = requests.get(
        "https://www.genome.jp/kegg/tables/br08606.html",
        timeout=30
    ).text
    html_path.write_text(html, encoding="utf-8")
else:
    html = html_path.read_text(encoding="utf-8")

# ---------- parse ----------
soup = BeautifulSoup(html, "lxml")          # fast parser
pairs = []
current_tag = None

for line in soup.get_text("\n").splitlines():
    if m := re.match(r"\[([^\]]+)]", line):     # e.g. [Anaerobic]
        current_tag = m.group(1).strip()
    elif line.startswith("map"):
        pid = line.split()[0].strip()           # map00680
        pairs.append((pid, current_tag or "Unknown"))

# ---------- save ----------
out_csv = INT / "pathway_env_tag.csv"
with out_csv.open("w", newline="") as fh:
    csv.writer(fh).writerows([("pathway", "tag")] + pairs)

print("✔ wrote", out_csv, f"({len(pairs)} rows)")
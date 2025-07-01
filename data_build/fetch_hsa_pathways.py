from pathlib import Path, read_text
import requests, csv

RAW = Path("data/raw"); RAW.mkdir(exist_ok=True, parents=True)
out = RAW / "kegg_hsa_pathways.csv"

print("⇢ downloading human pathway list …")
text = requests.get("https://rest.kegg.jp/list/pathway/hsa", timeout=30).text
rows = [ln.split("\t") for ln in text.strip().splitlines()]
with out.open("w", newline="") as fh:
    csv.writer(fh).writerows([("pathway","desc")] + rows)

print("saved", out, f"({len(rows)} rows)")
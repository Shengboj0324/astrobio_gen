"""
Combine every *.index / *.txt into one tidy CSV:

sample_id, population, platform, file_path

Only the first 7 columns of each .index are needed,
so memory stays low (≈ 200 MB parse, 20 MB output).
"""
import pandas as pd
from pathlib import Path, PurePosixPath
import re, gzip, csv

IND_DIR = Path("data/raw/1000g_indices")
out = Path("data/interim/1000g_sample_table.csv")
out.parent.mkdir(parents=True, exist_ok=True)

rows = []
for f in IND_DIR.iterdir():
    if not f.name.endswith(".index") and not f.name.endswith(".txt"):
        continue
    df = pd.read_table(
        f,
        nrows=None,
        comment="#",
        usecols=list(range(7)),
        names=["file", "md5", "size", "sample_id", "platform", "study", "population"],
        dtype=str,
    )
    rows.append(df[["sample_id", "population", "platform", "file"]])

pd.concat(rows, ignore_index=True).drop_duplicates().to_csv(out, index=False)
print("✔ sample table", out, "rows:", sum(len(r) for r in rows))
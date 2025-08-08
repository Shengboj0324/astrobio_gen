"""
Convert *.dirlist into one CSV:
sample_id, population, data_type, ftp_path, file_size
"""

import csv
import pathlib
import re

IND = pathlib.Path("data/raw/1000g_dirlists")
OUT = pathlib.Path("data/interim/1000g_master_index.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

rows = []
rx_file = re.compile(r"(\d+)\s+\S+\s+\S+\s+(.+)$")  # size  path
rx_sample = re.compile(r"/([A-Z0-9]{6,8})/")  # sample folder like HG00096

for dl in IND.glob("*.dirlist"):
    root = f"ftp://ftp.ncbi.nlm.nih.gov/{'/'.join(dl.stem.split('.')[:-1])}/"
    for line in dl.read_text().splitlines():
        m = rx_file.match(line)
        if not m:
            continue
        size, rel = m.groups()
        ftp_path = root + rel
        sample = rx_sample.search(rel)
        if not sample:
            continue
        sid = sample.group(1)
        pop = sid[:3]  # crude: HG0xxx → HG0 population unknown; refine later
        dtype = "cram" if rel.endswith(".cram") else "fastq"
        rows.append([sid, pop, dtype, ftp_path, size])

with OUT.open("w", newline="") as fh:
    csv.writer(fh).writerows([("sample_id", "population", "type", "ftp_path", "bytes")] + rows)

print("✔ master index saved:", OUT, len(rows), "rows")

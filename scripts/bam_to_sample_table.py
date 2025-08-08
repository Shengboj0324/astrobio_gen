#!/usr/bin/env python
"""
Scan every *.bam / *.cram in $ASTROBIO_GENOME_DIR
and write a lightweight sample table for the ML pipeline.

output → data/interim/genome_samples.csv
columns: sample,population,coverage,file
"""

import csv
import os
import re
import sys
from pathlib import Path

import pysam

GEN_DIR = Path(os.getenv("ASTROBIO_GENOME_DIR", ""))

if not GEN_DIR.is_dir():
    sys.exit(
        "ENV-VAR ASTROBIO_GENOME_DIR not set or path invalid → "
        "export ASTROBIO_GENOME_DIR=/full/path/to/DataSources(ISEF)"
    )

OUT = Path("data/interim/genome_samples.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

rows = []
bam_files = list(GEN_DIR.glob("*.bam")) + list(GEN_DIR.glob("*.cram"))
if not bam_files:
    sys.exit(f"No BAM/CRAM files found in {GEN_DIR}")


def guess_pop(sample_id: str) -> str:
    """
    Rough population guess from 1000 G naming convention.
    YRI/CEU/HG... etc → first three letters, else 'UNK'.
    """
    m = re.match(r"([A-Z]{3})", sample_id)
    return m.group(1) if m else "UNK"


for path in bam_files:
    try:
        bam = pysam.AlignmentFile(path, "rc" if path.suffix == ".cram" else "rb")
    except Exception as e:
        print(f"⚠  skip {path.name}: {e}")
        continue

    hd = bam.header.to_dict()
    # sample name from RG.SM or fall back to filename stem
    sid = hd.get("RG", [{}])[0].get("SM") or hd.get("PG", [{}])[0].get("ID") or path.stem
    pop = guess_pop(sid)
    coverage = round(bam.mapped / 3_000_000, 1)  # crude ≈× coverage

    rows.append([sid, pop, coverage, path.name])
    bam.close()

with OUT.open("w", newline="") as fh:
    csv.writer(fh).writerows([("sample", "population", "coverage", "file")] + rows)

print(f"✔  wrote {OUT}  ({len(rows)} samples)")

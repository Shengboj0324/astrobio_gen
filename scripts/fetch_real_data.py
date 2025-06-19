from __future__ import annotations
import argparse, json, subprocess, pathlib, shutil, datetime as dt

ROOT = pathlib.Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--top", type=int, default=50, help="keep N best HZ planets")
args = parser.parse_args()

# 1. Run step1 acquisition (full catalogue)
print("[fetch] running step1_data_acquisition.py …")
subprocess.run(["python", "step1_data_acquisition.py"], check=True, cwd=ROOT)

manifest = ROOT / "data/planets/planets.jsonl"
slim = manifest.with_name(f"top{args.top}.jsonl")

# 2. Keep first N lines (already sorted by distance inside step1)
print(f"[fetch] slicing top {args.top} rows →", slim.name)
with manifest.open() as src, slim.open("w") as dst:
    for i, line in enumerate(src):
        if i >= args.top:
            break
        dst.write(line)

print("[fetch] done ✔")
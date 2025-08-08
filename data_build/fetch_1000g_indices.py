"""
Stream-download all 1000 Genomes *.index and *.stats files listed in URLS
→ data/raw/1000g_indices/
(keeps a .done stamp so you can re-run safely)
"""

import hashlib
from pathlib import Path

import requests
import tqdm

URLS = [
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/1000G_2504_high_coverage/additional_698_related/1000G_698_related_high_coverage.sequence.index",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/1000G_2504_high_coverage/additional_698_related/20130606_g1k_3202_samples_ped_population.txt",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/1000G_2504_high_coverage/additional_698_related/1000G_698_related_high_coverage.sequence.index",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/1000G_2504_high_coverage/additional_698_related/20130606_g1k_3202_samples_ped_population.txt",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/1000G_2504_high_coverage/1000G_2504_high_coverage.sequence.index",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/complete_genomics_indices/20130725.sra.index",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/complete_genomics_indices/20130815.cg_ref_data.index",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/complete_genomics_indices/20130820.cg_data.index",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/alignment_indices/20091216.alignment.index",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20100208.sequence.index.stats",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20091216.sequence.index",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20091216.sequence.index.stats",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20100222.sequence.index",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20100222.sequence.index.stats",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20100311.sequence.index.stats",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20100429.sequence.index",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20100429.sequence.index.stats",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20100513.sequence.index",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20100804.sequence.index.stats",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20100513.sequence.index.stats",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20100517.sequence.index",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20100517.sequence.index.stats",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20100606.sequence.index",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20100606.sequence.index.stats",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20100611.sequence.index",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20100611.sequence.index.stats",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20100813.sequence.index",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20100710.sequence.index",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20100710.sequence.index.stats",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20100813.sequence.index.stats",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20100817.sequence.index",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20100817.sequence.index.stats",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20100901.sequence.index",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20100901.sequence.index.stats",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20100917.sequence.index",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20100917.sequence.index.stats",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20101004.sequence.index",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20101004.sequence.index.stats",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20101019.sequence.index",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20101019.sequence.index.stats",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20101025.sequence.index",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20101025.sequence.index.stats",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20101101.sequence.index",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20101101.sequence.index.stats",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20101123.sequence.index",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20101123.sequence.index.exome.stats",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20101123.sequence.index.low_coverage.stats",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20101123.sequence.index.stats",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20110124.sequence.index",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20110124.sequence.index.exome.stats",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20110124.sequence.index.low_coverage.stats",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20110228.sequence.index",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20110228.sequence.index.exome.stats",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20110228.sequence.index.low_coverage.stats",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20110411.sequence.index",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20110411.sequence.index.exome.stats",
    "https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/sequence_indices/20110411.sequence.index.low_coverage.stats",
]

OUT_DIR = Path("data/raw/1000g_indices")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for url in URLS:
    fname = OUT_DIR / url.split("/")[-1]
    if fname.with_suffix(".done").exists():
        print(fname.name, "✓")
        continue
    print("⇢", fname.name)
    with requests.get(url, stream=True, timeout=60) as r, open(fname, "wb") as fh:
        for chunk in tqdm.tqdm(r.iter_content(1024 * 1024)):
            fh.write(chunk)
    fname.with_suffix(".done").touch()
print("All index files present ✔")

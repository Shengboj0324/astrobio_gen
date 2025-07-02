#!/usr/bin/env bash
set -e
OUT=data/raw/1000g_dirlists
mkdir -p "$OUT"

# the two root URLs
URLS=(
  ftp://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/1000G_2504_high_coverage/additional_698_related/data/
  ftp://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/pilot_data/data/
)

for U in "${URLS[@]}"; do
  stem=$(basename "$(dirname "$U")")          # e.g., additional_698_related
  echo "⇢ listing $U …"
  lftp -c "open $U; cls -R --sort=date" > "$OUT/$stem.dirlist"
done
echo "All dir lists written to $OUT"
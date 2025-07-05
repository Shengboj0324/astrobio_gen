import os
from pathlib import Path

GENOME_DIR = Path(os.getenv("ASTROBIO_GENOME_DIR"))
if not GENOME_DIR.exists():
    raise FileNotFoundError(
        f"Genome dir {GENOME_DIR} not found. "
        "Set ASTROBIO_GENOME_DIR environment variable."
    )
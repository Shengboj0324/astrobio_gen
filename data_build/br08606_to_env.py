"""
Extract pathway-to-environment tags from br08606.html
→ data/interim/pathway_env_tag.csv : pathway_id,tag
"""

from bs4 import BeautifulSoup
from pathlib import Path
import csv, re

HTML = Path("data/raw/br08606.html").read_text()
soup = BeautifulSoup(HTML, "html.parser")

pairs = []
current_tag = None
for line in soup.get_text("\n").splitlines():
    if m := re.match(r"\[([^\]]+)]", line):          # e.g. [Anaerobic]
        current_tag = m.group(1).strip()
    elif line.startswith("map"):
        pid, desc = line.split(None, 1)
        pairs.append((pid, current_tag))

out = Path("data/interim/pathway_env_tag.csv")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as fh:
    csv.writer(fh).writerows([("pathway","tag")] + pairs)
print("saved", out, len(pairs), "rows")
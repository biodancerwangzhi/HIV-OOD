"""Download Stanford HIVDB genotype-phenotype Full tables into work/hivdb_full/.

Source:
  https://hivdb.stanford.edu/download/GenoPhenoDatasets/

Files:
  {PI,NRTI,NNRTI,INI,CAI}_DataSet.Full.txt

Note:
  HIVDB updates these tables over time. Numbers may differ slightly from a
  frozen paper snapshot. Cite HIVDB when using the data.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJ / "work" / "hivdb_full"
BASE_URL = "https://hivdb.stanford.edu/download/GenoPhenoDatasets"
FILES = [
    "PI_DataSet.Full.txt",
    "NRTI_DataSet.Full.txt",
    "NNRTI_DataSet.Full.txt",
    "INI_DataSet.Full.txt",
    "CAI_DataSet.Full.txt",
]


def download_one(name: str, force: bool = False) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / name
    if dest.exists() and not force and dest.stat().st_size > 0:
        print(f"skip existing {dest} ({dest.stat().st_size} bytes)")
        return dest
    url = f"{BASE_URL}/{name}"
    print(f"download {url}")
    tmp = dest.with_suffix(dest.suffix + ".partial")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    print(f"  -> {dest} ({dest.stat().st_size} bytes)")
    return dest


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true", help="re-download even if files exist")
    args = p.parse_args()
    for name in FILES:
        download_one(name, force=args.force)
    print(f"done. files in {OUT_DIR}")


if __name__ == "__main__":
    main()

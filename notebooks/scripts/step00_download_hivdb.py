"""Download Stanford HIVDB genotype-phenotype Full tables into work/hivdb_full/.

Source:
  https://hivdb.stanford.edu/download/GenoPhenoDatasets/

Files (saved locally as):
  {PI,NRTI,NNRTI,INSTI,CAI}_DataSet.Full.txt

Note:
  HIVDB serves the integrase class table as INI_DataSet.Full.txt. This project
  uses INSTI as the class name everywhere, so the remote file is fetched under
  its upstream name and saved locally as INSTI_DataSet.Full.txt.

  HIVDB updates these tables over time. Numbers may differ slightly from a
  frozen paper snapshot. Cite HIVDB when using the data.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJ / "work" / "hivdb_full"
BASE_URL = "https://hivdb.stanford.edu/download/GenoPhenoDatasets"
# local filename -> upstream filename on the HIVDB server.
# Only the integrase class differs: HIVDB calls it INI, this project calls it INSTI.
FILES = {
    "PI_DataSet.Full.txt": "PI_DataSet.Full.txt",
    "NRTI_DataSet.Full.txt": "NRTI_DataSet.Full.txt",
    "NNRTI_DataSet.Full.txt": "NNRTI_DataSet.Full.txt",
    "INSTI_DataSet.Full.txt": "INI_DataSet.Full.txt",
    "CAI_DataSet.Full.txt": "CAI_DataSet.Full.txt",
}


def download_one(name: str, remote_name: str | None = None, force: bool = False) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / name
    if dest.exists() and not force and dest.stat().st_size > 0:
        print(f"skip existing {dest} ({dest.stat().st_size} bytes)")
        return dest
    url = f"{BASE_URL}/{remote_name or name}"
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
    for name, remote_name in FILES.items():
        download_one(name, remote_name=remote_name, force=args.force)
    print(f"done. files in {OUT_DIR}")


if __name__ == "__main__":
    main()

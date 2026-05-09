"""Download Kraken master_q4 zip from Google Drive and extract a single CSV.

One-shot bootstrap. Re-runnable: skips download if zip cached, skips extract if
CSV present. Override with --force.

Defaults pull master_q4/BTCUSD_1.csv (1-minute OHLCVT) from the user's
Google Drive archive and write it to data/raw/BTCUSD_1.csv at the repo root.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

from gdown.download import download

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GDRIVE_ID = "1ptNqWYidLkhb2VAKuLCxmp2OXEfGO-AP"
DEFAULT_INNER_PATH = "master_q4/BTCUSD_1.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / ".cache"
DEFAULT_ZIP_STEM = "kraken_master_q4"


def download_zip(gdrive_id: str, dest: Path, force: bool) -> Path:
    if dest.exists() and not force:
        print(f"[skip-download] zip cached at {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    download(id=gdrive_id, output=str(dest), quiet=False)
    if not dest.exists():
        raise RuntimeError(f"gdown reported success but {dest} not found")
    return dest


def extract_member(zip_path: Path, member: str, out_dir: Path, force: bool) -> Path:
    out_path = out_dir / Path(member).name
    if out_path.exists() and not force:
        print(f"[skip-extract] {out_path} already present")
        return out_path
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if member not in names:
            sample = [n for n in names if "BTCUSD" in n][:8]
            raise FileNotFoundError(
                f"{member!r} not in zip. BTCUSD files present (first 8): {sample}"
            )
        with zf.open(member) as src, open(out_path, "wb") as dst:
            for chunk in iter(lambda: src.read(1 << 20), b""):
                dst.write(chunk)
    return out_path


def cache_zip_name(gdrive_id: str) -> str:
    return f"{DEFAULT_ZIP_STEM}_{gdrive_id}.zip"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gdrive-id", default=DEFAULT_GDRIVE_ID)
    ap.add_argument(
        "--inner-path",
        default=DEFAULT_INNER_PATH,
        help="Path inside the zip; e.g. master_q4/BTCUSD_1.csv",
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-extract even if files exist",
    )
    args = ap.parse_args()

    zip_path = args.cache_dir / cache_zip_name(args.gdrive_id)
    download_zip(args.gdrive_id, zip_path, args.force)
    out_path = extract_member(zip_path, args.inner_path, args.out_dir, args.force)
    print(f"[done] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

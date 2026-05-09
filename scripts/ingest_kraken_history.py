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
from contextlib import suppress
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from constants import DATA
from gdown.download import download

DEFAULT_OUT_DIR = REPO_ROOT / DATA.KRAKEN_HISTORY_OUT_DIR
DEFAULT_CACHE_DIR = REPO_ROOT / DATA.KRAKEN_HISTORY_CACHE_DIR


def normalize_member_path(member: str) -> str:
    return PurePosixPath(member.replace("\\", "/")).as_posix()


def temp_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".tmp")


def remove_if_exists(path: Path) -> None:
    with suppress(FileNotFoundError):
        path.unlink()


def download_zip(gdrive_id: str, dest: Path, force: bool) -> Path:
    if dest.exists() and not force:
        if zipfile.is_zipfile(dest):
            print(f"[skip-download] zip cached at {dest}")
            return dest
        print(f"[discard-download] invalid cached zip at {dest}")
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = temp_path(dest)
    remove_if_exists(temp_dest)
    try:
        download(id=gdrive_id, output=str(temp_dest), quiet=False)
        if not temp_dest.exists():
            raise RuntimeError(f"gdown reported success but {temp_dest} not found")
        if not zipfile.is_zipfile(temp_dest):
            raise RuntimeError(f"downloaded file at {temp_dest} is not a valid zip")
        temp_dest.replace(dest)
    except BaseException:
        remove_if_exists(temp_dest)
        raise
    return dest


def extract_member(zip_path: Path, member: str, out_dir: Path, force: bool) -> Path:
    normalized_member = normalize_member_path(member)
    out_path = out_dir / Path(normalized_member).name
    if out_path.exists() and not force:
        print(f"[skip-extract] {out_path} already present")
        return out_path
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if normalized_member not in names:
            sample = [n for n in names if "BTCUSD" in n][:8]
            raise FileNotFoundError(
                f"{normalized_member!r} not in zip. BTCUSD files present (first 8): {sample}"
            )
        temp_out_path = temp_path(out_path)
        remove_if_exists(temp_out_path)
        try:
            with zf.open(normalized_member) as src, open(temp_out_path, "wb") as dst:
                for chunk in iter(lambda: src.read(1 << 20), b""):
                    dst.write(chunk)
            temp_out_path.replace(out_path)
        except BaseException:
            remove_if_exists(temp_out_path)
            raise
    return out_path


def cache_zip_name(gdrive_id: str) -> str:
    return f"{DATA.KRAKEN_HISTORY_ZIP_STEM}_{gdrive_id}.zip"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gdrive-id", default=DATA.KRAKEN_HISTORY_GDRIVE_ID)
    ap.add_argument(
        "--inner-path",
        default=DATA.KRAKEN_HISTORY_INNER_PATH,
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

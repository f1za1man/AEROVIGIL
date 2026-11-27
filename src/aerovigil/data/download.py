r"""Download the UAV GPS-spoofing dataset (Anthony Finn) into data/raw.

The dataset is not redistributed with AEROVIGIL: its repository
(https://github.com/anthony-finn/UAV-GPS-Spoofing-Dataset) carries no licence, so the
default is "all rights reserved", and the author gave permission for non-commercial
academic use only (see docs/datasets/data_card.md). This script fetches the author's
own archive from the link in that repository's README, checks it is byte-for-byte the
archive the project was built on (SHA-256), and unpacks it.

    archive   PX4_Simulation_Data.zip, 2,577,647,220 bytes
    SHA-256   9E932C713AB2A880C167E6C9073CACB1B6C99561896B7C331E9AE3D98F8B668C
    unpacks   data/raw/uav_gps_spoofing/PX4_Simulation_Data/{Raw,Merged,...}

The download resumes if interrupted (run it again). Nothing is unpacked over an
existing copy unless --force is given.

Usage:
    python src\aerovigil\data\download.py
    python src\aerovigil\data\download.py --zip-only     (download and check only)
"""

import argparse
import hashlib
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
TARGET = RAW / "uav_gps_spoofing"
ZIP = RAW / "downloads" / "PX4_Simulation_Data.zip"
FILE_ID = "1jypbdKFP-4vN7HMwbnm4z4i-b0OhoPtS"
URL = ("https://drive.usercontent.google.com/download?"
       f"id={FILE_ID}&export=download&confirm=t")  # fmt: skip
SIZE = 2_577_647_220
SHA256 = "9E932C713AB2A880C167E6C9073CACB1B6C99561896B7C331E9AE3D98F8B668C"
CHUNK = 1 << 20


def download(path=ZIP, url=URL, size=SIZE):
    """Fetch the archive, resuming a partial file. Returns the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".zip.partial")
    if path.is_file() and path.stat().st_size == size:
        return path
    have = part.stat().st_size if part.is_file() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "dataset-download"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    started, last = time.time(), 0.0
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and r.status != 206:  # the server ignored the range: start again
            have = 0
        with open(part, "ab" if have else "wb") as f:
            done = have
            while chunk := r.read(CHUNK):
                f.write(chunk)
                done += len(chunk)
                if time.time() - last > 5:
                    last = time.time()
                    rate = (done - have) / max(last - started, 1e-9) / 1e6
                    print(f"  {done / 1e9:.2f} / {size / 1e9:.2f} GB  "
                          f"({rate:.1f} MB/s)", flush=True)  # fmt: skip
    if part.stat().st_size != size:
        raise SystemExit(f"download stopped at {part.stat().st_size:,} of {size:,} "
                         "bytes: run the script again to resume")  # fmt: skip
    part.replace(path)
    return path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK * 8):
            h.update(chunk)
    return h.hexdigest().upper()


def check(path=ZIP, expected=SHA256):
    got = sha256(path)
    if got != expected:
        raise SystemExit(f"{path} is not the expected archive (SHA-256 {got}); "
                         "delete it and download again")  # fmt: skip
    return got


def unpack(path=ZIP, target=TARGET, force=False):
    if target.is_dir() and any(target.iterdir()) and not force:
        print(f"{target} already holds the dataset; use --force to unpack again")
        return target
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as z:
        z.extractall(target)
    return target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip-only", action="store_true",
                    help="download and check the archive, do not unpack")  # fmt: skip
    ap.add_argument("--force", action="store_true", help="unpack over a copy")
    args = ap.parse_args()
    print(f"Downloading {ZIP.name} ({SIZE / 1e9:.2f} GB) to {ZIP.parent} ...")
    path = download()
    print("Checking SHA-256 ...")
    check(path)
    print("  matches the archive the project was built on")
    if not args.zip_only:
        print(f"Unpacking to {TARGET} ...")
        unpack(path, force=args.force)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

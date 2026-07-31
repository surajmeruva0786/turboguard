#!/usr/bin/env python
"""Download the real NASA IMS Bearing Dataset.

Source: NASA Prognostics Center of Excellence data repository. The dataset
is published as a single archive on S3:
https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip

Note: the archive is ~1 GB and expands to three run-to-failure test sets
(each a directory of 1-second snapshot files recorded every ~10 minutes).
This script downloads and extracts it; it does not run automatically as
part of the synthetic-data pipeline because of its size.

Usage:
    python scripts/download_ims.py --output_dir data/raw/ims
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from src.utils.io import ensure_dir
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

IMS_URL = "https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip"


def download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("Archive already downloaded, skipping: %s", dest)
        return
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                bar.update(len(chunk))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path("data/raw/ims"))
    parser.add_argument("--skip_extract", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)
    archive_path = out_dir / "4.Bearings.zip"

    logger.info("Downloading IMS bearing dataset (~1 GB) from %s", IMS_URL)
    download_file(IMS_URL, archive_path)

    if not args.skip_extract:
        logger.info("Extracting %s -> %s", archive_path, out_dir)
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(out_dir)

    logger.info("Done. Real IMS run-to-failure sets are under %s (gitignored).", out_dir)


if __name__ == "__main__":
    main()

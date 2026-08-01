#!/usr/bin/env python
"""Download the real CWRU Bearing Dataset from the Case Western Reserve
University Bearing Data Center.

The site (https://engineering.case.edu/bearingdatacenter) does not publish a
single bulk archive — each condition (fault location, fault diameter, motor
load) is an individual ``.mat`` file linked from one of four category pages.
This script scrapes those four pages for their real download links (each
link is literally ``<a href="https://engineering.case.edu/sites/default/
files/<N>.mat">LABEL_load</a>``, e.g. ``IR007_0`` = inner race, 0.007",
0 hp), parses the label into structured metadata, and downloads every
linked file, writing a ``manifest.csv`` alongside them.

Usage:
    python scripts/download_cwru.py --output_dir data/raw/cwru
    python scripts/download_cwru.py --output_dir data/raw/cwru --dry_run
    python scripts/download_cwru.py --output_dir data/raw/cwru --categories normal
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd
import requests
from src.utils.io import ensure_dir
from src.utils.logging_config import get_logger
from tqdm import tqdm

logger = get_logger(__name__)

BASE = "https://engineering.case.edu/bearingdatacenter"

CATEGORY_PAGES = {
    "normal": f"{BASE}/normal-baseline-data",
    "12k_drive_end": f"{BASE}/12k-drive-end-bearing-fault-data",
    "48k_drive_end": f"{BASE}/48k-drive-end-bearing-fault-data",
    "12k_fan_end": f"{BASE}/12k-fan-end-bearing-fault-data",
}

SAMPLE_RATE_BY_CATEGORY = {
    "normal": 12000,
    "12k_drive_end": 12000,
    "48k_drive_end": 48000,
    "12k_fan_end": 12000,
}

FAULT_LOCATION_CODE = {"IR": "inner_race", "OR": "outer_race", "B": "ball", "Normal": "healthy"}

LINK_RE = re.compile(r'<a\s+href="(https://engineering\.case\.edu/sites/default/files/\d+\.mat)">([^<]+)</a>')
LABEL_RE = re.compile(r"^(?P<code>IR|OR|B|Normal)(?P<diam>\d{3})?(@(?P<orientation>\d+))?_(?P<load>\d)$")


def fetch_category_manifest(category: str, url: str) -> list[dict]:
    """Scrape one category page for its real file links and parse each label."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    rows = []
    for file_url, label in LINK_RE.findall(resp.text):
        m = LABEL_RE.match(label)
        if not m:
            logger.warning("Unrecognized label %r on %s, skipping", label, url)
            continue
        g = m.groupdict()
        rows.append(
            {
                "category": category,
                "label": label,
                "fault_location": FAULT_LOCATION_CODE[g["code"]],
                "fault_diameter_in": (int(g["diam"]) / 1000) if g["diam"] else 0.0,
                "orientation_deg": g["orientation"],
                "load_hp": int(g["load"]),
                "sample_rate_hz": SAMPLE_RATE_BY_CATEGORY[category],
                "url": file_url,
                "filename": f"{label}.mat",
            }
        )
    return rows


def build_manifest(categories: list[str]) -> pd.DataFrame:
    rows = []
    for category in categories:
        page_url = CATEGORY_PAGES[category]
        logger.info("Scraping %s (%s)", category, page_url)
        rows.extend(fetch_category_manifest(category, page_url))
        time.sleep(0.5)  # be polite to the server between requests
    df = pd.DataFrame(rows).drop_duplicates(subset="url").reset_index(drop=True)
    return df


def download_file(url: str, dest: Path, chunk_size: int = 1 << 16) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("Already downloaded, skipping: %s", dest.name)
        return
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
        ) as bar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                bar.update(len(chunk))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path("data/raw/cwru"))
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=list(CATEGORY_PAGES),
        default=list(CATEGORY_PAGES),
        help="Which CWRU categories to fetch.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Only build/print the manifest; don't download.")
    parser.add_argument("--limit", type=int, default=None, help="Download only the first N files (for testing).")
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)
    manifest = build_manifest(args.categories)
    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    logger.info("Manifest: %d files -> %s", len(manifest), manifest_path)

    if args.dry_run:
        logger.info("Dry run - not downloading. Preview:\n%s", manifest.head(10).to_string())
        return

    rows = manifest.itertuples()
    if args.limit:
        rows = list(manifest.itertuples())[: args.limit]

    for row in rows:
        dest_dir = ensure_dir(out_dir / row.category)
        download_file(row.url, dest_dir / row.filename)

    logger.info("Done. Real CWRU .mat files are under %s (gitignored).", out_dir)


if __name__ == "__main__":
    main()

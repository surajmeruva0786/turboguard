#!/usr/bin/env python
"""Download the real NASA IMS Bearing Dataset.

Source: NASA Prognostics Center of Excellence data repository. The dataset
is published as a single archive on S3:
https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip

Note: the archive is ~1 GB and unpacks through *three* nested layers, none
of which are documented on the source page: the outer ``.zip`` contains one
``4. Bearings/IMS.7z``, which itself contains ``1st_test.rar``,
``2nd_test.rar``, ``3rd_test.rar`` (plus the official PDF Readme) — only
after extracting *those* do you get the three run-to-failure directories of
1-second snapshot files recorded every ~10 minutes. This script downloads
and extracts all three layers, so ``--output_dir`` ends up with
``1st_test/``, ``2nd_test/``, ``3rd_test/`` directly. RAR extraction needs
an ``unrar``-compatible executable on PATH (or, on Windows, a WinRAR
install) — see :func:`_find_unrar_exe`. It does not run automatically as
part of the synthetic-data pipeline because of its size.

Usage:
    python scripts/download_ims.py --output_dir data/raw/ims
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import py7zr
import requests
from src.utils.io import ensure_dir
from src.utils.logging_config import get_logger
from tqdm import tqdm

logger = get_logger(__name__)

IMS_URL = "https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip"

_WINRAR_FALLBACKS = [
    r"C:\Program Files\WinRAR\UnRAR.exe",
    r"C:\Program Files (x86)\WinRAR\UnRAR.exe",
]


def _find_unrar_exe() -> str:
    exe = shutil.which("unrar") or shutil.which("unar") or shutil.which("bsdtar")
    if exe:
        return exe
    for candidate in _WINRAR_FALLBACKS:
        if Path(candidate).exists():
            return candidate
    raise RuntimeError(
        "No unrar-compatible tool found (tried 'unrar'/'unar'/'bsdtar' on PATH and "
        f"{_WINRAR_FALLBACKS}). Install one, or pass --skip_rar_extract and unpack "
        "the *.rar files under --output_dir manually."
    )


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


def extract_nested_archive(out_dir: Path) -> None:
    """The outer zip contains one nested ``*.7z`` (e.g. ``4. Bearings/IMS.7z``)
    holding the actual ``1st_test``/``2nd_test``/``3rd_test`` directories.
    Extract it, move those directories up to ``out_dir``, and clean up the
    now-empty intermediate wrapper directory."""
    nested = next(out_dir.rglob("*.7z"), None)
    if nested is None:
        return
    wrapper_dir = nested.parent
    logger.info("Extracting nested archive %s -> %s", nested, wrapper_dir)
    with py7zr.SevenZipFile(nested, mode="r") as archive:
        archive.extractall(path=wrapper_dir)
    nested.unlink()
    for child in wrapper_dir.iterdir():
        shutil.move(str(child), str(out_dir / child.name))
    wrapper_dir.rmdir()


def _flatten_test_dir(out_dir: Path, expected_name: str, before: set[str]) -> None:
    """NASA's ``3rd_test.rar`` quirk: it extracts to ``4th_test/txt/<files>``
    instead of ``3rd_test/<files>`` like the other two archives — the
    snapshot files themselves match Set No. 3 from the bundled Readme (same
    date range, 4 channels), it's purely a stale top-level folder name plus
    one extra level of nesting. Detect any newly-created top-level entry
    that isn't already ``expected_name`` and normalize it to
    ``out_dir/<expected_name>/<files>`` so all three test sets present the
    same flat-directory shape."""
    after = {p.name for p in out_dir.iterdir()}
    new_entries = after - before - {expected_name}
    if not new_entries:
        return
    (stray_name,) = new_entries
    stray_dir = out_dir / stray_name
    # descend through any single-child wrapper directories to the real files
    source = stray_dir
    while source.is_dir():
        children = list(source.iterdir())
        if len(children) == 1 and children[0].is_dir():
            source = children[0]
        else:
            break
    logger.info("Normalizing %s -> %s (upstream archive names this set %r)", source, out_dir / expected_name, stray_name)
    target = ensure_dir(out_dir / expected_name)
    for child in source.iterdir():
        shutil.move(str(child), str(target / child.name))
    shutil.rmtree(stray_dir)


def extract_rar_archives(out_dir: Path) -> None:
    """Extract each ``<N>_test.rar`` left by :func:`extract_nested_archive`
    into a same-named directory, then delete the (large, redundant) .rar."""
    rar_files = sorted(out_dir.glob("*.rar"))
    if not rar_files:
        return
    unrar_exe = _find_unrar_exe()
    for rar_path in rar_files:
        expected_name = rar_path.stem
        before = {p.name for p in out_dir.iterdir()}
        logger.info("Extracting %s (via %s)", rar_path, unrar_exe)
        subprocess.run([unrar_exe, "x", "-y", str(rar_path), str(out_dir) + os.sep], check=True)
        rar_path.unlink()
        _flatten_test_dir(out_dir, expected_name, before)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path("data/raw/ims"))
    parser.add_argument("--skip_extract", action="store_true")
    parser.add_argument(
        "--skip_rar_extract", action="store_true",
        help="Stop after the .7z layer, leaving <N>_test.rar unextracted (e.g. if no unrar tool is available).",
    )
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)
    archive_path = out_dir / "4.Bearings.zip"

    logger.info("Downloading IMS bearing dataset (~1 GB) from %s", IMS_URL)
    download_file(IMS_URL, archive_path)

    if not args.skip_extract:
        logger.info("Extracting %s -> %s", archive_path, out_dir)
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(out_dir)
        extract_nested_archive(out_dir)
        if not args.skip_rar_extract:
            extract_rar_archives(out_dir)

    logger.info("Done. Real IMS run-to-failure sets are under %s (gitignored).", out_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Acquire the pinned Kaggle input, or verify an existing copy without network use."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile


DATASET_HANDLE = "arshkon/linkedin-job-postings/versions/13"
SOURCE_FILE = "postings.csv"
EXPECTED_BYTES = 516_843_769
EXPECTED_SHA256 = "de33f27c8d1b226014e0cede003404e9b0acb66db718671fda9b3e3538eca6ea"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "raw" / SOURCE_FILE


def sha256_file(path: Path) -> str:
    """Hash the exact bytes, preserving CSV encoding and line endings."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_file(path: Path) -> None:
    """Reject any input that differs from the recovered canonical raw file."""
    if not path.is_file():
        raise ValueError(f"Input file does not exist or is not a regular file: {path}")
    actual_bytes = path.stat().st_size
    if actual_bytes != EXPECTED_BYTES:
        raise ValueError(
            f"Size mismatch for {path}: expected {EXPECTED_BYTES} bytes, "
            f"found {actual_bytes}. Keep this file separate and review its provenance."
        )
    actual_hash = sha256_file(path)
    if actual_hash != EXPECTED_SHA256:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {EXPECTED_SHA256}, "
            f"found {actual_hash}. Do not silently substitute a different input."
        )


def acquire(output: Path, *, verify_only: bool = False) -> str:
    """Validate before use; never overwrite an existing path with a download."""
    output = output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        validate_file(output)
        return f"Verified existing raw input: {output}"
    if verify_only:
        raise ValueError(f"No input to verify: {output}")

    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "Install the official client first: python -m pip install kagglehub"
        ) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="linkedin-data-") as staging_dir:
        downloaded = Path(
            kagglehub.dataset_download(
                DATASET_HANDLE,
                path=SOURCE_FILE,
                output_dir=staging_dir,
            )
        )
        validate_file(downloaded)

        # Exclusive creation guards against a file appearing during the download.
        # A failed copy removes only the partial file created by this invocation.
        created_output = False
        try:
            with output.open("xb") as destination:
                created_output = True
                with downloaded.open("rb") as source:
                    shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
                destination.flush()
                os.fsync(destination.fileno())
            validate_file(output)
        except BaseException:
            if created_output:
                output.unlink(missing_ok=True)
            raise

    return f"Downloaded and verified {DATASET_HANDLE}/{SOURCE_FILE}: {output}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination or existing raw CSV (default: data/raw/postings.csv).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only validate the file; do not import KaggleHub or contact the network.",
    )
    args = parser.parse_args(argv)
    try:
        message = acquire(args.output, verify_only=args.verify_only)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Data verification failed: {exc}", file=sys.stderr)
        return 1
    print(message)
    print(f"SHA-256: {EXPECTED_SHA256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

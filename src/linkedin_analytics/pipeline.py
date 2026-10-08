"""Command-line entry point for the complete source-to-results workflow."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from threadpoolctl import threadpool_limits

from .data import load_and_clean
from .modeling import fit_interpretation, run_cross_validation


def _json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def run(data_path, output_dir="results", config_path="config.json", *, smoke_rows=None,
        allow_unverified_source=False, skip_interpretation=False):
    started = time.monotonic()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
    print("Loading and checking the source data...", flush=True)
    frame, audit = load_and_clean(Path(data_path))
    if not audit["matches_original_v13_fingerprint"] and not allow_unverified_source:
        raise ValueError("Source fingerprint differs from the recovered Kaggle v13 file. "
                         "Inspect the new source, then explicitly use --allow-unverified-source.")
    if smoke_rows:
        if output.name != "smoke":
            raise ValueError("Smoke output directory must be named 'smoke' to protect full results.")
        frame = frame.sample(n=min(smoke_rows, len(frame)), random_state=config["seed"]).reset_index(drop=True)
        audit["smoke_sample_rows"] = len(frame)
    _json(output / "data_audit.json", audit)
    project_root = Path(config_path).resolve().parent
    processed = project_root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(processed / ("smoke.parquet" if smoke_rows else "analysis.parquet"), index=False)
    print(f"Eligible records: {len(frame):,}; raw records: {audit['raw_rows']:,}", flush=True)
    with threadpool_limits(limits=config.get("n_threads", 3)):
        result = run_cross_validation(frame, config=config, output_dir=output,
                                      progress=lambda value: print(value, flush=True))
        if not skip_interpretation:
            print("Fitting descriptive full-sample models...", flush=True)
            fit_interpretation(frame, config=config, output_dir=output, cv_result=result)
    from .validation import verify_results
    checks = verify_results(output)
    _json(output / "independent_validation.json", checks)
    from .plotting import build_figures
    build_figures(output)
    distributions = ["numpy", "pandas", "scipy", "scikit-learn", "glum", "xgboost",
                     "statsmodels", "matplotlib", "pyarrow", "threadpoolctl"]
    metadata = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "python": platform.python_version(), "platform": platform.system(),
        "versions": {package: importlib.metadata.version(package) for package in distributions},
        "source_csv_sha256": audit["raw_dataset_sha256"],
        "source_matches_pinned_v13": audit["matches_original_v13_fingerprint"],
        "config": result["config"],
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "smoke_sample": bool(smoke_rows),
        "notes": ["Public outputs omit full job descriptions and personal/local paths.",
                  "Cross-validation targets held-out postings, not future dates or unseen employers."],
    }
    _json(output / "run_metadata.json", metadata)
    print(result["summary"].to_string(index=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Pinned postings.csv or its original ZIP")
    parser.add_argument("--output", default="results")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--smoke-rows", type=int)
    parser.add_argument("--allow-unverified-source", action="store_true")
    parser.add_argument("--skip-interpretation", action="store_true")
    args = parser.parse_args()
    run(args.data, args.output, args.config, smoke_rows=args.smoke_rows,
        allow_unverified_source=args.allow_unverified_source,
        skip_interpretation=args.skip_interpretation)


if __name__ == "__main__":
    main()

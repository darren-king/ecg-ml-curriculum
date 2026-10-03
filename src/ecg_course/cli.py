"""Command-line entry point: ``ecg-course fetch | export | verify | all``.

Default paths are relative to the current working directory and the
documented commands are run from the repository root.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import wfdb

from ecg_course.example import EXAMPLE, ExampleSpec
from ecg_course.export import (
    compare_asset_with_record,
    read_json_asset,
    save_all_leads_figure,
    save_lead_figure,
    write_json_asset,
)
from ecg_course.fetch import fetch_example, record_path, sha256_of
from ecg_course.loading import EcgRecord, RecordIntegrityError, load_record

DEFAULT_DATA_DIR = Path("data/raw/ptb-xl-1.0.3")
DEFAULT_SITE_DIR = Path("site")


def asset_paths(site_dir: Path, spec: ExampleSpec = EXAMPLE) -> dict[str, Path]:
    public = site_dir / "public"
    data = public / "data" / f"ptb-xl-{spec.dataset_version}"
    figures = public / "figures"
    return {
        "json": data / f"{spec.record_name}.json",
        "header": data / f"{spec.record_name}.hea",
        "licence": data / Path(spec.licence_file).name,
        "manifest": data / "manifest.json",
        "lead_svg": figures / f"{spec.record_name}_lead_{spec.figure_lead}.svg",
        "all_png": figures / f"{spec.record_name}_all_leads.png",
    }


def check_header(record: EcgRecord, spec: ExampleSpec = EXAMPLE) -> list[str]:
    """Compare loaded header values with the values verified upstream."""
    problems = []
    if record.fs != spec.expected_fs_hz:
        problems.append(f"fs {record.fs} != {spec.expected_fs_hz}")
    if record.n_samples != spec.expected_n_samples:
        problems.append(f"n_samples {record.n_samples} != {spec.expected_n_samples}")
    if record.lead_names != spec.expected_lead_names:
        problems.append(f"lead names {record.lead_names} differ from expectation")
    if set(record.units) != {spec.expected_units}:
        problems.append(f"units {set(record.units)} != {{{spec.expected_units!r}}}")
    if set(record.adc_gain) != {spec.expected_adc_gain}:
        problems.append(f"adc gain {set(record.adc_gain)} != {spec.expected_adc_gain}")
    if set(record.baseline) != {spec.expected_baseline}:
        problems.append(f"baseline {set(record.baseline)} != {spec.expected_baseline}")
    if set(record.fmt) != {spec.expected_fmt}:
        problems.append(f"format {set(record.fmt)} != {spec.expected_fmt!r}")
    return problems


def check_source_files(data_dir: Path, spec: ExampleSpec = EXAMPLE) -> list[str]:
    """Compare the local source files with the upstream checksums."""
    problems = []
    for relative in spec.files:
        path = data_dir / relative
        if not path.exists():
            problems.append(f"source file missing: {path}")
        elif sha256_of(path) != spec.upstream_sha256[relative]:
            problems.append(f"source file does not match upstream sha256: {path}")
    return problems


def cmd_fetch(args: argparse.Namespace, spec: ExampleSpec = EXAMPLE) -> int:
    manifest = fetch_example(args.data_dir, spec, force=args.force)
    for item in manifest.files:
        state = "downloaded" if item.downloaded_now else "cached"
        print(f"{state:10s} {item.relative_path}  {item.bytes} bytes  sha256 ok")
    print(f"manifest   {Path(args.data_dir) / 'manifest.json'}")
    return 0


def cmd_export(args: argparse.Namespace, spec: ExampleSpec = EXAMPLE) -> int:
    data_dir = Path(args.data_dir)
    site_dir = Path(args.site_dir)
    fetch_manifest_path = data_dir / "manifest.json"
    if not fetch_manifest_path.exists():
        print(
            f"no manifest at {fetch_manifest_path}; run `ecg-course fetch` first",
            file=sys.stderr,
        )
        return 2
    source_problems = check_source_files(data_dir, spec)
    if source_problems:
        print("source files do not match upstream:", *source_problems, sep="\n  ")
        return 1
    record = load_record(record_path(data_dir, spec))
    header_problems = check_header(record, spec)
    if header_problems:
        print(
            "header does not match the verified example:", *header_problems, sep="\n  "
        )
        return 1
    paths = asset_paths(site_dir, spec)
    fetch_manifest = json.loads(fetch_manifest_path.read_text(encoding="utf-8"))
    provenance = {
        "dataset": fetch_manifest["dataset"],
        "dataset_version": fetch_manifest["dataset_version"],
        "dataset_page_url": fetch_manifest["dataset_page_url"],
        "doi": fetch_manifest["doi"],
        "record_id": fetch_manifest["record_id"],
        "licence": fetch_manifest["licence"]["name"],
        "licence_url": fetch_manifest["licence"]["url"],
        "source_sha256": {
            item["relative_path"]: item["sha256_upstream"]
            for item in fetch_manifest["files"]
        },
        "processing": (
            "Read with wfdb.rdrecord(physical=True); values exported unchanged, "
            "one array per lead in header order."
        ),
    }
    write_json_asset(record, paths["json"], provenance)
    paths["header"].parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(record_path(data_dir, spec).with_suffix(".hea"), paths["header"])
    shutil.copyfile(data_dir / spec.licence_file, paths["licence"])
    save_lead_figure(record, spec.figure_lead, paths["lead_svg"])
    save_all_leads_figure(record, paths["all_png"])
    derived = []
    for key in ("json", "header", "licence", "lead_svg", "all_png"):
        path = paths[key]
        derived.append(
            {
                "path": str(path.relative_to(site_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256_of(path),
            }
        )
    export_manifest: dict[str, Any] = {
        **fetch_manifest,
        "licence": {
            **fetch_manifest["licence"],
            "local_file": str(paths["licence"].relative_to(site_dir)),
        },
        "export_tool": (
            f"ecg_course.cli export on Python {platform.python_version()}, "
            f"wfdb {wfdb.__version__}, numpy {np.__version__}, "
            f"matplotlib {matplotlib.__version__}"
        ),
        "conversion_steps": [
            "wfdb.rdrecord(record, physical=True) applies each lead's gain and "
            "baseline from the header to give values in the header's units.",
            "JSON: one array per lead, header order, Python repr floats, no "
            "rounding, filtering, resampling or reordering.",
            "Header and licence: the downloaded .hea and LICENSE.txt files copied "
            "unchanged so the site carries them with the derived data.",
            f"SVG: lead {spec.figure_lead} against time in seconds via Matplotlib "
            "(Figure API, Agg/SVG backends, fixed svg.hashsalt, no date metadata).",
            "PNG: all leads stacked in header order with one shared amplitude "
            "range chosen as min/max over every lead plus 5 % padding.",
        ],
        "derived_assets": derived,
    }
    paths["manifest"].write_text(
        json.dumps(export_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for item in derived:
        digest = item["sha256"][:12]
        print(f"wrote {item['path']}  {item['bytes']} bytes  sha256 {digest}…")
    print(f"wrote {paths['manifest'].relative_to(site_dir)}")
    return 0


def cmd_verify(args: argparse.Namespace, spec: ExampleSpec = EXAMPLE) -> int:
    data_dir = Path(args.data_dir)
    site_dir = Path(args.site_dir)
    paths = asset_paths(site_dir, spec)
    failures: list[str] = []
    source_problems = check_source_files(data_dir, spec)
    print(
        "source files match upstream sha256"
        if not source_problems
        else "source files differ from upstream"
    )
    failures.extend(source_problems)
    record = load_record(record_path(data_dir, spec))
    print(
        f"loaded {record.record_name}: shape {record.signal.shape}, "
        f"fs {record.fs:g} Hz, leads {list(record.lead_names)}"
    )
    header_problems = check_header(record, spec)
    print(
        "header matches verified values" if not header_problems else "header mismatch"
    )
    failures.extend(header_problems)
    if not paths["json"].exists():
        failures.append(f"missing asset {paths['json']}")
    else:
        asset = read_json_asset(paths["json"])
        result = compare_asset_with_record(asset, record, tolerance=args.tolerance)
        print(
            f"json asset vs loaded signal: max |difference| = "
            f"{result.max_abs_difference} (tolerance {args.tolerance})"
        )
        failures.extend(result.problems)
    if paths["manifest"].exists():
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        for item in manifest.get("derived_assets", []):
            path = site_dir / item["path"]
            if not path.exists():
                failures.append(f"derived asset missing: {item['path']}")
            elif sha256_of(path) != item["sha256"]:
                failures.append(f"derived asset changed since export: {item['path']}")
            else:
                print(f"derived asset unchanged: {item['path']}")
    else:
        failures.append(f"missing export manifest {paths['manifest']}")
    if failures:
        print("VERIFY FAILED:", *failures, sep="\n  ")
        return 1
    print("VERIFY OK")
    return 0


def cmd_all(args: argparse.Namespace, spec: ExampleSpec = EXAMPLE) -> int:
    for step in (cmd_fetch, cmd_export, cmd_verify):
        code = step(args, spec)
        if code:
            return code
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ecg-course",
        description="Fetch, export and verify the course's single PTB-XL example.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--site-dir", type=Path, default=DEFAULT_SITE_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="download the record, licence and manifest")
    fetch.add_argument("--force", action="store_true", help="re-download cached files")
    fetch.set_defaults(func=cmd_fetch)
    export = sub.add_parser("export", help="write the JSON asset and figures")
    export.set_defaults(func=cmd_export)
    verify = sub.add_parser("verify", help="compare exported asset with the record")
    verify.add_argument("--tolerance", type=float, default=0.0)
    verify.set_defaults(func=cmd_verify)
    everything = sub.add_parser("all", help="fetch, export then verify")
    everything.add_argument("--force", action="store_true")
    everything.add_argument("--tolerance", type=float, default=0.0)
    everything.set_defaults(func=cmd_all)
    return parser


def main(argv: list[str] | None = None, spec: ExampleSpec = EXAMPLE) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args, spec)
    except (FileNotFoundError, RecordIntegrityError, OSError) as error:
        print(f"ecg-course {args.command}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

import json
import py_compile
import re
from pathlib import Path

import numpy as np

from ecg_course import load_record
from ecg_course.cli import check_header
from ecg_course.example import EXAMPLE, ExampleSpec
from tests.conftest import FS, LEADS, N_SAMPLES


def test_check_header_accepts_matching_values(synthetic_record_path):
    record = load_record(synthetic_record_path)
    spec = ExampleSpec(
        **{
            **EXAMPLE.__dict__,
            "expected_fs_hz": float(FS),
            "expected_n_samples": N_SAMPLES,
            "expected_lead_names": LEADS,
        }
    )
    assert check_header(record, spec) == []
    problems = check_header(record, EXAMPLE)
    assert any("fs" in p for p in problems)
    assert any("n_samples" in p for p in problems)
    assert any("lead names" in p for p in problems)


def test_lesson_script_compiles_and_has_balanced_snippet_markers():
    script = Path(__file__).resolve().parents[1] / "examples" / "first_ecg.py"
    py_compile.compile(str(script), doraise=True)
    text = script.read_text()
    starts = re.findall(r"^# --8<-- \[start:([\w-]+)\]$", text, flags=re.M)
    ends = re.findall(r"^# --8<-- \[end:([\w-]+)\]$", text, flags=re.M)
    assert starts == ends
    assert len(starts) == len(set(starts)) >= 6


def test_numpy_column_matches_lead_accessor(synthetic_record_path):
    record = load_record(synthetic_record_path)
    assert np.array_equal(record.signal[:, 1], record.lead("II"))


def _synthetic_spec(data_dir: Path, record: Path) -> ExampleSpec:
    """A spec pointing at a SYNTHETIC record written into ``data_dir``."""
    import hashlib

    (data_dir / "LICENSE.txt").write_text("synthetic licence text\n")
    files = (
        f"{record.relative_to(data_dir)}.hea",
        f"{record.relative_to(data_dir)}.dat",
        "LICENSE.txt",
    )
    return ExampleSpec(
        **{
            **EXAMPLE.__dict__,
            "record_id": str(record.relative_to(data_dir)),
            "files": files,
            "upstream_sha256": {
                name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
                for name in files
            },
            "expected_fs_hz": float(FS),
            "expected_n_samples": N_SAMPLES,
            "expected_lead_names": LEADS,
        }
    )


def test_export_then_verify_round_trip_and_tamper_detection(tmp_path, capsys):
    from ecg_course.cli import main
    from ecg_course.fetch import fetch_example
    from tests.conftest import write_synthetic_record

    data_dir = tmp_path / "data"
    rec_dir = data_dir / "rec"
    rec_dir.mkdir(parents=True)
    record = write_synthetic_record(rec_dir)
    spec = _synthetic_spec(data_dir, record)
    site_dir = tmp_path / "site"

    def no_network(url: str, destination: Path) -> None:
        raise AssertionError(f"unexpected download of {url}")

    fetch_example(data_dir, spec, downloader=no_network)  # all files cached
    base = ["--data-dir", str(data_dir), "--site-dir", str(site_dir)]
    assert main([*base, "export"], spec) == 0
    assert main([*base, "verify"], spec) == 0
    assert "VERIFY OK" in capsys.readouterr().out
    public = site_dir / "public"
    assert (public / "data" / "ptb-xl-1.0.3" / "LICENSE.txt").exists()
    manifest = json.loads(
        (public / "data" / "ptb-xl-1.0.3" / "manifest.json").read_text()
    )
    assert [d["path"] for d in manifest["derived_assets"]] == [
        "public/data/ptb-xl-1.0.3/synthetic_ramp.json",
        "public/data/ptb-xl-1.0.3/synthetic_ramp.hea",
        "public/data/ptb-xl-1.0.3/LICENSE.txt",
        "public/figures/synthetic_ramp_lead_II.svg",
        "public/figures/synthetic_ramp_all_leads.png",
    ]
    assert "matplotlib" in manifest["export_tool"]

    # Tamper with one exported value: verify must fail.
    asset_path = public / "data" / "ptb-xl-1.0.3" / "synthetic_ramp.json"
    asset = json.loads(asset_path.read_text())
    asset["leads"][0]["values"][5] += 0.001
    asset_path.write_text(json.dumps(asset))
    assert main([*base, "verify"], spec) == 1
    out = capsys.readouterr().out
    assert "VERIFY FAILED" in out and "changed since export" in out


def test_verify_detects_source_file_drift(tmp_path, capsys):
    from ecg_course.cli import main
    from ecg_course.fetch import fetch_example
    from tests.conftest import write_synthetic_record

    data_dir = tmp_path / "data"
    rec_dir = data_dir / "rec"
    rec_dir.mkdir(parents=True)
    record = write_synthetic_record(rec_dir)
    spec = _synthetic_spec(data_dir, record)
    fetch_example(data_dir, spec, downloader=lambda u, d: None)
    base = ["--data-dir", str(data_dir), "--site-dir", str(tmp_path / "site")]
    assert main([*base, "export"], spec) == 0
    with open(f"{record}.dat", "r+b") as handle:
        handle.seek(10)
        handle.write(b"\x01")
    assert main([*base, "verify"], spec) == 1
    assert "does not match upstream" in capsys.readouterr().out


def test_cli_reports_missing_data_dir_without_traceback(tmp_path, capsys):
    from ecg_course.cli import main

    code = main(["--data-dir", str(tmp_path / "nowhere"), "verify"])
    assert code == 1
    assert "ecg-course verify" in capsys.readouterr().err

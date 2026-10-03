"""Export a loaded record to a browser JSON asset and static figures.

The JSON asset stores exactly the values WFDB produced, one array per lead in
header order, serialised with Python's shortest round-trip float repr. No
rounding, filtering or resampling happens here. Figures are rendered with
Matplotlib's object-oriented API, which never opens a window.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from matplotlib.figure import Figure

from ecg_course.loading import EcgRecord
from ecg_course.timing import last_sample_time_s, nominal_duration_s, time_axis

ASSET_SCHEMA_VERSION = 1
SVG_HASHSALT = "ecg-course"


def shared_amplitude_range(
    record: EcgRecord, pad_fraction: float = 0.05
) -> tuple[float, float]:
    """One y-axis range covering every lead, padded so no trace touches an edge."""
    if not np.isfinite(record.signal).all():
        raise ValueError("amplitude range is undefined for non-finite values")
    low = float(np.min(record.signal))
    high = float(np.max(record.signal))
    span = high - low
    if span == 0:
        span = 1.0
    pad = span * pad_fraction
    return (low - pad, high + pad)


def record_to_asset(
    record: EcgRecord, provenance: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build the JSON-ready dictionary consumed by the browser viewer."""
    if not np.isfinite(record.signal).all():
        raise ValueError("refusing to export a signal containing non-finite values")
    leads = [
        {
            "name": name,
            "units": record.units[column],
            "adc_gain": record.adc_gain[column],
            "baseline": record.baseline[column],
            "fmt": record.fmt[column],
            "adc_res_bits": record.adc_res[column],
            "values": record.signal[:, column].tolist(),
        }
        for column, name in enumerate(record.lead_names)
    ]
    return {
        "schema_version": ASSET_SCHEMA_VERSION,
        "record_name": record.record_name,
        "fs_hz": record.fs,
        "n_samples": record.n_samples,
        "n_leads": record.n_leads,
        "nominal_duration_s": nominal_duration_s(record.n_samples, record.fs),
        "last_sample_time_s": last_sample_time_s(record.n_samples, record.fs),
        "time_formula": "time_s[i] = i / fs_hz",
        "python_array_shape": [record.n_samples, record.n_leads],
        "python_array_layout": "axis 0 = samples (time), axis 1 = leads",
        "lead_order": list(record.lead_names),
        "leads": leads,
        "provenance": provenance or {},
    }


def write_json_asset(
    record: EcgRecord, path: str | Path, provenance: dict[str, Any] | None = None
) -> Path:
    """Write the asset deterministically (compact, UTF-8, trailing newline)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        record_to_asset(record, provenance),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    target.write_text(text + "\n", encoding="utf-8")
    return target


def read_json_asset(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass
class ComparisonResult:
    """Outcome of comparing an exported asset with a freshly loaded record."""

    max_abs_difference: float
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def compare_asset_with_record(
    asset: dict[str, Any], record: EcgRecord, tolerance: float = 0.0
) -> ComparisonResult:
    """Check that ``asset`` reproduces ``record`` sample for sample.

    The default tolerance is zero because the export is a lossless float
    round-trip; a non-zero tolerance is accepted for other serialisations.
    """
    problems: list[str] = []
    if asset.get("fs_hz") != record.fs:
        problems.append(f"fs_hz {asset.get('fs_hz')} != {record.fs}")
    if asset.get("n_samples") != record.n_samples:
        problems.append(f"n_samples {asset.get('n_samples')} != {record.n_samples}")
    if asset.get("lead_order") != list(record.lead_names):
        problems.append(
            f"lead_order {asset.get('lead_order')} != {list(record.lead_names)}"
        )
    if asset.get("n_leads") != record.n_leads:
        problems.append(f"n_leads {asset.get('n_leads')} != {record.n_leads}")
    expected_duration = nominal_duration_s(record.n_samples, record.fs)
    if asset.get("nominal_duration_s") != expected_duration:
        problems.append(
            f"nominal_duration_s {asset.get('nominal_duration_s')} != "
            f"{expected_duration}"
        )
    leads = asset.get("leads", [])
    if len(leads) != record.n_leads:
        problems.append(f"{len(leads)} leads in asset, {record.n_leads} in record")
        return ComparisonResult(float("nan"), problems)
    max_diff = 0.0
    for column, lead in enumerate(leads):
        name = record.lead_names[column]
        if lead.get("name") != name:
            problems.append(f"lead {column} named {lead.get('name')!r}, not {name!r}")
        if lead.get("units") != record.units[column]:
            problems.append(f"lead {name}: units {lead.get('units')!r} differ")
        for key, expected in (
            ("adc_gain", record.adc_gain[column]),
            ("baseline", record.baseline[column]),
            ("fmt", record.fmt[column]),
        ):
            if lead.get(key) != expected:
                problems.append(f"lead {name}: {key} {lead.get(key)!r} != {expected!r}")
        values = np.asarray(lead.get("values", []), dtype=np.float64)
        if values.shape != (record.n_samples,):
            problems.append(
                f"lead {name}: {values.size} values, not {record.n_samples}"
            )
            continue
        diff = float(np.max(np.abs(values - record.signal[:, column])))
        max_diff = max(max_diff, diff)
        if diff > tolerance:
            problems.append(
                f"lead {name}: max |asset - record| = {diff} exceeds {tolerance}"
            )
    return ComparisonResult(max_diff, problems)


def _title(record: EcgRecord, detail: str) -> str:
    return f"{record.record_name}: {detail} (teaching plot, not calibrated ECG paper)"


def save_lead_figure(
    record: EcgRecord,
    lead_name: str,
    path: str | Path,
    amplitude_range: tuple[float, float] | None = None,
) -> Path:
    """Save a single lead against time in seconds. Format follows the suffix."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if lead_name not in record.lead_names:
        raise KeyError(
            f"lead {lead_name!r} not in record; available: {list(record.lead_names)}"
        )
    column = record.lead_names.index(lead_name)
    y_range = amplitude_range or shared_amplitude_range(record)
    fig = Figure(figsize=(10, 3.2), dpi=100)
    ax = fig.subplots()
    ax.plot(time_axis(record.n_samples, record.fs), record.signal[:, column], lw=0.8)
    ax.set_xlim(0, nominal_duration_s(record.n_samples, record.fs))
    ax.set_ylim(*y_range)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Amplitude ({record.units[column]})")
    ax.set_title(_title(record, f"lead {lead_name} at {record.fs:g} Hz"), fontsize=10)
    ax.grid(True, lw=0.4, alpha=0.5)
    fig.tight_layout()
    _save(fig, target)
    return target


def save_all_leads_figure(record: EcgRecord, path: str | Path) -> Path:
    """Save every lead, stacked in header order, sharing one amplitude scale."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    y_range = shared_amplitude_range(record)
    t = time_axis(record.n_samples, record.fs)
    fig = Figure(figsize=(10, 1.1 * record.n_leads + 0.8), dpi=100)
    axes = fig.subplots(record.n_leads, 1, sharex=True, sharey=True)
    for column, (ax, name) in enumerate(zip(axes, record.lead_names, strict=True)):
        ax.plot(t, record.signal[:, column], lw=0.6)
        ax.set_ylim(*y_range)
        ax.set_ylabel(
            f"{name}\n({record.units[column]})",
            rotation=0,
            ha="right",
            va="center",
            fontsize=8,
        )
        ax.grid(True, lw=0.3, alpha=0.4)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlim(0, nominal_duration_s(record.n_samples, record.fs))
    axes[-1].set_xlabel("Time (s)")
    units = sorted(set(record.units))
    fig.suptitle(
        _title(record, f"all {record.n_leads} leads, shared {'/'.join(units)} scale"),
        fontsize=10,
    )
    fig.tight_layout()
    _save(fig, target)
    return target


def _save(fig: Figure, target: Path) -> None:
    # Fixed hash salt and no date metadata keep repeated exports byte-identical.
    with matplotlib.rc_context({"svg.hashsalt": SVG_HASHSALT}):
        if target.suffix.lower() == ".svg":
            fig.savefig(target, format="svg", metadata={"Date": None})
        else:
            fig.savefig(target)

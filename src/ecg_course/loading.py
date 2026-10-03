"""Load a WFDB record in physical units and check that it is self-consistent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb


class RecordIntegrityError(ValueError):
    """Raised when a record's signal and metadata disagree or contain gaps."""


@dataclass(frozen=True)
class EcgRecord:
    """One multi-lead recording as read by WFDB.

    ``signal`` has shape ``(n_samples, n_leads)``: axis 0 is time (one row per
    sample) and axis 1 is the lead (one column per channel, in header order).
    Values are in the physical units named in ``units`` for each column.
    """

    record_name: str
    fs: float
    n_samples: int
    lead_names: tuple[str, ...]
    units: tuple[str, ...]
    adc_gain: tuple[float, ...]
    baseline: tuple[int, ...]
    fmt: tuple[str, ...]
    adc_res: tuple[int, ...]
    signal: np.ndarray
    comments: tuple[str, ...] = ()

    @property
    def n_leads(self) -> int:
        return len(self.lead_names)

    def lead(self, name: str) -> np.ndarray:
        """Return one lead's samples by its header name (exact match)."""
        try:
            column = self.lead_names.index(name)
        except ValueError:
            raise KeyError(
                f"lead {name!r} not in record; available: {list(self.lead_names)}"
            ) from None
        return self.signal[:, column]


def load_record(record_path: str | Path) -> EcgRecord:
    """Read ``record_path`` (without extension) with WFDB in physical units.

    WFDB reads the header (``.hea``) for metadata and the signal file(s) it
    names (here a single ``.dat``) for samples, then applies each channel's
    gain and baseline to give physical values. The result is validated with
    :func:`validate_record` before it is returned.
    """
    path = Path(record_path)
    if path.suffix in {".hea", ".dat"}:
        path = path.with_suffix("")
    record = wfdb.rdrecord(str(path), physical=True)
    if record.p_signal is None:
        raise RecordIntegrityError(f"{path}: WFDB returned no physical signal")
    ecg = EcgRecord(
        record_name=str(record.record_name),
        fs=float(record.fs),
        n_samples=int(record.sig_len),
        lead_names=tuple(str(name) for name in record.sig_name),
        units=tuple(str(unit) for unit in record.units),
        adc_gain=tuple(float(gain) for gain in record.adc_gain),
        baseline=tuple(int(value) for value in record.baseline),
        fmt=tuple(str(value) for value in record.fmt),
        adc_res=tuple(int(value) for value in record.adc_res),
        signal=np.asarray(record.p_signal, dtype=np.float64),
        comments=tuple(str(line) for line in (record.comments or [])),
    )
    validate_record(ecg)
    return ecg


def validate_record(record: EcgRecord) -> None:
    """Raise :class:`RecordIntegrityError` if the record is not self-consistent.

    Checks: the array is two-dimensional, its row count equals the header's
    sample count, its column count equals the number of named leads, every
    per-lead metadata tuple has one entry per lead, the sampling frequency is
    positive, lead names are unique and every sample is finite. Nothing is
    repaired here; a failure is reported so the learner can see it.
    """
    signal = record.signal
    problems: list[str] = []
    if signal.ndim != 2:
        problems.append(f"signal has {signal.ndim} dimensions, expected 2")
    else:
        rows, columns = signal.shape
        if rows != record.n_samples:
            problems.append(
                f"signal has {rows} rows but header says {record.n_samples} samples"
            )
        if columns != record.n_leads:
            problems.append(
                f"signal has {columns} columns but header names {record.n_leads} leads"
            )
        if not np.isfinite(signal).all():
            count = int(np.size(signal) - np.count_nonzero(np.isfinite(signal)))
            problems.append(
                f"{count} non-finite sample value(s); WFDB marks invalid or "
                "missing samples as NaN"
            )
    for field_name in ("units", "adc_gain", "baseline", "fmt", "adc_res"):
        if len(getattr(record, field_name)) != record.n_leads:
            problems.append(f"{field_name} does not have one entry per lead")
    if not record.fs > 0:
        problems.append(f"sampling frequency must be positive, got {record.fs}")
    if len(set(record.lead_names)) != record.n_leads:
        problems.append(f"lead names are not unique: {record.lead_names}")
    if problems:
        raise RecordIntegrityError(f"{record.record_name}: " + "; ".join(problems))

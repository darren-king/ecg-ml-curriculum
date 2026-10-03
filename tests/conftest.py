"""Offline test fixtures.

Every record created here is SYNTHETIC: small integer ramps and a sine wave
written with WFDB into a temporary directory. They mirror the PTB-XL storage
conventions (format 16, 1000 ADC units per mV, baseline 0) so that the
loading code is exercised realistically, but they are not ECGs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import wfdb

FS = 250
N_SAMPLES = 1000
LEADS = ("I", "II", "V1")


def write_synthetic_record(
    directory: Path,
    name: str = "synthetic_ramp",
    *,
    invalid_sample: tuple[int, int] | None = None,
) -> Path:
    """Write a 3-lead synthetic WFDB record and return its path (no suffix)."""
    stored = np.zeros((N_SAMPLES, len(LEADS)), dtype=np.int64)
    stored[:, 0] = np.arange(N_SAMPLES) - 500  # ramp: -0.5 mV .. +0.499 mV
    stored[:, 1] = np.round(800 * np.sin(2 * np.pi * 1.0 * np.arange(N_SAMPLES) / FS))
    stored[:, 2] = 250  # constant +0.25 mV
    if invalid_sample is not None:
        row, column = invalid_sample
        stored[row, column] = -32768  # WFDB format 16 invalid-sample marker
    wfdb.wrsamp(
        name,
        fs=FS,
        units=["mV"] * len(LEADS),
        sig_name=list(LEADS),
        d_signal=stored,
        fmt=["16"] * len(LEADS),
        adc_gain=[1000.0] * len(LEADS),
        baseline=[0] * len(LEADS),
        write_dir=str(directory),
    )
    return directory / name


@pytest.fixture
def synthetic_record_path(tmp_path: Path) -> Path:
    return write_synthetic_record(tmp_path)


@pytest.fixture
def synthetic_record_with_gap(tmp_path: Path) -> Path:
    return write_synthetic_record(tmp_path, "synthetic_gap", invalid_sample=(7, 1))

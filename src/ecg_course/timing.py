"""Turning sample indices into seconds, and back.

The conventions used throughout the course:

* ``time[i] = i / fs`` for sample index ``i`` and sampling frequency ``fs``.
* The nominal duration of a recording is ``n_samples / fs``.
* The timestamp of the final sample is ``(n_samples - 1) / fs``, one sample
  interval short of the nominal duration. That is a counting convention, not
  missing data.
"""

from __future__ import annotations

import numpy as np


def _check(n_samples: int, fs: float) -> None:
    if n_samples < 1:
        raise ValueError(f"n_samples must be at least 1, got {n_samples}")
    if not fs > 0:
        raise ValueError(f"fs must be a positive number of hertz, got {fs}")


def time_axis(n_samples: int, fs: float) -> np.ndarray:
    """Return the time in seconds of every sample: ``time[i] = i / fs``."""
    _check(n_samples, fs)
    return np.arange(n_samples, dtype=np.float64) / float(fs)


def nominal_duration_s(n_samples: int, fs: float) -> float:
    """Nominal recording length in seconds, ``n_samples / fs``."""
    _check(n_samples, fs)
    return n_samples / float(fs)


def last_sample_time_s(n_samples: int, fs: float) -> float:
    """Timestamp of the final sample, ``(n_samples - 1) / fs``."""
    _check(n_samples, fs)
    return (n_samples - 1) / float(fs)


def sample_index_for_time(time_s: float, fs: float, n_samples: int) -> int:
    """Nearest sample index for a time in seconds, ``round(time_s * fs)``.

    Python's ``round`` sends exact half-sample times to the even index. Times
    outside ``0`` to ``(n_samples - 1) / fs`` raise ``ValueError``.
    """
    _check(n_samples, fs)
    last = last_sample_time_s(n_samples, fs)
    if time_s < 0 or time_s > last:
        raise ValueError(f"{time_s} s is outside the recording (0 to {last} s)")
    return int(round(time_s * float(fs)))

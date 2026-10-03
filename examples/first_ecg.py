"""Your first ECG in Python: the runnable version of the lesson.

Run from the repository root after `ecg-course fetch`:

    uv run --project python python/examples/first_ecg.py

The lesson page shows these sections verbatim. Markers of the form
`# --8<-- [start:name]` / `# --8<-- [end:name]` delimit each snippet.
"""

from __future__ import annotations

from pathlib import Path

# --8<-- [start:load]
import numpy as np

from ecg_course import load_record, nominal_duration_s, time_axis

record = load_record("data/raw/ptb-xl-1.0.3/records500/00000/00001_hr")

print(record.record_name)  # 00001_hr
print(record.fs)  # 500.0  samples per second (Hz)
print(record.n_samples)  # 5000
print(record.lead_names)  # ('I', 'II', 'III', 'AVR', ..., 'V6')
print(record.units)  # ('mV', 'mV', ..., 'mV')
# --8<-- [end:load]

# --8<-- [start:shape]
print(record.signal.shape)  # (5000, 12): rows are samples, columns are leads
print(record.signal.dtype)  # float64, in the units above

lead_ii = record.signal[:, 1]  # column 1 is the second lead listed, 'II'
print(np.array_equal(lead_ii, record.lead("II")))  # True
print(lead_ii[:5])  # the first five samples of lead II, in mV
# --8<-- [end:shape]

# --8<-- [start:time]
t = time_axis(record.n_samples, record.fs)  # t[i] = i / fs
print(t[:3])  # [0.    0.002 0.004]
print(t[-1])  # 9.998, the final sample's timestamp: (n_samples - 1) / fs
print(nominal_duration_s(record.n_samples, record.fs))  # 10.0 = n_samples / fs
# --8<-- [end:time]

# --8<-- [start:adc]
import wfdb

# The header says each lead was stored as 16-bit integers at 1000 units per mV
# with a baseline of 0. Reading with physical=False gives those integers as
# they sit in the .dat file, and WFDB's conversion to millivolts is
#     physical = (stored - baseline) / gain
print(record.adc_gain[1], record.baseline[1], record.fmt[1])  # 1000.0 0 16
raw = wfdb.rdrecord("data/raw/ptb-xl-1.0.3/records500/00000/00001_hr", physical=False)
stored_ii = raw.d_signal[:, 1]  # the stored integers of lead II
print(stored_ii[:5])  # [-50 -50 -50 -50 -50]
converted = (stored_ii - record.baseline[1]) / record.adc_gain[1]
print(np.array_equal(converted, lead_ii))  # True
# --8<-- [end:adc]

# --8<-- [start:table]
print("index  time_s  II_mV")
for i in (0, 1, 2, 500, 2500, 4999):
    print(f"{i:5d}  {t[i]:6.3f}  {lead_ii[i]:+.3f}")
# --8<-- [end:table]

# --8<-- [start:plot]
from ecg_course import save_all_leads_figure, save_lead_figure

out = Path("output")
save_lead_figure(record, "II", out / "00001_hr_lead_II.svg")
save_all_leads_figure(record, out / "00001_hr_all_leads.png")
print(sorted(p.name for p in out.iterdir()))  # the two files just written
# --8<-- [end:plot]

# --8<-- [start:exercise-solution]
from ecg_course import sample_index_for_time

# Exercise: which sample index corresponds to 2.5 s, and what is lead V2 there?
i_2500ms = sample_index_for_time(2.5, record.fs, record.n_samples)
v2 = record.lead("V2")
print(i_2500ms)  # 1250, because 2.5 s * 500 Hz = 1250
print(t[i_2500ms])  # 2.5
print(v2[i_2500ms])  # the value of V2 at that instant, in mV

# Second part: the largest absolute amplitude in lead II and when it occurs.
peak_index = int(np.argmax(np.abs(lead_ii)))
print(peak_index, t[peak_index], lead_ii[peak_index])
# --8<-- [end:exercise-solution]

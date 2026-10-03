# ECG ML Curriculum: course code

This is the Python companion to the **ECG ML Curriculum** website, a practical
course for doctors and AI engineers who want to work with ECG waveforms in
Python. The website is where you read; this folder is where you run code.

Lesson 1, *Your first ECG in Python*, loads one real 12-lead recording from the
public PTB-XL dataset, inspects its metadata, turns sample indices into seconds
and plots it. Everything the lesson shows comes from `examples/first_ecg.py`
and the small `ecg_course` package in `src/`.

## What you need

- A terminal and about fifteen minutes.
- [uv](https://docs.astral.sh/uv/getting-started/installation/), which
  installs Python 3.12 for you if you do not have it.
- Git, only if you prefer `git clone` to downloading a ZIP.

## Quick start

```sh
git clone https://github.com/darren-king/ecg-ml-curriculum.git
cd ecg-ml-curriculum

uv sync                               # create .venv with wfdb, numpy, matplotlib
uv run ecg-course fetch               # download the one example record from PhysioNet
uv run ecg-course check               # confirm the files match PhysioNet's checksums and load
uv run python examples/first_ecg.py   # the lesson, section by section
uv run pytest                         # optional: offline tests on synthetic data
```

If you downloaded a ZIP instead of cloning, unzip it and `cd` into the folder
it creates before running the commands.

## What is in the folder

| Path | What it is |
| --- | --- |
| `pyproject.toml`, `uv.lock` | Project definition and locked dependencies for uv |
| `src/ecg_course/` | The package: `example` (the record's verified facts), `fetch`, `loading`, `timing`, `export`, `cli` |
| `examples/first_ecg.py` | The lesson script, shown verbatim on the website |
| `tests/` | Offline tests that use small synthetic WFDB files, never the network |
| `data/raw/ptb-xl-1.0.3/` | Created by `fetch`: the record's header and signal files, the dataset licence and a manifest of checksums |
| `output/` | Created by the lesson script: two figures |

## What `fetch` downloads

Three files from PhysioNet, about 135 kB in total, for PTB-XL version 1.0.3
record `records500/00000/00001_hr`: the WFDB header (`.hea`), the signal file
(`.dat`) and `LICENSE.txt`. Each file's SHA-256 is compared with the value
PhysioNet publishes in the dataset's `SHA256SUMS.txt`, and a manifest records
what was fetched, from where and when. The whole dataset is never downloaded.

## Licences and attribution

The example record and everything derived from it are covered by the dataset's
licence, Creative Commons Attribution 4.0 International, which `fetch` saves as
`data/raw/ptb-xl-1.0.3/LICENSE.txt`. The citations PhysioNet asks for are in
`src/ecg_course/example.py` and on the website's Sources and data page:

- Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T. (2022).
  PTB-XL, a large publicly available electrocardiography dataset (version 1.0.3).
  PhysioNet. https://doi.org/10.13026/kfzx-aw45
- Wagner, P., et al. (2020). PTB-XL: A Large Publicly Available ECG Dataset.
  Scientific Data. https://doi.org/10.1038/s41597-020-0495-6
- Pollard, T., et al. (2026). PhysioNet as a global platform for biomedical
  research. Nature Health. https://doi.org/10.1038/s44360-026-00096-z

No licence has yet been chosen for the course's own code and prose.

## Scope

This is teaching material about ECG data as numbers. It does not interpret the
recording, does not label it as normal or abnormal and produces nothing meant
to inform care.

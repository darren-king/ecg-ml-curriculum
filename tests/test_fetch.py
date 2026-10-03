import hashlib
import json
from pathlib import Path

import pytest

from ecg_course.example import EXAMPLE, ExampleSpec
from ecg_course.fetch import ChecksumMismatchError, fetch_example, sha256_of

CONTENT = {
    "rec/synthetic.hea": (
        b"synthetic 1 100 10\nsynthetic.dat 16 1000(0)/mV 16 0 0 0 0 I\n"
    ),
    "rec/synthetic.dat": bytes(range(20)),
    "LICENSE.txt": b"synthetic licence text\n",
}


def _spec(**overrides) -> ExampleSpec:
    values = {
        **EXAMPLE.__dict__,
        "files_base_url": "https://example.invalid/base/",
        "record_id": "rec/synthetic",
        "files": tuple(CONTENT),
        "upstream_sha256": {
            name: hashlib.sha256(data).hexdigest() for name, data in CONTENT.items()
        },
    }
    values.update(overrides)
    return ExampleSpec(**values)


def _fake_downloader(log: list[str]):
    def download(url: str, destination: Path) -> None:
        log.append(url)
        relative = url.removeprefix("https://example.invalid/base/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(CONTENT[relative])

    return download


def test_sha256_of_matches_hashlib(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"abc")
    assert sha256_of(path) == hashlib.sha256(b"abc").hexdigest()


def test_fetch_downloads_each_file_checks_hashes_and_writes_manifest(tmp_path):
    log: list[str] = []
    manifest = fetch_example(tmp_path, _spec(), downloader=_fake_downloader(log))
    assert log == [
        "https://example.invalid/base/rec/synthetic.hea",
        "https://example.invalid/base/rec/synthetic.dat",
        "https://example.invalid/base/LICENSE.txt",
    ]
    assert all(item.matches_upstream and item.downloaded_now for item in manifest.files)
    written = json.loads((tmp_path / "manifest.json").read_text())
    assert written["record_id"] == "rec/synthetic"
    assert written["licence"]["name"].startswith("Creative Commons Attribution 4.0")
    assert len(written["citations"]) == 3
    assert [f["relative_path"] for f in written["files"]] == list(CONTENT)
    assert "SHA256SUMS.txt" in written["checksum_note"]


def test_fetch_reuses_cached_files_that_match(tmp_path):
    spec = _spec()
    fetch_example(tmp_path, spec, downloader=_fake_downloader([]))
    log: list[str] = []
    manifest = fetch_example(tmp_path, spec, downloader=_fake_downloader(log))
    assert log == []
    assert not any(item.downloaded_now for item in manifest.files)
    log_forced: list[str] = []
    fetch_example(tmp_path, spec, force=True, downloader=_fake_downloader(log_forced))
    assert len(log_forced) == 3


def test_fetch_raises_on_checksum_mismatch(tmp_path):
    bad = dict(_spec().upstream_sha256)
    bad["rec/synthetic.dat"] = "0" * 64
    with pytest.raises(ChecksumMismatchError, match="synthetic.dat"):
        fetch_example(
            tmp_path, _spec(upstream_sha256=bad), downloader=_fake_downloader([])
        )


def test_real_example_spec_is_internally_consistent():
    assert EXAMPLE.record_name == "00001_hr"
    assert set(EXAMPLE.upstream_sha256) == set(EXAMPLE.files)
    assert all(len(value) == 64 for value in EXAMPLE.upstream_sha256.values())
    assert EXAMPLE.url_for("LICENSE.txt").startswith("https://physionet.org/files/")
    assert len(EXAMPLE.expected_lead_names) == 12


def test_cached_fetch_keeps_retrieval_time_and_relative_paths(tmp_path):
    spec = _spec()
    first = fetch_example(tmp_path, spec, downloader=_fake_downloader([]))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["retrieved_at_utc"] = "2000-01-01T00:00:00+00:00"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    second = fetch_example(tmp_path, spec, downloader=_fake_downloader([]))
    assert first.files[0].local_path == "rec/synthetic.hea"
    assert second.retrieved_at_utc == "2000-01-01T00:00:00+00:00"
    assert second.checked_at_utc >= first.checked_at_utc
    forced = fetch_example(tmp_path, spec, force=True, downloader=_fake_downloader([]))
    assert forced.retrieved_at_utc != "2000-01-01T00:00:00+00:00"

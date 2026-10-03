"""Download the example record's files one by one and record their provenance.

Only the files named in :data:`ecg_course.example.EXAMPLE` are fetched: the
WFDB header, the signal file and the dataset licence. Each download is hashed
locally and compared with the checksum PhysioNet publishes for that file.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import wfdb

from ecg_course.example import EXAMPLE, ExampleSpec

Downloader = Callable[[str, Path], None]
USER_AGENT = "ecg-course/0.1 (+https://physionet.org/content/ptb-xl/1.0.3/)"


class ChecksumMismatchError(RuntimeError):
    """A downloaded file does not match the checksum published upstream."""


@dataclass(frozen=True)
class FetchedFile:
    relative_path: str
    url: str
    local_path: str
    bytes: int
    sha256_local: str
    sha256_upstream: str
    matches_upstream: bool
    downloaded_now: bool


@dataclass(frozen=True)
class FetchManifest:
    dataset: str
    dataset_version: str
    dataset_page_url: str
    doi: str
    record_id: str
    retrieved_at_utc: str
    checked_at_utc: str
    retrieval_tool: str
    files: tuple[FetchedFile, ...]
    checksum_note: str
    licence: dict[str, str]
    citations: tuple[str, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_of(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, timeout: float = 60.0) -> None:
    """Stream ``url`` to ``destination`` via a temporary file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with (
        urllib.request.urlopen(request, timeout=timeout) as response,
        open(temporary, "wb") as handle,
    ):
        shutil.copyfileobj(response, handle)
    temporary.replace(destination)


def fetch_example(
    data_dir: str | Path,
    spec: ExampleSpec = EXAMPLE,
    *,
    force: bool = False,
    downloader: Downloader = download,
) -> FetchManifest:
    """Ensure every example file exists locally and matches its upstream hash.

    Files already present with the right checksum are kept unless ``force`` is
    set. A file whose checksum differs after download is left in place for
    inspection and :class:`ChecksumMismatchError` is raised.
    """
    root = Path(data_dir)
    previous = _previous_retrieval_time(root / "manifest.json")
    fetched: list[FetchedFile] = []
    for relative in spec.files:
        local = root / relative
        expected = spec.upstream_sha256[relative]
        downloaded_now = False
        if force or not local.exists() or sha256_of(local) != expected:
            downloader(spec.url_for(relative), local)
            downloaded_now = True
        actual = sha256_of(local)
        if actual != expected:
            raise ChecksumMismatchError(
                f"{relative}: local sha256 {actual} does not match upstream "
                f"{expected}; the file was left at {local} for inspection"
            )
        fetched.append(
            FetchedFile(
                relative_path=relative,
                url=spec.url_for(relative),
                local_path=relative,
                bytes=local.stat().st_size,
                sha256_local=actual,
                sha256_upstream=expected,
                matches_upstream=True,
                downloaded_now=downloaded_now,
            )
        )
    now = datetime.now(UTC).isoformat(timespec="seconds")
    manifest = FetchManifest(
        dataset=spec.dataset_name,
        dataset_version=spec.dataset_version,
        dataset_page_url=spec.dataset_page_url,
        doi=spec.doi,
        record_id=spec.record_id,
        retrieved_at_utc=(
            now
            if any(item.downloaded_now for item in fetched) or not previous
            else previous
        ),
        checked_at_utc=now,
        retrieval_tool=(
            f"ecg_course.fetch (urllib) on Python {platform.python_version()}, "
            f"wfdb {wfdb.__version__}, {sys.platform}"
        ),
        files=tuple(fetched),
        checksum_note=(
            "sha256_local was computed here after download and compared with "
            f"sha256_upstream from {spec.upstream_checksums_source}. local_path "
            "is relative to the data directory. retrieved_at_utc is the time of "
            "the last download; checked_at_utc is the time of the last checksum "
            "check."
        ),
        licence={
            "name": spec.licence_name,
            "url": spec.licence_url,
            "local_file": spec.licence_file,
        },
        citations=spec.citations,
        notes=spec.notes,
    )
    write_manifest(manifest, root / "manifest.json")
    return manifest


def _previous_retrieval_time(manifest_path: Path) -> str | None:
    """Return ``retrieved_at_utc`` from an earlier manifest, if one exists."""
    if not manifest_path.exists():
        return None
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    retrieved = value.get("retrieved_at_utc") if isinstance(value, dict) else None
    return retrieved if isinstance(retrieved, str) else None


def write_manifest(manifest: FetchManifest, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


def record_path(data_dir: str | Path, spec: ExampleSpec = EXAMPLE) -> Path:
    """Path to the record (without extension) inside ``data_dir``."""
    return Path(data_dir) / spec.record_id

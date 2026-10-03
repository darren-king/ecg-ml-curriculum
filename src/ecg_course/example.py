"""Facts about the single public example record used by the course.

Every value below was checked against the upstream PhysioNet pages and files on
3 October 2026. ``UPSTREAM_SHA256`` is copied from the dataset's own
``SHA256SUMS.txt`` so that a local download can be compared with what PhysioNet
publishes, which is a genuine upstream comparison rather than a self-check.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExampleSpec:
    """Identity, provenance and expected header values for one record."""

    dataset_name: str
    dataset_version: str
    dataset_page_url: str
    files_base_url: str
    doi: str
    record_id: str
    files: tuple[str, ...]
    upstream_sha256: dict[str, str]
    upstream_checksums_source: str
    licence_name: str
    licence_url: str
    licence_file: str
    citations: tuple[str, ...]
    expected_fs_hz: float
    expected_n_samples: int
    expected_lead_names: tuple[str, ...]
    expected_units: str
    expected_adc_gain: float
    expected_baseline: int
    expected_fmt: str
    figure_lead: str = "II"
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def record_name(self) -> str:
        """The WFDB record name without its directory, e.g. ``00001_hr``."""
        return self.record_id.rsplit("/", 1)[-1]

    def url_for(self, relative_file: str) -> str:
        return self.files_base_url + relative_file


EXAMPLE = ExampleSpec(
    dataset_name="PTB-XL, a large publicly available electrocardiography dataset",
    dataset_version="1.0.3",
    dataset_page_url="https://physionet.org/content/ptb-xl/1.0.3/",
    files_base_url="https://physionet.org/files/ptb-xl/1.0.3/",
    doi="10.13026/kfzx-aw45",
    record_id="records500/00000/00001_hr",
    files=(
        "records500/00000/00001_hr.hea",
        "records500/00000/00001_hr.dat",
        "LICENSE.txt",
    ),
    upstream_sha256={
        "records500/00000/00001_hr.hea": (
            "c0ca9920530399329471389871e18828fb22f9eeea60932dccf077f33750c788"
        ),
        "records500/00000/00001_hr.dat": (
            "0797d0a8c43d5cc05bb0a73026a0f6d9f358b459849674ebd38abd9120241bcf"
        ),
        "LICENSE.txt": (
            "9a78e7f22742dde9f66ae235ec793ba2212019dc4d0ced75c4da09ced0b35fb2"
        ),
    },
    upstream_checksums_source=(
        "https://physionet.org/files/ptb-xl/1.0.3/SHA256SUMS.txt "
        "(lines copied on 2026-10-03)"
    ),
    licence_name="Creative Commons Attribution 4.0 International Public License",
    licence_url="https://physionet.org/content/ptb-xl/view-license/1.0.3/",
    licence_file="LICENSE.txt",
    citations=(
        "Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T. "
        "(2022). PTB-XL, a large publicly available electrocardiography dataset "
        "(version 1.0.3). PhysioNet. RRID:SCR_007345. "
        "https://doi.org/10.13026/kfzx-aw45",
        "Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, "
        "F.I., Samek, W., Schaeffter, T. (2020), PTB-XL: A Large Publicly "
        "Available ECG Dataset. Scientific Data. "
        "https://doi.org/10.1038/s41597-020-0495-6",
        "Pollard, T., Moody, B. E., Lehman, L., Gow, B., Fernandes, C., Xie, C., "
        "Johnson, A., Mark, R. G., & Heldt, T. (2026). PhysioNet as a global "
        "platform for biomedical research. Nature Health. "
        "https://doi.org/10.1038/s44360-026-00096-z",
    ),
    expected_fs_hz=500.0,
    expected_n_samples=5000,
    expected_lead_names=(
        "I",
        "II",
        "III",
        "AVR",
        "AVL",
        "AVF",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    ),
    expected_units="mV",
    expected_adc_gain=1000.0,
    expected_baseline=0,
    expected_fmt="16",
    notes=(
        "Only the header, signal file and licence are downloaded. The per-record "
        "clinical metadata table (ptbxl_database.csv) is not fetched and no "
        "diagnostic label is attached to this example.",
        "Lead names are reproduced exactly as written in the header (AVR, AVL, "
        "AVF in upper case).",
    ),
)

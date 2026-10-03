import numpy as np
import pytest

from ecg_course import EcgRecord, RecordIntegrityError, load_record
from ecg_course.loading import validate_record
from tests.conftest import FS, LEADS, N_SAMPLES


def test_loads_shape_metadata_and_physical_values(synthetic_record_path):
    record = load_record(synthetic_record_path)
    assert record.signal.shape == (N_SAMPLES, len(LEADS))
    assert record.n_samples == N_SAMPLES
    assert record.n_leads == len(LEADS)
    assert record.fs == FS
    assert record.lead_names == LEADS  # header order preserved
    assert record.units == ("mV",) * 3
    assert record.adc_gain == (1000.0,) * 3
    assert record.baseline == (0,) * 3
    assert record.fmt == ("16",) * 3
    # physical = (stored - baseline) / gain
    assert record.signal[0, 0] == pytest.approx(-0.5)
    assert record.signal[999, 0] == pytest.approx(0.499)
    np.testing.assert_allclose(record.signal[:, 2], 0.25)
    assert record.signal.dtype == np.float64


def test_lead_lookup_by_exact_name(synthetic_record_path):
    record = load_record(synthetic_record_path)
    np.testing.assert_array_equal(record.lead("V1"), record.signal[:, 2])
    with pytest.raises(KeyError):
        record.lead("v1")


def test_accepts_path_with_suffix(synthetic_record_path):
    record = load_record(str(synthetic_record_path) + ".hea")
    assert record.record_name == synthetic_record_path.name


def test_invalid_sample_is_reported_not_repaired(synthetic_record_with_gap):
    with pytest.raises(RecordIntegrityError, match="non-finite"):
        load_record(synthetic_record_with_gap)


def _record(**overrides) -> EcgRecord:
    base = dict(
        record_name="synthetic",
        fs=100.0,
        n_samples=4,
        lead_names=("I", "II"),
        units=("mV", "mV"),
        adc_gain=(1000.0, 1000.0),
        baseline=(0, 0),
        fmt=("16", "16"),
        adc_res=(16, 16),
        signal=np.zeros((4, 2)),
    )
    base.update(overrides)
    return EcgRecord(**base)


def test_validate_detects_mismatched_sample_count():
    with pytest.raises(RecordIntegrityError, match="rows"):
        validate_record(_record(n_samples=5))


def test_validate_detects_mismatched_lead_count():
    with pytest.raises(RecordIntegrityError, match="columns"):
        validate_record(_record(lead_names=("I",), units=("mV",)))


def test_validate_detects_duplicate_lead_names():
    with pytest.raises(RecordIntegrityError, match="unique"):
        validate_record(_record(lead_names=("I", "I")))


def test_validate_detects_bad_sampling_frequency():
    with pytest.raises(RecordIntegrityError, match="positive"):
        validate_record(_record(fs=0.0))


def test_non_canonical_lead_order_is_preserved(tmp_path):
    import wfdb

    stored = np.zeros((20, 2), dtype=np.int64)
    stored[:, 0] = 100  # "V1" first on purpose
    stored[:, 1] = -200
    wfdb.wrsamp(
        "synthetic_order",
        fs=100,
        units=["mV", "mV"],
        sig_name=["V1", "I"],
        d_signal=stored,
        fmt=["16", "16"],
        adc_gain=[1000.0, 1000.0],
        baseline=[0, 0],
        write_dir=str(tmp_path),
    )
    record = load_record(tmp_path / "synthetic_order")
    assert record.lead_names == ("V1", "I")
    assert record.signal[0, 0] == pytest.approx(0.1)
    assert record.signal[0, 1] == pytest.approx(-0.2)

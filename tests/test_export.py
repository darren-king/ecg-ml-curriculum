import json

import numpy as np
import pytest

from ecg_course import (
    compare_asset_with_record,
    load_record,
    record_to_asset,
    save_all_leads_figure,
    save_lead_figure,
    write_json_asset,
)
from ecg_course.export import read_json_asset, shared_amplitude_range


def test_asset_keeps_order_units_and_exact_values(synthetic_record_path):
    record = load_record(synthetic_record_path)
    asset = record_to_asset(record, {"dataset": "synthetic"})
    assert asset["schema_version"] == 2
    assert asset["amplitude_range"] == [-1.0, 1.0]
    assert asset["lead_order"] == ["I", "II", "V1"]
    assert [lead["name"] for lead in asset["leads"]] == ["I", "II", "V1"]
    assert asset["fs_hz"] == 250.0
    assert asset["n_samples"] == 1000
    assert asset["python_array_shape"] == [1000, 3]
    assert asset["nominal_duration_s"] == 4.0
    assert asset["last_sample_time_s"] == pytest.approx(3.996)
    for column, lead in enumerate(asset["leads"]):
        assert lead["units"] == "mV"
        assert lead["values"] == record.signal[:, column].tolist()
    assert asset["provenance"] == {"dataset": "synthetic"}


def test_json_round_trip_is_lossless_and_deterministic(synthetic_record_path, tmp_path):
    record = load_record(synthetic_record_path)
    first = write_json_asset(record, tmp_path / "a.json")
    second = write_json_asset(record, tmp_path / "b.json")
    assert first.read_bytes() == second.read_bytes()
    asset = read_json_asset(first)
    result = compare_asset_with_record(asset, record)
    assert result.ok, result.problems
    assert result.max_abs_difference == 0.0
    text = first.read_text()
    assert "NaN" not in text and text.endswith("\n")


def test_comparison_reports_tampered_values(synthetic_record_path, tmp_path):
    record = load_record(synthetic_record_path)
    path = write_json_asset(record, tmp_path / "a.json")
    asset = json.loads(path.read_text())
    asset["leads"][1]["values"][10] += 0.001
    asset["leads"][2]["name"] = "V2"
    result = compare_asset_with_record(asset, record)
    assert not result.ok
    assert any("lead II" in p for p in result.problems)
    assert any("named 'V2'" in p for p in result.problems)
    assert result.max_abs_difference == pytest.approx(0.001)


def test_comparison_reports_shape_and_metadata_drift(synthetic_record_path, tmp_path):
    record = load_record(synthetic_record_path)
    asset = read_json_asset(write_json_asset(record, tmp_path / "a.json"))
    asset["fs_hz"] = 500
    asset["amplitude_range"] = [-2.0, 2.0]
    asset["leads"][0]["values"].pop()
    result = compare_asset_with_record(asset, record)
    assert any("fs_hz" in p for p in result.problems)
    assert any("amplitude_range" in p for p in result.problems)
    assert any("999 values" in p for p in result.problems)


def test_export_refuses_non_finite_values(synthetic_record_path):
    record = load_record(synthetic_record_path)
    record.signal[3, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        record_to_asset(record)


def test_shared_amplitude_range_is_symmetric_and_tidy(synthetic_record_path):
    record = load_record(synthetic_record_path)  # peak |value| is 0.8 mV
    low, high = shared_amplitude_range(record)
    assert (low, high) == (-1.0, 1.0)
    assert high > np.abs(record.signal).max()
    assert (high / 0.5).is_integer()


@pytest.mark.parametrize(
    ("peak", "limit"),
    [(1.444, 1.5), (1.5, 2.0), (-0.49, 0.5), (0.0, 0.5), (2.0, 2.5)],
)
def test_shared_amplitude_range_rule(synthetic_record_path, peak, limit):
    record = load_record(synthetic_record_path)
    record.signal[:] = 0.0
    record.signal[0, 0] = peak
    assert shared_amplitude_range(record) == (-limit, limit)


def test_figures_are_written_and_svg_is_deterministic(synthetic_record_path, tmp_path):
    record = load_record(synthetic_record_path)
    svg_a = save_lead_figure(record, "II", tmp_path / "a" / "lead.svg")
    svg_b = save_lead_figure(record, "II", tmp_path / "b" / "lead.svg")
    png = save_all_leads_figure(record, tmp_path / "all.png")
    assert svg_a.stat().st_size > 1000 and png.stat().st_size > 1000
    assert svg_a.read_bytes() == svg_b.read_bytes()
    text = svg_a.read_text()
    assert "Time (s)" in text and "Amplitude (mV)" in text
    assert "dc:date" not in text

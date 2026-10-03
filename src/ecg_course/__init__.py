"""ECG ML Curriculum companion package.

Everything here works on one explicitly downloaded public PTB-XL record. The
functions are small and documented so that the lesson text can show them
directly. Nothing in this package filters, resamples, rescales or reorders the
signal: what WFDB reads in physical units is what gets exported.
"""

from ecg_course.example import EXAMPLE, ExampleSpec
from ecg_course.export import (
    compare_asset_with_record,
    record_to_asset,
    save_all_leads_figure,
    save_lead_figure,
    write_json_asset,
)
from ecg_course.loading import EcgRecord, RecordIntegrityError, load_record
from ecg_course.timing import (
    last_sample_time_s,
    nominal_duration_s,
    sample_index_for_time,
    time_axis,
)

__all__ = [
    "EXAMPLE",
    "EcgRecord",
    "ExampleSpec",
    "RecordIntegrityError",
    "compare_asset_with_record",
    "last_sample_time_s",
    "load_record",
    "nominal_duration_s",
    "record_to_asset",
    "sample_index_for_time",
    "save_all_leads_figure",
    "save_lead_figure",
    "time_axis",
    "write_json_asset",
]

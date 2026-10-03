import numpy as np
import pytest

from ecg_course import (
    last_sample_time_s,
    nominal_duration_s,
    sample_index_for_time,
    time_axis,
)


def test_time_axis_follows_index_over_fs():
    t = time_axis(5000, 500)
    assert t.shape == (5000,)
    assert t[0] == 0.0
    assert t[1] == pytest.approx(0.002)
    assert t[-1] == pytest.approx(9.998)
    np.testing.assert_allclose(t, np.arange(5000) / 500)


def test_duration_and_last_sample_differ_by_one_interval():
    assert nominal_duration_s(5000, 500) == 10.0
    assert last_sample_time_s(5000, 500) == pytest.approx(9.998)
    assert nominal_duration_s(5000, 500) - last_sample_time_s(5000, 500) == (
        pytest.approx(1 / 500)
    )


def test_sample_index_for_time_rounds_to_nearest():
    assert sample_index_for_time(2.5, 500, 5000) == 1250
    assert sample_index_for_time(0.0, 500, 5000) == 0
    assert sample_index_for_time(9.998, 500, 5000) == 4999
    assert sample_index_for_time(0.003, 500, 5000) == 2  # exact half rounds to even
    with pytest.raises(ValueError):
        sample_index_for_time(10.0, 500, 5000)
    with pytest.raises(ValueError):
        sample_index_for_time(9.999, 500, 5000)
    with pytest.raises(ValueError):
        sample_index_for_time(-0.001, 500, 5000)


@pytest.mark.parametrize("bad", [(0, 500), (10, 0), (10, -1.0)])
def test_rejects_degenerate_inputs(bad):
    with pytest.raises(ValueError):
        time_axis(*bad)

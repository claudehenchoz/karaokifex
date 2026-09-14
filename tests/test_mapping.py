import numpy as np
import pytest

from karaokifex.mapping import TimeMap, fit_time_map, xcorr_offset

HOP = 0.02


def test_xcorr_finds_the_intro_offset():
    starts = [5.0, 9.0, 14.5, 20.0, 31.0, 36.0]
    envelope = np.zeros(3000)
    for start in starts:
        envelope[round((start + 12.0) / HOP)] = 10.0
    envelope[500] = 3.0  # an ad-lib that isn't a line start
    assert xcorr_offset(starts, envelope, HOP) == pytest.approx(12.0, abs=HOP)


def test_xcorr_needs_enough_lines_and_signal():
    assert xcorr_offset([1.0, 2.0], np.ones(100), HOP) is None
    assert xcorr_offset([1.0, 2.0, 3.0, None], np.zeros(100), HOP) is None


def test_fit_recovers_speed_change_despite_outliers():
    anchors = [(t, 1.02 * t + 5.0) for t in range(10, 200, 10)]
    anchors[3] = (40.0, 90.0)  # a wrong match
    anchors[7] = (80.0, 20.0)
    mapping = fit_time_map(anchors)
    assert mapping.scale == pytest.approx(1.02, abs=1e-3)
    assert mapping.offset == pytest.approx(5.0, abs=0.05)
    assert mapping.corrections == ()


def test_few_anchors_fall_back_to_the_median_or_prior():
    assert fit_time_map([(1.0, 13.0), (5.0, 17.1), (9.0, 21.0)]).offset == pytest.approx(12.0, abs=0.1)
    assert fit_time_map([(1.0, 13.0)], prior_offset=11.5) == TimeMap(1.0, 11.5)


def test_cut_verse_becomes_a_section_correction():
    # The video drops 20 s of the record between lrclib 100 s and 120 s.
    anchors = [(t, t + 3.0) for t in range(10, 100, 10)] + [(t, t + 3.0 - 20.0) for t in range(120, 200, 10)]
    mapping = fit_time_map(anchors)
    assert mapping.map(50.0) == pytest.approx(53.0, abs=0.2)
    assert mapping.map(150.0) == pytest.approx(133.0, abs=0.2)
    assert len(mapping.corrections) == 1


def test_roundtrip():
    mapping = TimeMap(1.01, -2.0, ((100.0, -20.0),))
    assert TimeMap.from_dict(mapping.to_dict()) == mapping
    assert "sections" in mapping.describe()

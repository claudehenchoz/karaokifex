import numpy as np
import pytest

from karaokifex.activity import Activity

RATE = 8000


def signal(duration: float, *voiced: tuple[float, float]) -> np.ndarray:
    rng = np.random.default_rng(0)
    samples = rng.normal(0, 1e-4, round(duration * RATE))
    t = np.arange(len(samples)) / RATE
    for start, end in voiced:
        mask = (t >= start) & (t < end)
        samples[mask] += 0.3 * np.sin(2 * np.pi * 220 * t[mask])
    return samples


@pytest.fixture
def activity():
    # A phrase from 1.0 to 2.0 s with a 50 ms breath inside, and a 40 ms click at 2.5 s.
    return Activity.compute(signal(4.0, (1.0, 1.5), (1.55, 2.0), (2.5, 2.54)), RATE)


def test_voiced_intervals_fill_breaths_and_drop_clicks(activity):
    [(start, end)] = activity.voiced_intervals(0.0, 4.0)
    assert start == pytest.approx(1.0, abs=0.03)
    assert end == pytest.approx(2.0, abs=0.03)


def test_queries(activity):
    assert activity.voiced_fraction(1.2, 1.8) == 1.0
    assert activity.voiced_fraction(0.0, 0.9) == 0.0
    assert activity.is_voiced(1.5) and not activity.is_voiced(3.0)
    assert activity.next_voiced(0.5, 3.0) == pytest.approx(1.0, abs=0.03)
    assert activity.next_voiced(2.2, 3.0) is None
    assert activity.onset_near(1.1, 0.3) == pytest.approx(1.0, abs=0.03)
    assert activity.onset_near(3.0, 0.3) is None
    assert activity.phrase_end(1.5, 3.0) == pytest.approx(2.0, abs=0.03)
    assert activity.phrase_end(1.5, 1.7) == 1.7
    assert activity.phrase_end(0.5, 3.0) == 0.5


def test_onset_envelope_peaks_at_the_phrase_start(activity):
    envelope = activity.onset_envelope()
    assert abs(np.argmax(envelope) * activity.hop - 1.0) < 0.05


def test_silence_is_never_voiced():
    silent = Activity.compute(signal(2.0), RATE)
    assert not silent.voiced.any()
    assert Activity.compute(np.zeros(0), RATE).duration == 0


def test_stereo_input_and_save_load(activity, tmp_path):
    stereo = np.stack([signal(2.0, (0.5, 1.5))] * 2, axis=1)
    assert Activity.compute(stereo, RATE).is_voiced(1.0)
    path = tmp_path / "activity.npz"
    activity.save(path)
    loaded = Activity.load(path)
    assert np.array_equal(loaded.voiced, activity.voiced) and loaded.hop == activity.hop

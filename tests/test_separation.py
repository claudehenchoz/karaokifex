import numpy as np
import soundfile as sf

from karaokifex.steps import separation


def song(seed=1, n=44100 * 3):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 44100
    band = 0.3 * np.sin(2 * np.pi * 55 * t)[:, None] * [1.0, 0.9] + 0.05 * rng.standard_normal((n, 2))
    voice = 0.2 * np.sin(2 * np.pi * 440 * t + np.sin(2 * np.pi * 3 * t))[:, None] * [1.0, 1.0]
    return band.astype(np.float32), voice.astype(np.float32)


def test_fit_gains_recovers_how_the_stems_were_scaled():
    band, voice = song()
    a, b = separation.fit_gains(band + voice, band * 0.6, voice * 0.9)
    assert abs(a - 1 / 0.6) < 1e-3 and abs(b - 1 / 0.9) < 1e-3


def test_fit_gains_with_a_silent_stem():
    band, voice = song()
    assert separation.fit_gains(band, band, np.zeros_like(voice)) == (1.0, 1.0)


def test_combine_puts_the_lead_back_at_the_songs_level_and_keeps_the_rest_whole(tmp_path):
    band, voice = song()
    mix = band + voice
    sf.write(tmp_path / "audio.wav", mix, 44100, subtype="FLOAT")
    # like audio-separator: the mix scaled before separating, each stem scaled again after
    stems = []
    for i, (scale_in, scale_lead, scale_rest, error) in enumerate([(0.9, 0.95, 0.62, 0.02), (0.9, 1.0, 0.7, -0.02)]):
        lead = voice * scale_in * scale_lead * (1 + error)
        rest = (mix * scale_in - voice * scale_in * (1 + error)) * scale_rest
        sf.write(tmp_path / f"lead.{i}.wav", lead, 44100, subtype="FLOAT")
        sf.write(tmp_path / f"backing.{i}.wav", rest, 44100, subtype="FLOAT")
        stems.append((tmp_path / f"lead.{i}.wav", tmp_path / f"backing.{i}.wav"))
    gains = separation.combine(tmp_path / "audio.wav", stems, backing=tmp_path / "backing.wav", lead=tmp_path / "lead.wav")
    backing, _ = sf.read(tmp_path / "backing.wav", dtype="float32")
    lead, _ = sf.read(tmp_path / "lead.wav", dtype="float32")
    assert len(gains) == 2
    # the two models' errors cancel in the average: the lead is the voice, the backing the band, at their own level
    assert np.sqrt(np.mean((lead - voice) ** 2)) < 1e-3
    assert np.sqrt(np.mean((backing - band) ** 2)) < 1e-3
    assert abs(np.sqrt(np.mean(backing ** 2)) / np.sqrt(np.mean(band ** 2)) - 1) < 0.01

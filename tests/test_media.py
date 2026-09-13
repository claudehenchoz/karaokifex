import pytest

from karaokifex.steps import media


@pytest.fixture(autouse=True)
def fresh_cache():
    media.find_ffmpeg.cache_clear()
    yield
    media.find_ffmpeg.cache_clear()


def fake_toolchain(monkeypatch, *, candidates, libass, nvenc):
    monkeypatch.setattr(media, "_ffmpegs_on_path", lambda: list(candidates))
    monkeypatch.setattr(media, "_has_libass", lambda binary: binary in libass)
    monkeypatch.setattr(media, "_can_encode", lambda binary, encoder: encoder.gpu and binary in nvenc)


def test_prefers_the_first_ffmpeg_that_can_encode_on_the_gpu(monkeypatch):
    fake_toolchain(monkeypatch, candidates=["old", "new"], libass={"old", "new"}, nvenc={"new"})
    assert media.find_ffmpeg() == media.FfmpegBinary("new", media.NVENC)


def test_falls_back_to_cpu_encoding(monkeypatch):
    fake_toolchain(monkeypatch, candidates=["no-ass", "old"], libass={"old"}, nvenc=set())
    assert media.find_ffmpeg() == media.FfmpegBinary("old", media.X264)


def test_explicit_ffmpeg_is_the_only_candidate(monkeypatch):
    fake_toolchain(monkeypatch, candidates=["new"], libass={"new", "mine"}, nvenc={"new", "mine"})
    assert media.find_ffmpeg("mine").path == "mine"


def test_fails_without_libass(monkeypatch):
    fake_toolchain(monkeypatch, candidates=["a"], libass=set(), nvenc={"a"})
    with pytest.raises(media.FfmpegError, match="libass"):
        media.find_ffmpeg()


def test_encoder_options_become_arguments():
    assert media._as_args({"preset": "p5", "b:v": 0}) == ["-preset", "p5", "-b:v", "0"]

from pathlib import Path

import ffmpeg
import pytest

from karaokifex.steps import media
from karaokifex.steps.media import Encoder, FfmpegBinary, SourceInfo

ALL_SOFTWARE = frozenset({"libsvtav1", "libx265", "libx264", "libopus", "aac"})


@pytest.fixture(autouse=True)
def fresh_cache():
    media.find_ffmpeg.cache_clear()
    yield
    media.find_ffmpeg.cache_clear()


def fake_toolchain(monkeypatch, *, candidates, libass, gpu):
    """gpu maps binary -> the NVENC formats it can encode."""
    monkeypatch.setattr(media, "_ffmpegs_on_path", lambda: list(candidates))
    monkeypatch.setattr(media, "_has_libass", lambda binary: binary in libass)
    monkeypatch.setattr(media, "_software_encoders", lambda binary: ALL_SOFTWARE)
    monkeypatch.setattr(media, "_can_encode",
                        lambda binary, codec: any(media.GPU_ENCODERS[f] == codec for f in gpu.get(binary, ())))


def test_prefers_the_first_ffmpeg_that_can_encode_on_the_gpu(monkeypatch):
    fake_toolchain(monkeypatch, candidates=["old", "new"], libass={"old", "new"}, gpu={"new": {"hevc", "h264"}})
    tool = media.find_ffmpeg()
    assert (tool.path, tool.gpu_formats) == ("new", frozenset({"hevc", "h264"}))


def test_falls_back_to_cpu_encoding(monkeypatch):
    fake_toolchain(monkeypatch, candidates=["no-ass", "old"], libass={"old"}, gpu={"no-ass": {"h264"}})
    tool = media.find_ffmpeg()
    assert (tool.path, tool.gpu) == ("old", False)


def test_explicit_ffmpeg_is_the_only_candidate(monkeypatch):
    fake_toolchain(monkeypatch, candidates=["new"], libass={"new", "mine"}, gpu={"new": {"h264"}, "mine": {"h264"}})
    assert media.find_ffmpeg("mine").path == "mine"


def test_fails_without_libass(monkeypatch):
    fake_toolchain(monkeypatch, candidates=["a"], libass=set(), gpu={"a": {"h264"}})
    with pytest.raises(media.FfmpegError, match="libass"):
        media.find_ffmpeg()


RTX_3070 = FfmpegBinary("ffmpeg", frozenset({"hevc", "h264"}), ALL_SOFTWARE)


@pytest.mark.parametrize(
    ("source", "tool", "expected"),
    [
        ("av1", RTX_3070, Encoder("hevc", gpu=True)),  # no AV1 on the GPU: next most efficient format
        ("h264", RTX_3070, Encoder("h264", gpu=True)),  # same format as the source when possible
        ("vp9", RTX_3070, Encoder("hevc", gpu=True)),  # no VP9 encoder at all
        (None, RTX_3070, Encoder("hevc", gpu=True)),
        ("av1", FfmpegBinary("ffmpeg", frozenset({"av1", "hevc"}), ALL_SOFTWARE), Encoder("av1", gpu=True)),
        ("av1", FfmpegBinary("ffmpeg", frozenset(), ALL_SOFTWARE), Encoder("av1", gpu=False)),
        ("av1", FfmpegBinary(), Encoder("h264", gpu=False)),
    ],
)
def test_choose_encoder(source, tool, expected):
    assert media.choose_encoder(source, tool) == expected


def test_cpu_fallback_keeps_the_format_order():
    assert media.choose_encoder("av1", RTX_3070, gpu=False) == Encoder("av1", gpu=False)


def test_target_bitrate_scales_with_format_efficiency():
    source = SourceInfo("av1", 10_000_000, "opus")
    assert media.target_bitrate(source, "av1") == 10_000_000
    assert media.target_bitrate(source, "hevc") == 13_000_000
    assert media.target_bitrate(SourceInfo("av1", None), "hevc") is None


@pytest.mark.parametrize(
    ("source_height", "target_height", "expected"),
    [(720, 1080, "scale=-2:1080"), (1080, 1080, None), (2160, 1080, None), (None, 1080, None), (720, None, None)],
)
def test_scale_filter_only_upsamples_smaller_sources(source_height, target_height, expected):
    assert media.scale_filter(source_height, target_height) == expected


def test_encoder_options():
    gpu = Encoder("hevc", gpu=True).options(13_000_000)
    assert gpu["b:v"] == 13_000_000 and gpu["maxrate"] == 26_000_000 and gpu["preset"] == "p4"
    assert Encoder("hevc", gpu=True).options(None)["cq"] == 23
    assert Encoder("h264", gpu=False).options(None) == {"preset": "medium", "crf": 20}


def test_audio_follows_the_source_codec():
    assert media.audio_encoder("opus", RTX_3070) == ("libopus", "160k")
    assert media.audio_encoder("mp3", RTX_3070) == ("aac", "256k")
    assert media.audio_encoder("opus", FfmpegBinary()) == ("aac", "256k")  # no libopus in this build


def test_browser_friendly_encodes_only_h264():
    assert media.choose_encoder("av1", RTX_3070, formats=("h264",)) == Encoder("h264", gpu=True)
    assert media.choose_encoder("av1", RTX_3070, gpu=False, formats=("h264",)) == Encoder("h264", gpu=False)
    assert media.audio_encoder("opus", RTX_3070, browser=True) == ("aac", "256k")


@pytest.mark.parametrize(
    ("source", "ready"),
    [
        (SourceInfo("h264", pixel_format="yuv420p", profile="High"), True),
        (SourceInfo("h264", pixel_format="yuv420p", profile="Constrained Baseline"), True),
        (SourceInfo("h264", pixel_format="yuv420p10le", profile="High 10"), False),
        (SourceInfo("h264", pixel_format="yuv444p", profile="High 4:4:4 Predictive"), False),
        (SourceInfo("vp9", pixel_format="yuv420p", profile="Profile 0"), False),
        (SourceInfo(), False),
    ],
)
def test_browser_ready_sources(source, ready):
    assert source.browser_ready is ready


def fake_render(monkeypatch, tmp_path, subtitles, source, **options):
    """Runs media.render with ffmpeg replaced; returns its description and the ffmpeg arguments."""
    calls = []

    def run(stream, *, binary, cwd, duration, on_progress):
        calls.append(args := ffmpeg.compile(stream))
        (cwd / args[-1]).write_text("x")

    monkeypatch.setattr(media, "run", run)
    description = media.render(tmp_path / "video.mkv", tmp_path / "backing.wav", subtitles, tmp_path / "out.mkv",
                               tool=RTX_3070, source=source, **options)
    return description, calls[-1]


def test_render_burns_in_the_subtitles(monkeypatch, tmp_path):
    description, args = fake_render(monkeypatch, tmp_path, tmp_path / "lyrics.ass", SourceInfo("h264", video_height=1080))
    assert description.startswith("h264_nvenc")
    graph = args[args.index("-filter_complex") + 1]
    assert "eq=brightness=-0.08" in graph and "subtitles=lyrics.ass" in graph


def test_render_without_subtitles_copies_the_video(monkeypatch, tmp_path):
    description, args = fake_render(monkeypatch, tmp_path, None, SourceInfo("vp9", video_height=1080, audio_format="opus"))
    assert description == "vp9 copied + libopus"
    assert args[args.index("-vcodec") + 1] == "copy"
    assert "-filter_complex" not in args and "-hwaccel" not in args


def test_render_without_subtitles_still_upscales(monkeypatch, tmp_path):
    description, args = fake_render(monkeypatch, tmp_path, None, SourceInfo("h264", video_height=720))
    assert description.startswith("h264_nvenc")
    graph = args[args.index("-filter_complex") + 1]
    assert "scale=-2:1080" in graph and "subtitles" not in graph and "eq=" not in graph


H264 = SourceInfo("h264", 4_000_000, "opus", 1920, 1080, "yuv420p", "High")


def test_browser_friendly_copies_h264_and_encodes_aac(monkeypatch, tmp_path):
    description, args = fake_render(monkeypatch, tmp_path, None, H264, browser=True)
    assert description == "h264 copied + aac"
    assert args[args.index("-vcodec") + 1] == "copy"
    assert args[args.index("-acodec") + 1] == "aac"
    assert args[args.index("-movflags") + 1] == "+faststart"


def test_browser_friendly_encodes_other_formats_to_h264(monkeypatch, tmp_path):
    source = SourceInfo("av1", 2_000_000, "opus", 1920, 1080, "yuv420p", "Main")
    description, args = fake_render(monkeypatch, tmp_path, None, source, browser=True)
    assert description.startswith("h264_nvenc (GPU) at 3.6 Mbit/s + aac")
    assert args[args.index("-profile:v") + 1] == "high" and args[args.index("-pix_fmt") + 1] == "yuv420p"
    assert args[args.index("-movflags") + 1] == "+faststart"


@pytest.mark.parametrize("subtitles, height", [("lyrics.ass", 1080), (None, 720)])
def test_browser_friendly_encodes_when_the_picture_changes(monkeypatch, tmp_path, subtitles, height):
    source = SourceInfo("h264", 4_000_000, "opus", 1280, height, "yuv420p", "High")
    subtitles = subtitles and tmp_path / subtitles
    description, args = fake_render(monkeypatch, tmp_path, subtitles, source, browser=True)
    assert description.startswith("h264_nvenc")
    assert "-filter_complex" in args


def test_the_original_keeps_its_own_audio(monkeypatch, tmp_path):
    source = SourceInfo("vp9", 2_000_000, "opus", 1920, 1080, "yuv420p", "Profile 0")
    description, args = fake_render(monkeypatch, tmp_path, None, source, copy_audio=True)
    assert description == "vp9 copied + opus copied"
    assert args[args.index("-acodec") + 1] == "copy" and "-b:a" not in args


def test_the_original_for_browsers_gets_aac(monkeypatch, tmp_path):
    source = SourceInfo("h264", 4_000_000, "opus", 1920, 1080, "yuv420p", "High")
    description, args = fake_render(monkeypatch, tmp_path, None, source, copy_audio=True, browser=True)
    assert description == "h264 copied + aac"
    source = SourceInfo("h264", 4_000_000, "aac", 1920, 1080, "yuv420p", "High")
    description, args = fake_render(monkeypatch, tmp_path, None, source, copy_audio=True, browser=True)
    assert description == "h264 copied + aac copied"


def test_loudness_reads_ffmpegs_summary(monkeypatch):
    summary = "  Integrated loudness:\n    I:         -11.4 LUFS\n    Threshold: -21.6 LUFS\n"
    monkeypatch.setattr(media.subprocess, "run", lambda *a, **k: type("R", (), {"stderr": summary})())
    assert media.loudness(Path("x.wav")) == -11.4
    monkeypatch.setattr(media.subprocess, "run", lambda *a, **k: type("R", (), {"stderr": "no audio"})())
    assert media.loudness(Path("x.wav")) is None

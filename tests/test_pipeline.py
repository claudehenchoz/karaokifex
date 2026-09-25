import json
from dataclasses import replace

import numpy as np
import pytest
import requests

from karaokifex import pipeline
from karaokifex.config import Config
from karaokifex.models import LyricLine, Lyrics, TimedWord, VideoInfo
from karaokifex.runner import TaskRunner
from karaokifex.steps.lyrics import save_lyrics
from karaokifex.steps.transcription import save_transcript
from karaokifex.workspace import Workspace


@pytest.fixture
def job(tmp_path):
    config = Config(url="https://youtu.be/x", output_dir=tmp_path)
    workspace = Workspace.create(tmp_path, "Artist - Song")
    info = VideoInfo(id="x", title="Artist - Song", duration=200, width=1280, height=720)
    return pipeline.Job(config, workspace, info, "Artist", "Song", "cpu")


def tasks_of(job):
    tasks = {task.name: task for task in pipeline.build_tasks(job)}
    TaskRunner(list(tasks.values()))  # validates dependencies and cycles
    return tasks


def test_task_graph_is_valid_and_wired(job):
    tasks = tasks_of(job)
    assert {name for name, task in tasks.items() if not task.deps} == {"lyrics", "download", "load_whisper"}
    assert {name for name, task in tasks.items() if task.gpu} == {"separate_karaoke", "transcribe", "force_align"}
    assert set(tasks["transcribe"].deps) == {"separate_karaoke", "load_whisper", "lyrics"}  # lyrics prompt whisper
    assert set(tasks["vocal_activity"].deps) == {"separate_karaoke"}
    assert set(tasks["force_align"].deps) == {"lyrics", "transcribe", "vocal_activity"}
    assert set(tasks["subtitles"].deps) == {"lyrics", "transcribe", "vocal_activity", "force_align"}
    assert set(tasks["render"].deps) == {"extract_video", "separate_karaoke", "subtitles"}
    assert tasks["render"].outputs == (job.workspace.final_video,)
    assert "transcribe_mix" not in tasks
    assert "palette" not in tasks
    assert "original" not in tasks


def test_keep_source_renders_the_original_in_the_output_format(job):
    job = replace(job, config=replace(job.config, keep_source=True, browser_friendly=True))
    tasks = tasks_of(job)
    # the song's loudness first (--lift-quiet, on by default): a quiet song's original is lifted too
    assert set(tasks["original"].deps) == {"download", "extract_video", "extract_audio"}
    assert tasks["original"].outputs == (job.workspace.root / "Artist - Song (Original).mp4",)
    unlifted = replace(job, config=replace(job.config, lift_quiet=False))
    assert set(tasks_of(unlifted)["original"].deps) == {"download", "extract_video"}


def test_mix_vote_and_debug_change_the_graph(job):
    job = replace(job, config=replace(job.config, mix_vote=True, debug_ass=True))
    tasks = tasks_of(job)
    assert "transcribe" in tasks["transcribe_mix"].deps  # takes over the loaded model
    assert "transcribe_mix" in tasks["subtitles"].deps
    assert job.workspace.transcript_mix_json in tasks["load_whisper"].outputs
    assert tasks["render"].outputs == (job.workspace.debug_video,)


def test_without_burned_in_lyrics_the_render_does_not_wait_for_them(job):
    job = replace(job, config=replace(job.config, burn_lyrics=False))
    tasks = tasks_of(job)
    assert set(tasks["render"].deps) == {"extract_video", "separate_karaoke"}
    assert "subtitles" in tasks  # the lyrics files are still written
    assert tasks["render"].outputs == (job.workspace.plain_video,)


def test_browser_friendly_renders_an_mp4(job):
    job = replace(job, config=replace(job.config, browser_friendly=True))
    assert tasks_of(job)["render"].outputs == (job.workspace.final_video.with_suffix(".mp4"),)
    job = replace(job, config=replace(job.config, burn_lyrics=False))
    assert job.output_video.name == "Artist - Song (Karaoke, no lyrics).mp4"


def test_palette_step_writes_the_dominant_colours(job, monkeypatch):
    job = replace(job, config=replace(job.config, palette=True))
    assert tasks_of(job)["palette"].deps == ("extract_video",)
    frames = np.zeros((4, 36, 64, 3), dtype=np.uint8)
    frames[:, :9], frames[:, 9:27] = (255, 0, 0), (0, 0, 255)  # the black rows below are a letterbox bar
    monkeypatch.setattr(pipeline.media, "sample_frames", lambda video, **options: frames)
    pipeline._palette(job, ctx=Ctx())
    metadata = json.loads(job.workspace.metadata_json.read_text(encoding="utf-8"))
    assert metadata["palette"] == {"frames": 4, "colors": [{"hex": "#0000ff", "rgb": [0, 0, 255], "weight": 0.6667},
                                                           {"hex": "#ff0000", "rgb": [255, 0, 0], "weight": 0.3333}]}


class Ctx:
    def note(self, text):
        pass


class FakeResponse:
    status_code = 200

    def __init__(self, recordings):
        self._recordings = recordings

    def raise_for_status(self):
        pass

    def json(self):
        return {"recordings": self._recordings}


PUMPKINS = [{"title": "Mayonaise", "artist-credit": [{"name": "Smashing Pumpkins", "joinphrase": "",
                                                      "artist": {"name": "The Smashing Pumpkins"}}]}]


def test_canonical_names_come_from_musicbrainz(monkeypatch):
    monkeypatch.setattr(pipeline.musicbrainz, "MIN_INTERVAL", 0.0)
    info = VideoInfo(id="x", title='Smashing Pumpkins "Mayonaise"', uploader="ag4321")
    guess = ("Smashing Pumpkins", "Mayonaise")
    get = lambda url, **options: FakeResponse(PUMPKINS)  # noqa: E731
    assert pipeline.canonical_names(info, None, None, guess, get=get) == ("The Smashing Pumpkins", "Mayonaise")
    assert pipeline.canonical_names(info, None, "mayonaise", guess, get=get) == ("The Smashing Pumpkins", "mayonaise")
    assert pipeline.canonical_names(info, None, None, guess, get=lambda url, **o: FakeResponse([])) == guess


def test_canonical_names_fall_back_when_offline(monkeypatch):
    monkeypatch.setattr(pipeline.musicbrainz, "MIN_INTERVAL", 0.0)

    def offline(url, **options):
        raise requests.ConnectionError("no network")

    info = VideoInfo(id="x", title="Culture Beat - Mr. Vain")
    assert pipeline.canonical_names(info, None, None, ("Culture Beat", "Mr. Vain"), get=offline) == \
        ("Culture Beat", "Mr. Vain")


WORDS = [TimedWord("hello", 5.0, 5.4), TimedWord("world", 5.5, 6.0), TimedWord("again", 8.0, 8.6)]


def lyrics(id, *lines):
    return Lyrics(id, "Artist", "Song", None, 200.0, True, tuple(LyricLine(start, text) for start, text in lines))


def test_subtitles_step_uses_lyrics(job):
    ws = job.workspace
    save_lyrics([lyrics(1, (5.0, "Hello world"), (8.0, "Again"))], ws.lyrics_json)
    save_transcript(WORDS, "en", ws.transcript_json)
    assert pipeline._subtitles(job, ctx=None).startswith("lrclib #1")
    ass = ws.subtitles.read_text(encoding="utf-8")
    assert "PlayResX: 1280" in ass
    assert "Hello" in ass and "Again" in ass
    assert "Legend" in ws.debug_subtitles.read_text(encoding="utf-8")
    timings = json.loads(ws.timings_json.read_text(encoding="utf-8"))
    assert [w["source"] for line in timings["lines"] for w in line] == ["whisper"] * 3


def test_subtitles_step_picks_the_lyrics_that_fit(job):
    ws = job.workspace
    save_lyrics([lyrics(1, (5.0, "Something else entirely"), (8.0, "Not it")),
                 lyrics(2, (5.0, "Hello world"), (8.0, "Again"))], ws.lyrics_json)
    save_transcript(WORDS, "en", ws.transcript_json)
    assert pipeline._subtitles(job, ctx=None).startswith("lrclib #2")


def test_subtitles_step_falls_back_to_transcription(job):
    ws = job.workspace
    save_lyrics([], ws.lyrics_json)
    save_transcript(WORDS, "en", ws.transcript_json)
    pipeline._subtitles(job, ctx=None)
    assert "hello" in ws.subtitles.read_text(encoding="utf-8")


def test_subtitles_step_fails_without_anything_to_show(job):
    save_lyrics([], job.workspace.lyrics_json)
    save_transcript([], "en", job.workspace.transcript_json)
    with pytest.raises(RuntimeError, match="nothing to display"):
        pipeline._subtitles(job, ctx=None)


def test_describe_runs_after_the_lyrics_only_when_asked(job):
    assert "describe" not in tasks_of(job)
    described = replace(job, config=replace(job.config, describe=True))
    task = tasks_of(described)["describe"]
    assert task.deps == ("lyrics",) and task.outputs == (job.workspace.song_json,)


def test_describe_song_keeps_the_names_and_a_language_when_musicbrainz_is_away(job, monkeypatch):
    monkeypatch.setattr(pipeline.musicbrainz, "MIN_INTERVAL", 0.0)

    def away(*_, **__):
        raise requests.ConnectionError("down")

    about = pipeline.describe_song(job.workspace.song_json, "Artist", "Song", "de", get=away)
    assert about == {"artist": "Artist", "song": "Song", "musicbrainz": None, "language": "de"}
    assert json.loads(job.workspace.song_json.read_text(encoding="utf-8")) == about


def test_quality_reads_the_download_before_it_goes(job):
    assert "quality" not in tasks_of(job)
    measured = replace(job, config=replace(job.config, quality=True, keep_source=True))
    task = tasks_of(measured)["quality"]
    assert task.deps == ("download", "render", "original") and task.outputs == (job.workspace.quality_json,)
    assert job.workspace.quality_json in job.workspace.artifacts()


def test_upscaling_is_for_burned_in_lyrics(job):
    config = job.config
    assert config.burn_lyrics and config.target_height == config.resolution
    assert replace(config, burn_lyrics=False).target_height is None
    assert replace(config, burn_lyrics=False, upscale=True).target_height == config.resolution
    assert replace(config, upscale=False).target_height is None


def test_separation_runs_each_model_and_resumes_after_the_last_one_done(job, monkeypatch):
    import numpy as np
    import soundfile as sf

    sf.write(job.workspace.audio, np.zeros((4410, 2), dtype=np.float32), 44100, subtype="FLOAT")
    ran = []

    def fake_separate(audio, *, model, stems, **_):
        ran.append(model)
        for path in stems.values():
            sf.write(path, np.zeros((4410, 2), dtype=np.float32), 44100, subtype="FLOAT")

    monkeypatch.setattr(pipeline.separation, "separate", fake_separate)
    ensemble = replace(job, config=replace(job.config, karaoke_models=("a.ckpt", "b.ckpt", "c.ckpt")))
    (job.workspace.stems_dir / "lead.0.wav").write_bytes(b"")  # a lead without its backing: model a again
    for name in ("lead.1.wav", "backing.1.wav"):
        sf.write(job.workspace.stems_dir / name, np.zeros((4410, 2), dtype=np.float32), 44100, subtype="FLOAT")
    pipeline._separate_karaoke(ensemble, type("Ctx", (), {"note": lambda self, text: None})())
    assert ran == ["a.ckpt", "c.ckpt"]
    assert job.workspace.karaoke_backing.exists() and job.workspace.karaoke_lead.exists()

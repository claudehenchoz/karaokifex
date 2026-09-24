import json
from dataclasses import replace

import numpy as np
import pytest

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

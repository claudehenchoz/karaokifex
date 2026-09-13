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


def test_task_graph_is_valid_and_wired(job):
    tasks = {task.name: task for task in pipeline.build_tasks(job)}
    TaskRunner(list(tasks.values()))  # validates dependencies and cycles
    assert {name for name, task in tasks.items() if not task.deps} == {"lyrics", "download", "load_whisper"}
    assert {name for name, task in tasks.items() if task.gpu} == {"separate_karaoke", "transcribe"}
    assert set(tasks["transcribe"].deps) == {"separate_karaoke", "load_whisper"}  # lead vocals come from the karaoke pass
    assert set(tasks["subtitles"].deps) == {"lyrics", "transcribe"}
    assert set(tasks["render"].deps) == {"extract_video", "separate_karaoke", "subtitles"}


WORDS = [TimedWord("hello", 5.0, 5.4), TimedWord("world", 5.5, 6.0), TimedWord("again", 8.0, 8.6)]


def test_subtitles_step_uses_lyrics(job):
    ws = job.workspace
    save_lyrics(Lyrics(1, "Artist", "Song", None, 200.0, True,
                       (LyricLine(5.0, "Hello world"), LyricLine(8.0, "Again"))), ws.lyrics_json)
    save_transcript(WORDS, "en", ws.transcript_json)
    pipeline._subtitles(job, ctx=None)
    ass = ws.subtitles.read_text(encoding="utf-8")
    assert "PlayResX: 1280" in ass
    assert "Hello" in ass and "Again" in ass


def test_subtitles_step_falls_back_to_transcription(job):
    ws = job.workspace
    save_lyrics(None, ws.lyrics_json)
    save_transcript(WORDS, "en", ws.transcript_json)
    pipeline._subtitles(job, ctx=None)
    assert "hello" in ws.subtitles.read_text(encoding="utf-8")


def test_subtitles_step_fails_without_anything_to_show(job):
    save_lyrics(None, job.workspace.lyrics_json)
    save_transcript([], "en", job.workspace.transcript_json)
    with pytest.raises(RuntimeError, match="nothing to display"):
        pipeline._subtitles(job, ctx=None)

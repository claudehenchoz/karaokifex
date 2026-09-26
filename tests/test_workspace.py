from pathlib import Path

import pytest

from karaokifex.workspace import Workspace, partial_path, sanitize_name


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("AC/DC: Back In Black?", "ACDC Back In Black"),
        ("  spaced   out  ", "spaced out"),
        ("trailing dots...", "trailing dots"),
        ('a<b>c|d*e"f', "abcdef"),
        ("", "untitled"),
        ("CON", "_CON"),
    ],
)
def test_sanitize_name(name, expected):
    assert sanitize_name(name) == expected


def test_sanitize_name_truncates():
    assert len(sanitize_name("x" * 300)) == 100


def test_partial_path():
    assert partial_path(Path("a/audio.wav")) == Path("a/audio.partial.wav")


def test_cleanup_keeps_artifacts(tmp_path):
    ws = Workspace.create(tmp_path, "Artist - Song")
    temp = [ws.source, ws.audio, ws.video, ws.karaoke_lead, ws.transcript_json]
    for path in [*temp, *ws.artifacts()]:
        path.write_text("x")
    assert set(ws.temp_files()) == set(temp)
    assert set(ws.cleanup()) == set(temp)
    assert all(path.exists() for path in ws.artifacts())
    assert not any(path.exists() for path in temp)


def test_cleanup_keeps_the_lyrics_data(tmp_path):
    ws = Workspace.create(tmp_path, "Artist - Song")
    assert {ws.timings_json, ws.info_json, ws.metadata_json, ws.lyrics_json, ws.subtitles} <= ws.artifacts()


def test_cleanup_keeps_the_original_but_not_the_download(tmp_path):
    ws = Workspace.create(tmp_path, "Artist - Song")
    original = ws.original_video.with_suffix(".mp4")
    for path in (ws.source, original):
        path.write_text("x")
    assert ws.cleanup() == [ws.source]
    assert original.exists()


def test_cleanup_keeps_karaoke_videos_from_earlier_runs(tmp_path):
    ws = Workspace.create(tmp_path, "Artist [Live] - Song")  # glob metacharacters in the name
    earlier = ws.final_video.with_suffix(".mp4")
    half_written = ws.root / f"{ws.final_video.stem}.partial.mkv"
    for path in (earlier, half_written, ws.source):
        path.write_text("x")
    assert set(ws.temp_files()) == {half_written, ws.source}
    ws.cleanup()
    assert earlier.exists()


def test_cleanup_removes_empty_folders(tmp_path):
    ws = Workspace.create(tmp_path, "Artist - Song")
    ws.karaoke_lead.write_text("x")
    ws.cleanup()
    assert not ws.stems_dir.exists()
    assert ws.root.exists()

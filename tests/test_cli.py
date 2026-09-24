import pytest
from click.testing import CliRunner

from karaokifex import cli
from karaokifex.config import Config
from karaokifex.models import VideoInfo
from karaokifex.pipeline import Job, PipelineResult
from karaokifex.runner import RunReport, Status, TaskOutcome
from karaokifex.workspace import Workspace

URL = "https://youtu.be/x"


@pytest.fixture
def fake_run(monkeypatch, tmp_path):
    """Replaces the real pipeline; returns a dict that records the Config and lets tests pick success/failure."""
    state = {"ok": True}
    workspace = Workspace.create(tmp_path, "Artist - Song")
    workspace.source.write_text("temp")
    workspace.final_video.write_text("final")
    state["workspace"] = workspace

    def run(config):
        state["config"] = config
        render = (TaskOutcome(Status.DONE, 2.0, "libx264") if state["ok"]
                  else TaskOutcome(Status.FAILED, 2.0, error=RuntimeError("boom")))
        report = RunReport({"lyrics": TaskOutcome(Status.DONE, 1.0, "lrclib #1"), "render": render})
        job = Job(config, workspace, VideoInfo(id="x", title="t"), "Artist", "Song", "cpu")
        return PipelineResult(job, report)

    monkeypatch.setattr(cli, "run_pipeline", run)
    return state


def test_options_become_config(fake_run):
    args = [URL, "-a", "Artist", "-s", "Song", "--gpu-jobs", "2", "--overlap", "4", "--fp32", "--keep-temp",
            "--mix-vote", "--debug-ass"]
    result = CliRunner().invoke(cli.main, args)
    assert result.exit_code == 0, result.output
    config: Config = fake_run["config"]
    assert (config.url, config.artist, config.song, config.gpu_jobs, config.keep_temp) == (URL, "Artist", "Song", 2, True)
    assert (config.separation_overlap, config.fp16) == (4, False)
    assert (config.mix_vote, config.debug_ass) == (True, True)


def test_resolution_option_becomes_config(fake_run):
    result = CliRunner().invoke(cli.main, [URL, "--resolution", "720"])
    assert result.exit_code == 0, result.output
    assert fake_run["config"].resolution == 720


def test_lead_volume_option_becomes_config(fake_run):
    result = CliRunner().invoke(cli.main, [URL, "--lead-volume", "0.35"])
    assert result.exit_code == 0, result.output
    assert fake_run["config"].lead_volume == 0.35


def test_burn_lyrics_option_becomes_config(fake_run):
    assert CliRunner().invoke(cli.main, [URL]).exit_code == 0
    assert fake_run["config"].burn_lyrics is True
    result = CliRunner().invoke(cli.main, [URL, "--no-burn-lyrics"])
    assert result.exit_code == 0, result.output
    assert fake_run["config"].burn_lyrics is False


def test_browser_friendly_option_becomes_config(fake_run):
    result = CliRunner().invoke(cli.main, [URL, "--browser-friendly"])
    assert result.exit_code == 0, result.output
    assert fake_run["config"].browser_friendly is True


def test_palette_option_becomes_config(fake_run):
    result = CliRunner().invoke(cli.main, [URL, "--palette"])
    assert result.exit_code == 0, result.output
    assert fake_run["config"].palette is True


def test_debug_ass_needs_burned_in_lyrics(fake_run):
    result = CliRunner().invoke(cli.main, [URL, "--no-burn-lyrics", "--debug-ass"])
    assert result.exit_code == 2
    assert "--debug-ass" in result.output


def test_temp_files_are_deleted_without_asking(fake_run):
    ws = fake_run["workspace"]
    ws.karaoke_lead.write_text("temp")
    result = CliRunner().invoke(cli.main, [URL])
    assert result.exit_code == 0, result.output
    assert not ws.source.exists() and not ws.karaoke_lead.exists()
    assert ws.final_video.exists()


def test_keep_source_keeps_the_download(fake_run):
    ws = fake_run["workspace"]
    ws.karaoke_lead.write_text("temp")
    result = CliRunner().invoke(cli.main, [URL, "--keep-source"])
    assert result.exit_code == 0, result.output
    assert ws.source.exists() and not ws.karaoke_lead.exists()


def test_keep_temp_keeps_everything(fake_run):
    result = CliRunner().invoke(cli.main, [URL, "--keep-temp"])
    assert result.exit_code == 0, result.output
    assert fake_run["workspace"].source.exists()


def test_autodelete_is_still_accepted(fake_run):
    result = CliRunner().invoke(cli.main, [URL, "--autodelete"])
    assert result.exit_code == 0, result.output
    assert not fake_run["workspace"].source.exists()


def test_failed_run_exits_nonzero_and_keeps_files(fake_run):
    fake_run["ok"] = False
    result = CliRunner().invoke(cli.main, [URL])
    assert result.exit_code == 1
    assert fake_run["workspace"].source.exists()

"""The karaokifex pipeline: which steps exist, what they need, and what can run in parallel.

    probe ─┬─ lyrics ──────────────────────────────────────────────┐
           ├─ load_whisper ──────────────────────────┐             │
           └─ download ─┬─ extract_audio ─ separate_karaoke ─ transcribe ─ subtitles ─ render
                        └─ extract_video ─────────────────────────────────────────────────┘

One separation pass yields both stems the rest needs: the backing track (for
render) and the lead vocals (for transcribe). `probe` runs first on its own
(its metadata names the song folder); the task runner handles the rest.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import partial

import ffmpeg
from rich.live import Live

from karaokifex.ass import build_ass
from karaokifex.config import Config
from karaokifex.console import TaskBoard, console, register_tasks
from karaokifex.gpu import free_gpu_memory
from karaokifex.metadata import guess_artist_song
from karaokifex.models import VideoInfo
from karaokifex.runner import RunReport, Task, TaskContext, TaskRunner, current_task
from karaokifex.steps import download, lyrics, media, separation, transcription
from karaokifex.timing import align_lyrics, lines_from_words
from karaokifex.workspace import Workspace, partial_path

log = logging.getLogger(__name__)

LOW_MATCH_WARNING = 0.3  # below this share of whisper-timed words the lyrics are probably a different version


@dataclass(frozen=True)
class Job:
    """Everything the steps need to know about the song being processed."""

    config: Config
    workspace: Workspace
    info: VideoInfo
    artist: str
    song: str
    device: str
    ffmpeg: media.FfmpegBinary = media.FfmpegBinary()

    @property
    def title(self) -> str:
        return f"{self.artist} – {self.song}"


@dataclass(frozen=True)
class PipelineResult:
    job: Job
    report: RunReport

    @property
    def ok(self) -> bool:
        return self.report.ok


def prepare(config: Config) -> Job:
    """Probe the video and set up its working folder."""
    token = current_task.set("probe")
    try:
        log.info("looking up %s", config.url)
        info = download.probe(config.url)
        artist, song = guess_artist_song(info, config.artist, config.song)
        workspace = Workspace.create(config.output_dir, f"{artist} - {song}")
        workspace.info_json.write_text(json.dumps(info.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        device = config.resolve_device()
        ffmpeg = media.find_ffmpeg(config.ffmpeg)
        log.info("“%s” (%s) → %s – %s", info.title, _format_length(info.duration), artist, song)
        log.info("working folder: %s · device: %s", workspace.root, device)
        log.info("ffmpeg: %s · %s", ffmpeg.path, ffmpeg.describe())
        return Job(config, workspace, info, artist, song, device, ffmpeg)
    finally:
        current_task.reset(token)


def build_tasks(job: Job) -> list[Task]:
    ws, cfg = job.workspace, job.config
    return [
        Task("lyrics", partial(_lyrics, job), outputs=(ws.lyrics_json,), description="lrclib lookup"),
        Task("download", partial(_download, job), outputs=(ws.source,), description="yt-dlp: best video + audio"),
        Task("load_whisper", partial(_load_whisper, job), outputs=(ws.transcript_json,),
             description=f"whisperx {cfg.whisper_model}"),
        Task("extract_audio", partial(_extract_audio, job), deps=("download",), outputs=(ws.audio,),
             description="ffmpeg → audio.wav"),
        Task("extract_video", partial(_extract_video, job), deps=("download",), outputs=(ws.video,),
             description="ffmpeg → video.mkv (no audio)"),
        Task("separate_karaoke", partial(_separate_karaoke, job), deps=("extract_audio",),
             outputs=(ws.karaoke_backing, ws.karaoke_lead), gpu=True, description=cfg.karaoke_model),
        Task("transcribe", partial(_transcribe, job), deps=("separate_karaoke", "load_whisper"),
             outputs=(ws.transcript_json,), gpu=True, description="whisperx on the lead vocals"),
        Task("subtitles", partial(_subtitles, job), deps=("lyrics", "transcribe"), outputs=(ws.subtitles,),
             description="lyrics + word timings → karaoke ASS"),
        Task("render", partial(_render, job), deps=("extract_video", "separate_karaoke", "subtitles"),
             outputs=(ws.final_video,), gpu=job.ffmpeg.gpu,
             description="darken, karaoke audio, burn in subtitles"),
    ]


def run_pipeline(config: Config) -> PipelineResult:
    job = prepare(config)
    tasks = build_tasks(job)
    register_tasks(task.name for task in tasks)
    board = TaskBoard(job.title, [(task.name, task.description) for task in tasks])
    runner = TaskRunner(tasks, gpu_slots=config.gpu_jobs, force=config.force, observer=board)
    with Live(board, console=console, refresh_per_second=8):
        report = runner.run()
    return PipelineResult(job, report)


# --- task implementations ------------------------------------------------------------


def _lyrics(job: Job, ctx: TaskContext) -> str:
    ctx.note(f"searching “{job.artist} – {job.song}”…")
    found = lyrics.fetch_lyrics(job.artist, job.song, job.info.duration)
    lyrics.save_lyrics(found, job.workspace.lyrics_json)
    if found is None:
        log.warning("no lyrics on lrclib — the whisperx transcription will be used instead")
        return "whisperx transcription (lrclib miss)"
    kind = "synced" if found.synced else "plain"
    log.info("found %s lyrics: %s – %s (%d lines, %s)", kind, found.artist, found.track, len(found.lines),
             _format_length(found.duration))
    return f"lrclib #{found.lrclib_id}: {found.artist} – {found.track} ({kind})"


def _download(job: Job, ctx: TaskContext) -> None:
    def on_progress(fraction: float | None, note: str) -> None:
        ctx.progress(fraction)
        ctx.note(note)

    download.download(job.config.url, job.workspace.source, on_progress)
    log.info("downloaded %s (%.0f MiB)", job.workspace.source.name, job.workspace.source.stat().st_size / 1_048_576)


def _extract_audio(job: Job, ctx: TaskContext) -> None:
    media.extract_audio(job.workspace.source, job.workspace.audio, binary=job.ffmpeg.path,
                        duration=job.info.duration, on_progress=ctx.progress)


def _extract_video(job: Job, ctx: TaskContext) -> None:
    media.extract_video(job.workspace.source, job.workspace.video, binary=job.ffmpeg.path,
                        duration=job.info.duration, on_progress=ctx.progress)


def _separate_karaoke(job: Job, ctx: TaskContext) -> None:
    ws, cfg = job.workspace, job.config
    separation.separate(ws.audio, model=cfg.karaoke_model, model_dir=cfg.model_dir,
                        stems={"instrumental": ws.karaoke_backing, "vocals": ws.karaoke_lead},
                        overlap=cfg.separation_overlap, fp16=cfg.fp16, verbose=cfg.verbose, on_stage=ctx.note)


def _load_whisper(job: Job, ctx: TaskContext) -> object:
    ctx.note("loading model (downloaded on first use)…")
    return transcription.load_model(job.config.whisper_model, job.device, job.config.language)


def _transcribe(job: Job, ctx: TaskContext) -> None:
    # take(): the runner must not keep the model alive — it would hog GPU memory during render.
    model = ctx.take("load_whisper")
    if model is None:  # load_whisper was skipped because an old transcript existed, but vocals changed
        ctx.note("loading model…")
        model = transcription.load_model(job.config.whisper_model, job.device, job.config.language)
    words, language = transcription.transcribe(model, job.workspace.karaoke_lead, job.device,
                                               language=job.config.language, on_stage=ctx.note)
    del model
    free_gpu_memory()
    transcription.save_transcript(words, language, job.workspace.transcript_json)
    log.info("heard %d words (language: %s)", len(words), language)


def _subtitles(job: Job, ctx: TaskContext) -> None:
    ws = job.workspace
    found = lyrics.load_lyrics(ws.lyrics_json)
    words, _ = transcription.load_transcript(ws.transcript_json)
    if found is not None:
        alignment = align_lyrics(found.lines, words)
        lines = alignment.lines
        log.info("%d of %d lyric words (%.0f%%) timed by whisperx, the rest interpolated",
                 alignment.matched, alignment.total, alignment.match_ratio * 100)
        if alignment.match_ratio < LOW_MATCH_WARNING:
            log.warning("few words matched — the lyrics may belong to a different version of the song")
    else:
        lines = lines_from_words(words)
        log.info("built %d lines from the transcription", len(lines))
    if not lines:
        raise RuntimeError("nothing to display: no lyrics found and no words transcribed")

    width, height = job.info.width or 1920, job.info.height or 1080
    partial_file = partial_path(ws.subtitles)
    partial_file.write_text(build_ass(lines, width=width, height=height, title=job.title), encoding="utf-8")
    partial_file.replace(ws.subtitles)
    log.info("wrote %d karaoke lines to %s", len(lines), ws.subtitles.name)


def _render(job: Job, ctx: TaskContext) -> str:
    ws = job.workspace
    try:
        source = media.probe_source(ws.video, ws.source, ffprobe=job.ffmpeg.ffprobe)
    except (ffmpeg.Error, OSError) as error:
        log.warning("couldn't inspect the source video (%s) — encoding at constant quality", error)
        source = media.SourceInfo()
    encoding = media.render(ws.video, ws.karaoke_backing, ws.subtitles, ws.final_video, tool=job.ffmpeg,
                            source=source, darken=job.config.darken, duration=job.info.duration,
                            on_progress=ctx.progress)
    size = ws.final_video.stat().st_size / 1_048_576
    log.info("rendered %s (%.0f MiB) with %s", ws.final_video.name, size, encoding)
    return encoding


def _format_length(seconds: float | None) -> str:
    if not seconds:
        return "unknown length"
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes}:{secs:02d}"

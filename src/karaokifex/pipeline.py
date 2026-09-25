"""The karaokifex pipeline: which steps exist, what they need, and what can run in parallel.

    probe ─┬─ lyrics ─────────────────────┬──────────────┬──────────────┐
           ├─ load_whisper ───────────────┤              │              │
           └─ download ─┬─ extract_audio ─ separate_karaoke ─┬─ transcribe ─ force_align ─ subtitles ─ render
                        │                                    └─ vocal_activity ┘ (also → subtitles)  │
                        └─ extract_video ────────────────────────────────────────────────────────────┘

One separation pass yields both stems the rest needs: the backing track (for
render) and the lead vocals (for transcription, vocal activity and forced
alignment). With --mix-vote, `transcribe_mix` also transcribes the full mix.
With --no-burn-lyrics, `render` doesn't wait for `subtitles`, which still writes
the lyrics files. With --palette, `palette` samples frames of the extracted video
for its dominant colours. With --describe, `describe` asks MusicBrainz about the
song (album, year, genres, writers, language) once the lyrics are in. With --keep-source, `original` puts the download's own
sound back under the video, in the output format.
`probe` runs first on its own (its metadata names the song folder); the task
runner handles the rest.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import ffmpeg
import requests
from rich.live import Live

from karaokifex.activity import Activity
from karaokifex.ass import build_ass
from karaokifex.config import Config
from karaokifex.console import TaskBoard, console, register_tasks
from karaokifex.gpu import free_gpu_memory
from karaokifex.metadata import guess_artist_song, name_guesses, title_segments
from karaokifex.models import Lyrics, TimedWord, VideoInfo
from karaokifex.palette import dominant_colors, without_bars
from karaokifex.runner import RunReport, Task, TaskContext, TaskRunner, current_task
from karaokifex.steps import download, lyrics, media, musicbrainz, separation, transcription
from karaokifex.timing import (
    Alignment,
    AlignmentPlan,
    align_lyrics,
    filter_heard,
    forced_requests,
    kept_lines,
    lines_from_words,
    plan_alignment,
)
from karaokifex.workspace import Workspace, partial_path

log = logging.getLogger(__name__)

LOW_MATCH_WARNING = 0.3  # below this quality the lyrics are probably a different version of the song


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

    @property
    def output_video(self) -> Path:
        if not self.config.burn_lyrics:
            video = self.workspace.plain_video
        else:
            video = self.workspace.debug_video if self.config.debug_ass else self.workspace.final_video
        return self._in_output_format(video)

    @property
    def original_video(self) -> Path:
        return self._in_output_format(self.workspace.original_video)

    def _in_output_format(self, video: Path) -> Path:
        return video.with_suffix(".mp4") if self.config.browser_friendly else video


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
        if config.musicbrainz and not (config.artist and config.song):
            artist, song = canonical_names(info, config.artist, config.song, (artist, song))
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


def canonical_names(info: VideoInfo, artist: str | None, song: str | None, guess: tuple[str, str],
                    *, get: musicbrainz.HttpGet = requests.get) -> tuple[str, str]:
    """Artist and song as MusicBrainz spells them, if it confirms them; otherwise `guess`. Given names win."""
    segments = title_segments(info.title)
    # A title that doesn't split into parts may still hold both names ('Smashing Pumpkins Mayonaise').
    text = segments[0] if len(segments) == 1 and not (artist or song or info.artist) else None
    try:
        match = musicbrainz.lookup(name_guesses(info, artist, song), text, get=get)
    except (requests.RequestException, ValueError) as error:
        log.warning("MusicBrainz lookup failed (%s) — keeping “%s – %s”", error, *guess)
        return guess
    if match is None:
        log.info("MusicBrainz isn't sure about this one — keeping “%s – %s”", *guess)
        return guess
    names = (artist or match.artist, song or match.song)
    log.info("MusicBrainz: “%s – %s” (%d matching recordings%s)", *names, match.recordings,
             "" if names == guess else f"; the video's own guess was “{guess[0]} – {guess[1]}”")
    return names


def build_tasks(job: Job) -> list[Task]:
    ws, cfg = job.workspace, job.config
    transcripts = (ws.transcript_json, ws.transcript_mix_json) if cfg.mix_vote else (ws.transcript_json,)
    tasks = [
        Task("lyrics", partial(_lyrics, job), outputs=(ws.lyrics_json,), description="lrclib lookup"),
        Task("download", partial(_download, job), outputs=(ws.source,),
             description="yt-dlp: best video" + (" (H.264 if as good)" if cfg.browser_friendly else "") + " + audio"),
        Task("load_whisper", partial(_load_whisper, job), outputs=transcripts,
             description=f"whisperx {cfg.whisper_model}"),
        Task("extract_audio", partial(_extract_audio, job), deps=("download",), outputs=(ws.audio,),
             description="ffmpeg → audio.wav"),
        Task("extract_video", partial(_extract_video, job), deps=("download",), outputs=(ws.video,),
             description="ffmpeg → video.mkv (no audio)"),
        Task("separate_karaoke", partial(_separate_karaoke, job), deps=("extract_audio",),
             outputs=(ws.karaoke_backing, ws.karaoke_lead), gpu=True, description=cfg.karaoke_model),
        Task("vocal_activity", partial(_vocal_activity, job), deps=("separate_karaoke",),
             outputs=(ws.lead_activity,), description="when the lead vocals are audible"),
        Task("transcribe", partial(_transcribe, job), deps=("separate_karaoke", "load_whisper", "lyrics"),
             outputs=(ws.transcript_json,), gpu=True, description="whisperx on the lead vocals"),
    ]
    if cfg.mix_vote:
        # Depends on transcribe so it can take over the loaded model instead of loading it twice.
        tasks.append(Task("transcribe_mix", partial(_transcribe_mix, job),
                          deps=("extract_audio", "load_whisper", "lyrics", "transcribe"),
                          outputs=(ws.transcript_mix_json,), gpu=True, description="whisperx on the full mix"))
    if cfg.keep_source:
        tasks.append(Task("original", partial(_original, job), deps=("download", "extract_video"),
                          outputs=(job.original_video,), gpu=job.ffmpeg.gpu,
                          description="the original video with its own sound"))
    if cfg.palette:
        tasks.append(Task("palette", partial(_palette, job), deps=("extract_video",), outputs=(ws.metadata_json,),
                          description="dominant colours of the video"))
    if cfg.describe:
        tasks.append(Task("describe", partial(_describe, job), deps=("lyrics",), outputs=(ws.song_json,),
                          description="MusicBrainz: album, year, genres, writers, language"))
    subtitle_deps = ("lyrics", "transcribe", "vocal_activity", "force_align")
    # Without burned-in lyrics the render doesn't wait for them; the subtitles step still writes the lyrics files.
    render_deps = ("extract_video", "separate_karaoke") + (("subtitles",) if cfg.burn_lyrics else ())
    if not cfg.burn_lyrics:
        render_description = "karaoke audio, picture as it is (no lyrics burned in)"
    else:
        render_description = "darken, karaoke audio, burn in subtitles" + (" (debug colours)" if cfg.debug_ass else "")
    tasks += [
        Task("force_align", partial(_force_align, job), deps=("lyrics", "transcribe", "vocal_activity"),
             outputs=(ws.forced_json,), gpu=True, description="wav2vec2 alignment of the known lyrics"),
        Task("subtitles", partial(_subtitles, job),
             deps=subtitle_deps + (("transcribe_mix",) if cfg.mix_vote else ()),
             outputs=(ws.subtitles, ws.debug_subtitles, ws.timings_json),
             description="lyrics + word timings → karaoke ASS"),
        Task("render", partial(_render, job), deps=render_deps, outputs=(job.output_video,), gpu=job.ffmpeg.gpu,
             description=render_description),
    ]
    return tasks


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
    if not found:
        log.warning("no lyrics on lrclib — the whisperx transcription will be used instead")
        return "whisperx transcription (lrclib miss)"
    for candidate in found:
        log.info("found %s (%d lines, %s)", lyrics.describe(candidate), len(candidate.lines),
                 _format_length(candidate.duration))
    others = f" (+{len(found) - 1} alternatives)" if len(found) > 1 else ""
    return lyrics.describe(found[0]) + others


def _download(job: Job, ctx: TaskContext) -> None:
    def on_progress(fraction: float | None, note: str) -> None:
        ctx.progress(fraction)
        ctx.note(note)

    download.download(job.config.url, job.workspace.source, on_progress, prefer_h264=job.config.browser_friendly)
    log.info("downloaded %s (%.0f MiB)", job.workspace.source.name, job.workspace.source.stat().st_size / 1_048_576)


def _extract_audio(job: Job, ctx: TaskContext) -> None:
    media.extract_audio(job.workspace.source, job.workspace.audio, binary=job.ffmpeg.path,
                        duration=job.info.duration, on_progress=ctx.progress)


def _extract_video(job: Job, ctx: TaskContext) -> None:
    media.extract_video(job.workspace.source, job.workspace.video, binary=job.ffmpeg.path,
                        duration=job.info.duration, on_progress=ctx.progress)


def _palette(job: Job, ctx: TaskContext) -> None:
    ws = job.workspace
    ctx.note("sampling frames…")
    frames = media.sample_frames(ws.video, binary=job.ffmpeg.path, ffprobe=job.ffmpeg.ffprobe,
                                 duration=job.info.duration)
    if not len(frames):
        raise RuntimeError(f"no frame of {ws.video.name} could be decoded")
    swatches = dominant_colors(without_bars(frames))
    _write_json(ws.metadata_json, {"palette": {"colors": [swatch.to_dict() for swatch in swatches],
                                               "frames": len(frames)}})
    log.info("dominant colours: %s", ", ".join(f"{swatch.hex} {swatch.weight:.0%}" for swatch in swatches))


def _describe(job: Job, ctx: TaskContext) -> str:
    ctx.note("asking MusicBrainz…")
    candidates = lyrics.load_lyrics(job.workspace.lyrics_json)
    details = describe_song(job.workspace.song_json, job.artist, job.song,
                            job.config.language or (lyrics.guess_language(candidates[0].lines) if candidates else None))
    return ", ".join(str(x) for x in (details.get("album"), details.get("year"), *details.get("genres", [])[:2]) if x) \
        or "MusicBrainz knows nothing more"


def describe_song(target: Path, artist: str, song: str, language: str | None = None, *,
                  get: musicbrainz.HttpGet = requests.get) -> dict[str, Any]:
    """Write song.json: the song's names and what MusicBrainz knows of it. The language is the one
    the song's work is in, else `language` (the lyrics' or the singing's). MusicBrainz not answering
    leaves the rest out, and says so."""
    about: dict[str, Any] = {"artist": artist, "song": song}
    try:
        details = musicbrainz.describe(artist, song, get=get)
    except (requests.RequestException, ValueError) as error:
        log.warning("MusicBrainz didn't answer (%s): no album, year, genres or writers this time", error)
        about["musicbrainz"] = None
    else:
        if details is None:
            log.info("MusicBrainz knows no recording of “%s – %s”", artist, song)
        else:
            about.update(details.to_dict())
            log.info("MusicBrainz: first out %s, on %s · %s · by %s · sung in %s", details.year or "?", details.album or "no album",
                     ", ".join(details.genres) or "no genres", ", ".join(w.name for w in details.writers) or "?",
                     details.language or "?")
    about["language"] = about.get("language") or language
    _write_json(target, about)
    return about


def _separate_karaoke(job: Job, ctx: TaskContext) -> None:
    ws, cfg = job.workspace, job.config
    separation.separate(ws.audio, model=cfg.karaoke_model, model_dir=cfg.model_dir,
                        stems={"instrumental": ws.karaoke_backing, "vocals": ws.karaoke_lead},
                        overlap=cfg.separation_overlap, fp16=cfg.fp16, verbose=cfg.verbose, on_stage=ctx.note)


def _vocal_activity(job: Job, ctx: TaskContext) -> None:
    import soundfile  # only needed here

    samples, rate = soundfile.read(str(job.workspace.karaoke_lead), dtype="float32")
    activity = Activity.compute(samples, rate)
    partial_file = partial_path(job.workspace.lead_activity)
    activity.save(partial_file)
    partial_file.replace(job.workspace.lead_activity)
    log.info("lead vocals audible %.0f%% of the time", 100 * float(activity.voiced.mean()) if activity.duration else 0)


def _load_whisper(job: Job, ctx: TaskContext) -> object:
    ctx.note("loading model (downloaded on first use)…")
    return transcription.load_model(job.config.whisper_model, job.device, job.config.language)


def _song_language(job: Job, candidates: list[Lyrics]) -> str | None:
    """--language wins; otherwise the lyrics' language, which is more reliable than listening to singing."""
    if job.config.language or not candidates:
        return job.config.language
    if language := lyrics.guess_language(candidates[0].lines):
        log.info("language from the lyrics: %s", language)
    return language


def _whisper_model(job: Job, ctx: TaskContext, *upstream: str) -> Any:
    # take(): the runner must not keep the model alive — it would hog GPU memory during render.
    for task in upstream:
        if (model := ctx.take(task)) is not None:
            return model
    ctx.note("loading model…")  # upstream was cached (an old transcript existed) but we must transcribe again
    return transcription.load_model(job.config.whisper_model, job.device, job.config.language)


def _transcribe_into(job: Job, ctx: TaskContext, model: Any, audio: Path, target: Path) -> None:
    candidates = lyrics.load_lyrics(job.workspace.lyrics_json)
    prompt = lyrics.prompt_text(candidates[0].lines) if candidates else None
    words, language = transcription.transcribe(model, audio, job.device, language=_song_language(job, candidates),
                                               prompt=prompt, on_stage=ctx.note)
    transcription.save_transcript(words, language, target)
    log.info("heard %d words in %s (language: %s)", len(words), audio.name, language)


def _transcribe(job: Job, ctx: TaskContext) -> Any:
    model = _whisper_model(job, ctx, "load_whisper")
    _transcribe_into(job, ctx, model, job.workspace.karaoke_lead, job.workspace.transcript_json)
    if job.config.mix_vote:
        return model  # handed over to transcribe_mix, which frees it
    del model
    free_gpu_memory()
    return None


def _transcribe_mix(job: Job, ctx: TaskContext) -> None:
    model = _whisper_model(job, ctx, "transcribe", "load_whisper")
    _transcribe_into(job, ctx, model, job.workspace.audio, job.workspace.transcript_mix_json)
    del model
    free_gpu_memory()


def _force_align(job: Job, ctx: TaskContext) -> None:
    ws = job.workspace
    candidates = lyrics.load_lyrics(ws.lyrics_json)
    words, language = transcription.load_transcript(ws.transcript_json)
    activity = Activity.load(ws.lead_activity)
    plans = [plan_alignment(c.lines, words, activity, language=language) for c in candidates]
    requests = [((number, index), line_words, window)
                for number, (candidate, plan) in enumerate(zip(candidates, plans))
                for index, line_words, window in forced_requests(candidate.lines, plan)]
    results = transcription.force_align(requests, ws.karaoke_lead, language, job.device, on_stage=ctx.note)

    entries = []
    for number, (candidate, plan) in enumerate(zip(candidates, plans)):
        lines: list[list[dict[str, Any]] | None] = [None] * len(kept_lines(candidate.lines))
        for (owner, index), placed in results.items():
            if owner == number and placed is not None:
                lines[index] = [asdict(word) for word in placed]
        aligned = sum(line is not None for line in lines)
        log.info("%s: %s · %d lines force-aligned, %d cut", lyrics.describe(candidate), plan.time_map.describe(),
                 aligned, len(plan.cut))
        entries.append({"lrclib_id": candidate.lrclib_id, "plan": plan.to_dict(), "lines": lines})
    _write_json(ws.forced_json, {"candidates": entries})


def load_forced(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["candidates"] if path.exists() else []


def forced_lines(entry: dict[str, Any]) -> list[list[TimedWord] | None]:
    return [None if line is None else [TimedWord(**word) for word in line] for line in entry["lines"]]


def align_candidates(candidates: list[Lyrics], words: list[TimedWord], language: str | None,
                     activity: Activity | None, forced: list[dict[str, Any]], *,
                     mix: list[TimedWord] | None = None, use_word_tags: bool = True
                     ) -> list[tuple[Lyrics, Alignment]]:
    """Align every lyrics candidate; best (by how well it fits the audio) first."""
    results = []
    for number, candidate in enumerate(candidates):
        entry = forced[number] if number < len(forced) and forced[number]["lrclib_id"] == candidate.lrclib_id else None
        plan = AlignmentPlan.from_dict(entry["plan"]) if entry else None
        alignment = align_lyrics(candidate.lines, words, plan=plan, forced=forced_lines(entry) if entry else None,
                                 activity=activity, language=language, mix_heard=mix, use_word_tags=use_word_tags)
        results.append((candidate, alignment))
    return sorted(results, key=lambda item: -item[1].quality)  # stable: lrclib's ranking breaks ties


def _subtitles(job: Job, ctx: TaskContext) -> str:
    ws = job.workspace
    candidates = lyrics.load_lyrics(ws.lyrics_json)
    words, language = transcription.load_transcript(ws.transcript_json)
    activity = Activity.load(ws.lead_activity) if ws.lead_activity.exists() else None
    mix = transcription.load_transcript(ws.transcript_mix_json)[0] if job.config.mix_vote else None
    if candidates:
        ranked = align_candidates(candidates, words, language, activity, load_forced(ws.forced_json), mix=mix)
        for candidate, alignment in ranked:
            log.info("%s: quality %.2f (%.0f%% heard, forced score %s, %d lines cut)", lyrics.describe(candidate),
                     alignment.quality, alignment.match_ratio * 100,
                     "–" if alignment.forced_score is None else f"{alignment.forced_score:.2f}", alignment.cut)
        chosen, alignment = ranked[0]
        lines = alignment.lines
        sources = Counter(word.source for line in lines for word in line.words)
        log.info("word timing sources: %s", ", ".join(f"{name} {count}" for name, count in sources.most_common()))
        if alignment.quality < LOW_MATCH_WARNING:
            log.warning("the lyrics fit the audio poorly — they may belong to a different version of the song")
        description = lyrics.describe(chosen)
    else:
        lines = lines_from_words(filter_heard(words, activity))
        log.info("built %d lines from the transcription", len(lines))
        description = "whisperx transcription (lrclib miss)"
    if not lines:
        raise RuntimeError("nothing to display: no lyrics found and no words transcribed")

    width, height = job.info.width or 1920, job.info.height or 1080
    for path, debug in ((ws.subtitles, False), (ws.debug_subtitles, True)):
        _write_text(path, build_ass(lines, width=width, height=height, title=job.title, debug=debug))
    _write_json(ws.timings_json, {"lyrics": description, "language": language,
                                  "lines": [[asdict(word) for word in line.words] for line in lines]})
    log.info("wrote %d karaoke lines to %s", len(lines), ws.subtitles.name)
    return description


def _source_info(job: Job) -> media.SourceInfo:
    try:
        return media.probe_source(job.workspace.video, job.workspace.source, ffprobe=job.ffmpeg.ffprobe)
    except (ffmpeg.Error, OSError) as error:
        log.warning("couldn't inspect the source video (%s) — encoding at constant quality", error)
        return media.SourceInfo()


def _original(job: Job, ctx: TaskContext) -> str:
    """The download in the output format: the same picture treatment as the karaoke video, the source's own sound."""
    ws = job.workspace
    encoding = media.render(ws.video, ws.source, None, job.original_video, tool=job.ffmpeg, source=_source_info(job),
                            target_height=job.config.resolution, browser=job.config.browser_friendly,
                            copy_audio=True, duration=job.info.duration, on_progress=ctx.progress)
    size = job.original_video.stat().st_size / 1_048_576
    log.info("wrote %s (%.0f MiB) with %s", job.original_video.name, size, encoding)
    return encoding


def _render(job: Job, ctx: TaskContext) -> str:
    ws = job.workspace
    source = _source_info(job)
    if not job.config.burn_lyrics:
        subtitles = None
    else:
        subtitles = ws.debug_subtitles if job.config.debug_ass else ws.subtitles
    encoding = media.render(ws.video, ws.karaoke_backing, subtitles, job.output_video, tool=job.ffmpeg,
                            source=source, lead=ws.karaoke_lead, lead_volume=job.config.lead_volume,
                            darken=job.config.darken, target_height=job.config.resolution,
                            browser=job.config.browser_friendly, duration=job.info.duration,
                            on_progress=ctx.progress)
    size = job.output_video.stat().st_size / 1_048_576
    log.info("rendered %s (%.0f MiB) with %s", job.output_video.name, size, encoding)
    return encoding


def _write_text(path: Path, text: str) -> None:
    partial_file = partial_path(path)
    partial_file.write_text(text, encoding="utf-8")
    partial_file.replace(path)


def _write_json(path: Path, data: Any) -> None:
    _write_text(path, json.dumps(data, indent=1, ensure_ascii=False))


def _format_length(seconds: float | None) -> str:
    if not seconds:
        return "unknown length"
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes}:{secs:02d}"

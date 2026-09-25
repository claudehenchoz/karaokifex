# AGENTS.md

This file provides guidance to agents when working with code in this repository.

Karaokifex turns a YouTube music video into a karaoke video (lead vocals removed, lyrics burned in with a
word-by-word highlight). See README.md for user-facing usage, the step table and the installation instructions.

## Commands

- `uv sync`: install everything. torch/torchaudio/torchvision come from the PyTorch **cu128** index
  (`[tool.uv.sources]` in pyproject.toml); PyPI only has CPU wheels on Windows.
- `uv run pytest`: the full suite takes about 1 s and needs no GPU, network or ffmpeg.
- Single test: `uv run pytest tests/test_timing.py::test_repeated_chorus_matches_the_right_occurrence`
- `uv run karaokifex <url> -a <artist> -s <song> [-l en] [-o <dir>] [--force] [--keep-source] [--keep-temp] [--palette] [--debug-ass] [--mix-vote] [-v]`
- `uv run karaokifex-eval <song folder>... [--recompute [--baseline]]`: word onset error against a
  `reference.ass`/`reference.lrc` in the folder. `--recompute` needs the folder's temp files and no GPU.
- No linter or formatter is configured.

While a karaokifex run is active, use `uv run --no-sync …`. On Windows the running process locks DLLs,
so `uv sync` fails halfway. If a sync ever uninstalls `onnxruntime-gpu`, the shared `onnxruntime`
package folder gets deleted too: repair it with `uv sync --reinstall-package onnxruntime`.

End-to-end runs use the user's single 8 GB GPU. Never start one while the user's run is active
(`Get-Process karaokifex`); overlapping runs made both many times slower. Send test runs to a scratch
folder with `-o`, never into the user's song folders.

## Architecture

**Task graph + runner.** `pipeline.prepare()` probes the video synchronously; its metadata names the
per-song folder and it picks the ffmpeg binary. `pipeline.build_tasks()` then declares the steps as
`runner.Task`s with `deps`, `outputs` and a `gpu` flag. `runner.TaskRunner` is generic: it starts every
task whose deps are done on a thread pool, shares a GPU semaphore (`--gpu-jobs`), skips the dependents
of failed tasks, and reports state to an observer (the live board).

**Caching / resume.** A task is marked "cached" only if all its `outputs` exist *and* every dependency
was cached too. Correctness therefore depends on outputs being written atomically: write to
`workspace.partial_path(p)`, then rename. Keep that pattern in every new step. `load_whisper` declares
the transcript as its output, so the model isn't loaded when the transcript already exists.

**Results and GPU memory.** Task return values stay in the runner until the run ends. A task consuming
a GPU object must use `ctx.take("task")` rather than `ctx.result(...)`, and then free it
(`gpu.free_gpu_memory()`); see `_transcribe` in pipeline.py.

**Module roles.** `steps/` wraps one external tool per module (yt-dlp, ffmpeg, audio-separator, lrclib,
MusicBrainz, whisperx). Heavy imports (torch, whisperx, audio_separator) happen *inside* functions. Keep
it that way: CLI startup stays fast, and the tests never import them (test_transcription.py injects a
fake `whisperx` module). The pure logic, where the tests concentrate, lives in:
- `timing.py`: times every lyric word. `plan_alignment` maps lrclib onto the video (`mapping.py`),
  derives each line's search window, and marks lines cut from the video. `align_lyrics` then takes, per
  word, the first available of: enhanced-LRC tag (`lrc-tag`), confident forced alignment (`forced`), a
  DP match with a heard word (`whisper`; pairs only inside the line's window, so repeated choruses match
  the right occurrence), and packing into the voiced parts of the gap (`lrc-line`/`interpolated`).
  With `activity` it drops heard words in silence and snaps starts and held-note ends to the voice.
  Every new argument is optional; without them it is plain whisper matching (`karaokifex-eval --baseline`).
- `mapping.py`: `video = scale·lrclib + offset`, fitted by deterministic RANSAC over anchors (heard
  first words of lines), with a cross-correlation prior and piecewise corrections for sections that jumped.
- `activity.py`: RMS voice activity on the lead stem (numpy), stored as `stems/lead_activity.npz`.
- `evaluate.py`: the golden-set metric. Its references live in the user's song folders, never in the
  repo (lyrics are copyrighted); tests use made-up words only.
- `ass.py`: `\kf` karaoke tags. Durations are differences of rounded absolute times, so the tags always
  sum to the line length. Lines alternate between an upper and a lower slot.
- `metadata.py`: artist/song from yt-dlp metadata or the video title, plus the (artist, song) guesses
  (`name_guesses`: every ordered pair of title parts) that `steps/musicbrainz.py` checks.
- `palette.py`: `--palette`, deterministic k-means (fixed-seed k-means++) over frames that
  `media.sample_frames` grabs with one fast seek each (dav1d ignores `-skip_frame nokey`, so decoding
  only keyframes doesn't work); black bars are cropped first. Written to `metadata.json`.
- `steps/musicbrainz.py` also has `describe`: the confirmed recordings are the song; the earliest official
  album of the artist's own gives the album, its recording the genres (weighed with the album's and the
  artist's) and the work, whose relations give the writers and whose language the song's. Written to
  `song.json` by the `describe` step (`--describe`) and by `karaokifex-describe` (`describe.py`).
- `quality.py`: `summary` reads a file's streams with ffprobe; the `quality` step (`--quality`) reads the
  download before cleanup deletes it, and `karaokifex-quality` makes do with `info.json` and the original.

**Logging.** `runner.current_task` (a ContextVar set inside each worker) tags every log line with its
task (`console.TaskLogHandler`). Some libraries install their own console handlers: `setup_logging`
pre-configures the `whisperx` logger, and `console.adopt_loggers` re-routes lightning after import. Known
harmless messages are dropped by `_DropKnownNoise`. `cli.py` sets `TQDM_DISABLE` and
`HF_HUB_DISABLE_PROGRESS_BARS` *before* any other import, so progress bars don't garble the live board.

## Decisions and environment facts (don't undo without reason)

- **One separation pass.** The karaoke model (`mel_band_roformer_karaoke_gabox.ckpt`) outputs the
  backing track (for render) *and* the lead vocals, which are what whisperx transcribes. A separate
  vocal-isolation pass was removed at the user's request because it was very slow.
- **Separation speed.** Separation runs with `use_native_fp16` and overlap 2 (model configs default to 8).
  That is about 4.6× faster; `--overlap 8 --fp32` restores the defaults. `use_autocast` has no effect on
  Roformers.
- **No `[gpu]` extra for audio-separator.** It pulls `onnxruntime-gpu` built for CUDA 13, which conflicts
  with torch's CUDA 12.8 and only matters for ONNX models; Roformers run on CUDA through torch.
  `audioread`, `hf-xet` and `yt-dlp[default]` are explicit dependencies for real reasons (see the
  pyproject comments).
- **ffmpeg selection** (`steps/media.find_ffmpeg`). PATH holds several ffmpeg builds; ImageMagick's 4.2.3
  comes first and can't drive NVENC on current drivers. Each candidate is test-driven (it needs libass
  and a working NVENC encode); `--ffmpeg` / `KARAOKIFEX_FFMPEG` override the choice.
- **Rendering.** CUDA decode plus NVENC (preset p4). eq and subtitles run on the CPU; ffmpeg runs at
  below-normal priority. ffmpeg runs with `cwd` set to the song folder and relative paths, because the
  subtitles filter can't handle Windows drive letters. Output is MKV: the source's codec if the GPU can
  encode it, otherwise the most efficient one it can (the RTX 3070 has no AV1 NVENC, so AV1 sources
  become HEVC). The bitrate is the source's, scaled by `BITRATE_FACTOR`, and the audio keeps the
  source's codec (Opus). With `--no-burn-lyrics` nothing touches the picture (no eq, no subtitles), so
  the video stream is copied and not upscaled (`Config.target_height` is None unless `--upscale`); the
  lyrics files are still written.
- **Browser-friendly** (`--browser-friendly`): MP4 with `+faststart`, H.264 High yuv420p (the only
  format `choose_encoder` may pick then) and AAC. `SourceInfo.browser_ready` decides whether an untouched
  picture can be copied. The download sorts `res,fps,vcodec:h264`, so H.264 wins only when it costs no
  resolution or frame rate; YouTube's H.264 stops at 1080p, so bigger sources get encoded.
- **yt-dlp** needs a JavaScript runtime for YouTube. Node is enabled via `js_runtimes` in `steps/download.py`.
- **Language.** whisperx language auto-detection (first 30 s) is unreliable on singing (it heard Björk as
  Welsh). `-l/--language` forces the language; otherwise it is detected from the lyrics text (langdetect).
- **Forced alignment** uses whisperx's own wav2vec2 aligner, one `whisperx.align` call per line on
  normalised words (characters outside the model's alphabet would become wildcards). CTC places every
  word inside the window even when it's wrong, so windows are clamped to heard neighbours and the per-word
  `score` decides trust (`FORCED_LINE_MIN`/`FORCED_WORD_MIN`; calibrate them with karaokifex-eval).
- **Whisper tweaks.** The lyrics are the `initial_prompt`, set per call via `dataclasses.replace` on
  `model.options`, so `load_whisper` stays independent of `lyrics`. `suppress_numerals` spells numbers out.
  Phonetic matching (jellyfish metaphone; it has no double metaphone) applies to English only.
- **VAD** is RMS thresholding on the clean karaoke stem, not silero/pyannote: deterministic and testable.
  numpy is therefore imported at CLI startup (via timing → mapping), which is acceptable.
- **Names.** `prepare()` asks MusicBrainz (one recording search, all guesses OR-ed as phrase pairs, at
  most 1 request/s with a contact in the User-Agent) before the folder is named. Only recordings whose
  artist and title both match a guess count (`normalize_name`: case, accents, punctuation, a leading
  "The", "&"/"and"); no confirmation keeps the heuristic guess, so offline runs still work. The artist
  is the credited name (in the artist entry's spelling when only that differs), typographic
  punctuation becomes plain. A lookup that changes its mind names a different folder, so a re-run
  that must resume should pass `-a`/`-s`.
- **Lyrics candidates.** `lyrics.json` stores up to 3 lrclib versions; `subtitles` aligns each and keeps
  the best `Alignment.quality`. `load_lyrics` still reads the old single-match format.
- **Cleanup.** `Workspace.artifacts()` defines what survives cleanup, including karaoke videos from
  earlier runs; everything else in the song folder is temporary. Temporary files are deleted after every
  successful run without asking (`--keep-temp` keeps them); failed runs keep everything so they can
  resume. `--keep-source` doesn't keep `source.mkv` itself: the `original` task renders it like the
  karaoke video (same `media.render`, `copy_audio=True`) as `(Original).mkv`/`.mp4`, which is an artifact.

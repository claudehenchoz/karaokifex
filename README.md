# 🎤 Karaokifex

Turn a YouTube music video into a karaoke video: the lead vocals are removed (backing vocals stay), the
picture is slightly darkened, and the lyrics are burned in with a word-by-word highlight.

```
uv run karaokifex "https://www.youtube.com/watch?v=..." --artist "Artist" --song "Song"
```

Artist and song are optional. If you leave them out, they are taken from the video's metadata or title.

## How it works

| Step               | Tool                                                            | Output                                      |
|--------------------|-----------------------------------------------------------------|---------------------------------------------|
| `download`         | yt-dlp, best video + best audio                                 | `source.mkv`                                |
| `extract_audio`    | ffmpeg                                                          | `audio.wav`                                 |
| `extract_video`    | ffmpeg (stream copy)                                            | `video.mkv`                                 |
| `separate_karaoke` | audio-separator, `mel_band_roformer_karaoke_gabox.ckpt`         | `stems/karaoke_backing.wav`, `stems/karaoke_lead.wav` |
| `lyrics`           | lrclib.net, closest match by song length                        | `lyrics.json`                               |
| `load_whisper`     | whisperx model load (runs early, while everything else works)   | –                                           |
| `transcribe`       | whisperx transcription + word alignment on the lead vocals      | `transcript.json`                           |
| `subtitles`        | lrclib lines + whisperx word times → karaoke ASS (`\kf` tags)   | `lyrics.ass`                                |
| `render`           | ffmpeg: darken, burn in subtitles, karaoke audio (GPU decode + NVENC) | `<Artist - Song> (Karaoke).mkv`       |

Each step starts as soon as its inputs exist, so the lyrics lookup, the download, and the whisperx model
load all run at the same time. GPU-heavy steps take turns (`--gpu-jobs` raises that limit). A live task
board shows every step's state, and each log line is tagged with the step that wrote it.

All files for a song go into a folder named `Artist - Song` in the current directory (or `--output-dir`).
If you re-run the same command, any step whose output already exists is skipped, so a failed run picks
up where it stopped (`--force` redoes everything). At the end, karaokifex asks before deleting the
temporary files (`--autodelete` skips the question). It keeps the video, the ASS file, the karaoke audio
track, and the lyrics.

If lrclib has no lyrics for the song, the lines are built from the whisperx transcription instead.

There is a single stem separation pass. The karaoke model's lead-vocal stem doubles as whisperx's input,
which also keeps backing vocals out of the transcription. The Roformer model runs in half precision
with an overlap of 2, about 4.6× faster than audio-separator's defaults (fp32, overlap 8) in a benchmark
on an RTX 3070. `--overlap 8 --fp32` restores those defaults if you want the last bit of quality.

Run `uv run karaokifex --help` for all options.

## Code layout

```
src/karaokifex/
  cli.py          command line (click), cleanup prompt
  pipeline.py     the task graph: which step needs what
  runner.py       generic parallel dependency-graph runner
  console.py      rich logging + live task board
  workspace.py    per-song folder and file names
  timing.py       lyrics + word timestamps → timed lines (pure)
  ass.py          timed lines → karaoke ASS (pure)
  metadata.py     artist/song from video metadata (pure)
  steps/          one module per external tool: download, media, separation, lyrics, transcription
```

## Development

```
uv sync
uv run pytest
```

Rendering decodes and encodes on the GPU (`-hwaccel cuda` + NVENC); only the darkening and subtitle
filters run on the CPU. The output is an MKV, like the download. The video keeps the source's format when
the GPU can encode it; otherwise it uses the most efficient format the GPU can encode. For example, an
RTX 30xx can't encode AV1, so AV1 sources become HEVC. The bitrate follows the source's, scaled by how
efficient the output format is, so the file ends up about the size of the original. The audio keeps
the source's codec (usually Opus). PATH often holds several ffmpeg builds (ImageMagick ships an old one),
so karaokifex test-drives each one and uses the first that has libass and can encode with NVENC. If none
can, it falls back to x264 on the CPU at below-normal priority. Use `--ffmpeg` or `KARAOKIFEX_FFMPEG` to
pick a specific one.

Requirements: ffmpeg with libass on `PATH` (4.3 or newer for NVENC on current drivers), and Node.js or Deno (yt-dlp needs one for YouTube). An NVIDIA
GPU is strongly recommended; PyTorch is installed from the CUDA 12.8 index. On first use, the separation
models (~1 GB) and whisper large-v3 (~3 GB) are downloaded.

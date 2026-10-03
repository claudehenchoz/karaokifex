# 🎤 Karaokifex

Turn a YouTube music video into a karaoke video: the lead vocals are removed (backing vocals stay), the
picture is slightly darkened, and the lyrics are burned in with a word-by-word highlight.

```
karaokifex "https://www.youtube.com/watch?v=..." --artist "Artist" --song "Song"
```

Artist and song are optional. If you leave them out, they are taken from the video's metadata or title
and checked against [MusicBrainz](https://musicbrainz.org), which also settles their spelling. Titles like
`Say Anything • In Your Eyes • Peter Gabriel` or `Smashing Pumpkins "Mayonaise"` work too: every part of
the title is tried as artist and as song, and the pair that MusicBrainz recordings confirm wins. When
MusicBrainz isn't sure (or can't be reached), the names from the video are used as they are.
`--no-musicbrainz` skips the lookup; names given with `--artist`/`--song` always win. The song folder and
every file in it are named `Artist - Song` after the result.

## Installation

Prerequisites:

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- An NVIDIA GPU with a current driver (strongly recommended; PyTorch is installed as a CUDA 12.8 build)
- [ffmpeg](https://ffmpeg.org/download.html) on `PATH`, built with libass; version 4.3 or newer for GPU encoding
- [Node.js](https://nodejs.org/) or [Deno](https://deno.com/) on `PATH` (yt-dlp needs one for YouTube)

Install karaokifex as a command-line tool:

```
uv tool install --python 3.12 git+https://github.com/claudehenchoz/karaokifex
```

Afterwards `karaokifex` is available in any terminal. If it isn't found, run `uv tool update-shell` and
open a new terminal. On first use, the separation model (~1 GB) and whisper large-v3 (~3 GB) are
downloaded. Each song gets its own folder in the directory you run karaokifex from.

Upgrade or remove it with:

```
uv tool upgrade karaokifex
uv tool uninstall karaokifex
```

To run it from a checkout instead:

```
git clone https://github.com/claudehenchoz/karaokifex
cd karaokifex
uv sync
uv run karaokifex "https://www.youtube.com/watch?v=..."
```

## How it works

| Step               | Tool                                                            | Output                                      |
|--------------------|-----------------------------------------------------------------|---------------------------------------------|
| `download`         | yt-dlp, best video + best audio (`--browser-friendly`: H.264 if as good) | `source.mkv`                       |
| `extract_audio`    | ffmpeg                                                          | `audio.wav`                                 |
| `extract_video`    | ffmpeg (stream copy)                                            | `video.mkv`                                 |
| `original`         | only with `--keep-source`: the video with its own sound, made like the karaoke video | `<Artist - Song> (Original).mkv` |
| `palette`          | only with `--palette`: the video's dominant colours (k-means over 32 sampled frames) | `metadata.json`        |
| `describe`         | only with `--describe`: MusicBrainz's album, year, genres, writers, language, artist's country | `song.json` |
| `quality`          | only with `--quality`: the download's and the renders' resolution, frame rate, codecs, bitrates | `quality.json` |
| `separate_karaoke` | audio-separator, `mel_band_roformer_karaoke_gabox.ckpt` (or several, averaged); the backing is the song minus the lead | `stems/karaoke_backing.wav`, `stems/karaoke_lead.wav` |
| `lyrics`           | lrclib.net, up to 3 versions (the one that fits the audio wins) | `lyrics.json`                               |
| `load_whisper`     | whisperx model load (runs early, while everything else works)   | –                                           |
| `vocal_activity`   | energy envelope of the lead vocals: when someone is singing     | `stems/lead_activity.npz`                   |
| `transcribe`       | whisperx on the lead vocals, prompted with the lyrics           | `transcript.json`                           |
| `transcribe_mix`   | only with `--mix-vote`: whisperx on the full mix                | `transcript_mix.json`                       |
| `force_align`      | lrclib → video time map, then wav2vec2 forced alignment of each lyric line | `forced.json`                    |
| `subtitles`        | best timing source per word, snapped to the voice → karaoke ASS (`\kf` tags) | `lyrics.ass`, `lyrics.debug.ass`, `timings.json` |
| `render`           | ffmpeg: darken, burn in subtitles, karaoke audio (GPU decode + NVENC) | `<Artist - Song> (Karaoke).mkv` (or `.mp4`) |

Each step starts as soon as its inputs exist, so the lyrics lookup, the download, and the whisperx model
load all run at the same time. GPU-heavy steps take turns (`--gpu-jobs` raises that limit). Songs made at
once on one machine can share a lock file (`--gpu-lock`, or `KARAOKIFEX_GPU_LOCK`): their separation,
transcription and alignment then take turns on the GPU too, instead of filling its memory together, while
one song's download or render overlaps another's models. A live task board shows every step's state, and
each log line is tagged with the step that wrote it.

All files for a song go into a folder named `Artist - Song` in the current directory (or `--output-dir`).
If you re-run the same command, any step whose output already exists is skipped, so a failed run picks
up where it stopped (`--force` redoes everything). After a successful run, karaokifex deletes the
temporary files. It keeps the video, the ASS file, the karaoke audio track, the lyrics with their word
timings (`timings.json`), and the video's metadata (`info.json`, `metadata.json`). `--keep-source` also
keeps the original: the video with its own sound (vocals and all), as `<Artist - Song> (Original).mkv`.
It is made like the karaoke video, in the same format and resolution (an MP4 with `--browser-friendly`),
and its audio is copied when that format takes it. The raw download (`source.mkv`) is still deleted;
`--keep-temp` keeps every file.

### How the words get their timing

The lyrics are known, so the job is finding *when* each word is sung:

1. **Time map.** Music videos often have a longer intro, are sped up a few percent, or cut a verse.
   lrclib's line starts are first cross-correlated with the vocal onsets, giving a global offset. A robust
   line fit (`video = scale × lrclib + offset`) through the lines whisperx heard then refines it, and
   sections that jumped get their own shift. Lines whose place in the video is silent, or that don't fit
   between their heard neighbours, were cut from the video and are not shown.
2. **Forced alignment.** Each lyric line is aligned inside its window on the lead vocals by wav2vec2 (the
   same aligner whisperx uses), so every word gets a time and a confidence score.
3. **Merging.** Per word, the best source wins: enhanced-LRC word tags from lrclib, then a confident forced
   alignment, then the words whisperx heard (matched by spelling, sound, and "alright"/"all right"-style
   merges), and finally interpolation into the parts of the gap where someone sings.
4. **The voice decides.** Words whisperx "heard" in silence are dropped, words don't start in silence,
   and a line's last word lasts until the held note ends.

Whisper is prompted with the lyrics and set to the lyrics' language (detected from their text, or
`--language`). `--mix-vote` additionally transcribes the full mix and lets both transcriptions vote,
at the cost of a second whisperx pass. If lrclib has no lyrics for the song, the lines are built from
the whisperx transcription instead.

To check timing, `--debug-ass` renders `(Karaoke debug).mkv` with each word coloured by what timed it:
violet LRC tag, green forced alignment, cyan whisperx, orange lrclib line start, red interpolated.
Low-confidence words are underlined. `lyrics.debug.ass` is written on every run, so
`mpv video.mkv --sub-file=lyrics.debug.ass` shows the same thing without rendering.

To measure timing, put a `reference.ass` (for example timed in Aegisub against the video) or an enhanced
`reference.lrc` into a song folder and keep its temporary files (`--keep-temp`).
`karaokifex-eval <song folder>...` then reports the median word onset error and the share of words within
100/300 ms, per timing source. `--recompute` re-runs the alignment from the cached files (no GPU needed),
and `--baseline` compares against whisper-only matching.

There is a single stem separation pass. The karaoke model's lead-vocal stem doubles as whisperx's input,
which also keeps backing vocals out of the transcription. The Roformer model runs in half precision
with an overlap of 2, about 4.6× faster than audio-separator's defaults (fp32, overlap 8) in a benchmark
on an RTX 3070. `--overlap 8 --fp32` restores those defaults if you want the last bit of quality.

The karaoke is the song minus its lead vocal, not the model's own "instrumental". audio-separator scales
the mix before separating and each stem after, so that stem came out a few dB quieter than the song across
the board, bass and air included: a thinner copy. The lead is fitted back to the song's own level (least
squares), and the backing is what is left of the song, at its level and fullness. `--karaoke-model` can be
given several times: each model separates the song, and their leads are averaged (an ensemble), which the
top of MVSEP's lead/back-vocals leaderboard does too. For example
`--karaoke-model bs_roformer_karaoke_frazer_becruily.ckpt --karaoke-model bs_roformer_karaoke_anvuew.ckpt
--karaoke-model mel_band_roformer_karaoke_gabox.ckpt`. Each model is a pass of its own: those three with
`--overlap 8 --fp32` took about nine times as long to separate a song as the first of them alone with
`--overlap 4` (6.6 minutes against 43 s on an RTX 4070 Ti), about five times as long for the whole song.

Without its lead voice the song is quieter; `--match-loudness` brings the karaoke to the original's
loudness (EBU R128), a limiter keeping its peaks under -1 dBFS, so switching between the two doesn't change
the level.

Some songs, old masters mostly, are mastered far quieter than the rest (-17 LUFS and below, where most sit
between -8 and -14) and play noticeably softer. `--lift-quiet`, on by default, lifts a song quieter than
`--quiet-floor` (-12 LUFS, a couple of dB under most songs) up to it, by at most 12 dB, the original and the karaoke alike, the same limiter keeping the peaks
under -1 dBFS; louder songs are left as they are (`--no-lift-quiet` turns it off). A quiet song's original is
then re-encoded rather than having its sound copied. `karaokifex-lift <song folder>... [--dry]` does the same
for songs made before it: both renders' sound lifted by the same amount, the picture copied.
 
Rendering decodes and encodes on the GPU (`-hwaccel cuda` + NVENC); only the darkening and subtitle
filters run on the CPU. With burned-in lyrics the default output height is 1080p: sources below
1080p are upscaled proportionally, so the lyrics render sharply, while larger sources keep their
original resolution. Use `--resolution` to choose a different minimum output height. The isolated lead vocal is omitted by default; use `--lead-volume`
between 0 and 1 to mix it back into the karaoke audio.

`--no-burn-lyrics` leaves the picture as it is: no lyrics, no darkening, and no upscaling -- the player
scales the picture anyway. The video stream is then copied without re-encoding. `--upscale` scales a
smaller source up to `--resolution` even so, and `--no-upscale` keeps every source's size with burned-in
lyrics too. The result is
`<Artist - Song> (Karaoke, no lyrics).mkv`, and the lyrics are still written to `lyrics.ass` and
`timings.json` (every word with its start and end), for a player that shows them itself.

`--browser-friendly` writes an MP4 that every browser's `<video>` plays: H.264 (High profile, 8-bit
4:2:0), AAC audio, and the index at the front of the file ("fast start"), so playback begins while it
loads. The download then prefers H.264 whenever YouTube offers it at the best resolution and frame rate.
Such a video is copied, whatever its size, whenever nothing touches the picture -- no lyrics burned in
and no upscaling -- and only the audio is encoded: a render takes seconds. Burned-in lyrics, other
formats (VP9, AV1) and upscaling mean an H.264 encode.

`--palette` finds the video's five dominant colours and writes them to `metadata.json`, most common
first, each with its share of the picture (black letterbox and pillarbox bars don't count):

```
{"palette": {"colors": [{"hex": "#282522", "rgb": [40, 37, 34], "weight": 0.2803}, ...], "frames": 32}}
```

The colours come from k-means over 32 small frames spread across the video, one fast seek each, so the
step takes seconds and runs while the stems are separated.

`--describe` asks MusicBrainz what it knows of the song and writes it to `song.json`: the album it
first came out on (the earliest official album of the artist's own, no compilation or live record),
the year it first came out, its genres (the recording's, its album's and its artist's votes), its
writers with their roles, the language it is sung in (ISO 639-1; the lyrics' guess when MusicBrainz has
none) and where the artist is from. It runs once the lyrics are in, at most six requests a second
apart. `karaokifex-describe <song folder>...` fills `song.json` in for songs made before.

`--quality` writes `quality.json`: the download's resolution, frame rate, codecs and bitrates, read
before the download is deleted, the same for each render, and whether a render was upscaled (the
download's own size is otherwise lost under an upscale). `karaokifex-quality <song folder>...` fills it
in for songs made before, from what is left: `info.json`'s resolution, and the original's streams where
the original is the download's picture copied.

lrclib is asked again when it answers busy (502, 503, 504) or not at all, waiting 2, 4, 8 and 16 s.

Without `--browser-friendly`, the output is an MKV, like the download. The video keeps the source's
format when the GPU can encode it; otherwise it uses the most efficient format the GPU can encode. For
example, an RTX 30xx can't encode AV1, so AV1 sources become HEVC. The bitrate follows the source's,
scaled by how efficient the output format is, so the file ends up about the size of the original. The
audio keeps the source's codec (usually Opus). PATH often holds several ffmpeg builds (ImageMagick ships an old one),
so karaokifex test-drives each one and uses the first that has libass and can encode with NVENC. If none
can, it falls back to x264 on the CPU at below-normal priority. Use `--ffmpeg` or `KARAOKIFEX_FFMPEG` to
pick a specific one.

Run `karaokifex --help` for all options.

## Code layout

```
src/karaokifex/
  cli.py          command line (click), cleanup
  pipeline.py     the task graph: which step needs what
  runner.py       generic parallel dependency-graph runner
  console.py      rich logging + live task board
  workspace.py    per-song folder and file names
  timing.py       lyrics + word timestamps → timed lines (pure)
  mapping.py      lrclib → video time map (pure)
  activity.py     when the lead vocals are audible (pure)
  ass.py          timed lines → karaoke ASS (pure)
  evaluate.py     karaokifex-eval: timing accuracy against a reference
  metadata.py     artist/song guesses from video metadata (pure)
  palette.py      dominant colours of the video (pure)
  describe.py     karaokifex-describe: song.json for songs made before --describe
  quality.py      --quality and karaokifex-quality: the streams of the download and the renders
  level.py        --lift-quiet and karaokifex-lift: how much a quiet song is lifted, and the lift itself
  steps/          one module per external tool: download, media, separation, lyrics, transcription,
                  musicbrainz
```

## Development

```
uv sync
uv run pytest
```

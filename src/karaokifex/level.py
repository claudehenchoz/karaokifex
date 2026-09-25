"""How loud a song comes out: the karaoke matched to the song (--match-loudness), a song too quiet lifted.

    karaokifex-lift <song folder>... [--floor -16] [--dry]

Some songs, old masters mostly, are mastered far quieter than the rest (-20 LUFS and below where
most sit near -10), so they play noticeably softer. --lift-quiet, on by default, brings a song
quieter than QUIET_FLOOR up to it, by at most MOST dB, the original and the karaoke alike, so
switching between them keeps the level; the render's limiter keeps the peaks under -1 dBFS.
Louder songs are left as they are. `gains()` is the arithmetic; `main` lifts songs made before
it: their renders' sound re-encoded with the gain, the picture copied.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import click

log = logging.getLogger("karaokifex")

QUIET_FLOOR = -16.0   # LUFS: a song quieter than this is lifted to it
MOST = 12.0           # dB: the most a song is lifted


def lift(level: float | None, floor: float = QUIET_FLOOR) -> float:
    """The dB a song at `level` LUFS is lifted by: up to `floor`, at most MOST; none when it is loud enough."""
    return min(MOST, floor - level) if level is not None and level < floor else 0.0


def gains(song: float | None, karaoke: float | None, *, match: bool, lift_quiet: bool,
          floor: float = QUIET_FLOOR) -> tuple[float, float]:
    """The dB the original and the karaoke get. Matched, the karaoke comes to the song's loudness, and a
    quiet song lifts both alike; unmatched, each is lifted from its own loudness."""
    matched = match and song is not None and karaoke is not None
    original = lift(song, floor) if lift_quiet else 0.0
    karaoke_gain = song - karaoke if matched else 0.0
    if lift_quiet:
        karaoke_gain += original if matched else lift(karaoke, floor)
    return original, karaoke_gain


def _renders(folder: Path) -> dict[str, Path]:
    """The song's karaoke and original renders, the browser-friendly MP4 first."""
    out = {}
    for name, marker in (("karaoke", "(Karaoke"), ("original", "(Original)")):
        found = sorted((p for p in folder.iterdir() if marker in p.name and p.suffix in (".mp4", ".mkv")
                        and ".partial" not in p.name), key=lambda p: p.suffix != ".mp4")
        if found:
            out[name] = found[0]
    return out


def relevel(path: Path, gain_db: float, *, binary: str = "ffmpeg") -> None:
    """The file's sound `gain_db` louder, peaks limited under -1 dBFS, re-encoded as AAC; the picture copied."""
    from karaokifex.workspace import partial_path

    partial = partial_path(path)
    args = [binary, "-hide_banner", "-nostats", "-loglevel", "error", "-y", "-i", str(path), "-map", "0:v:0?", "-map", "0:a:0",
            "-c:v", "copy", "-af", f"volume={gain_db:.2f}dB,alimiter=limit=0.891:level=0", "-c:a", "aac", "-b:a", "256k",
            *(["-movflags", "+faststart"] if path.suffix == ".mp4" else []), "-f", "mp4" if path.suffix == ".mp4" else "matroska",
            str(partial)]
    subprocess.run(args, check=True, stdin=subprocess.DEVNULL)
    partial.replace(path)


@click.command()
@click.argument("folders", nargs=-1, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--floor", type=float, default=QUIET_FLOOR, show_default=True, help="Lift songs quieter than this (LUFS).")
@click.option("--dry", is_flag=True, help="Only say which songs are too quiet and by how much they would be lifted.")
@click.option("--ffmpeg", "binary", default="ffmpeg", show_default=True, help="ffmpeg executable.")
def main(folders: tuple[Path, ...], floor: float, dry: bool, binary: str) -> None:
    """Lift songs made before --lift-quiet that are too quiet: both renders by the same dB, the picture copied."""
    from karaokifex.console import setup_logging
    from karaokifex.steps.media import loudness

    setup_logging(False)
    for folder in folders:
        renders = _renders(folder)
        reference = renders.get("original") or renders.get("karaoke")
        if reference is None:
            continue
        level = loudness(reference, binary=binary)
        gain = lift(level, floor)
        if not gain:
            log.info("%s: %s LUFS, loud enough", folder.name, f"{level:.1f}" if level is not None else "unknown")
            continue
        log.info("%s: %.1f LUFS, %slifted %+.1f dB", folder.name, level, "would be " if dry else "", gain)
        if dry:
            continue
        for path in renders.values():
            try:
                relevel(path, gain, binary=binary)
            except subprocess.CalledProcessError as error:
                log.warning("%s: %s could not be lifted (%s)", folder.name, path.name, error)


if __name__ == "__main__":
    main()

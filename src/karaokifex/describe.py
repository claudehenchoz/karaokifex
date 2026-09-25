"""Describe songs already made: what MusicBrainz knows of each, written to its song.json.

    karaokifex-describe <song folder>... [--force]

For songs made before --describe (or when MusicBrainz didn't answer). The names are the folder's
("Artist - Song"); the language falls back to the one timings.json heard.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from karaokifex.console import setup_logging
from karaokifex.pipeline import describe_song
from karaokifex.workspace import Workspace

log = logging.getLogger("karaokifex")


@click.command(context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 100})
@click.argument("folders", nargs=-1, required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--force", is_flag=True, help="Ask again for songs whose song.json MusicBrainz has already filled.")
@click.option("-v", "--verbose", is_flag=True, help="Show debug output.")
def main(folders: tuple[Path, ...], force: bool, verbose: bool) -> None:
    """Write song.json into each song folder: album, year, genres, writers and language from MusicBrainz."""
    setup_logging(verbose)
    for folder in folders:
        target = Workspace(folder, folder.name).song_json
        if target.exists() and not force and json.loads(target.read_text(encoding="utf-8")).get("musicbrainz"):
            log.info("%s: described already", folder.name)
            continue
        artist, _, song = folder.name.partition(" - ")
        if not song:
            log.warning("%s: not named “Artist - Song”, skipped", folder.name)
            continue
        timings = folder / "timings.json"
        heard = json.loads(timings.read_text(encoding="utf-8")).get("language") if timings.exists() else None
        log.info("%s", folder.name)
        describe_song(target, artist, song, heard)


if __name__ == "__main__":
    main()

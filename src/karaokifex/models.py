"""Plain data types shared between the pipeline steps."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class VideoInfo:
    """The bits of yt-dlp metadata the pipeline cares about."""

    id: str
    title: str
    duration: float | None = None
    width: int | None = None
    height: int | None = None
    artist: str | None = None
    track: str | None = None
    uploader: str | None = None

    @classmethod
    def from_ytdlp(cls, info: dict[str, Any]) -> VideoInfo:
        artists = info.get("artists")
        return cls(
            id=info.get("id", ""),
            title=info.get("title", ""),
            duration=info.get("duration"),
            width=info.get("width"),
            height=info.get("height"),
            artist=info.get("artist") or (", ".join(artists) if artists else None) or info.get("creator"),
            track=info.get("track"),
            uploader=info.get("uploader") or info.get("channel"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoInfo:
        return cls(**data)


@dataclass(frozen=True)
class LyricLine:
    """One line of lyrics; `start` is only known for synced (LRC) lyrics.

    Enhanced LRC also times single words: `word_starts` then has one entry per
    `timing.split_words(text)` word (None for untagged words), and `end` is the
    closing tag after the last word, if any.
    """

    start: float | None
    text: str
    word_starts: tuple[float | None, ...] | None = None
    end: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LyricLine:
        starts = data.get("word_starts")
        return cls(data["start"], data["text"], None if starts is None else tuple(starts), data.get("end"))


@dataclass(frozen=True)
class Lyrics:
    """A lyrics match from lrclib."""

    lrclib_id: int | None
    artist: str
    track: str
    album: str | None
    duration: float | None
    synced: bool
    lines: tuple[LyricLine, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Lyrics:
        lines = tuple(LyricLine.from_dict(line) for line in data["lines"])
        return cls(**{**data, "lines": lines})


@dataclass(frozen=True)
class TimedWord:
    """A word with start/end time in seconds (from whisperx, or after merging).

    `score` is the aligner's confidence (0–1) when known. `source` says which
    mechanism timed a merged word: forced, whisper, lrc-tag, lrc-line or interpolated.
    """

    text: str
    start: float
    end: float
    score: float | None = None
    source: str = ""


@dataclass(frozen=True)
class TimedLine:
    """A display line of the karaoke track."""

    words: tuple[TimedWord, ...]

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words)

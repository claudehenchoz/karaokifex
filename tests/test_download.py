from karaokifex.steps import download


class FakeYoutubeDL:
    options: dict = {}

    def __init__(self, options):
        FakeYoutubeDL.options = options

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def download(self, urls):
        pass


def test_browser_friendly_download_prefers_h264(monkeypatch, tmp_path):
    monkeypatch.setattr(download.yt_dlp, "YoutubeDL", FakeYoutubeDL)
    target = tmp_path / "source.mkv"
    target.write_text("x")
    download.download("https://youtu.be/x", target, lambda fraction, note: None)
    assert "format_sort" not in FakeYoutubeDL.options
    download.download("https://youtu.be/x", target, lambda fraction, note: None, prefer_h264=True)
    assert FakeYoutubeDL.options["format_sort"] == ["res", "fps", "vcodec:h264"]
    assert FakeYoutubeDL.options["format"] == "bv*+ba/b"

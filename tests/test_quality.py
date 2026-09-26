import json

from karaokifex import quality


def probe_of(width, height, *, vcodec="h264", fps="30000/1001", vrate="4000000", acodec="aac", arate="128000"):
    return {"format": {"format_name": "mov,mp4", "size": "1000000", "duration": "200.0", "bit_rate": "4128000"},
            "streams": [{"codec_type": "video", "codec_name": vcodec, "profile": "High", "width": width, "height": height,
                         "avg_frame_rate": fps, "bit_rate": vrate, "pix_fmt": "yuv420p"},
                        {"codec_type": "audio", "codec_name": acodec, "bit_rate": arate, "sample_rate": "48000",
                         "channels": 2}]}


def test_summary_reads_the_streams(tmp_path, monkeypatch):
    monkeypatch.setattr(quality.ffmpeg, "probe", lambda path, cmd: probe_of(1920, 1080))
    s = quality.summary(tmp_path / "x.mp4")
    assert s["video"] == {"codec": "h264", "profile": "High", "width": 1920, "height": 1080, "fps": 29.97,
                          "bitrate": 4000000, "pix_fmt": "yuv420p"}
    assert s["audio"] == {"codec": "aac", "bitrate": 128000, "sample_rate": 48000, "channels": 2}
    assert s["bitrate"] == 4128000 and s["duration"] == 200.0


def test_write_quality_tells_an_upscale(tmp_path, monkeypatch):
    monkeypatch.setattr(quality.ffmpeg, "probe", lambda path, cmd: probe_of(1920, 1440))
    render = tmp_path / "Song (Karaoke).mp4"
    render.write_bytes(b"x")
    source = {"video": {"width": 640, "height": 480}}
    q = quality.write_quality(tmp_path / "quality.json", source=source, renders={"karaoke": render,
                                                                                  "original": tmp_path / "missing.mp4"})
    assert q["upscaled"] and "original" not in q
    assert json.loads((tmp_path / "quality.json").read_text())["karaoke"]["video"]["height"] == 1440


def test_after_the_fact_takes_the_original_when_it_is_the_download_copied(tmp_path, monkeypatch):
    (tmp_path / "info.json").write_text(json.dumps({"width": 1280, "height": 720}))
    original = tmp_path / "Song (Original).mp4"
    original.write_bytes(b"x")
    monkeypatch.setattr(quality.ffmpeg, "probe", lambda path, cmd: probe_of(1280, 720, vrate="2500000"))
    source = quality.after_the_fact(tmp_path, {"original": original})
    assert source["video"]["bitrate"] == 2500000 and "copied" in source["from"]
    monkeypatch.setattr(quality.ffmpeg, "probe", lambda path, cmd: probe_of(1920, 1080))  # upscaled: not the download
    assert quality.after_the_fact(tmp_path, {"original": original}) == {"from": "info.json",
                                                                        "video": {"width": 1280, "height": 720}}

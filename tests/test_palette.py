import numpy as np

from karaokifex.palette import Swatch, dominant_colors, without_bars

RED, BLUE, YELLOW = (200, 30, 40), (20, 40, 180), (250, 240, 10)


def striped_frames(noise: int = 0) -> np.ndarray:
    """Ten 64×40 frames: half red, 30% blue, 20% yellow (by rows), with optional pixel noise."""
    frames = np.zeros((10, 40, 64, 3), dtype=np.int64)
    frames[:, :20], frames[:, 20:32], frames[:, 32:] = RED, BLUE, YELLOW
    if noise:
        frames += np.random.default_rng(1).integers(-noise, noise + 1, frames.shape)
    return frames.clip(0, 255).astype(np.uint8)


def test_finds_the_colours_by_their_share_of_the_picture():
    swatches = dominant_colors(striped_frames(noise=4), count=3)
    assert [s.weight for s in swatches] == [0.5, 0.3, 0.2]
    for swatch, expected in zip(swatches, (RED, BLUE, YELLOW)):
        assert max(abs(a - b) for a, b in zip(swatch.rgb, expected)) <= 2


def test_is_deterministic():
    assert dominant_colors(striped_frames(noise=30)) == dominant_colors(striped_frames(noise=30))


def test_weights_add_up_and_come_sorted():
    swatches = dominant_colors(striped_frames(noise=30))
    assert len(swatches) == 5
    assert abs(sum(s.weight for s in swatches) - 1) < 1e-9
    assert [s.weight for s in swatches] == sorted((s.weight for s in swatches), reverse=True)


def test_fewer_colours_than_asked_for():
    assert dominant_colors(striped_frames()) == [Swatch(RED, 0.5), Swatch(BLUE, 0.3), Swatch(YELLOW, 0.2)]
    assert dominant_colors(np.zeros((2, 8, 8, 3), dtype=np.uint8)) == [Swatch((0, 0, 0), 1.0)]
    assert dominant_colors(np.zeros((0, 8, 8, 3), dtype=np.uint8)) == []


def test_swatch_as_json():
    assert Swatch((255, 16, 0), 0.123456).to_dict() == {"hex": "#ff1000", "rgb": [255, 16, 0], "weight": 0.1235}


def test_black_bars_are_cropped():
    frames = np.zeros((3, 36, 64, 3), dtype=np.uint8)
    frames[:, 5:31, 8:56] = RED  # letterbox and pillarbox bars around the picture
    frames[1, 10, 20] = (0, 0, 0)  # a black pixel inside the picture stays
    cropped = without_bars(frames)
    assert cropped.shape == (3, 26, 48, 3)
    assert dominant_colors(cropped)[0] == Swatch(RED, 1 - 1 / cropped[..., 0].size)


def test_black_frames_are_not_cropped_away():
    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    assert without_bars(frames).shape == frames.shape

import pytest

from fall_detection.app.media_capture import frame_delay_ms, parse_viewport


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1440x1000", {"width": 1440, "height": 1000}),
        ("390X844", {"width": 390, "height": 844}),
    ],
)
def test_parse_viewport(value, expected):
    assert parse_viewport(value) == expected


@pytest.mark.parametrize("value", ["", "1440", "0x100", "wide"])
def test_parse_viewport_rejects_invalid_sizes(value):
    with pytest.raises(ValueError):
        parse_viewport(value)


def test_frame_delay_ms_uses_requested_fps():
    assert frame_delay_ms(8.0) == 125


@pytest.mark.parametrize("fps", [0, -1])
def test_frame_delay_ms_rejects_nonpositive_fps(fps):
    with pytest.raises(ValueError):
        frame_delay_ms(fps)

from pathlib import Path
from unittest.mock import Mock

from fall_detection.io.video import H264VideoWriter


def test_constructor_closes_mkstemp_file_descriptor(monkeypatch, tmp_path):
    temp_path = tmp_path / "temporary.mp4"
    fake_writer = Mock()
    fake_writer.isOpened.return_value = True
    close_fd = Mock()
    monkeypatch.setattr(
        "fall_detection.io.video.tempfile.mkstemp", lambda **_kwargs: (47, str(temp_path))
    )
    monkeypatch.setattr("fall_detection.io.video.os.close", close_fd)
    monkeypatch.setattr("fall_detection.io.video.cv2.VideoWriter", Mock(return_value=fake_writer))
    monkeypatch.setattr("fall_detection.io.video.cv2.VideoWriter_fourcc", Mock(return_value=0))

    H264VideoWriter(tmp_path / "output.mp4", 30.0, 640, 480)

    close_fd.assert_called_once_with(47)


def test_close_retries_windows_tempfile_cleanup(monkeypatch, tmp_path):
    writer = object.__new__(H264VideoWriter)
    writer.out_path = tmp_path / "output.mp4"
    writer._tmp = tmp_path / "locked.mp4"
    writer._tmp.write_bytes(b"video")
    writer._writer = Mock()

    monkeypatch.setattr("fall_detection.io.video.reencode_h264", Mock())
    original_unlink = Path.unlink
    calls = 0

    def flaky_unlink(path, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise PermissionError("file is still in use")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)

    writer.close()

    writer._writer.release.assert_called_once_with()
    assert calls == 2
    assert not writer._tmp.exists()

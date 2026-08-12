import subprocess
import sys


def test_cli_help_is_utf8_decodable_on_windows():
    """Removing UTF-8 stream setup must reproduce mojibake on Windows."""
    completed = subprocess.run(
        [sys.executable, "-m", "fall_detection.cli", "evaluate", "--help"],
        check=True,
        capture_output=True,
    )

    output = completed.stdout.decode("utf-8")
    assert "模型快取根目錄" in output
    assert "快取子目錄名稱" in output

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_published_test_count_matches_collected_suite():
    """A new offline test must not leave the public evidence count stale."""
    metrics = json.loads(
        (REPO_ROOT / "eval" / "metrics.json").read_text(encoding="utf-8")
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-m",
            "not inference",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    match = re.search(r"(\d+)(?:/\d+)? tests collected", completed.stdout)

    assert match is not None
    assert metrics["repository_evidence"]["offline_tests"] == int(match.group(1))

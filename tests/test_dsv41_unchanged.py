"""The generalized simulator must still produce the DeepSeek-V4.1 study's results, byte for byte.

`tests/reference/dsv41_results.sha256` holds the SHA-256 of that study's committed result files.
The fast test checks the copies committed here; the slow test re-runs the experiments.
"""
import hashlib
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
REF = {line.split()[1]: line.split()[0]
       for line in (ROOT / "tests" / "reference" / "dsv41_results.sha256").read_text().splitlines()
       if line and not line.startswith("#")}


def _digests() -> dict:
    return {name: hashlib.sha256((ROOT / "results" / "dsv41" / name).read_bytes()).hexdigest() for name in REF}


def test_committed_dsv41_results_match_reference():
    assert len(REF) == 5
    assert _digests() == REF


@pytest.mark.slow
def test_rerun_dsv41_results_match_reference():
    subprocess.run([sys.executable, "-m", "experiments.dsv41.run_all"], cwd=ROOT, check=True, capture_output=True)
    assert _digests() == REF

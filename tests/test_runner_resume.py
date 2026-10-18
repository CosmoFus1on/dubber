import json
import subprocess
import sys

import pytest

from conftest import ROOT

pytestmark = pytest.mark.slow


def run(clip, work, *args):
    p = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "dubber.py"), str(clip), "--to", "hi",
                        "--work", str(work), "--out", str(work / "o.mp4"), *args],
                       capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    assert p.returncode == 0, p.stderr[-2000:]
    return p.stderr


def mtimes(work):
    files = [work / "stems" / "vocals.wav", work / "stems" / "no_vocals.wav", work / "transcript.json",
             work / "diarization.json", work / "refs" / "SPEAKER_00.wav"]
    return {f.name: f.stat().st_mtime_ns for f in files}


def test_resume(clip_single, tmp_path):
    w = tmp_path / "w"
    for s in ("separate", "transcribe", "speakers"):
        run(clip_single, w, "--only", s)
    before = mtimes(w)
    for s in ("separate", "transcribe", "speakers"):
        assert "up to date" in run(clip_single, w, "--only", s)
    assert mtimes(w) == before
    run(clip_single, w, "--only", "segment")
    assert "up to date" in run(clip_single, w, "--only", "segment")
    assert mtimes(w) == before
    seg_before = (w / "segments.json").stat().st_mtime_ns
    log = run(clip_single, w, "--from", "segment", "--only", "segment")
    assert "[segment] running" in log and "up to date" not in log
    assert (w / "segments.json").stat().st_mtime_ns != seg_before
    assert mtimes(w) == before
    state = json.loads((w / "state.json").read_text(encoding="utf-8"))
    assert all(state[s]["done"] for s in ("separate", "transcribe", "speakers", "segment"))

"""Acceptance: editing one segment's text in segments.json and rerunning redoes only that segment's
translation / TTS / fit; stages 0-2 outputs are untouched (mtime check). Needs a completed run in
.cache/work/clip (python dubber.py input/clip.mp4 --to hi), the LLM proxy and the Chatterbox env."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / ".cache" / "work" / "clip"
WORK = ROOT / ".cache" / "work" / "clip_resume"
EDIT_ID = 1
NEW_TEXT = "I have always wanted to ride those old yellow trams across the whole city."


def mtimes(paths):
    return {str(p): p.stat().st_mtime_ns for p in paths if p.exists()}


@pytest.mark.slow
def test_edit_one_segment_redoes_only_that_segment():
    if not (SRC / "mix.wav").exists():
        pytest.skip("run the full pipeline on input/clip.mp4 first")
    if WORK.exists():
        shutil.rmtree(WORK)
    shutil.copytree(SRC, WORK)
    seg_path = WORK / "segments.json"
    data = json.loads(seg_path.read_text(encoding="utf-8"))
    target = next(s for s in data["segments"] if s["id"] == EDIT_ID)
    old_translation = target["translation"]
    target["text"] = NEW_TEXT
    seg_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    untouched = [WORK / "audio.wav", WORK / "stems" / "vocals.wav", WORK / "stems" / "no_vocals.wav",
                 WORK / "transcript.json", WORK / "diarization.json", *sorted((WORK / "refs").glob("*.wav"))]
    other_tts = [p for p in (WORK / "tts").glob("*.wav") if p.stem != f"{EDIT_ID:05d}"]
    other_fit = [p for p in (WORK / "fit").glob("*.wav") if p.stem != f"{EDIT_ID:05d}"]
    before_untouched, before_tts, before_fit = mtimes(untouched), mtimes(other_tts), mtimes(other_fit)
    edited_tts_before = (WORK / "tts" / f"{EDIT_ID:05d}.wav").stat().st_mtime_ns

    r = subprocess.run([sys.executable, "-X", "utf8", "dubber.py", "input/clip.mp4", "--to", "hi",
                        "--work", str(WORK), "--out", "output/clip_resume_hi.mp4"],
                       cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr[-2000:]

    assert mtimes(untouched) == before_untouched, "stages 0-2 outputs were touched"
    assert mtimes(other_tts) == before_tts, "other segments were re-synthesized"
    assert mtimes(other_fit) == before_fit, "other segments were re-fitted"
    assert (WORK / "tts" / f"{EDIT_ID:05d}.wav").stat().st_mtime_ns != edited_tts_before, "edited segment not re-synthesized"

    after = json.loads(seg_path.read_text(encoding="utf-8"))
    seg = next(s for s in after["segments"] if s["id"] == EDIT_ID)
    assert seg["text"] == NEW_TEXT, "edit was overwritten by a stage rerun"
    assert seg["translation"] and seg["translation"] != old_translation
    assert seg["verify"]["attempts"] >= 1 and seg["verify"]["passed"] is not None
    assert seg["tts"]["fitted_wav"]
    state = json.loads((WORK / "state.json").read_text(encoding="utf-8"))
    assert all(state[s]["done"] for s in ("separate", "transcribe", "speakers", "segment"))

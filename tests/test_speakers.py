import json
import subprocess
import sys

import pytest

from conftest import ROOT
from dubber import audio


def _ov(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


@pytest.mark.slow
def test_two_speakers_on_clip():
    clip = ROOT / "input" / "clip.mp4"
    work = ROOT / ".cache" / "work" / "clip"
    for stage in ("separate", "transcribe"):  # cached when up to date
        subprocess.run([sys.executable, "dubber.py", str(clip), "--to", "hi", "--only", stage], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "dubber.py", str(clip), "--to", "hi", "--from", "speakers", "--only", "speakers"],
                   cwd=ROOT, check=True)
    subprocess.run([sys.executable, "dubber.py", str(clip), "--to", "hi", "--from", "segment", "--only", "segment"],
                   cwd=ROOT, check=True)

    proj = json.loads((work / "segments.json").read_text(encoding="utf-8"))
    truth = json.loads((ROOT / "tests" / "clip_truth.json").read_text(encoding="utf-8"))
    turns = json.loads((work / "diarization.json").read_text(encoding="utf-8"))["turns"]
    assert len(proj["speakers"]) == 2

    t_gender = {t["id"]: t["gender"] for t in truth["speakers"]}
    mapping = {}
    for name in proj["speakers"]:
        mine = [(u["start"], u["end"]) for u in turns if u["speaker"] == name]
        mapping[name] = max(truth["speakers"], key=lambda t: sum(
            _ov(a, b, c, d) for a, b in mine for c, d in t["turns"]))["id"]
    assert sorted(mapping.values()) == ["A", "B"]
    for name, sp in proj["speakers"].items():
        assert sp["gender"] == t_gender[mapping[name]], (name, sp)
        assert 10.0 <= audio.duration(work / sp["ref_clip"]) <= 30.0

    segs = proj["segments"]
    for line in truth["lines"]:
        mid = (line["start"] + line["end"]) / 2
        hit = [s for s in segs if s["start"] <= mid <= s["end"]]
        assert hit and mapping[hit[0]["speaker"]] == line["speaker"], line["text"]

    s0, s1 = truth["song"]
    song_segs = [s for s in segs if _ov(s["start"], s["end"], s0, s1) > 0]
    tr = json.loads((work / "transcript.json").read_text(encoding="utf-8"))
    song_words = [w for w in tr["words"] if s0 <= (w["s"] + w["e"]) / 2 <= s1]
    assert song_segs or not song_words

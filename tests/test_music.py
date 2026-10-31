import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dubber import config
from dubber.music import smooth, vote

ROOT = Path(__file__).resolve().parent.parent


def frames(*runs, n=100):
    """bool array on the 100 ms grid, True inside (start_s, end_s) runs."""
    m = np.zeros(n, bool)
    for a, b in runs:
        m[int(a * 10):int(b * 10)] = True
    return m


def test_vote_and_smooth():
    no = frames()
    ina, energy = frames((1, 5)), frames((0, 10))
    whisper = frames((3, 5.5), (6.2, 8))           # 0.7 s gap is bridged, 2-of-3 elsewhere
    m = vote(ina, no, energy, whisper)
    assert smooth(m) == [(1.0, 8.0)]
    # ina abstaining everywhere: energy alone is not enough, energy + whisper is
    assert smooth(vote(no, ~no, energy, no)) == []
    assert smooth(vote(no, ~no, energy, whisper)) == [(3.0, 8.0)]
    # runs shorter than MUSIC_MIN_S are dropped; gaps of MUSIC_BRIDGE_S or more are not bridged
    assert smooth(frames((0, config.MUSIC_MIN_S - 0.2))) == []
    assert smooth(frames((0, 2.5), (2.5 + config.MUSIC_BRIDGE_S, 6))) == [(0.0, 2.5), (3.5, 6.0)]


def run_segment(clip: Path) -> list:
    if not clip.exists():
        pytest.skip(f"{clip.name} missing")
    subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "dubber.py"), str(clip), "--to", "hi",
                    "--from", "segment", "--only", "segment"], cwd=ROOT, check=True)
    d = json.loads((ROOT / ".cache" / "work" / clip.stem / "segments.json").read_text(encoding="utf-8"))
    return [(s["start"], s["end"]) for s in d["segments"] if s["type"] == "music"]


@pytest.mark.slow
def test_clip_song_detected():
    truth = json.loads((ROOT / "tests" / "clip_truth.json").read_text(encoding="utf-8"))
    music = run_segment(ROOT / "input" / "clip.mp4")
    t = np.arange(0, truth["duration"], 0.01)
    det = np.zeros(len(t), bool)
    for a, b in music:
        det |= (t >= a) & (t < b)
    song = (t >= truth["song"][0]) & (t < truth["song"][1])
    assert (det & song).sum() / (det | song).sum() > 0.8
    for line in truth["lines"]:
        mid = (line["start"] + line["end"]) / 2
        assert not any(a <= mid <= b for a, b in music), line["text"]


@pytest.mark.slow
def test_clip_single_no_music():
    assert run_segment(ROOT / "input" / "clip_single.mp4") == []

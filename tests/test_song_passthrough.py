"""Acceptance: the song section of tests/clip.mp4 is untouched in the dubbed output (corr > 0.999 with the
original, allowing only a constant loudness gain). Needs a completed run in work/clip and out/clip_hi.mp4."""
import json
from pathlib import Path

import numpy as np
import pytest

from dubber import audio, ffmpeg

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.slow
def test_song_region_is_original_audio():
    work, out = ROOT / "work" / "clip", ROOT / "out" / "clip_hi.mp4"
    if not (work / "audio.wav").exists() or not out.exists():
        pytest.skip("run: python dubber.py tests/clip.mp4 --to hi --out out/clip_hi.mp4")
    truth = json.loads((ROOT / "tests" / "clip_truth.json").read_text(encoding="utf-8"))
    a, b = truth["song"]
    sr = 48000
    extracted = work / "_out_audio.wav"
    ffmpeg.run(["-i", str(out), "-map", "0:a:0", "-ac", "2", "-ar", str(sr), str(extracted)])
    o, _ = audio.read(work / "audio.wav", sr=sr, mono=True)
    d, _ = audio.read(extracted, sr=sr, mono=True)
    i, j = int((a + 0.3) * sr), int((b - 0.3) * sr)
    x, y = o[i:j], d[i:j]
    corr = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))
    assert corr > 0.999, corr
    segs = json.loads((work / "segments.json").read_text(encoding="utf-8"))["segments"]
    music = [s for s in segs if s["type"] == "music"]
    assert music and not any(s["translation"] for s in music)
    assert all(s["type"] == "music" or not (s["start"] < b and s["end"] > a) for s in segs if s["type"] == "speech"), \
        "a speech segment overlaps the song"

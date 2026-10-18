"""Stage 1b -- M1 STUB. Single speaker SPEAKER_00 covering the whole file.

An Opus agent replaces this with pyannote diarization + gender detection later. It must keep the same outputs:
work/diarization.json {"turns", "exclusive"}, work/refs/<SPEAKER>.wav and Project.speakers.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from . import audio, config
from .schema import Speaker

log = logging.getLogger("dubber.speakers")
SPK = "SPEAKER_00"


def _best_window(x: np.ndarray, sr: int, words: list, lo: float, hi: float) -> tuple[float, float]:
    """Contiguous window (up to hi seconds) with the highest median RMS over whisper words."""
    total = len(x) / sr
    if total <= lo:
        return 0.0, total
    win = min(hi, total)
    rms = []
    for w in words:
        seg = x[int(w["s"] * sr):int(w["e"] * sr)]
        rms.append((w["s"], float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0))
    best, best_t = -1.0, 0.0
    for t0 in np.arange(0, total - win + 1e-6, 1.0):
        vals = [r for t, r in rms if t0 <= t < t0 + win]
        m = float(np.median(vals)) if vals else 0.0
        if m > best:
            best, best_t = m, float(t0)
    return best_t, best_t + win


def run(project, work, opts) -> None:
    work = Path(work)
    dur = project.source.get("duration") or audio.duration(work / "audio.wav")
    turn = [{"speaker": SPK, "start": 0.0, "end": dur}]
    (work / "diarization.json").write_text(json.dumps({"turns": turn, "exclusive": turn}), encoding="utf-8")
    x, sr = audio.read(work / "stems" / "vocals.wav", sr=16000, mono=True)
    words = json.loads((work / "transcript.json").read_text(encoding="utf-8"))["words"]
    t0, t1 = _best_window(x, sr, words, config.REF_CLIP_MIN_S, config.REF_CLIP_MAX_S)
    audio.write(work / "refs" / f"{SPK}.wav", audio.peak_normalize(x[int(t0 * sr):int(t1 * sr)]), sr)
    project.speakers = {SPK: Speaker(gender="unknown", gender_confidence=0.0, ref_clip=f"refs/{SPK}.wav")}
    log.info("stub: one speaker, ref clip %.1f-%.1fs", t0, t1)

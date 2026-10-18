"""Music/song detection: 2-of-3 detector vote on a 100 ms grid, then smoothing.

Detectors (each a bool per frame):
  ina      inaSpeechSegmenter said "music". ina runs on the vocals stem, so where it saw nothing
           (noEnergy) or was unavailable it abstains, and the other two must both fire.
  energy   accompaniment: the bed (no_vocals) is above MUSIC_BED_FLOOR_DB and no more than
           MUSIC_BED_MARGIN_DB below the vocals. Fires under quiet dialogue too; the vote handles that.
  whisper  Whisper doesn't believe it heard speech: a whisper segment with bad stats / repeated
           n-grams, or no transcribed word while there is sound nobody transcribed: vocals stem
           energy (singing Whisper's VAD skipped) or the bed rising clearly above its level under
           dialogue (an instrumental passage; Demucs puts all of it in no_vocals).
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np

from . import audio, config

log = logging.getLogger("dubber.music")
HOP = config.MUSIC_HOP_S
SR = 16000


def frame_db(x: np.ndarray, n: int, win: int = 1) -> np.ndarray:
    """RMS dBFS per HOP frame (n frames), power-averaged over `win` frames (centered) for 'sustained'."""
    k = int(SR * HOP)
    x = np.pad(x[: n * k], (0, max(0, n * k - len(x))))
    p = (x.reshape(n, k).astype(np.float64) ** 2).mean(axis=1)
    if win > 1:
        p = np.convolve(p, np.ones(win) / win, mode="same")
    return 10 * np.log10(np.maximum(p, 1e-18))


def _mask(n: int, spans) -> np.ndarray:
    m = np.zeros(n, bool)
    for a, b in spans:
        m[max(0, int(round(a / HOP))):max(0, int(round(b / HOP)))] = True
    return m


def _repetitive(text: str) -> bool:
    w = text.lower().split()
    grams = Counter(zip(w, w[1:], w[2:]))
    return bool(grams) and max(grams.values()) >= config.MUSIC_NGRAM_REPEATS


def detectors(work: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame (ina, ina_abstains, energy, whisper) bool arrays."""
    work = Path(work)
    voc, _ = audio.read(work / "stems" / "vocals.wav", sr=SR, mono=True)
    bed, _ = audio.read(work / "stems" / "no_vocals.wav", sr=SR, mono=True)
    n = int(np.ceil(max(len(voc), len(bed)) / (SR * HOP)))
    win = max(1, int(round(config.MUSIC_SUSTAIN_S / HOP)))
    v, b = frame_db(voc, n, win), frame_db(bed, n, win)

    ina = json.loads((work / "ina.json").read_text(encoding="utf-8"))
    labels = ina.get("labels", []) if ina.get("available") else []
    ina_music = _mask(n, [(l["start"], l["end"]) for l in labels if l["label"] == "music"])
    heard = _mask(n, [(l["start"], l["end"]) for l in labels if l["label"] != "noEnergy"])
    abstain = ~heard  # unavailable -> nothing heard -> abstains everywhere

    energy = (b >= config.MUSIC_BED_FLOOR_DB) & (b >= v - config.MUSIC_BED_MARGIN_DB)

    tr = json.loads((work / "transcript.json").read_text(encoding="utf-8"))
    pad = config.MUSIC_SUSTAIN_S / 2 + HOP  # the sustain window smears speech energy this far past a word
    words = _mask(n, [(w["s"] - pad, w["e"] + pad) for w in tr["words"]])
    bad = _mask(n, [(s["start"], s["end"]) for s in tr.get("whisper_segments", [])
                    if s["avg_logprob"] < config.MUSIC_LOGPROB_MAX or s["compression_ratio"] > config.MUSIC_COMPRESSION_MIN
                    or s["no_speech_prob"] > config.MUSIC_NO_SPEECH_MIN or _repetitive(s.get("text", ""))])
    base = float(np.median(b[words])) if words.any() else float(np.median(b))
    untranscribed = ~words & ((v >= config.MUSIC_VOCALS_MIN_DB) | (b >= base + config.MUSIC_BED_RISE_DB))
    return ina_music, abstain, energy, bad | untranscribed


def vote(ina, abstain, energy, whisper) -> np.ndarray:
    """2 of 3; where ina abstains the other two must agree."""
    votes = ina.astype(int) + energy + whisper
    return np.where(abstain, energy & whisper, votes >= config.MUSIC_VOTES_NEEDED)


def smooth(m: np.ndarray, hop: float = HOP) -> list[tuple[float, float]]:
    """Bridge gaps < MUSIC_BRIDGE_S, then drop runs < MUSIC_MIN_S. Returns [(start, end)] seconds."""
    runs, i, n = [], 0, len(m)
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]:
                j += 1
            runs.append([i, j])
            i = j
        else:
            i += 1
    merged: list[list[int]] = []
    for a, b in runs:
        if merged and (a - merged[-1][1]) * hop < config.MUSIC_BRIDGE_S:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(round(a * hop, 2), round(b * hop, 2)) for a, b in merged if (b - a) * hop >= config.MUSIC_MIN_S]


def detect_regions(project, work) -> list[tuple[float, float]]:
    return smooth(vote(*detectors(work)))


def apply(segs, work) -> list:
    """Tag segments mostly inside music regions, append uncovered regions as music segments. Returns segs."""
    from .schema import MUSIC, Segment
    ina, abstain, energy, whisper = detectors(work)
    regions = smooth(vote(ina, abstain, energy, whisper))
    votes = ina.astype(int) + energy + whisper
    n = len(votes)
    in_music = _mask(n, regions)

    def scores(a, b):
        sl = slice(max(0, int(a / HOP)), min(n, max(int(a / HOP) + 1, int(np.ceil(b / HOP)))))
        f = lambda x: round(float(np.mean(x[sl])), 3) if x[sl].size else 0.0
        return {"ina": f(ina), "energy": f(energy), "whisper": f(whisper), "votes": f(votes)}, f(in_music)

    for s in segs:
        s.extra["music"], frac = scores(s.start, s.end)
        if frac > config.MUSIC_SEGMENT_FRAC:
            s.type = MUSIC
    nxt = max((s.id for s in segs), default=-1) + 1
    for a, b in regions:
        if not any(s.start < b and s.end > a for s in segs):
            seg = Segment(id=nxt, speaker="MUSIC", gender="unknown", start=a, end=b, type=MUSIC, text="")
            seg.extra["music"] = scores(a, b)[0]
            segs.append(seg)
            nxt += 1
    log.info("music regions: %s", ", ".join(f"{a:.1f}-{b:.1f}s" for a, b in regions) or "none")
    return segs

"""Stage 2: merge transcript words into segments (pause / speaker change / length cap)."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from . import config
from .schema import F_OVERLAP, MUSIC, SPEECH, Segment, Word

log = logging.getLogger("dubber.segment")
_END = re.compile(r"[.!?。！？…]$")
_SENT = re.compile(r"[.!?。！？]$")


def detect_music(segs: list[Segment], project, work) -> list[Segment]:
    """Tag music segments and append uncovered music regions (see music.py for the 3-detector vote)."""
    from . import music
    return music.apply(segs, work)


def _speaker_at(t: float, turns: list) -> str:
    for u in turns:
        if u["start"] <= t <= u["end"]:
            return u["speaker"]
    return min(turns, key=lambda u: min(abs(t - u["start"]), abs(t - u["end"])))["speaker"]


def _join(words: list[Word]) -> str:
    return re.sub(r"\s+", " ", "".join(w.w if w.w.startswith(" ") else " " + w.w for w in words)).strip()


def build(transcript: dict, turns: list, speakers: dict) -> list[Segment]:
    words = sorted((Word(w["w"], w["s"], w["e"], w.get("p", 1.0)) for w in transcript["words"]), key=lambda w: w.s)
    groups: list[tuple[str, list[Word]]] = []
    cur: list[Word] = []
    cur_spk = None
    for w in words:
        spk = _speaker_at((w.s + w.e) / 2, turns) if turns else "SPEAKER_00"
        gap = w.s - cur[-1].e if cur else 0.0
        sentence = (cur and _SENT.search(cur[-1].w.strip()) and gap >= config.SENTENCE_GAP_S
                    and cur[-1].e - cur[0].s >= config.SENTENCE_MIN_S)
        if cur and (spk != cur_spk or gap > config.MERGE_PAUSE_S or sentence):
            groups.append((cur_spk, cur))
            cur = []
        if cur and w.e - cur[0].s > config.SEGMENT_CAP_S:
            # split after the last sentence end in the window, else right here
            k = next((i + 1 for i in range(len(cur) - 1, 0, -1) if _END.search(cur[i].w.strip())), len(cur))
            groups.append((cur_spk, cur[:k]))
            cur = cur[k:]
        cur_spk = spk
        cur.append(w)
    if cur:
        groups.append((cur_spk, cur))

    wsegs = transcript.get("whisper_segments", [])
    out = []
    for i, (spk, ws) in enumerate(groups):
        s, e = ws[0].s, ws[-1].e
        ov = [x for x in wsegs if x["start"] < e and x["end"] > s]
        extra = {}
        if ov:
            extra = {"avg_logprob": sum(x["avg_logprob"] for x in ov) / len(ov),
                     "compression_ratio": max(x["compression_ratio"] for x in ov),
                     "no_speech_prob": sum(x["no_speech_prob"] for x in ov) / len(ov)}
        sp = speakers.get(spk)
        out.append(Segment(id=i, speaker=spk, gender=sp.gender if sp else "unknown", start=s, end=e,
                           text=_join(ws), words=ws, extra=extra))
    return out


def flag_overlaps(segs: list[Segment], turns: list) -> None:
    """Flag segments that another speaker's (non-exclusive) turn overlaps by more than OVERLAP_MIN_S."""
    for seg in segs:
        ov: dict = {}
        for u in turns:
            if u["speaker"] != seg.speaker:
                ov[u["speaker"]] = ov.get(u["speaker"], 0.0) + max(0.0, min(seg.end, u["end"]) - max(seg.start, u["start"]))
        other = max(ov, key=ov.get) if ov else None
        if other and ov[other] > config.OVERLAP_MIN_S:
            seg.flag(F_OVERLAP)
            seg.extra["overlaps_with"] = other


def carry_over(old: list, new: list, tol: float = 0.05) -> int:
    """Re-segmentation must not throw away work: a new segment with the same text, type and (within tol)
    the same start keeps the old one's translation, verification, TTS state, flags and extras."""
    by_text = {}
    for o in old:
        by_text.setdefault((o.type, o.text.strip()), []).append(o)
    kept = 0
    for n in new:
        for o in by_text.get((n.type, n.text.strip()), []):
            if abs(o.start - n.start) <= tol:
                n.translation, n.budget_chars, n.verify, n.tts = o.translation, o.budget_chars, o.verify, o.tts
                n.flags, n.extra = list(o.flags), dict(o.extra, **n.extra)
                kept += 1
                break
    if old:
        log.info("re-segmentation kept the work of %d/%d segments", kept, len(new))
    return kept


def run(project, work, opts) -> None:
    work = Path(work)
    transcript = json.loads((work / "transcript.json").read_text(encoding="utf-8"))
    dia = json.loads((work / "diarization.json").read_text(encoding="utf-8"))
    segs = build(transcript, dia["exclusive"], project.speakers)
    flag_overlaps(segs, dia["turns"])
    for s in segs:
        s.type = SPEECH
    segs = detect_music(segs, project, work)
    carry_over(project.segments, segs)
    project.segments = segs
    log.info("%d segments, %d overlapping, %d music", len(segs), sum(F_OVERLAP in s.flags for s in segs),
             sum(s.type == MUSIC for s in segs))

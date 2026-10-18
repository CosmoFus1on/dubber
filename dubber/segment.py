"""Stage 2: merge transcript words into segments (pause / speaker change / length cap)."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from . import config
from .schema import MUSIC, SPEECH, Segment, Word

log = logging.getLogger("dubber.segment")
_END = re.compile(r"[.!?。！？…]$")


def detect_music(segment, project, work) -> bool:
    """M3 fills this in (3-detector vote). Always speech for now."""
    return False


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
        if cur and (spk != cur_spk or w.s - cur[-1].e > config.MERGE_PAUSE_S):
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


def run(project, work, opts) -> None:
    work = Path(work)
    transcript = json.loads((work / "transcript.json").read_text(encoding="utf-8"))
    turns = json.loads((work / "diarization.json").read_text(encoding="utf-8"))["exclusive"]
    segs = build(transcript, turns, project.speakers)
    for s in segs:
        s.type = MUSIC if detect_music(s, project, work) else SPEECH
    project.segments = segs
    log.info("%d segments", len(segs))

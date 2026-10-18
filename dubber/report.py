"""report.json writer: measured timings, flags, engines used."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from . import config
from .schema import MUSIC, SPEECH

log = logging.getLogger("dubber.report")


def write(project, work, opts, state) -> Path:
    work = Path(work)
    flags: dict = {}
    for s in project.segments:
        for f in s.flags:
            flags.setdefault(f, []).append(s.id)
    total = round(sum(project.timings.values()), 2)
    dur = project.source.get("duration") or 0.0
    rep = {
        "video": project.source.get("video"),
        "target": project.target_language,
        "duration_s": dur,
        "stages": {k: round(v, 2) for k, v in project.timings.items()},
        "total_s": total,
        "seconds_per_video_minute": round(total / (dur / 60), 2) if dur else None,
        "note": "stage seconds are from the most recent run of each stage; a resume reports the resume",
        "llm_calls": {k: v.get("llm_calls", 0) for k, v in state.items()},
        "models": {"llm_translate": config.LLM_MODEL, "llm_fast": config.LLM_FAST,
                   "whisper": config.WHISPER_CPU if opts.cpu else config.WHISPER_GPU,
                   "separation": config.DEMUCS_MODEL, "diarization": config.DIARIZATION_MODEL,
                   "embedder": config.EMBEDDER},
        "llm_model": config.LLM_MODEL,
        "engines_used": sorted({s.tts.engine for s in project.segments if s.tts.engine}),
        "flags": flags,
        "speaker_flags": {k: v.flags for k, v in project.speakers.items() if v.flags},
        "segments_flagged": [{"id": s.id, "start": s.start, "end": s.end, "flags": s.flags,
                              "translation": s.translation} for s in project.segments if s.flags],
        "counts": {"segments": len(project.segments),
                   "speech": sum(s.type == SPEECH for s in project.segments),
                   "music": sum(s.type == MUSIC for s in project.segments),
                   "speakers": len(project.speakers)},
    }
    path = work / "report.json"
    path.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    return path

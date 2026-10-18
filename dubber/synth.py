"""Stage 4: voice-preserving TTS. One engine family in VRAM at a time; engines with conflicting pins
run in their own venv (see tts/base.py). Incremental per segment via Segment.tts_key.

Prosody hints: the original segment's loudness relative to the speaker's median maps to
`exaggeration` (emotion intensity); the speaking rate (chars/s vs the language's typical) to `cfg_weight`.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from . import audio, config
from .schema import F_NO_REF, F_TTS_FALLBACK, SPEECH, Project, Segment, sha1
from .tts.base import TTSJob, chain_for, run_engine

log = logging.getLogger("dubber.synth")


def prosody_hints(project: Project, work: Path) -> dict[int, dict]:
    """Per-segment {exaggeration, cfg_weight} from the vocals stem. ponytail: linear maps; a learned mapping is the upgrade."""
    hints = {}
    try:
        x, sr = audio.read(work / "stems" / "vocals.wav", sr=16000, mono=True)
    except Exception:
        return {s.id: {"exaggeration": 0.5, "cfg_weight": 0.5} for s in project.segments}
    per_speaker: dict[str, list[float]] = {}
    rms = {}
    for s in project.speech():
        a, b = int(s.start * sr), int(s.end * sr)
        seg = x[a:b]
        db = audio.rms_db(seg) if len(seg) else -60.0
        rms[s.id] = db
        per_speaker.setdefault(s.speaker, []).append(db)
    med = {k: float(np.median(v)) for k, v in per_speaker.items()}
    for s in project.speech():
        loud = rms[s.id] - med.get(s.speaker, rms[s.id])          # dB above the speaker's usual level
        exag = float(np.clip(0.5 + loud / 24.0, 0.3, 0.8))
        rate = len(s.text) / max(s.duration, 0.2)                 # chars/s in the source language
        fast = rate / config.cps(project.source.get("language") or "en")
        cfg = float(np.clip(0.5 - (fast - 1.0) * 0.2, 0.3, 0.6))  # faster original -> lower cfg (quicker delivery)
        hints[s.id] = {"exaggeration": round(exag, 2), "cfg_weight": round(cfg, 2)}
    return hints


def synthesize(segs: list[Segment], project: Project, work: Path, opts: config.Options,
               hints: dict[int, dict] | None = None) -> None:
    """Synthesize the given segments (unconditionally), trying the language's engine chain."""
    if not segs:
        return
    hints = hints or prosody_hints(project, work)
    tts_dir = work / "tts"
    tts_dir.mkdir(exist_ok=True)
    chain = chain_for(opts.target, cpu=opts.cpu)
    if not chain:
        raise RuntimeError(f"no TTS engine available for '{opts.target}'. Run setup.ps1 -Engines chatterbox")
    pending = list(segs)
    for n, engine in enumerate(chain):
        jobs = []
        for s in pending:
            spk = project.speakers.get(s.speaker)
            ref = (work / spk.ref_clip) if spk and spk.ref_clip and (work / spk.ref_clip).exists() else None
            if ref is None:
                s.flag(F_NO_REF)
            h = hints.get(s.id, {"exaggeration": 0.5, "cfg_weight": 0.5})
            jobs.append(TTSJob(str(s.id), s.translation, opts.target, str((tts_dir / f"{s.id:05d}.wav").resolve()),
                               ref_wav=str(ref) if ref else None, gender=s.gender, **h))
        log.info("engine %s: %d segments", engine, len(jobs))
        results = {r.id: r for r in run_engine(engine, jobs, opts.device)}
        still = []
        for s in pending:
            r = results.get(str(s.id))
            if r and r.wav and Path(r.wav).exists():
                s.tts.engine = engine
                s.tts.wav = str(Path(r.wav).relative_to(work.resolve())) if Path(r.wav).is_relative_to(work.resolve()) else r.wav
                s.tts.raw_duration = round(audio.duration(r.wav), 3)
                s.tts.fitted_wav = None
                s.tts.stretch = None
                s.unflag_prefix(F_TTS_FALLBACK)
                if n > 0:
                    s.flag(F_TTS_FALLBACK + engine)
                s.extra["tts_key"] = tts_key(s, engine, project, work, hints)
            else:
                log.warning("engine %s failed on segment %d: %s", engine, s.id, r.error if r else "no result")
                still.append(s)
        pending = still
        if not pending:
            break
    for s in pending:
        log.error("segment %d: every engine failed; it will be left silent", s.id)
        s.flag("tts_failed")


def tts_key(s: Segment, engine: str, project: Project, work: Path, hints: dict) -> str:
    spk = project.speakers.get(s.speaker)
    ref = work / spk.ref_clip if spk and spk.ref_clip else None
    ref_hash = sha1(ref) if ref and ref.exists() else "none"
    return s.tts_key(engine, ref_hash, hints.get(s.id, {}))


def run(project: Project, work: Path, opts: config.Options) -> None:
    hints = prosody_hints(project, work)
    chain = chain_for(opts.target, cpu=opts.cpu)
    primary = chain[0] if chain else "none"
    todo = []
    for s in project.sorted_segments():
        if s.type != SPEECH or not s.translation:
            continue
        wav = work / s.tts.wav if s.tts.wav else None
        if wav and wav.exists() and s.extra.get("tts_key") == tts_key(s, s.tts.engine or primary, project, work, hints):
            continue
        todo.append(s)
    log.info("tts: %d/%d segments need synthesis (chain %s)", len(todo), len(project.speech()), chain)
    synthesize(todo, project, work, opts, hints)
    project.save(work / "segments.json")

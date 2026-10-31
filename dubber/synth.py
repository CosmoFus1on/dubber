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
from .schema import F_NO_REF, F_TTS_FAILED, F_TTS_FALLBACK, SPEECH, Project, Segment, sha1
from .tts.base import TTSJob, chain_for, run_engine

log = logging.getLogger("dubber.synth")


def pitch_rise(seg: np.ndarray, sr: int) -> float:
    """Pitch at the end of the line vs. the rest, in semitones (rising = a question-like contour)."""
    import librosa
    if len(seg) < sr // 4:
        return 0.0
    f0 = librosa.yin(seg.astype(np.float32), fmin=60, fmax=400, sr=sr, frame_length=1024, hop_length=256)
    f0 = f0[(f0 > 60) & (f0 < 400)]
    if len(f0) < 8:
        return 0.0
    k = max(2, len(f0) // 4)
    return float(12 * np.log2(np.median(f0[-k:]) / np.median(f0[:-k])))


def prosody_hints(project: Project, work: Path) -> dict[int, dict]:
    """Per-segment {exaggeration, cfg_weight} for the engine.

    exaggeration (emotion intensity) rises with: loudness above the speaker's usual level, loudness swings
    inside the line, a rising pitch at the end, and '?'/'!' in the text. cfg_weight (pacing/adherence) drops
    for fast or emphatic lines so delivery gets livelier. ponytail: hand-tuned linear maps; a learned mapping
    (or an emotion classifier on the original line) is the upgrade."""
    hints = {}
    try:
        x, sr = audio.read(work / "stems" / "vocals.wav", sr=16000, mono=True)
    except Exception:
        return {s.id: {"exaggeration": 0.5, "cfg_weight": 0.5} for s in project.segments}
    per_speaker: dict[str, list[float]] = {}
    feat = {}
    for s in project.speech():
        seg = x[int(s.start * sr):int(s.end * sr)]
        db = audio.rms_db(seg) if len(seg) else -60.0
        hop = sr // 20
        frames = [audio.rms_db(seg[i:i + hop]) for i in range(0, max(0, len(seg) - hop), hop)] if len(seg) > hop else [db]
        swing = float(np.std([f for f in frames if f > -50])) if len(frames) > 2 else 0.0   # dB spread of the loud frames
        feat[s.id] = (db, swing, pitch_rise(seg, sr))
        per_speaker.setdefault(s.speaker, []).append(db)
    med = {k: float(np.median(v)) for k, v in per_speaker.items()}
    src = project.source.get("language") or "en"
    for s in project.speech():
        db, swing, rise = feat[s.id]
        text = (s.text or "").strip()
        loud = db - med.get(s.speaker, db)                      # dB above the speaker's usual level
        exag = 0.5 + loud / 24.0 + min(swing, 6.0) / 30.0      # louder and more dynamic -> more emotion
        if rise > 2.0 or text.endswith(("?", "？")):
            exag += 0.15                                       # questions: lift the contour
        if text.endswith(("!", "！")) or loud > 4.0:
            exag += 0.2                                        # exclamations and shouts
        rate = len(text) / max(s.duration, 0.2)
        fast = rate / config.cps(src)
        cfg = 0.5 - (fast - 1.0) * 0.2 - (0.1 if exag > 0.7 else 0.0)   # faster or emphatic -> lower cfg
        # coarse steps: the speaker median shifts a little whenever a segment changes, and that must not
        # invalidate every other segment's tts_key
        hints[s.id] = {"exaggeration": round(float(np.clip(exag, 0.3, 0.9)), 1),
                       "cfg_weight": round(float(np.clip(cfg, 0.2, 0.6)), 1)}
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
        # Engines babble on very short text now and then; one retry fixes most of it.
        retry = [j for j in jobs if _babbled(results.get(j.id), j)]
        if retry:
            log.info("engine %s: re-synthesizing %d implausibly slow clip(s)", engine, len(retry))
            results.update({r.id: r for r in run_engine(engine, retry, opts.device)})
        still = []
        for s in pending:
            r = results.get(str(s.id))
            if r and r.wav and Path(r.wav).exists():
                trim_silence(Path(r.wav))
                s.tts.engine = engine
                wav = Path(r.wav)
                s.tts.wav = wav.relative_to(work.resolve()).as_posix() if wav.is_relative_to(work.resolve()) else r.wav
                s.tts.raw_duration = round(audio.duration(r.wav), 3)
                s.tts.fitted_wav = None
                s.tts.stretch = None
                s.extra.pop("fit_key", None)
                s.unflag_prefix(F_TTS_FALLBACK)
                s.unflag_prefix(F_TTS_FAILED)
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
        s.tts.wav = s.tts.fitted_wav = None  # never let fit/mix pick up audio of an older translation
        s.tts.raw_duration = s.tts.stretch = None
        s.extra.pop("tts_key", None)
        s.extra.pop("fit_key", None)
        s.flag(F_TTS_FAILED)


def trim_silence(wav: Path, top_db: float = 35.0, keep_s: float = 0.05) -> float:
    """Trim leading/trailing silence in place (keeping a short margin). Returns the new duration."""
    import librosa
    x, sr = audio.read(wav)
    mono = x if x.ndim == 1 else x.mean(axis=1)
    _, (a, b) = librosa.effects.trim(mono, top_db=top_db)
    k = int(keep_s * sr)
    a, b = max(0, a - k), min(len(x), b + k)
    if b - a < len(x):
        audio.write(wav, x[a:b], sr)
    return (b - a) / sr


def _babbled(r, job: TTSJob) -> bool:
    """True when the clip is far longer than the text could take (engine hallucination)."""
    if not r or not r.wav or not Path(r.wav).exists():
        return False
    expect = len(job.text) / max(config.cps(job.lang), 1.0)
    return trim_silence(Path(r.wav)) > max(1.5, expect * config.BABBLE_RATIO)


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
        if (not opts.force and wav and wav.exists()
                and s.extra.get("tts_key") == tts_key(s, s.tts.engine or primary, project, work, hints)):
            continue
        todo.append(s)
    log.info("tts: %d/%d segments need synthesis (chain %s)", len(todo), len(project.speech()), chain)
    synthesize(todo, project, work, opts, hints)
    project.save(work / "segments.json")

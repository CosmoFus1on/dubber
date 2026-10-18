"""Stage 5a: timing fit. Each dubbed clip must fit its slot (end - start) plus a small spill into the
following gap. Order of remedies: shorter translation (up to 2 rounds) -> rubberband tempo up to 1.15
(flag stretch_over_15 above that) -> cap at 1.25 and hard-trim with a 40 ms fade (flag truncated).
Shorter clips are left as-is; placement at `start` with silence padding happens in mix. Never slow down.
"""
from __future__ import annotations

import logging
import math
import shutil
from pathlib import Path

import numpy as np

from . import audio, config, ffmpeg, prompts, synth
from .llm import LLMError, default_provider
from .schema import F_STRETCH_15, F_TRUNCATED, SPEECH, Project, Segment, sha1

log = logging.getLogger("dubber.fit")


def allowed_length(project: Project, seg: Segment) -> float:
    nxt = project.next_start_after(seg)
    gap = (nxt - seg.end) if nxt is not None else config.SPILL_MAX_S
    return seg.duration + max(0.0, min(config.SPILL_MAX_S, gap))


def shorten(llm, seg: Segment, src: str, tgt: str, target_chars: int) -> str | None:
    try:
        d = llm.chat_json(prompts.shorten_messages(src, tgt, seg.text, seg.translation, target_chars),
                          config.LLM_MODEL, fallback=config.LLM_FALLBACK_CHAIN)
    except LLMError as e:
        log.warning("segment %d: shorten failed (%s)", seg.id, e)
        return None
    t = str(d.get("translation", "")).strip() if isinstance(d, dict) else ""
    return t if t and t != seg.translation else None


def trim_with_fade(src: Path, dst: Path, seconds: float) -> None:
    x, sr = audio.read(src)
    n = int(seconds * sr)
    x = x[:n]
    k = min(len(x), int(sr * config.FADE_MS / 1000))
    if k > 0:
        ramp = np.linspace(1.0, 0.0, k, dtype=np.float32)
        x[-k:] = x[-k:] * (ramp if x.ndim == 1 else ramp[:, None])
    audio.write(dst, x, sr)


def fit_segment(project: Project, seg: Segment, work: Path, opts: config.Options, llm, translator) -> None:
    fit_dir = work / "fit"
    fit_dir.mkdir(exist_ok=True)
    src_lang = config.norm_lang(project.source.get("language") or "en")
    allowed = allowed_length(project, seg)
    raw_path = work / seg.tts.wav
    raw = audio.duration(raw_path)
    rounds = 0
    max_rounds = 0 if opts.demo_fast else config.SHORTEN_ROUNDS
    while raw > allowed and rounds < max_rounds and seg.translation:
        target = max(4, math.floor(len(seg.translation) * allowed / raw * 0.95))
        new = shorten(llm, seg, src_lang, opts.target, target)
        rounds += 1
        if not new:
            break
        log.info("segment %d: shortened %d -> %d chars (raw %.2fs > allowed %.2fs)", seg.id, len(seg.translation), len(new), raw, allowed)
        seg.translation = new
        seg.verify.passed = None
        synth.synthesize([seg], project, work, opts)
        if translator is not None:
            translator.verify(seg)
        raw_path = work / seg.tts.wav
        raw = audio.duration(raw_path)
    seg.tts.shorten_rounds = rounds
    seg.unflag_prefix(F_STRETCH_15)
    seg.unflag_prefix(F_TRUNCATED)
    out = fit_dir / f"{seg.id:05d}.wav"
    tempo = 1.0
    if raw > allowed + 0.005:
        tempo = min(raw / allowed, config.STRETCH_HARD)
        if tempo > config.STRETCH_SOFT:
            seg.flag(F_STRETCH_15)
        ffmpeg.rubberband(raw_path, out, tempo)
        if audio.duration(out) > allowed + 0.005:
            trim_with_fade(out, out, allowed)
            seg.flag(F_TRUNCATED)
            log.warning("segment %d: truncated to %.2fs even at %.2fx", seg.id, allowed, tempo)
    else:
        shutil.copyfile(raw_path, out)
    seg.tts.stretch = round(tempo, 3)
    seg.tts.fitted_wav = str(out.relative_to(work))
    seg.extra["fit_key"] = fit_key(seg, allowed, raw_path)


def fit_key(seg: Segment, allowed: float, raw_path: Path) -> str:
    st = raw_path.stat()
    return sha1(seg.translation, round(allowed, 3), st.st_size, int(st.st_mtime))


def run(project: Project, work: Path, opts: config.Options) -> None:
    llm = default_provider()
    try:
        from .translate import Translator
        translator = Translator(project, opts, llm)
    except Exception:  # verification is best-effort after a shorten
        translator = None
    todo = []
    for s in project.sorted_segments():
        if s.type != SPEECH or not s.tts.wav or not (work / s.tts.wav).exists():
            continue
        fitted = work / s.tts.fitted_wav if s.tts.fitted_wav else None
        if fitted and fitted.exists() and s.extra.get("fit_key") == fit_key(s, allowed_length(project, s), work / s.tts.wav):
            continue
        todo.append(s)
    log.info("fit: %d/%d segments to fit", len(todo), len(project.speech()))
    for n, s in enumerate(todo, 1):
        fit_segment(project, s, work, opts, llm, translator)
        if n % 10 == 0:
            project.save(work / "segments.json")
    project.save(work / "segments.json")
    stretched = [s for s in project.speech() if (s.tts.stretch or 1.0) > 1.0]
    log.info("fit done: %d stretched, %d over 1.15, %d truncated", len(stretched),
             sum(F_STRETCH_15 in s.flags for s in project.speech()), sum(F_TRUNCATED in s.flags for s in project.speech()))

"""Stage 5a: timing fit. Each dubbed clip must fit its slot (end - start) plus a small spill into the
following gap. Order of remedies: shorter translation (up to 2 rounds, batched so the TTS engine loads
once per round) -> rubberband tempo up to 1.15 (flag stretch_over_15 above that) -> cap at 1.25 and
hard-trim with a 40 ms fade (flag truncated). Shorter clips are left as-is; placement at `start` with
silence padding happens in mix. Never slow down.
"""
from __future__ import annotations

import logging
import math
import shutil
from pathlib import Path

import numpy as np

from . import audio, config, ffmpeg, synth
from .llm import default_provider
from .schema import F_STRETCH_15, F_TRUNCATED, SPEECH, Project, Segment, Verify, sha1
from .translate import Translator, hand_edited, shorten

log = logging.getLogger("dubber.fit")


def allowed_length(project: Project, seg: Segment) -> float:
    nxt = project.next_start_after(seg)
    end = project.source.get("duration")
    gap = (nxt - seg.end) if nxt is not None else config.SPILL_MAX_S
    if end:
        gap = min(gap, end - seg.end)  # never past the end of the video
    return max(0.05, seg.duration + max(0.0, min(config.SPILL_MAX_S, gap)))


def trim_with_fade(src: Path, dst: Path, seconds: float) -> None:
    x, sr = audio.read(src)
    n = int(seconds * sr)
    x = x[:n]
    k = min(len(x), int(sr * config.FADE_MS / 1000))
    if k > 0:
        ramp = np.linspace(1.0, 0.0, k, dtype=np.float32)
        x[-k:] = x[-k:] * (ramp if x.ndim == 1 else ramp[:, None])
    audio.write(dst, x, sr)


def fit_key(seg: Segment, allowed: float, raw_path: Path) -> str:
    st = raw_path.stat()
    return sha1(seg.translation, round(allowed, 3), st.st_size, int(st.st_mtime))


def fit_audio(project: Project, seg: Segment, work: Path) -> None:
    """Stretch / trim / copy the raw clip into fit/<id>.wav. No LLM, no TTS."""
    fit_dir = work / "fit"
    fit_dir.mkdir(exist_ok=True)
    allowed = allowed_length(project, seg)
    raw_path = work / seg.tts.wav
    raw = audio.duration(raw_path)
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
    seg.tts.fitted_wav = out.relative_to(work).as_posix()
    seg.extra["fit_key"] = fit_key(seg, allowed, raw_path)


def shorten_rounds(project: Project, todo: list[Segment], work: Path, opts: config.Options) -> None:
    """Batched: shorten every over-long translation, re-synthesize them in one engine call, verify, repeat."""
    max_rounds = 0 if opts.demo_fast else config.SHORTEN_ROUNDS
    if not max_rounds:
        return
    llm = default_provider()
    try:
        translator = Translator(project, opts, llm)
    except Exception:  # verification after a shorten is best-effort
        translator = None
    src_lang = config.norm_lang(project.source.get("language") or "en")
    for rnd in range(1, max_rounds + 1):
        changed = []
        for s in todo:
            if hand_edited(s) or not s.translation or not s.tts.wav:
                continue  # a human's text is never shortened; it goes to tempo/truncate
            allowed, raw = allowed_length(project, s), audio.duration(work / s.tts.wav)
            if raw <= allowed:
                continue
            target = max(4, math.floor(len(s.translation) * allowed / raw))
            new = shorten(llm, s, src_lang, opts.target, target)
            if not new:
                continue
            prev = (s.translation, s.verify, list(s.flags))
            s.translation, s.verify = new, Verify()
            # judge the shorter line BEFORE spending TTS on it; never let verify re-translate here (that
            # would desync audio and text). A rejected shortening is simply reverted, flags included.
            if translator is not None and not translator.verify(s, retranslate=False):
                log.info("segment %d round %d: shortened line rejected by the judge (%s); keeping previous",
                         s.id, rnd, s.verify.issue)
                s.translation, s.verify, s.flags = prev
                s.extra["shorten_rejected"] = s.verify.issue or "judge rejected the shorter line"
                continue
            log.info("segment %d round %d: shortened %d -> %d chars (raw %.2fs > allowed %.2fs)",
                     s.id, rnd, len(prev[0]), len(new), raw, allowed)
            s.tts.shorten_rounds = rnd
            changed.append(s)
        if not changed:
            return
        synth.synthesize(changed, project, work, opts)
        project.save(work / "segments.json")


def run(project: Project, work: Path, opts: config.Options) -> None:
    todo = []
    for s in project.sorted_segments():
        if s.type != SPEECH or not s.tts.wav or not (work / s.tts.wav).exists():
            continue
        fitted = work / s.tts.fitted_wav if s.tts.fitted_wav else None
        if (not opts.force and fitted and fitted.exists()
                and s.extra.get("fit_key") == fit_key(s, allowed_length(project, s), work / s.tts.wav)):
            continue
        todo.append(s)
    log.info("fit: %d/%d segments to fit", len(todo), len(project.speech()))
    shorten_rounds(project, todo, work, opts)
    for s in todo:
        fit_audio(project, s, work)
    project.save(work / "segments.json")
    log.info("fit done: %d stretched, %d over 1.15, %d truncated",
             sum((s.tts.stretch or 1.0) > 1.0 for s in project.speech()),
             sum(F_STRETCH_15 in s.flags for s in project.speech()), sum(F_TRUNCATED in s.flags for s in project.speech()))

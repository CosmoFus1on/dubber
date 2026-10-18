"""Stage 1b: pyannote diarization, gender per speaker (F0 + inaSpeechSegmenter), reference clip per speaker.

Writes work/diarization.json {"turns", "exclusive"}, work/ina.json, work/refs/<SPEAKER>.wav and Project.speakers.
"""
from __future__ import annotations

import json
import logging
import tempfile
import time
from pathlib import Path

import numpy as np

from . import audio, config, ffmpeg, gpu
from .schema import F_LOW_GENDER, F_NO_REF, Speaker

log = logging.getLogger("dubber.speakers")
SR = 16000          # analysis rate
REF_SR = 24000      # reference clip rate
FADE_S = 0.03
MIN_PIECE_S = 1.0
F0_MAX_AUDIO_S = 60.0


def decide_gender(f0_median, ina_male_ratio) -> tuple[str, float]:
    """Cross-check the F0 vote against inaSpeechSegmenter's male ratio. Returns (gender, confidence)."""
    if f0_median is None:
        return "unknown", 0.0
    f0_vote = "male" if f0_median < config.F0_MALE_MAX_HZ else "female"
    f0_strength = min(1.0, abs(f0_median - config.F0_MALE_MAX_HZ) / 60)
    if ina_male_ratio is None:
        return f0_vote, min(0.55, 0.3 + 0.3 * f0_strength)
    ina_vote = "male" if ina_male_ratio > 0.5 else "female"
    if ina_vote != f0_vote:
        return f0_vote, 0.4
    return f0_vote, 0.6 + 0.4 * min(f0_strength, abs(ina_male_ratio - 0.5) * 2)


# ---- interval helpers ([start, end] pairs in seconds) ----

def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _subtract(pieces: list, cuts: list) -> list:
    """pieces minus every cut range."""
    for c0, c1 in cuts:
        nxt = []
        for p0, p1 in pieces:
            if c1 <= p0 or c0 >= p1:
                nxt.append((p0, p1))
                continue
            if c0 > p0:
                nxt.append((p0, c0))
            if c1 < p1:
                nxt.append((c1, p1))
        pieces = nxt
    return pieces


# ---- sub-steps ----

def _diarize(x16: np.ndarray, opts) -> tuple[list, list]:
    import torch
    from pyannote.audio import Pipeline
    pipe = Pipeline.from_pretrained(config.DIARIZATION_MODEL, token=config.HF_TOKEN)
    pipe.to(torch.device(opts.device))
    kw = {"num_speakers": opts.num_speakers} if opts.num_speakers else {}
    out = pipe({"waveform": torch.from_numpy(x16).float().unsqueeze(0), "sample_rate": SR}, **kw)
    raw = [(s.start, s.end, spk) for s, _, spk in out.speaker_diarization.itertracks(yield_label=True)]
    exc = [(s.start, s.end, spk) for s, _, spk in out.exclusive_speaker_diarization.itertracks(yield_label=True)]
    del pipe, out
    gpu.unload()
    names: dict = {}
    for _, _, spk in sorted(raw + exc):
        names.setdefault(spk, f"SPEAKER_{len(names):02d}")
    fmt = lambda tr: [{"speaker": names[k], "start": round(a, 3), "end": round(b, 3)} for a, b, k in sorted(tr)]
    return fmt(raw), fmt(exc)


def _ina(vocals: Path):
    """[(label, start, end)] or None when TensorFlow / the model is unavailable."""
    try:
        from inaSpeechSegmenter import Segmenter
        with tempfile.TemporaryDirectory(prefix="dubber_") as td:
            wav = Path(td) / "vocals16k.wav"
            ffmpeg.extract_audio(vocals, wav, SR, 1)
            seg = Segmenter(detect_gender=True)
            labels = [(str(l), float(a), float(b)) for l, a, b in seg(str(wav))]
            del seg
        return labels
    except Exception as e:  # noqa: BLE001 - any TF/model failure falls back to F0 only
        log.warning("inaSpeechSegmenter unavailable (%s: %s); gender from F0 only", type(e).__name__, e)
        return None


def _f0_median(x16: np.ndarray, turns: list):
    import librosa
    chunks, total = [], 0.0
    for a, b in turns:
        take = min(b - a, F0_MAX_AUDIO_S - total)
        if take <= 0:
            break
        chunks.append(x16[int(a * SR):int((a + take) * SR)])
        total += take
    y = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    if len(y) < SR // 4:
        return None
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=400, sr=SR)
    f0 = f0[voiced & np.isfinite(f0)]
    return float(np.median(f0)) if f0.size else None


def _ref_clip(x16, bg16, pieces: list) -> tuple[np.ndarray, list]:
    """Pick the cleanest, longest pieces and concatenate them (30 ms fades) to MIN..MAX seconds."""
    scored = []
    for a, b in pieces:
        v, n = x16[int(a * SR):int(b * SR)], bg16[int(a * SR):int(b * SR)]
        ratio_db = 10 * np.log10((np.mean(v ** 2) + 1e-12) / (np.mean(n ** 2) + 1e-12))
        # ponytail: clamp + linear length bonus; a learned quality score (DNSMOS) is the upgrade
        scored.append((min(ratio_db, 40.0) + 2.0 * min(b - a, 10.0), a, b))
    chosen, total = [], 0.0
    for _, a, b in sorted(scored, reverse=True):
        if total >= config.REF_CLIP_MIN_S:
            break
        b = min(b, a + config.REF_CLIP_MAX_S - total)  # total < MIN < MAX, so b > a
        chosen.append((a, b))
        total += b - a
    chosen.sort()
    fade = int(FADE_S * REF_SR)
    ramp = np.linspace(0, 1, fade, dtype=np.float32)
    x24 = None
    out = []
    for a, b in chosen:
        if x24 is None:
            import librosa
            x24 = librosa.resample(x16, orig_sr=SR, target_sr=REF_SR)
        p = x24[int(a * REF_SR):int(b * REF_SR)].copy()
        if len(p) > 2 * fade:
            p[:fade] *= ramp
            p[-fade:] *= ramp[::-1]
        out.append(p)
    return (np.concatenate(out) if out else np.zeros(0, np.float32)), chosen


def run(project, work, opts) -> None:
    work = Path(work)
    stems = work / "stems"
    x16, _ = audio.read(stems / "vocals.wav", sr=SR, mono=True)
    bg16, _ = audio.read(stems / "no_vocals.wav", sr=SR, mono=True)
    dur = len(x16) / SR

    t = time.perf_counter()
    turns, exclusive = _diarize(x16, opts)
    if not turns:
        log.warning("pyannote found no speech; one speaker over the whole file")
        turns = exclusive = [{"speaker": "SPEAKER_00", "start": 0.0, "end": round(dur, 3)}]
    (work / "diarization.json").write_text(json.dumps({"turns": turns, "exclusive": exclusive}, indent=1),
                                           encoding="utf-8")
    t_dia = time.perf_counter() - t

    t = time.perf_counter()
    labels = _ina(stems / "vocals.wav")
    (work / "ina.json").write_text(json.dumps({"available": labels is not None, "labels": [
        {"label": l, "start": round(a, 3), "end": round(b, 3)} for l, a, b in labels or []]}, indent=1),
        encoding="utf-8")
    t_ina = time.perf_counter() - t

    bad = [(a, b) for l, a, b in labels or [] if l in ("music", "noise")]
    t_f0 = t_ref = 0.0
    project.speakers = {}
    for name in sorted({u["speaker"] for u in turns}):
        mine = [(u["start"], u["end"]) for u in exclusive if u["speaker"] == name]
        flags = []

        ina_ratio = None
        if labels is not None:
            sec = {"male": 0.0, "female": 0.0}
            for l, a, b in labels:
                if l in sec:
                    sec[l] += sum(_overlap(a, b, s, e) for s, e in mine)
            if sec["male"] + sec["female"] >= 1.0:
                ina_ratio = round(sec["male"] / (sec["male"] + sec["female"]), 3)

        t = time.perf_counter()
        f0 = _f0_median(x16, mine)
        t_f0 += time.perf_counter() - t
        gender, conf = decide_gender(f0, ina_ratio)
        if conf < config.GENDER_CONF_MIN:
            flags.append(F_LOW_GENDER)

        t = time.perf_counter()
        pieces = [(a, b) for a, b in _subtract(mine, bad) if b - a >= MIN_PIECE_S]
        if not pieces:  # < 1 s clean: use whatever this speaker has
            pieces = [(a, b) for a, b in mine if b - a > 0.1]
        clip, used = _ref_clip(x16, bg16, pieces)
        ref = None
        if len(clip):
            ref = f"refs/{name}.wav"
            audio.write(work / ref, audio.peak_normalize(clip), REF_SR)
        else:
            flags.append(F_NO_REF)
        t_ref += time.perf_counter() - t

        project.speakers[name] = Speaker(gender, round(conf, 3), None if f0 is None else round(f0, 1), ina_ratio,
                                         ref, flags)
        log.info("%s: %s conf=%.2f f0=%s ina_male=%s ref=%.1fs from %s %s", name, gender, conf,
                 None if f0 is None else round(f0, 1), ina_ratio, len(clip) / REF_SR,
                 [(round(a, 2), round(b, 2)) for a, b in used], flags or "")
    log.info("timing: diarization %.1fs, ina %.1fs, f0 %.1fs, refs %.1fs", t_dia, t_ina, t_f0, t_ref)

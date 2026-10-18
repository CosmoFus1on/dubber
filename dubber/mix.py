"""Stage 5b: mix and mux.

voice track  = fitted clips placed at absolute sample offsets (overlaps simply sum)
background   = no_vocals stem, swapped for the original mix inside music segments (50 ms crossfades)
mix          = duck(background by voice) + voice  ->  two-pass loudnorm -16 LUFS  ->  exact length
output       = original video stream copied + dubbed AAC (+ optional original track) + .srt
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from . import audio, config, ffmpeg
from .schema import MUSIC, SPEECH, Project

log = logging.getLogger("dubber.mix")


def _fit_length(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    pad = np.zeros((n - len(x),) + x.shape[1:], dtype=x.dtype)
    return np.concatenate([x, pad])


def _stereo(x: np.ndarray) -> np.ndarray:
    return np.stack([x, x], axis=1) if x.ndim == 1 else x[:, :2]


def music_mask(project: Project, n: int, sr: int) -> np.ndarray:
    """1.0 inside music segments, 0.0 elsewhere, with linear crossfade ramps."""
    mask = np.zeros(n, dtype=np.float32)
    for s in project.segments:
        if s.type == MUSIC:
            mask[max(0, int(s.start * sr)):min(n, int(s.end * sr))] = 1.0
    k = max(1, int(sr * config.MUSIC_XFADE_MS / 1000))
    if mask.any() and k > 1:
        kernel = np.ones(k, dtype=np.float32) / k
        mask = np.convolve(mask, kernel, mode="same").astype(np.float32)
    return mask


def build_voice(project: Project, work: Path, n: int, sr: int) -> np.ndarray:
    voice = np.zeros((n, 2), dtype=np.float32)
    placed = 0
    for s in project.speech():
        if not s.tts.fitted_wav:
            continue
        x, _ = audio.read(work / s.tts.fitted_wav, sr=sr, mono=False)
        x = _stereo(x)
        a = int(round(s.start * sr))
        x = x[:max(0, n - a)]
        voice[a:a + len(x)] += x
        placed += 1
    log.info("placed %d clips", placed)
    return voice


def run(project: Project, work: Path, opts: config.Options) -> None:
    sr = config.MIX_SR
    dur = float(project.source["duration"])
    n = int(round(dur * sr))
    mix_dir = work / "mix"
    mix_dir.mkdir(exist_ok=True)

    voice = build_voice(project, work, n, sr)
    voice_path = mix_dir / "voice.wav"
    audio.write(voice_path, voice, sr)

    bg, _ = audio.read(work / "stems" / "no_vocals.wav", sr=sr, mono=False)
    orig, _ = audio.read(work / "audio.wav", sr=sr, mono=False)
    bg, orig = _fit_length(_stereo(bg), n), _fit_length(_stereo(orig), n)
    m = music_mask(project, n, sr)[:, None]
    bg = bg * (1.0 - m) + orig * m
    bg_path = mix_dir / "background.wav"
    audio.write(bg_path, bg, sr)

    ducked_path = mix_dir / "background_ducked.wav"
    if voice.any():
        ffmpeg.duck(bg_path, voice_path, ducked_path)
        bg, _ = audio.read(ducked_path, sr=sr, mono=False)
        bg = _fit_length(_stereo(bg), n)
        bg = bg * (1.0 - m) + orig * m   # music segments are never ducked: original audio, untouched
    del orig
    mix = bg + voice
    peak = float(np.abs(mix).max()) if len(mix) else 0.0
    if peak > 0.99:
        mix *= 0.99 / peak
    raw_path = mix_dir / "mix_raw.wav"
    audio.write(raw_path, mix, sr)
    del mix, bg, voice

    norm_path = mix_dir / "mix_norm.wav"
    ffmpeg.loudnorm(raw_path, norm_path, config.TARGET_LUFS)
    final, _ = audio.read(norm_path, sr=sr, mono=False)
    final = _fit_length(_stereo(final), n)           # loudnorm's resampling can shift length by a few ms
    final_path = work / "mix.wav"
    audio.write(final_path, final, sr)
    got = audio.duration(final_path)
    if abs(got - dur) > config.LENGTH_TOLERANCE_S:
        raise RuntimeError(f"mix length {got:.3f}s != video {dur:.3f}s")

    out = Path(opts.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    srt = out.with_suffix(".srt")
    ffmpeg.write_srt([s for s in project.sorted_segments() if s.type == SPEECH or s.text], srt)
    ffmpeg.mux(opts.video, final_path, out, keep_original_audio=opts.keep_original_audio, srt=srt)
    vdur, adur = ffmpeg.duration(opts.video), ffmpeg.duration(out)
    log.info("wrote %s (video %.3fs, output %.3fs) and %s", out, vdur, adur, srt)
    if abs(vdur - adur) > 0.1:
        log.warning("container durations differ by %.3fs", abs(vdur - adur))

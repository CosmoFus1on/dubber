"""Audio I/O helpers: float32 numpy via soundfile."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger("dubber.audio")


def read(path, sr=None, mono: bool = False):
    x, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    if mono:
        x = x.mean(axis=1, keepdims=True)
    if sr and sr != file_sr:
        import librosa
        x = librosa.resample(x.T, orig_sr=file_sr, target_sr=sr).T
        file_sr = sr
    x = np.ascontiguousarray(x, dtype=np.float32)
    return (x[:, 0] if mono else x), file_sr


def write(path, data, sr: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sr, subtype="PCM_16")


def duration(path) -> float:
    return sf.info(str(path)).duration


def rms_db(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(20 * np.log10(max(np.sqrt(np.mean(x ** 2)) if x.size else 0.0, 1e-9)))


def peak_normalize(x, db: float = -1.0):
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak < 1e-9:
        return x
    return (x * (10 ** (db / 20) / peak)).astype(np.float32)

"""Stage 0: extract audio and split vocals / no_vocals with Demucs htdemucs."""
from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np
import torch

from . import audio, config, ffmpeg, gpu

log = logging.getLogger("dubber.separate")
SR = 48000


def run(project, work, opts) -> None:
    work = Path(work)
    wav = work / "audio.wav"
    ffmpeg.extract_audio(opts.video, wav, SR, 2)
    project.source["duration"] = ffmpeg.duration(opts.video)
    project.source["sample_rate"] = SR

    from demucs.api import Separator
    segment = None
    if opts.device == "cpu" or (torch.cuda.is_available() and
                                torch.cuda.get_device_properties(0).total_memory < 8 * 2**30):
        segment = 7  # htdemucs max segment is 7.8 s
    sep = Separator(model=config.DEMUCS_MODEL, device=opts.device, segment=segment, progress=False)
    x, _ = audio.read(wav, sr=sep.samplerate)  # (n, 2) float32
    _, stems = sep.separate_tensor(torch.from_numpy(x.T.copy()), sep.samplerate)
    vocals = stems["vocals"].cpu().numpy()
    rest = sum(v.cpu().numpy() for k, v in stems.items() if k != "vocals")
    out = work / "stems"
    for name, a in (("vocals", vocals), ("no_vocals", rest)):
        a = librosa.resample(np.asarray(a), orig_sr=sep.samplerate, target_sr=SR)
        audio.write(out / f"{name}.wav", np.clip(a.T, -1, 1), SR)
    del sep, stems
    gpu.unload()

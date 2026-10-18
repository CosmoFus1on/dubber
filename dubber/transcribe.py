"""Stage 1a: faster-whisper transcription with word timestamps."""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from . import config, ffmpeg, gpu

log = logging.getLogger("dubber.transcribe")


def model_name(device: str) -> str:
    return config.WHISPER_GPU if device == "cuda" else config.WHISPER_CPU


def run(project, work, opts) -> None:
    work = Path(work)
    from faster_whisper import BatchedInferencePipeline, WhisperModel
    with tempfile.TemporaryDirectory(prefix="dubber_") as td:
        wav16 = Path(td) / "vocals16k.wav"
        ffmpeg.extract_audio(work / "stems" / "vocals.wav", wav16, 16000, 1)
        cuda = opts.device == "cuda"
        model = WhisperModel(model_name(opts.device), device=opts.device,
                             compute_type="float16" if cuda else "int8")
        kw = dict(word_timestamps=True, vad_filter=True, language=project.source.get("language") or None)
        if cuda:
            segs, info = BatchedInferencePipeline(model).transcribe(str(wav16), batch_size=8, **kw)
        else:
            segs, info = model.transcribe(str(wav16), **kw)
        words, wsegs = [], []
        for s in segs:
            wsegs.append({"start": s.start, "end": s.end, "text": s.text, "avg_logprob": s.avg_logprob,
                          "compression_ratio": s.compression_ratio, "no_speech_prob": s.no_speech_prob})
            for w in s.words or []:
                words.append({"w": w.word, "s": w.start, "e": w.end, "p": w.probability})
        del model
    out = {"language": info.language, "language_probability": info.language_probability,
           "words": words, "whisper_segments": wsegs}
    (work / "transcript.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    project.source["language"] = info.language
    log.info("language=%s (%.2f), %d words", info.language, info.language_probability, len(words))
    gpu.unload()

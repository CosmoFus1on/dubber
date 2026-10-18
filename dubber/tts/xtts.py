"""XTTS-v2 via coqui-tts 0.27 (Idiap fork). Code MPL-2.0; model weights Coqui Public Model License
(non-commercial only). Zero-shot cloning from the reference clip, 17 languages.

Runs in envs/xtts (torch 2.8.0+cu128: 2.9+ would need torchcodec). COQUI_TOS_AGREED=1 accepts the CPML
prompt non-interactively. Speed: XTTS `speed` = job.speed.
"""
from __future__ import annotations

import logging
import os

from .base import TTSEngine, TTSJob, TTSResult, main

log = logging.getLogger("dubber.tts.xtts")

LANGS = frozenset("en es fr de it pt pl tr ru nl cs ar zh ja hu ko hi".split())
MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
XTTS_LANG = {"zh": "zh-cn"}


class XTTS(TTSEngine):
    name = "xtts"
    languages = LANGS
    clones = True

    def check(self) -> bool:
        try:
            import TTS  # noqa: F401
            return True
        except Exception:
            return False

    def synth(self, jobs: list[TTSJob], device: str) -> list[TTSResult]:
        os.environ["COQUI_TOS_AGREED"] = "1"
        import torch
        from TTS.api import TTS

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        tts = TTS(MODEL).to(device)
        results = []
        for j in jobs:
            try:
                if not j.ref_wav:
                    raise ValueError("xtts needs a reference clip")
                tts.tts_to_file(text=j.text, speaker_wav=j.ref_wav, language=XTTS_LANG.get(j.lang, j.lang),
                                file_path=j.out_wav, speed=float(j.speed))
                results.append(TTSResult(j.id, j.out_wav, self.name))
            except Exception as e:
                log.exception("xtts failed on job %s", j.id)
                results.append(TTSResult(j.id, None, self.name, f"{type(e).__name__}: {e}"))
        del tts
        if device == "cuda":
            torch.cuda.empty_cache()
        return results


ENGINE = XTTS()

if __name__ == "__main__":
    main(ENGINE)

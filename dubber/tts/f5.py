"""F5-TTS v1 Base (SWivid/F5-TTS). Code MIT; model weights CC-BY-NC-4.0 (non-commercial). zh + en only.

Runs in .envs/f5 (torch 2.8.0+cu128). F5 needs the reference clip's transcript: we pass ref_text="" and
f5-tts transcribes the (auto-clipped, <=12 s) reference with Whisper itself, cached per clip.
Speed: F5 `speed` = job.speed.
"""
from __future__ import annotations

import logging

from .base import TTSEngine, TTSJob, TTSResult, main

log = logging.getLogger("dubber.tts.f5")


class F5(TTSEngine):
    name = "f5"
    languages = frozenset({"zh", "en"})
    clones = True

    def check(self) -> bool:
        try:
            import f5_tts  # noqa: F401
            return True
        except Exception:
            return False

    def synth(self, jobs: list[TTSJob], device: str) -> list[TTSResult]:
        import torch
        from f5_tts.api import F5TTS

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        model = F5TTS(model="F5TTS_v1_Base", device=device)
        results = []
        for j in jobs:
            try:
                if not j.ref_wav:
                    raise ValueError("f5 needs a reference clip")
                model.infer(ref_file=j.ref_wav, ref_text="", gen_text=j.text, file_wave=j.out_wav,
                            speed=float(j.speed), show_info=lambda *a, **k: None)
                results.append(TTSResult(j.id, j.out_wav, self.name))
            except Exception as e:
                log.exception("f5 failed on job %s", j.id)
                results.append(TTSResult(j.id, None, self.name, f"{type(e).__name__}: {e}"))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        return results


ENGINE = F5()

if __name__ == "__main__":
    main(ENGINE)

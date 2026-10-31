"""Chatterbox Multilingual v3 (Resemble AI, MIT). Zero-shot voice cloning, 23 languages.

Runs in .envs/chatterbox (pins torch 2.6.0 / transformers 5.2.0). Every output carries Resemble's
PerTh watermark; we keep it. Prosody hints: exaggeration (emotion intensity) and cfg_weight (pacing).
"""
from __future__ import annotations

import logging

from .base import TTSEngine, TTSJob, TTSResult, main

log = logging.getLogger("dubber.tts.chatterbox")

LANGS = frozenset("ar da de el en es fi fr he hi it ja ko ms nl no pl pt ru sv sw tr zh".split())


class Chatterbox(TTSEngine):
    name = "chatterbox"
    languages = LANGS
    clones = True

    def check(self) -> bool:
        try:
            import chatterbox  # noqa: F401
            return True
        except Exception:
            return False

    def synth(self, jobs: list[TTSJob], device: str) -> list[TTSResult]:
        import soundfile as sf
        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        model = getattr(self, "_model", None)
        if model is None:
            try:
                model = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v3")
            except TypeError:  # older package without the v3 switch
                model = ChatterboxMultilingualTTS.from_pretrained(device=device)
            if self.keep_loaded:
                self._model = model   # warm worker: ~25 s load paid once per run
        # ponytail: torch.load on this checkpoint is fine on CPU too, just slow (~10x realtime)
        results = []
        for j in jobs:
            try:
                wav = model.generate(j.text, language_id=j.lang, audio_prompt_path=j.ref_wav,
                                     exaggeration=float(j.exaggeration), cfg_weight=float(j.cfg_weight))
                sf.write(j.out_wav, wav.squeeze(0).cpu().numpy(), model.sr)
                results.append(TTSResult(j.id, j.out_wav, self.name))
            except Exception as e:  # never let one segment kill the batch
                log.exception("chatterbox failed on job %s", j.id)
                results.append(TTSResult(j.id, None, self.name, f"{type(e).__name__}: {e}"))
        if not self.keep_loaded:
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        return results


ENGINE = Chatterbox()

if __name__ == "__main__":
    main(ENGINE)

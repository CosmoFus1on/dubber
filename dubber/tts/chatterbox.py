"""Chatterbox Multilingual v3 (Resemble AI, MIT). Zero-shot voice cloning, 23 languages.

Runs in .envs/chatterbox (pins torch 2.6.0 / transformers 5.2.0). Every output carries Resemble's
PerTh watermark; we keep it. Prosody hints: exaggeration (emotion intensity) and cfg_weight (pacing).
"""
from __future__ import annotations

import logging
from pathlib import Path

from .. import config
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

    @staticmethod
    def _load_pack(lang: str, repo: str, device: str):
        """Assemble a from_local() folder for a single-language pack. chatterbox-tts 0.1.7 hard-codes the file
        names, so the pack's T3 takes the multilingual T3's name; the pack's v3 decoder is overlaid non-strictly
        (it only lacks two computed mel buffers)."""
        import shutil
        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        from huggingface_hub import snapshot_download
        d = config.MODELS_DIR / f"chatterbox-{lang}"
        if not (d / "s3gen.pt").exists():
            base = Path(snapshot_download("ResembleAI/chatterbox", allow_patterns=[
                "ve.pt", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json"]))
            pack = Path(snapshot_download(repo))
            d.mkdir(parents=True, exist_ok=True)
            for n in ("ve.pt", "s3gen.pt", "conds.pt", "grapheme_mtl_merged_expanded_v1.json", "Cangjie5_TC.json"):
                shutil.copyfile(base / n, d / n)
            shutil.copyfile(pack / f"t3_{lang}.safetensors", d / "t3_mtl23ls_v2.safetensors")
            if (pack / "s3gen_v3.pt").exists():
                shutil.copyfile(pack / "s3gen_v3.pt", d / "s3gen_v3.pt")
        model = ChatterboxMultilingualTTS.from_local(d, device)
        if (d / "s3gen_v3.pt").exists():
            res = model.s3gen.load_state_dict(torch.load(d / "s3gen_v3.pt", map_location=device, weights_only=True),
                                              strict=False)
            log.info("chatterbox %s pack loaded (decoder v3; %d computed buffers kept)", lang, len(res.missing_keys))
        return model

    def synth(self, jobs: list[TTSJob], device: str) -> list[TTSResult]:
        import soundfile as sf
        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        # one model per target language: the dedicated pack when Resemble ships one, else the 23-language model
        lang = jobs[0].lang if jobs else ""
        pack = config.CHATTERBOX_LANG_PACKS.get(lang)
        model = getattr(self, "_model", None) if getattr(self, "_model_lang", None) == lang else None
        if model is None:
            if pack:
                model = self._load_pack(lang, pack, device)
            else:
                try:
                    model = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v3")
                except TypeError:  # chatterbox-tts 0.1.7 has no v3 switch and loads the v2 checkpoint
                    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
            if self.keep_loaded:
                self._model, self._model_lang = model, lang   # warm worker: ~25 s load paid once per run
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

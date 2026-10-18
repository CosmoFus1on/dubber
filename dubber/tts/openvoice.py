"""OpenVoice V2 (MyShell, MIT) on top of MeloTTS (MyShell, MIT). en es fr zh ja ko.

MeloTTS speaks the line in its base voice for the language, then OpenVoice's ToneColorConverter moves the
timbre to the reference clip's speaker (tone colour only: accent and rhythm stay Melo's).
Runs in envs/openvoice (torch 2.8.0+cu128, MeloTTS pins transformers 4.27.4; OpenVoice installed --no-deps
because its own pins are stale; Japanese needs `python -m unidic download`). Checkpoints from
myshell-ai/OpenVoiceV2 go to models/openvoice/. Watermarking (wavmark) is off. Speed: Melo `speed` = job.speed.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from .. import config
from .base import TTSEngine, TTSJob, TTSResult, main

log = logging.getLogger("dubber.tts.openvoice")

CKPT = config.ROOT / "models" / "openvoice"
# our code -> (Melo language, Melo speaker key, base-speaker SE file)
MELO = {"en": ("EN", "EN-US", "en-us"), "es": ("ES", "ES", "es"), "fr": ("FR", "FR", "fr"),
        "zh": ("ZH", "ZH", "zh"), "ja": ("JP", "JP", "jp"), "ko": ("KR", "KR", "kr")}


class OpenVoice(TTSEngine):
    name = "openvoice"
    languages = frozenset(MELO)
    clones = True

    def check(self) -> bool:
        try:
            import melo.api  # noqa: F401
            import openvoice.api  # noqa: F401
            return True
        except Exception:
            return False

    def synth(self, jobs: list[TTSJob], device: str) -> list[TTSResult]:
        import torch
        from huggingface_hub import snapshot_download
        from melo.api import TTS
        from openvoice.api import ToneColorConverter

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        snapshot_download("myshell-ai/OpenVoiceV2", local_dir=str(CKPT),
                          allow_patterns=["converter/*", "base_speakers/ses/*.pth"])
        conv = ToneColorConverter(str(CKPT / "converter" / "config.json"), device=device, enable_watermark=False)
        conv.load_ckpt(str(CKPT / "converter" / "checkpoint.pth"))
        melo: dict[str, TTS] = {}
        tgt_se: dict[str, object] = {}
        results = []
        with tempfile.TemporaryDirectory(prefix="dubber_ov_") as td:
            for j in jobs:
                try:
                    if not j.ref_wav:
                        raise ValueError("openvoice needs a reference clip")
                    mlang, spk, se_name = MELO[j.lang]
                    if mlang not in melo:
                        melo[mlang] = TTS(language=mlang, device=device)
                    m = melo[mlang]
                    if j.ref_wav not in tgt_se:
                        tgt_se[j.ref_wav] = conv.extract_se([j.ref_wav])
                    src_se = torch.load(str(CKPT / "base_speakers" / "ses" / f"{se_name}.pth"), map_location=device)
                    base = str(Path(td) / f"{j.id}.wav")
                    m.tts_to_file(j.text, m.hps.data.spk2id[spk], base, speed=float(j.speed), quiet=True)
                    conv.convert(audio_src_path=base, src_se=src_se, tgt_se=tgt_se[j.ref_wav], output_path=j.out_wav)
                    results.append(TTSResult(j.id, j.out_wav, self.name))
                except Exception as e:
                    log.exception("openvoice failed on job %s", j.id)
                    results.append(TTSResult(j.id, None, self.name, f"{type(e).__name__}: {e}"))
        del conv, melo, tgt_se
        if device == "cuda":
            torch.cuda.empty_cache()
        return results


ENGINE = OpenVoice()

if __name__ == "__main__":
    main(ENGINE)

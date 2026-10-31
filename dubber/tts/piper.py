"""Piper (rhasspy/OHF-Voice, piper-tts 1.8). Code GPL-3.0; the voices used here (rhasspy/piper-voices) are MIT.

No cloning: a fixed voice per (language, gender). Runs in the main venv (ONNX, CPU is fast enough).
Voices download lazily into .cache/models/piper/ on first use. zh needs piper-tts[zh] (g2pW, also downloaded
into .cache/models/piper/), ja needs piper-tts[ja] (pyopenjtalk-plus). Speed: length_scale = default / job.speed.
"""
from __future__ import annotations

import logging

from .. import config
from .base import TTSEngine, TTSJob, TTSResult, main

log = logging.getLogger("dubber.tts.piper")

VOICE_DIR = config.MODELS_DIR / "piper"
# (voice name, speaker_id) per language and gender; "unknown" falls back to female.
VOICES = {
    "en": {"male": ("en_US-ryan-medium", None), "female": ("en_US-lessac-medium", None)},
    "hi": {"male": ("hi_IN-rohan-medium", None), "female": ("hi_IN-priyamvada-medium", None)},
    "zh": {"male": ("zh_CN-chaowen-medium", None), "female": ("zh_CN-huayan-medium", None)},
    "ja": {"male": ("ja_JP-hi_fi_captain-medium", 1), "female": ("ja_JP-hi_fi_captain-medium", 0)},
}


def voice_for(lang: str, gender: str) -> tuple[str, int | None]:
    table = VOICES[lang]
    return table.get(gender, table["female"])


class Piper(TTSEngine):
    name = "piper"
    languages = frozenset(VOICES)
    clones = False

    def check(self) -> bool:
        try:
            import piper  # noqa: F401
            return True
        except Exception:
            return False

    def synth(self, jobs: list[TTSJob], device: str) -> list[TTSResult]:
        import wave

        from piper import PiperVoice, SynthesisConfig
        from piper.download_voices import download_voice

        VOICE_DIR.mkdir(parents=True, exist_ok=True)
        loaded: dict[str, PiperVoice] = {}
        results = []
        for j in jobs:
            try:
                name, sid = voice_for(j.lang, j.gender)
                if name not in loaded:
                    onnx = VOICE_DIR / f"{name}.onnx"
                    if not onnx.exists():
                        download_voice(name, VOICE_DIR)
                    # ponytail: CPU only; onnxruntime-gpu needs matching CUDA DLLs and piper is already ~50x realtime
                    loaded[name] = PiperVoice.load(onnx, download_dir=VOICE_DIR)
                v = loaded[name]
                scale = (v.config.length_scale or 1.0) / max(float(j.speed), 0.1)
                with wave.open(j.out_wav, "wb") as wf:
                    v.synthesize_wav(j.text, wf, syn_config=SynthesisConfig(speaker_id=sid, length_scale=scale))
                results.append(TTSResult(j.id, j.out_wav, self.name))
            except Exception as e:
                log.exception("piper failed on job %s", j.id)
                results.append(TTSResult(j.id, None, self.name, f"{type(e).__name__}: {e}"))
        loaded.clear()
        return results


ENGINE = Piper()

if __name__ == "__main__":
    main(ENGINE)

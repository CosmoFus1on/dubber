"""A stub TTS engine for testing the warm-worker protocol: writes a short tone, counts model loads."""
from __future__ import annotations

import sys
import time

from dubber.tts.base import TTSEngine, TTSJob, TTSResult, main


class Fake(TTSEngine):
    name = "fake"
    languages = frozenset({"en", "hi"})
    loads = 0

    def synth(self, jobs: list[TTSJob], device: str) -> list[TTSResult]:
        import numpy as np
        import soundfile as sf
        if getattr(self, "_model", None) is None:
            time.sleep(0.2)                      # pretend to load
            Fake.loads += 1
            if self.keep_loaded:
                self._model = object()
        print("chatter on stdout must not break the protocol")  # libraries do this
        out = []
        for j in jobs:
            sf.write(j.out_wav, 0.1 * np.sin(np.arange(2400) / 10.0).astype("float32"), 24000)
            out.append(TTSResult(j.id, j.out_wav, self.name))
        sys.stderr.write(f"fake loads={Fake.loads}\n")
        return out


ENGINE = Fake()

if __name__ == "__main__":
    main(ENGINE)

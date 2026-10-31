"""TTSEngine contract, per-language fallback selection, and the subprocess runner.

An engine module (dubber/tts/<name>.py) must:
  * import only stdlib + this module at import time (heavy imports inside synth()),
  * define ENGINE = <Engine instance> with .name, .languages (frozenset of 2-letter codes), .clones,
  * end with:  if __name__ == "__main__": main(ENGINE)
If .envs/<name>/Scripts/python.exe exists, the engine runs in that venv via subprocess (used when
its dependency pins conflict with the main venv); otherwise it runs in-process.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import os
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .. import config
from ..config import norm_lang

log = logging.getLogger("dubber.tts")

ENGINE_MODULES = {
    "chatterbox": "dubber.tts.chatterbox",
    "xtts": "dubber.tts.xtts",
    "openvoice": "dubber.tts.openvoice",
    "f5": "dubber.tts.f5",
    "piper": "dubber.tts.piper",
}


@dataclass
class TTSJob:
    id: str
    text: str
    lang: str                      # 2-letter target code (config.norm_lang applied by caller)
    out_wav: str                   # absolute path to write (engine's native sample rate is fine)
    ref_wav: Optional[str] = None  # speaker reference clip; None -> engine default voice
    gender: str = "unknown"        # for engines without cloning (voice pick)
    exaggeration: float = 0.5      # prosody hints from the original segment; engines may ignore
    cfg_weight: float = 0.5
    speed: float = 1.0


@dataclass
class TTSResult:
    id: str
    wav: Optional[str]
    engine: str = ""
    error: Optional[str] = None


class TTSEngine(ABC):
    name: str = ""
    languages: frozenset = frozenset()
    clones: bool = True

    def supports(self, lang: str) -> bool:
        return norm_lang(lang) in self.languages

    def check(self) -> bool:
        """Cheap availability probe for in-process engines (imports, voice files). Override as needed."""
        return True

    @abstractmethod
    def synth(self, jobs: list[TTSJob], device: str) -> list[TTSResult]:
        """Synthesize every job (load the model once, free it before returning). Never raise per job."""


def engine_python(name: str) -> Optional[Path]:
    for rel in ("Scripts/python.exe", "bin/python"):
        p = config.ENGINE_VENVS / name / rel
        if p.exists():
            return p
    return None


def available(name: str) -> bool:
    if engine_python(name):
        return True
    try:
        return importlib.util.find_spec(ENGINE_MODULES[name]) is not None and _importable_in_process(name)
    except Exception:
        return False


def _importable_in_process(name: str) -> bool:
    try:
        mod = importlib.import_module(ENGINE_MODULES[name])
        return bool(mod.ENGINE.check())
    except Exception:
        return False


def chain_for(lang: str, cpu: bool = False) -> list[str]:
    chain = config.TTS_CPU_CHAIN if cpu else config.TTS_FALLBACK.get(norm_lang(lang), config.TTS_DEFAULT_CHAIN)
    return [e for e in chain if available(e)]


def run_engine(name: str, jobs: list[TTSJob], device: str) -> list[TTSResult]:
    """Run jobs on one engine, in its own venv when present. Returns one result per job, in order."""
    if not jobs:
        return []
    py = engine_python(name)
    if py is None:
        mod = importlib.import_module(ENGINE_MODULES[name])
        return mod.ENGINE.synth(jobs, device)
    with tempfile.TemporaryDirectory(prefix="dubber_tts_") as td:
        jobs_path, out_path = Path(td) / "jobs.json", Path(td) / "out.json"
        jobs_path.write_text(json.dumps({"device": device, "jobs": [asdict(j) for j in jobs]}, ensure_ascii=False),
                             encoding="utf-8")
        env = dict(os.environ, PYTHONPATH=str(config.ROOT), PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        proc = subprocess.run([str(py), "-m", ENGINE_MODULES[name], str(jobs_path), str(out_path)],
                              cwd=str(config.ROOT), env=env, capture_output=True, text=True, encoding="utf-8",
                              errors="replace")
        if proc.returncode != 0 or not out_path.exists():
            tail = (proc.stderr or "")[-2000:]
            log.error("engine %s subprocess failed (rc=%s):\n%s", name, proc.returncode, tail)
            return [TTSResult(j.id, None, name, f"subprocess rc={proc.returncode}: {tail[-300:]}") for j in jobs]
        results = json.loads(out_path.read_text(encoding="utf-8"))
        return [TTSResult(**r) for r in results]


def main(engine: TTSEngine) -> None:
    """CLI entry used by run_engine: python -m dubber.tts.<name> jobs.json out.json"""
    jobs_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    spec = json.loads(jobs_path.read_text(encoding="utf-8"))
    jobs = [TTSJob(**j) for j in spec["jobs"]]
    results = engine.synth(jobs, spec.get("device", "cpu"))
    out_path.write_text(json.dumps([asdict(r) for r in results], ensure_ascii=False), encoding="utf-8")

"""TTSEngine contract, per-language fallback selection, and the subprocess runner.

An engine module (dubber/tts/<name>.py) must:
  * import only stdlib + this module at import time (heavy imports inside synth()),
  * define ENGINE = <Engine instance> with .name, .languages (frozenset of 2-letter codes), .clones,
  * end with:  if __name__ == "__main__": main(ENGINE)
If .envs/<name>/Scripts/python.exe exists, the engine runs in that venv via subprocess (used when
its dependency pins conflict with the main venv); otherwise it runs in-process.
"""
from __future__ import annotations

import atexit
import importlib
import importlib.util
import json
import logging
import os
import subprocess
import sys
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
    keep_loaded: bool = False   # set by --serve: synth() may cache its model on self between calls

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


_workers: dict[str, subprocess.Popen] = {}   # one warm engine process per engine name, see _worker()


def _spawn(name: str, py: Path) -> subprocess.Popen:
    env = dict(os.environ, PYTHONPATH=str(config.ROOT), PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    return subprocess.Popen([str(py), "-m", ENGINE_MODULES[name], "--serve"], cwd=str(config.ROOT), env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8",
                            errors="replace", bufsize=1)


def _worker(name: str, py: Path) -> subprocess.Popen:
    """A warm engine process: the model loads once and serves every later call (Chatterbox: ~25 s saved per call).
    Only one engine stays resident, so VRAM never holds two TTS models."""
    w = _workers.get(name)
    if w is not None and w.poll() is None:
        return w
    stop_workers()
    if not _workers and not getattr(stop_workers, "_registered", False):
        atexit.register(stop_workers)
        stop_workers._registered = True  # type: ignore[attr-defined]
    _workers[name] = _spawn(name, py)
    return _workers[name]


def stop_workers() -> None:
    for name, w in list(_workers.items()):
        try:
            if w.poll() is None:
                w.stdin.close()
                w.wait(timeout=10)
        except Exception:
            w.kill()
        _workers.pop(name, None)


def run_engine(name: str, jobs: list[TTSJob], device: str) -> list[TTSResult]:
    """Run jobs on one engine, in its own venv when present. Returns one result per job, in order."""
    if not jobs:
        return []
    py = engine_python(name)
    if py is None:
        mod = importlib.import_module(ENGINE_MODULES[name])
        return mod.ENGINE.synth(jobs, device)
    spec = json.dumps({"device": device, "jobs": [asdict(j) for j in jobs]}, ensure_ascii=False)
    for attempt in (1, 2):  # a crashed worker gets one fresh process
        w = _worker(name, py)
        try:
            w.stdin.write(spec + "\n")
            w.stdin.flush()
            line = w.stdout.readline()
        except (OSError, ValueError):
            line = ""
        if line:
            return [TTSResult(**r) for r in json.loads(line)]
        rc = w.poll()
        log.error("engine %s worker died (rc=%s) on attempt %d", name, rc, attempt)
        _workers.pop(name, None)
    return [TTSResult(j.id, None, name, "engine worker died; see its stderr above") for j in jobs]


def main(engine: TTSEngine) -> None:
    """CLI entry used by run_engine.
    --serve:            read one JSON batch per stdin line, answer one JSON line on stdout, keep the model loaded.
    jobs.json out.json: one-shot file mode (benchmarks, smoke tests)."""
    if sys.argv[1:2] == ["--serve"]:
        out = sys.stdout
        sys.stdout = sys.stderr   # anything a library prints must not corrupt the protocol
        engine.keep_loaded = True
        for line in sys.stdin:
            if not line.strip():
                continue
            spec = json.loads(line)
            results = engine.synth([TTSJob(**j) for j in spec["jobs"]], spec.get("device", "cpu"))
            out.write(json.dumps([asdict(r) for r in results], ensure_ascii=False) + "\n")
            out.flush()
        return
    jobs_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    spec = json.loads(jobs_path.read_text(encoding="utf-8"))
    jobs = [TTSJob(**j) for j in spec["jobs"]]
    results = engine.synth(jobs, spec.get("device", "cpu"))
    out_path.write_text(json.dumps([asdict(r) for r in results], ensure_ascii=False), encoding="utf-8")

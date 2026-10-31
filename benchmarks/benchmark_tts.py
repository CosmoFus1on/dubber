"""TTS benchmark: every available engine x {hi, ms, ja, zh} it supports, 5 fixed sentences (benchmarks/sentences.json).

    .venv\Scripts\python.exe -X utf8 bench\benchmark_tts.py                 # all engines
    .venv\Scripts\python.exe -X utf8 bench\benchmark_tts.py xtts piper      # just these (merged into results.json)

Per (engine, lang): one 1-sentence warm-up run_engine call, then one 5-sentence call, on cuda.
  load_s   = wall time of the warm-up call (subprocess start + imports + model load + 1 sentence)
  rtf      = synth seconds / trimmed audio seconds. Subprocess engines pay the load in both calls, so
             rtf = (five - warm) / audio of sentences 1-4; in-process engines (piper) only pay it in the
             warm-up, so rtf = five / audio of all 5.
  err_ms   = |trimmed duration - slot| (librosa trim top_db=35), mean and max
  sim      = mean cosine similarity of resemblyzer embeddings, output vs the speaker's reference clip
  vram_mb  = peak device-wide nvidia-smi memory.used during the 5-sentence call, minus the idle baseline
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import importlib  # noqa: E402

import librosa  # noqa: E402
import numpy as np  # noqa: E402

from dubber import audio, gpu  # noqa: E402
from dubber.tts.base import ENGINE_MODULES, TTSJob, available, engine_python, run_engine  # noqa: E402

BENCH = ROOT / "bench"
LANGS = ["hi", "ms", "ja", "zh"]
ENGINES = ["chatterbox", "xtts", "openvoice", "f5", "piper"]
REF = {"A": ROOT / ".cache" / "work" / "clip" / "refs" / "SPEAKER_00.wav", "B": ROOT / ".cache" / "work" / "clip" / "refs" / "SPEAKER_01.wav"}


def vram_used() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.split()[0])
    except Exception:
        return -1


class VramPeak(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.base, self.peak, self.stop = vram_used(), 0, threading.Event()

    def run(self):
        while not self.stop.is_set():
            self.peak = max(self.peak, vram_used())
            self.stop.wait(0.5)

    def result(self) -> int | None:
        self.stop.set()
        self.join()
        return (self.peak - self.base) if self.base >= 0 else None


def trimmed(wav: str) -> tuple[np.ndarray, int]:
    x, sr = audio.read(wav, mono=True)
    y, _ = librosa.effects.trim(x, top_db=35)
    return y, sr


def jobs_for(engine: str, lang: str, sents: list[dict], tag: str) -> list[TTSJob]:
    out = BENCH / "out" / engine / lang
    out.mkdir(parents=True, exist_ok=True)
    return [TTSJob(f"{tag}{s['id']}", s["text"][lang], lang, str(out / f"{tag}{s['id']}.wav"),
                   ref_wav=str(REF[s["speaker"]]), gender=s["gender"]) for s in sents]


def bench_pair(engine: str, lang: str, sents: list[dict], encoder, ref_emb: dict) -> dict:
    from resemblyzer import preprocess_wav

    t = time.perf_counter()
    run_engine(engine, jobs_for(engine, lang, sents[:1], "warm"), "cuda")
    warm = time.perf_counter() - t
    gpu.unload()
    vp = VramPeak()
    vp.start()
    t = time.perf_counter()
    res = run_engine(engine, jobs_for(engine, lang, sents, ""), "cuda")
    five = time.perf_counter() - t
    vram = vp.result()
    gpu.unload()
    sub = engine_python(engine) is not None
    synth = (five - warm) if sub else five
    rows, fails, secs, errors = [], 0, 0.0, []
    for s, r in zip(sents, res):
        row = {"id": s["id"], "text": s["text"][lang], "slot": s["slot"], "error": r.error}
        if r.wav and Path(r.wav).exists():
            y, sr = trimmed(r.wav)
            dur = len(y) / sr
            if dur < 0.2 or audio.rms_db(y) < -45:
                row["error"] = row["error"] or "silent output"
            else:
                emb = encoder.embed_utterance(preprocess_wav(y, source_sr=sr))
                row.update(wav=str(Path(r.wav).relative_to(ROOT)), dur=round(dur, 3),
                           err_ms=round(abs(dur - s["slot"]) * 1000), sim=round(float(emb @ ref_emb[s["speaker"]]), 3))
                secs += dur if (s["id"] > 0 or not sub) else 0.0
                errors.append(row["err_ms"])
        if "dur" not in row:
            fails += 1
        rows.append(row)
    sims = [r["sim"] for r in rows if "sim" in r]
    return {"engine": engine, "lang": lang, "subprocess": sub, "load_s": round(warm, 1),
            "five_wall_s": round(five, 1), "rtf_audio_s": round(secs, 2),
            "rtf": round(max(synth, 0.0) / secs, 3) if secs else None,
            "err_mean_ms": round(float(np.mean(errors))) if errors else None,
            "err_max_ms": max(errors) if errors else None,
            "sim": round(float(np.mean(sims)), 3) if sims else None,
            "fails": fails, "vram_mb": vram, "lines": rows}


def table(results: list[dict]) -> str:
    out = ["| engine | lang | load s (warm-up) | RTF | timing err mean / max ms | speaker sim | fails | VRAM peak MB |",
           "|---|---|---|---|---|---|---|---|"]
    order = {e: i for i, e in enumerate(ENGINES)}
    for r in sorted(results, key=lambda r: (order.get(r["engine"], 99), LANGS.index(r["lang"]))):
        f = lambda v, fmt="{}": "-" if v is None else fmt.format(v)  # noqa: E731
        out.append(f"| {r['engine']} | {r['lang']} | {r['load_s']} | {f(r['rtf'], '{:.2f}')} | "
                   f"{f(r['err_mean_ms'])} / {f(r['err_max_ms'])} | {f(r['sim'], '{:.3f}')} | {r['fails']}/5 | "
                   f"{f(r['vram_mb'])} |")
    return "\n".join(out)


def write_md(results: list[dict]) -> None:
    md = BENCH / "results.md"
    a, b = "<!-- table -->", "<!-- /table -->"
    text = md.read_text(encoding="utf-8") if md.exists() else f"# TTS benchmark\n\n{a}\n{b}\n"
    head, rest = text.split(a, 1)
    md.write_text(f"{head}{a}\n{table(results)}\n{b}{rest.split(b, 1)[1]}", encoding="utf-8")


def main(names: list[str]) -> None:
    from resemblyzer import VoiceEncoder, preprocess_wav

    sents = json.loads((BENCH / "sentences.json").read_text(encoding="utf-8"))["sentences"]
    rj = BENCH / "results.json"
    results = json.loads(rj.read_text(encoding="utf-8")) if rj.exists() else []
    encoder = VoiceEncoder("cpu")  # CPU: keeps the GPU free for the engine under test
    ref_emb = {k: encoder.embed_utterance(preprocess_wav(p)) for k, p in REF.items()}
    for name in names:
        eng = importlib.import_module(ENGINE_MODULES[name]).ENGINE
        if not available(name):
            print(f"{name}: not available, skipped")
            continue
        for lang in [l for l in LANGS if eng.supports(l)]:
            print(f"{name} {lang} ...", flush=True)
            r = bench_pair(name, lang, sents, encoder, ref_emb)
            print(f"  load {r['load_s']}s rtf {r['rtf']} err {r['err_mean_ms']}/{r['err_max_ms']} ms "
                  f"sim {r['sim']} fails {r['fails']} vram {r['vram_mb']}", flush=True)
            results = [x for x in results if (x["engine"], x["lang"]) != (name, lang)] + [r]
            rj.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
            write_md(results)
        gpu.unload()


if __name__ == "__main__":
    main(sys.argv[1:] or ENGINES)

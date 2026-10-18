"""Smoke test: one sentence per language through run_engine; checks a non-silent wav is written.

    .venv\Scripts\python.exe -X utf8 bench\smoke_tts.py piper hi zh ja en
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dubber import audio  # noqa: E402
from dubber.tts.base import TTSJob, run_engine  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TEXT = {"en": "Hey, I was thinking we should finally plan that trip.",
        "hi": "मैं सोच रहा था कि हमें आखिरकार वह यात्रा प्लान करनी चाहिए।",
        "zh": "我在想，我们终于应该计划那次旅行了。",
        "ja": "ついにあの旅行の計画を立てるべきだと思っていたんだ。",
        "ms": "Saya terfikir kita patut rancang perjalanan itu akhirnya."}

if __name__ == "__main__":
    name, langs = sys.argv[1], sys.argv[2:] or ["en"]
    out = ROOT / "bench" / "out" / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    ref = str(ROOT / "work" / "clip" / "refs" / "SPEAKER_01.wav")
    jobs = [TTSJob(l, TEXT[l], l, str(out / f"{name}_{l}.wav"), ref_wav=ref, gender="female") for l in langs]
    t = time.time()
    res = run_engine(name, jobs, "cuda")
    print(f"{name}: {time.time() - t:.1f}s")
    ok = True
    for r in res:
        if not r.wav:
            print(" ", r.id, "FAIL", r.error); ok = False; continue
        x, sr = audio.read(r.wav, mono=True)
        db = audio.rms_db(x)
        print(f"  {r.id}: {len(x) / sr:.2f}s rms {db:.1f} dB")
        ok &= db > -45
    sys.exit(0 if ok else 1)

"""Warm engine worker: one process serves repeated run_engine calls, survives stdout chatter, restarts after a crash."""
import sys
from pathlib import Path

from dubber.tts import base
from dubber.tts.base import TTSJob, run_engine


def test_worker_is_reused_and_restarted(tmp_path, monkeypatch):
    monkeypatch.setitem(base.ENGINE_MODULES, "fake", "tests.fake_engine")
    monkeypatch.setattr(base, "engine_python", lambda name: Path(sys.executable) if name == "fake" else None)
    base.stop_workers()
    jobs = [TTSJob(str(i), "hello", "en", str(tmp_path / f"{i}.wav")) for i in range(3)]
    r1 = run_engine("fake", jobs[:2], "cpu")
    pid = base._workers["fake"].pid
    r2 = run_engine("fake", jobs[2:], "cpu")
    assert [r.wav for r in r1 + r2] == [j.out_wav for j in jobs] and all(Path(j.out_wav).exists() for j in jobs)
    assert base._workers["fake"].pid == pid, "second call must reuse the warm worker"
    base._workers["fake"].kill()
    base._workers["fake"].wait()
    r3 = run_engine("fake", jobs[:1], "cpu")
    assert r3[0].wav and base._workers["fake"].pid != pid, "a dead worker is replaced"
    base.stop_workers()
    assert not base._workers

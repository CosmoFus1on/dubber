"""M3 review: each test reproduces one finding in review/M3-review.md and FAILS until it is fixed.
Fast: no models, no network (LLM, TTS and stage modules are faked)."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from conftest import ROOT
from dubber import audio, config, fit, llm, mix, synth, translate
from dubber.llm import LLMError, LLMProvider
from dubber.schema import F_VERIFY_FAILED, STAGES, Project, Segment, TTSInfo, Verify
from dubber.tts.base import TTSResult

SR = 48000


def tone(seconds, amp=0.3):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def project(*segs, dur=20.0):
    p = Project(source={"video": "x.mp4", "duration": dur, "sample_rate": SR, "language": "en"}, target_language="hi")
    p.segments.extend(segs)
    return p


def opts(work, **kw):
    return config.Options(video=Path("x.mp4"), target="hi", out=work / "o.mp4", work=work, device="cpu", **kw)


# ---------------------------------------------------------------- llm.py
def _provider(tmp_path, posts, behaviour):
    p = llm.OpenAICompatible(cache_dir=tmp_path / "cache", backoff=[1, 2])

    def post(model, messages, json_mode, temperature, max_tokens):
        posts.append(model)
        return behaviour(model)
    p._post = post
    return p


def test_llm_non_json_answer_is_not_cached(tmp_path, monkeypatch):
    """F1: a non-JSON answer is cached, so chat_json fails identically on every rerun forever."""
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    posts, answer = [], ["Sure! Here is the translation you asked for."]
    p = _provider(tmp_path, posts, lambda m: (answer[0], ""))
    msgs = [{"role": "user", "content": "x"}]
    with pytest.raises(LLMError):
        p.chat_json(msgs, "m")
    answer[0] = '{"a": 1}'
    assert p.chat_json(msgs, "m") == {"a": 1}


def test_llm_non_retryable_error_not_retried_with_backoff(tmp_path, monkeypatch):
    """F6: HTTP 400/401 is retried 1+len(backoff) rounds over the whole chain (31 s of sleeps by default)."""
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    posts = []

    def bad(m):
        raise LLMError(f"{m}: HTTP 400", retryable=False, status=400)
    p = _provider(tmp_path, posts, bad)
    with pytest.raises(LLMError):
        p.chat([{"role": "user", "content": "x"}], "m", fallback=["f1", "f2"])
    assert posts == ["m", "f1", "f2"]  # each model once; it falls through, but never backs off on a 400


# ---------------------------------------------------------------- translate.py
class FakeLLM(LLMProvider):
    """'dubbing translator' -> translate (or the retry text), 'tighten' -> SHORT, judge fails lines containing SHORT."""

    def __init__(self, translate_raises=False, shape=None):
        self.translate_raises, self.shape, self.calls = translate_raises, shape, 0

    def chat(self, messages, model, json_mode=False, temperature=0.3, max_tokens=None, fallback=()):
        self.calls += 1
        sys_msg, user = messages[0]["content"], messages[-1]["content"]
        if "dubbing translator" in sys_msg:
            if self.translate_raises:
                raise LLMError("all models failed")
            lines = json.loads(user.split("Fit max_chars.\n", 1)[1].split("\n\nA previous attempt")[0])
            out = [{"id": l["id"], "translation": "RETRANSLATED " + l["text"]} for l in lines]
            if self.shape == "object":
                return json.dumps({"translations": out[0]})
            return json.dumps({"translations": out})
        if "literal translator" in sys_msg:
            return json.dumps({"back": user})
        if "review dubbing" in sys_msg:
            dubbed = user.split("Dubbed line", 1)[1].split("\n", 1)[0]
            return json.dumps({"score": 0.3 if "SHORT" in dubbed else 0.95, "fluency": 0.9, "issue": "lost meaning"})
        if "tighten" in sys_msg:
            return json.dumps({"translation": "SHORT"})
        raise AssertionError("unexpected prompt")


def test_translate_all_survives_llm_outage(tmp_path, monkeypatch):
    """F2: translate_batch is unguarded; one LLMError (or one non-JSON answer) kills the whole stage."""
    s = Segment(id=0, speaker="A", gender="male", start=0, end=2, text="hello there")
    p = project(s)
    tr = translate.Translator(p, opts(tmp_path), FakeLLM(translate_raises=True))
    tr.translate_all([s], lambda: None)
    assert F_VERIFY_FAILED in s.flags


def test_translate_accepts_single_object_reply(tmp_path):
    """F9: a one-line request answered as {"translations": {"id":0,...}} is dropped and the line flagged."""
    s = Segment(id=0, speaker="A", gender="male", start=0, end=2, text="hello there", budget_chars=30)
    tr = translate.Translator(project(s), opts(tmp_path), FakeLLM(shape="object"))
    assert tr.translate_batch([s]) == {0: "RETRANSLATED hello there"}


# ---------------------------------------------------------------- fit.py shorten loop
def _shorten_env(tmp_path, monkeypatch, seg):
    work = tmp_path / "w"
    (work / "tts").mkdir(parents=True)
    audio.write(work / seg.tts.wav, tone(3.0), SR)  # 3 s raw for a 1 s slot
    p = project(seg, Segment(id=1, speaker="A", gender="male", start=2.0, end=3.0, text="next"))
    fake = FakeLLM()
    monkeypatch.setattr(fit, "default_provider", lambda: fake)
    monkeypatch.setattr(translate, "_embed_sim", lambda a, b: 0.9)
    spoken = {}

    def fake_synth(segs, project, work, opts, hints=None):
        for s in segs:
            spoken[s.id] = s.translation
            audio.write(work / s.tts.wav, tone(0.5), SR)
    monkeypatch.setattr(fit.synth, "synthesize", fake_synth)
    return p, work, spoken


def test_shorten_then_verify_retranslation_desyncs_audio(tmp_path, monkeypatch):
    """F3: verify() after a shorten may re-translate; the new text is never synthesized, so the dub
    speaks 'SHORT' while segments.json / the .srt say 'RETRANSLATED ...'."""
    seg = Segment(id=0, speaker="A", gender="male", start=0.0, end=1.0, text="a long line", translation="LONG " * 10,
                  verify=Verify(passed=True, attempts=1), tts=TTSInfo(engine="fake", wav="tts/00000.wav"))
    p, work, spoken = _shorten_env(tmp_path, monkeypatch, seg)
    fit.shorten_rounds(p, [seg], work, opts(work))
    # whatever the dub speaks must be the stored translation: either the shortened line was synthesized,
    # or it was rejected by the judge and the previous text (whose audio already exists) was kept
    assert spoken.get(0, "LONG " * 10) == seg.translation


def test_shorten_never_touches_hand_edited_translation(tmp_path, monkeypatch):
    """F4: a hand-set translation (verify.attempts == -1) is shortened, re-verified and re-translated."""
    seg = Segment(id=0, speaker="A", gender="male", start=0.0, end=1.0, text="a long line",
                  translation="MANUAL hand edit " * 3, verify=Verify(attempts=-1),
                  tts=TTSInfo(engine="fake", wav="tts/00000.wav"))
    p, work, _ = _shorten_env(tmp_path, monkeypatch, seg)
    fit.shorten_rounds(p, [seg], work, opts(work))
    assert seg.translation == "MANUAL hand edit " * 3 and seg.verify.attempts == -1


def test_fit_zero_allowed_length(tmp_path):
    """F7: zero-duration segment whose neighbour starts at its end -> allowed 0 -> ZeroDivisionError."""
    work = tmp_path / "w"
    (work / "tts").mkdir(parents=True)
    a = Segment(id=0, speaker="A", gender="male", start=1.0, end=1.0, text="x", translation="x",
                tts=TTSInfo(engine="fake", wav="tts/00000.wav"))
    b = Segment(id=1, speaker="B", gender="male", start=1.0, end=2.0, text="y")
    audio.write(work / a.tts.wav, tone(0.5), SR)
    fit.fit_audio(project(a, b), a, work)


def test_last_clip_may_not_spill_past_video_end():
    """F10: the last segment gets SPILL_MAX_S past the video end; mix then hard-cuts it without a fade."""
    s = Segment(id=0, speaker="A", gender="male", start=9.0, end=10.0, text="x")
    assert fit.allowed_length(project(s, dur=10.0), s) <= 1.0 + 1e-9


# ---------------------------------------------------------------- mix.py
def test_mix_negative_start_does_not_crash(tmp_path):
    """F8: a (hand-edited or padded) start < 0 makes voice[a:a+len] an empty/short slice -> broadcast error."""
    work = tmp_path / "w"
    audio.write(work / "fit" / "00000.wav", tone(0.5), SR)
    s = Segment(id=0, speaker="A", gender="male", start=-0.01, end=0.5, text="x", translation="x",
                tts=TTSInfo(fitted_wav="fit/00000.wav"))
    v = mix.build_voice(project(s, dur=2.0), work, 2 * SR, SR)
    assert np.abs(v[:SR // 10]).max() > 0.1


# ---------------------------------------------------------------- synth.py
def test_failed_resynthesis_does_not_keep_stale_audio(tmp_path, monkeypatch):
    """F5: when every engine fails for a changed translation, the old tts/fit wavs stay referenced and
    mix places audio of the OLD translation; 'tts_failed' is also never cleared on a later success."""
    work = tmp_path / "w"
    for d in ("tts", "fit"):
        audio.write(work / d / "00000.wav", tone(1.0), SR)
    s = Segment(id=0, speaker="A", gender="male", start=0.0, end=2.0, text="x", translation="NEW text",
                tts=TTSInfo(engine="fake", wav="tts/00000.wav", fitted_wav="fit/00000.wav"),
                extra={"tts_key": "old", "fit_key": "old"})
    p = project(s)
    monkeypatch.setattr(synth, "chain_for", lambda lang, cpu=False: ["fake"])
    monkeypatch.setattr(synth, "run_engine", lambda name, jobs, dev: [TTSResult(j.id, None, name, "boom") for j in jobs])
    hints = {0: {"exaggeration": 0.5, "cfg_weight": 0.5}}
    synth.synthesize([s], p, work, opts(work), hints)
    assert "tts_failed" in s.flags
    assert s.tts.fitted_wav is None and s.tts.wav is None  # nothing stale left for fit/mix to pick up

    def ok(name, jobs, dev):
        for j in jobs:
            audio.write(j.out_wav, tone(1.0), SR)
        return [TTSResult(j.id, j.out_wav, name) for j in jobs]
    monkeypatch.setattr(synth, "run_engine", ok)
    synth.synthesize([s], p, work, opts(work), hints)
    assert "tts_failed" not in s.flags


# ---------------------------------------------------------------- schema.py (Windows editors)
def test_segments_json_with_bom_loads(tmp_path):
    """F11: Notepad / PowerShell 5.1 'Set-Content -Encoding utf8' save with a BOM -> json.loads fails."""
    path = tmp_path / "segments.json"
    project(Segment(id=0, speaker="A", gender="male", start=0, end=1, text="x")).save(path)
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    assert Project.load(path).segments[0].text == "x"


# ---------------------------------------------------------------- dubber.py runner
def _runner(tmp_path, monkeypatch, state_done=True):
    spec = importlib.util.spec_from_file_location("dubber_cli_review", ROOT / "dubber.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    calls = []
    for st in STAGES:
        m = types.ModuleType(f"fake_{st}")
        m.run = lambda project, work, opts, st=st: calls.append(st)
        monkeypatch.setitem(sys.modules, f"fake_{st}", m)
    monkeypatch.setattr(cli, "MODULES", {st: f"fake_{st}" for st in STAGES})
    monkeypatch.setattr(cli, "input_hash", lambda *a: "h")
    monkeypatch.setattr(cli.gpu, "device", lambda cpu=False: "cpu")
    monkeypatch.setattr(cli.gpu, "unload", lambda *a: None)
    monkeypatch.setattr(cli.report, "write", lambda *a: None)
    work = tmp_path / "w"
    for f in ("stems/vocals.wav", "stems/no_vocals.wav", "transcript.json", "diarization.json"):
        (work / f).parent.mkdir(parents=True, exist_ok=True)
        (work / f).write_text("{}", encoding="utf-8")
    if state_done:
        (work / "state.json").write_text(json.dumps({st: {"input_hash": "h", "done": True} for st in STAGES}),
                                         encoding="utf-8")
    base = [str(tmp_path / "in.mp4"), "--to", "hi", "--work", str(work), "--out", str(tmp_path / "o.mp4")]
    return cli, calls, work, base


def test_from_before_only_still_forces(tmp_path, monkeypatch):
    """F12: `--from transcribe --only speakers` should force speakers; `forced` is computed after the --only skip."""
    cli, calls, work, base = _runner(tmp_path, monkeypatch)
    project(Segment(id=0, speaker="A", gender="male", start=0, end=1, text="x")).save(work / "segments.json")
    cli.main(base + ["--from", "transcribe", "--only", "speakers"])
    assert calls == ["speakers"]


def test_from_translate_forces_retranslation(tmp_path, monkeypatch):
    """F13: `--from translate` is a no-op for incremental stages; translate.run never learns it was forced."""
    cli, calls, work, base = _runner(tmp_path, monkeypatch)
    project(Segment(id=0, speaker="A", gender="male", start=0, end=1, text="x")).save(work / "segments.json")
    seen = {}
    sys.modules["fake_translate"].run = lambda project, work, opts: seen.update(force=getattr(opts, "force", None))
    cli.main(base + ["--from", "translate", "--only", "translate"])
    assert seen.get("force"), "stage has no way to know it must ignore its per-segment cache"


def test_missing_segments_json_reruns_project_writers(tmp_path, monkeypatch):
    """F14: deleting segments.json (a natural 'reset') keeps separate/transcribe/speakers skipped, so the
    fresh Project has duration=None (mix crashes), language=None and no speakers (no ref clips, gender unknown)."""
    cli, calls, work, base = _runner(tmp_path, monkeypatch)
    cli.main(base)
    assert {"separate", "transcribe", "speakers"} <= set(calls)

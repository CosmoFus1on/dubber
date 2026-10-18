"""Translate/verify loop with a fake LLM: budgets, windowing, retry-on-fail, hand-edit and key-based skips."""
import json
from pathlib import Path

from dubber import config, translate
from dubber.llm import LLMProvider
from dubber.schema import F_VERIFY_FAILED, Project, Segment, SPEECH


class FakeLLM(LLMProvider):
    """Translates by prefixing 'HI:'; judge fails any translation containing 'bad' once, then the retry passes."""

    def __init__(self):
        self.calls = 0
        self.log = []

    def chat(self, messages, model, json_mode=False, temperature=0.3, max_tokens=None, fallback=()):
        self.calls += 1
        sys_msg, user = messages[0]["content"], messages[-1]["content"]
        self.log.append(model)
        if "dubbing translator" in sys_msg:
            lines = json.loads(user.split("Fit max_chars.\n", 1)[1].split("\n\nA previous attempt")[0])
            retry = "A previous attempt" in user
            out = [{"id": l["id"], "translation": ("HI:" + l["text"]).replace("bad", "good" if retry else "bad")} for l in lines]
            return json.dumps({"translations": out})
        if "literal translator" in sys_msg:
            return json.dumps({"back": user.replace("HI:", "")})
        if "review dubbing" in sys_msg:
            return json.dumps({"score": 0.3 if "bad" in user.split("Back-translation")[1] else 0.95, "fluency": 0.9, "issue": "meaning drift"})
        if "tighten" in sys_msg:
            return json.dumps({"translation": "HI:short"})
        raise AssertionError("unexpected prompt")


def make_project(n=25):
    p = Project(source={"video": "x.mp4", "duration": 100.0, "sample_rate": 48000, "language": "en"}, target_language="hi")
    for i in range(n):
        p.segments.append(Segment(id=i, speaker="SPEAKER_00", gender="male", start=i * 2.0, end=i * 2.0 + 1.5,
                                  text=f"line {i} is bad" if i == 3 else f"line {i}"))
    return p


def test_translate_and_verify(tmp_path, monkeypatch):
    monkeypatch.setattr(translate, "_embed_sim", lambda a, b: 0.9)
    p = make_project()
    opts = config.Options(video=Path("x.mp4"), target="hi", out=tmp_path / "o.mp4", work=tmp_path)
    llm = FakeLLM()
    tr = translate.Translator(p, opts, llm)
    for s in p.segments:
        s.budget_chars = translate.budget_for(s, "hi")
    assert p.segments[0].budget_chars == int(1.5 * config.cps("hi") * config.BUDGET_FACTOR)
    todo = [s for s in p.segments if tr.needs(s)]
    assert len(todo) == 25
    tr.translate_all(todo, lambda: None)
    assert llm.calls == 2  # two windows of 20 + 5
    assert all(s.translation.startswith("HI:") for s in p.segments)
    assert all(m == config.LLM_MODEL for m in llm.log)
    for s in p.segments:
        tr.verify(s)
    bad = p.segments[3]
    assert bad.verify.attempts == 2 and bad.verify.passed and "good" in bad.translation
    assert F_VERIFY_FAILED not in bad.flags
    assert all(s.verify.passed for s in p.segments)
    # nothing to redo on a rerun
    assert not any(tr.needs(s) for s in p.segments)
    # editing text invalidates only that segment
    p.segments[7].text = "edited line"
    assert [s.id for s in p.segments if tr.needs(s)] == [7]
    # hand-edited translation is never touched
    p.segments[8].translation, p.segments[8].verify.attempts = "manual", -1
    p.segments[8].text = "changed too"
    assert not tr.needs(p.segments[8])


def test_verify_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setattr(translate, "_embed_sim", lambda a, b: 0.9)

    class AlwaysBad(FakeLLM):
        def chat(self, messages, model, **kw):
            r = super().chat(messages, model, **kw)
            return r.replace("good", "bad") if "translations" in r else r

    p = make_project(5)
    opts = config.Options(video=Path("x.mp4"), target="hi", out=Path("o.mp4"), work=Path("."))
    tr = translate.Translator(p, opts, AlwaysBad())
    for s in p.segments:
        s.budget_chars = 20
    tr.translate_all(p.segments, lambda: None)
    assert tr.verify(p.segments[3]) is False
    assert p.segments[3].verify.attempts == config.VERIFY_MAX_ATTEMPTS
    assert F_VERIFY_FAILED in p.segments[3].flags
    assert p.segments[3].verify.sim == 0.9


def test_low_similarity_fails(monkeypatch):
    monkeypatch.setattr(translate, "_embed_sim", lambda a, b: 0.2)
    p = make_project(2)
    opts = config.Options(video=Path("x.mp4"), target="hi", out=Path("o.mp4"), work=Path("."), demo_fast=True)
    tr = translate.Translator(p, opts, FakeLLM())
    for s in p.segments:
        s.budget_chars = 20
    tr.translate_all(p.segments, lambda: None)
    assert tr.verify(p.segments[0]) is False and p.segments[0].verify.attempts == 1

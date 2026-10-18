"""Stage 3: duration-budgeted translation + verification loop.

Per segment: budget_chars from the slot length and the target language's chars/s; translation in
windows of ~20 segments with the transcript as context; verification = back-translation + LLM judge +
embedding similarity; failures are re-translated alone with the judge's issue, up to 3 attempts.
Everything is incremental: a segment is only re-translated when its translate_key changes.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

from . import config, prompts
from .llm import LLMError, LLMProvider, default_provider
from .schema import F_VERIFY_FAILED, SPEECH, Project, Segment, Verify, sha1

log = logging.getLogger("dubber.translate")

_embedder = None


def load_glossary(path: Optional[Path]) -> dict[str, str]:
    """One term per line: 'Term' (keep as-is) or 'Term=Translation'."""
    if not path or not Path(path).exists():
        return {}
    out = {}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def hand_edited(seg: Segment) -> bool:
    """A translation typed by a human (verify.attempts == -1) is final: never re-translated, shortened or judged."""
    return seg.verify.attempts == -1 and bool(seg.translation)


def budget_for(seg: Segment, tgt: str) -> int:
    return max(4, math.floor(seg.duration * config.cps(tgt) * config.BUDGET_FACTOR))


def context_lines(project: Project, around: Optional[list[int]] = None) -> list[str]:
    segs = project.sorted_segments()
    if len(segs) > config.CONTEXT_FULL_MAX and around:
        lo, hi = min(around) - config.CONTEXT_HALF, max(around) + config.CONTEXT_HALF
        segs = [s for s in segs if lo <= s.id <= hi]
    return [f"{s.id} | {s.speaker} | {s.gender} | {'[music] ' if s.type != SPEECH else ''}{s.text}" for s in segs]


def _line(seg: Segment) -> dict:
    return {"id": seg.id, "speaker": seg.speaker, "gender": seg.gender, "max_chars": seg.budget_chars, "text": seg.text}


def shorten(llm: LLMProvider, seg: Segment, src: str, tgt: str, target_chars: int) -> Optional[str]:
    """Ask for a shorter rewrite of seg.translation. Returns None when unavailable or unchanged."""
    try:
        d = llm.chat_json(prompts.shorten_messages(src, tgt, seg.text, seg.translation, target_chars),
                          config.LLM_MODEL, fallback=config.LLM_FALLBACK_CHAIN)
    except LLMError as e:
        log.warning("segment %d: shorten failed (%s)", seg.id, e)
        return None
    t = str(d.get("translation", "")).strip() if isinstance(d, dict) else ""
    return t if t and t != seg.translation else None


def _embed_sim(a: str, b: str) -> float:
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(config.EMBEDDER, device="cpu")  # small; keep VRAM for the big models
    v = _embedder.encode([a, b], normalize_embeddings=True)
    return float(v[0] @ v[1])


class Translator:
    def __init__(self, project: Project, opts: config.Options, llm: Optional[LLMProvider] = None):
        self.p = project
        self.opts = opts
        self.src = config.norm_lang(project.source.get("language") or "en")
        self.tgt = config.norm_lang(opts.target)
        self.llm = llm or default_provider()
        self.glossary = load_glossary(opts.glossary)
        self.model = config.LLM_MODEL
        self.max_attempts = 1 if opts.demo_fast else config.VERIFY_MAX_ATTEMPTS

    # ---- keys -------------------------------------------------------------------------------
    def key(self, seg: Segment) -> str:
        # ponytail: neighbours' text is deliberately NOT in the key, so editing one segment re-translates only it.
        # The prompt text is: a prompt change re-translates everything, which is what you want.
        ctx = sha1(self.glossary, seg.speaker, seg.gender, prompts.TRANSLATE_SYSTEM)
        return seg.translate_key(ctx, self.tgt, self.model)

    hand_edited = staticmethod(lambda seg: hand_edited(seg))

    def needs(self, seg: Segment) -> bool:
        if seg.type != SPEECH or not seg.text.strip():
            return False
        if self.hand_edited(seg):
            return False
        if self.opts.force:
            return True
        return seg.translation is None or seg.extra.get("translate_key") != self.key(seg)

    # ---- translation ------------------------------------------------------------------------
    def translate_batch(self, segs: list[Segment], retry: Optional[dict] = None) -> dict[int, str]:
        ids = [s.id for s in segs]
        msgs = prompts.translate_messages(self.src, self.tgt, self.glossary, context_lines(self.p, ids),
                                          [_line(s) for s in segs], retry)
        try:
            data = self.llm.chat_json(msgs, self.model, fallback=config.LLM_FALLBACK_CHAIN)
        except LLMError as e:
            log.error("translation request for ids %s failed: %s", ids, e)
            return {}
        items = data.get("translations", data) if isinstance(data, dict) else data
        if isinstance(items, dict) and "id" in items:  # single-line answers come back as a bare object
            items = [items]
        out = {}
        for it in items if isinstance(items, list) else []:
            try:
                i, t = int(it["id"]), str(it["translation"]).strip()
            except (KeyError, TypeError, ValueError):
                continue
            if i in ids and t:
                out[i] = t
        return out

    def translate_all(self, todo: list[Segment], save) -> None:
        for i in range(0, len(todo), config.TRANSLATE_WINDOW):
            window = todo[i:i + config.TRANSLATE_WINDOW]
            got = self.translate_batch(window)
            missing = [s for s in window if s.id not in got]
            for s in missing:  # one more try alone, then leave untranslated + flagged
                got.update(self.translate_batch([s]))
            for s in window:
                if s.id in got:
                    s.translation = got[s.id]
                    # models often overshoot max_chars; a text-only shorten here is far cheaper than re-synthesis
                    if s.budget_chars and len(s.translation) > s.budget_chars * config.PRESHORTEN_OVER:
                        new = shorten(self.llm, s, self.src, self.tgt, s.budget_chars)
                        if new:
                            log.info("segment %d: pre-shortened %d -> %d chars (budget %d)", s.id, len(s.translation), len(new), s.budget_chars)
                            s.translation = new
                    s.extra["translate_key"] = self.key(s)
                    s.verify = Verify()
                    s.unflag_prefix(F_VERIFY_FAILED)
                else:
                    log.error("segment %d: no translation returned", s.id)
                    s.flag(F_VERIFY_FAILED)
            log.info("translated %d/%d", min(i + len(window), len(todo)), len(todo))
            save()

    # ---- verification -----------------------------------------------------------------------
    def check(self, seg: Segment) -> tuple[float, float, float, str, str]:
        """Return (judge, fluency, sim, issue, back) for the segment's current translation."""
        back = self.llm.chat_json(prompts.back_messages(self.src, self.tgt, seg.translation), config.LLM_FAST,
                                  fallback=config.LLM_FAST_FALLBACK)
        back = str(back.get("back", "")) if isinstance(back, dict) else str(back)
        # the judge reads the target text itself, so it is run on the strong pinned model, not auto:fast
        j = self.llm.chat_json(prompts.judge_messages(self.src, self.tgt, seg.text, seg.translation, back, seg.gender),
                               config.LLM_MODEL, fallback=config.LLM_FALLBACK_CHAIN)
        def num(k):
            try:
                return float(j.get(k, 0.0))
            except (TypeError, ValueError, AttributeError):
                return 0.0
        issue = str(j.get("issue", "")) if isinstance(j, dict) else ""
        return num("score"), num("fluency"), _embed_sim(seg.text, back), issue, back

    def verify(self, seg: Segment, retranslate: bool = True) -> bool:
        """Verify, re-translating with the judge's issue up to max_attempts. Returns passed.
        retranslate=False judges the current text once and never replaces it (used after a shorten)."""
        if self.hand_edited(seg):
            return True
        while True:
            try:
                score, fluency, sim, issue, back = self.check(seg)
            except LLMError as e:
                log.warning("segment %d: verify call failed (%s); keeping translation unverified", seg.id, e)
                seg.verify.issue = f"verify unavailable: {e}"[:200]
                seg.flag(F_VERIFY_FAILED)
                return False
            seg.verify.attempts += 1  # only completed checks count
            seg.verify.back_translation, seg.verify.judge, seg.verify.sim, seg.verify.issue = back, score, round(sim, 3), issue
            seg.verify.fluency = fluency
            passed = score >= config.JUDGE_MIN and fluency >= config.FLUENCY_MIN and sim >= config.SIM_MIN
            seg.verify.passed = passed
            if passed:
                seg.unflag_prefix(F_VERIFY_FAILED)
                return True
            if not retranslate or seg.verify.attempts >= self.max_attempts:
                seg.flag(F_VERIFY_FAILED)
                log.warning("segment %d failed verification after %d attempts: %s", seg.id, seg.verify.attempts, issue)
                return False
            why = issue or f"meaning {score:.2f}, fluency {fluency:.2f}, similarity {sim:.2f}"
            if fluency < config.FLUENCY_MIN:
                why += " The line must be complete, natural spoken language, not shorthand."
            retry = {"id": seg.id, "prev": seg.translation, "issue": why}
            try:
                got = self.translate_batch([seg], retry)
            except LLMError as e:
                log.warning("segment %d: retranslate failed (%s)", seg.id, e)
                got = {}
            if got.get(seg.id):
                seg.translation = got[seg.id]
                seg.extra["translate_key"] = self.key(seg)
                seg.tts.wav = seg.tts.fitted_wav = None  # the old audio no longer matches the text


def run(project: Project, work: Path, opts: config.Options) -> None:
    tr = Translator(project, opts)
    path = work / "segments.json"
    save = lambda: project.save(path)  # noqa: E731
    for s in project.segments:
        if s.type == SPEECH:
            s.budget_chars = budget_for(s, tr.tgt)
        if s.type != SPEECH or not s.text.strip():
            if s.translation and not tr.hand_edited(s):  # text cleared or retyped as music: drop the dub
                s.translation, s.tts = None, type(s.tts)()
                s.verify = Verify()
    todo = [s for s in project.sorted_segments() if tr.needs(s)]
    log.info("translate %s -> %s with %s: %d segments to translate", tr.src, tr.tgt, tr.model, len(todo))
    tr.translate_all(todo, save)
    pending = [s for s in project.sorted_segments()
               if s.type == SPEECH and s.translation and s.verify.passed is None and s.verify.attempts != -1]
    for n, s in enumerate(pending, 1):
        tr.verify(s)
        if n % 10 == 0:
            save()
    save()
    passed = sum(1 for s in project.speech() if s.verify.passed or s.verify.attempts == -1)
    log.info("verification: %d/%d passed, %d flagged, %d LLM calls", passed, len(project.speech()),
             sum(1 for s in project.segments if F_VERIFY_FAILED in s.flags), getattr(tr.llm, "calls", -1))

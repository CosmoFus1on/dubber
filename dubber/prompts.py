"""Prompt templates for translate / shorten / back-translate / judge. Text only, never audio."""
from __future__ import annotations

import json

LANG_NAMES = {
    "en": "English", "es": "Spanish", "de": "German", "fr": "French", "it": "Italian", "pt": "Portuguese",
    "nl": "Dutch", "pl": "Polish", "ru": "Russian", "tr": "Turkish", "sv": "Swedish", "da": "Danish",
    "no": "Norwegian", "fi": "Finnish", "el": "Greek", "he": "Hebrew", "ar": "Arabic", "hi": "Hindi",
    "ms": "Malay", "sw": "Swahili", "ja": "Japanese", "zh": "Mandarin Chinese (Simplified)", "ko": "Korean",
    "id": "Indonesian", "vi": "Vietnamese", "th": "Thai", "ta": "Tamil", "bn": "Bengali", "ur": "Urdu",
}


def lang_name(code: str) -> str:
    return LANG_NAMES.get(code, code)


TRANSLATE_SYSTEM = """You are a professional dubbing translator working from {src} into {tgt}.
Your lines will be spoken by a voice actor over the original video, so each line must fit its time slot.

Rules:
1. Translate meaning, tone and register, never word by word. Idioms, slang, jokes and sarcasm must sound natural in {tgt}.
2. Keep names, brand terms and every glossary term exactly as given (do not transliterate glossary terms unless the glossary says so).
3. Respect the speaker's gender for grammatical agreement; respect who is being addressed when the context makes it clear.
4. Each line has max_chars. Stay within it: prefer shorter synonyms, drop fillers ("well", "you know"), never drop meaning. Count characters including spaces.
5. Use the full transcript only as context for consistency (names, pronouns, formality, running jokes). Translate only the requested ids.
6. Never add explanations, romanisation or brackets. Use the natural script of {tgt}.
7. Output JSON only: {{"translations": [{{"id": <int>, "translation": "<text>"}}, ...]}} with exactly one entry per requested id."""

TRANSLATE_USER = """Glossary (keep as-is):
{glossary}

Full transcript for context (id | speaker | gender | text):
{context}

Translate these lines. Fit max_chars.
{lines}"""

RETRY_SUFFIX = """

A previous attempt for id {id} was rejected by a reviewer.
Previous translation: {prev}
Reviewer issue: {issue}
Produce a corrected translation that fixes the issue while staying within max_chars."""

SHORTEN_SYSTEM = """You shorten dubbing lines in {tgt}. Keep the meaning, tone and any names. Return JSON only: {{"translation": "<text>"}}."""

SHORTEN_USER = """Original {src} line: {source}
Current {tgt} translation ({current_len} chars): {current}
Rewrite it in at most {target_chars} characters. Drop fillers and redundancy, keep every piece of meaning you can, keep the register."""

BACK_SYSTEM = """You are a literal translator. Translate the given {tgt} text into {src} faithfully and completely, preserving tone, without adding or omitting anything. Return JSON only: {{"back": "<text>"}}."""

JUDGE_SYSTEM = """You review dubbing translations. You are shown the original line and a back-translation of the dubbed line into the same language.
Score how well the dubbed line preserves meaning and tone, from 0 to 1:
1.0 = same meaning and tone; 0.85 = tiny nuance lost or slightly different wording; 0.7 = a minor detail lost but intent intact;
0.5 = noticeable meaning drift or missing/added content; 0.2 = wrong meaning or wrong addressee; 0 = unrelated.
Differences caused only by literal back-translation (word order, synonyms, article/particle choices) must not lower the score.
Return JSON only: {{"score": <float>, "issue": "<one short sentence describing the problem, or empty string>"}}."""

JUDGE_USER = """Original ({src}): {original}
Back-translation of the dub ({src}): {back}
Target language was {tgt}. Speaker gender: {gender}."""


def translate_messages(src: str, tgt: str, glossary: dict, context_lines: list[str], lines: list[dict],
                       retry: dict | None = None) -> list[dict]:
    gl = "\n".join(f"- {k}" + (f" => {v}" if v else "") for k, v in glossary.items()) or "(none)"
    user = TRANSLATE_USER.format(glossary=gl, context="\n".join(context_lines),
                                 lines=json.dumps(lines, ensure_ascii=False, indent=0))
    if retry:
        user += RETRY_SUFFIX.format(**retry)
    return [{"role": "system", "content": TRANSLATE_SYSTEM.format(src=lang_name(src), tgt=lang_name(tgt))},
            {"role": "user", "content": user}]


def shorten_messages(src: str, tgt: str, source: str, current: str, target_chars: int) -> list[dict]:
    return [{"role": "system", "content": SHORTEN_SYSTEM.format(tgt=lang_name(tgt))},
            {"role": "user", "content": SHORTEN_USER.format(src=lang_name(src), tgt=lang_name(tgt), source=source,
                                                            current=current, current_len=len(current),
                                                            target_chars=target_chars)}]


def back_messages(src: str, tgt: str, text: str) -> list[dict]:
    return [{"role": "system", "content": BACK_SYSTEM.format(src=lang_name(src), tgt=lang_name(tgt))},
            {"role": "user", "content": text}]


def judge_messages(src: str, tgt: str, original: str, back: str, gender: str) -> list[dict]:
    return [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_USER.format(src=lang_name(src), tgt=lang_name(tgt), original=original,
                                                          back=back, gender=gender)}]

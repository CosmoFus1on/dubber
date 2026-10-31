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
Your lines will be spoken by a voice actor over the original video, so each line must sound like real dialogue and fit its time slot.

Rules:
1. Translate meaning, intent, tone and register, never word by word. Idioms, slang, jokes and sarcasm must land naturally in {tgt}.
2. Keep every name, brand and glossary term exactly as given (do not transliterate glossary terms unless the glossary gives a translation).
3. Grammatical gender: "gender" is the speaker's own gender; use it for every word that agrees with the speaker (first-person verbs, adjectives, participles). Work out who is being addressed from the surrounding lines and agree with that person too.
4. Register: pick the formality (e.g. tu/vous, du/Sie, tum/aap) that fits the relationship shown in the transcript and keep it consistent for the same pair of speakers throughout.
5. Punctuation drives the voice actor's delivery: keep every question as a question ending in "?" and every exclamation ending in "!", in the target script's convention; never flatten them into statements.
6. Length: max_chars is a soft target so the dub fits the slot. Reach it by choosing compact natural phrasing and dropping fillers and hesitations ("well", "you know", "I mean", "honestly"). Keep every concrete fact: names, numbers, times, places, negations, questions. Going up to 20% over max_chars is better than losing meaning or sounding clipped.
7. Fluency first. Every line must be complete, natural spoken {tgt} that a native speaker would actually say: full verbs, connectives, particles and politeness markers as the language requires. Never write headline style, note-taking shorthand or bare noun lists.
8. Use the full transcript only as context (names, pronouns, gender, formality, running jokes). Translate only the requested ids, each one in full; never merge or split lines.
9. No explanations, notes, romanisation or brackets. Use the normal script of {tgt}.
10. Output JSON only: {{"translations": [{{"id": <int>, "translation": "<text>"}}, ...]}} with exactly one entry per requested id.

Example (English to Spanish, for illustration only):
Line: {{"id": 7, "gender": "female", "max_chars": 50, "text": "Well, honestly, I'm so tired, I think I'll just stay home tonight."}}
Good (52 chars, slightly over, complete and natural): "Estoy agotada, creo que esta noche me quedo en casa."
Rejected, telegraphic: "Agotada. Esta noche, casa."
Rejected, wrong gender for a female speaker: "Estoy agotado, creo que esta noche me quedo en casa.\""""

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
Produce a corrected translation that fixes the issue: complete, natural spoken dialogue, close to max_chars (a little over is fine, telegraphic is not)."""

SHORTEN_SYSTEM = """You tighten dubbing lines in {tgt} so they fit a time slot.
The result must still be one complete, natural spoken {tgt} line: full sentences with verbs and connectives, the same meaning, tone, register and grammatical gender, and every name, number, time, place, negation and question kept.
Shorten only by choosing more compact wording and dropping fillers, hesitations or repetition. Never drop grammar, never write fragments, headline style or bare noun lists. If the target length cannot be reached without that, return the shortest fluent version even if it is longer than asked.
Return JSON only: {{"translation": "<text>"}}.

Example (Spanish, for illustration only), asked for about 50 characters:
Current: "Bueno, la verdad es que estoy muy agotada y creo que esta noche me voy a quedar en casa."
Good: {{"translation": "Estoy agotada, creo que esta noche me quedo en casa."}}
Rejected (fragments): {{"translation": "Agotada; esta noche, casa."}}"""

SHORTEN_USER = """Original {src} line: {source}
Current {tgt} translation ({current_len} chars): {current}
Rewrite it to about {target_chars} characters (a little over is fine if needed for fluency)."""

BACK_SYSTEM = """You are a literal translator. Translate the given {tgt} text into {src} faithfully and completely, close to word for word, preserving tone.
Do not add, omit, explain or smooth anything: if the {tgt} text is fragmentary, ungrammatical or uses the wrong gender, let that show in the translation. Return JSON only: {{"back": "<text>"}}."""

JUDGE_SYSTEM = """You review dubbing translations as a bilingual native speaker of {tgt}. You are shown the original line, the dubbed line in {tgt}, a literal back-translation of the dub, and the speaker's gender.
Judge the dubbed {tgt} line itself; the back-translation is only a hint about its meaning.
Give two scores from 0 to 1.
"score" = meaning and tone preserved: 1.0 same meaning and tone; 0.85 tiny nuance lost; 0.7 a minor detail lost but intent intact; 0.6 a concrete fact lost (name, number, time, place, negation) or a question turned into a statement; 0.5 noticeable drift or added content; 0.2 wrong meaning or addressee; 0 unrelated.
Dropping fillers and hesitations ("well", "you know", "I mean") is expected for timing and is not meaning loss. Differences caused only by the literal back-translation (word order, odd synonyms, articles, particles, stiff phrasing) must not lower either score.
"fluency" = how natural the {tgt} line is as spoken dialogue: 1.0 a native speaker would say exactly this; 0.8 natural with a slightly odd word; 0.6 understandable but stiff or literal; 0.4 telegraphic, missing verbs or connectives, headline style, or wrong register; 0.3 wrong grammatical gender for the speaker or addressee; 0.2 ungrammatical.
Return JSON only: {{"score": <float>, "fluency": <float>, "issue": "<one short sentence naming the main problem, or empty string>"}}.

Examples (English to Spanish, for illustration only). Original: "Well, honestly, I'm so tired, I think I'll just stay home tonight." Speaker: female.
Dub "Estoy agotada, creo que esta noche me quedo en casa." / back "I am exhausted, I believe this night I stay in house." -> {{"score": 1.0, "fluency": 1.0, "issue": ""}}
Dub "Agotada. Esta noche, casa." / back "Exhausted. Tonight, home." -> {{"score": 0.85, "fluency": 0.4, "issue": "telegraphic fragments without verbs"}}
Dub "Estoy agotada, me quedo en casa." / back "I am exhausted, I stay at home." -> {{"score": 0.6, "fluency": 1.0, "issue": "drops 'tonight'"}}"""

JUDGE_USER = """Original ({src}): {original}
Dubbed line ({tgt}): {translation}
Back-translation of the dub ({src}): {back}
Speaker gender: {gender}."""


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


def judge_messages(src: str, tgt: str, original: str, translation: str, back: str, gender: str) -> list[dict]:
    return [{"role": "system", "content": JUDGE_SYSTEM.format(tgt=lang_name(tgt))},
            {"role": "user", "content": JUDGE_USER.format(src=lang_name(src), tgt=lang_name(tgt), original=original,
                                                          translation=translation, back=back, gender=gender)}]

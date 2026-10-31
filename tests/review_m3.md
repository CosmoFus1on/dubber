# M3 review: llm / translate / prompts / synth / fit / mix / schema / config / runner / report

Reviewer: independent. Scope: correctness. Ranked by severity, worst first.
CONFIRMED = reproduced by a failing test in `tests/test_review_m3.py` (or a check against the real run in
`work/clip/`). PLAUSIBLE = found by reading, not reproduced.
`pytest tests -q -m "not slow"`: the 17 existing tests pass, and the 14 new review tests fail as intended.

---

## HIGH

### F1. A non-JSON LLM answer is cached and then fails the same way on every run. CONFIRMED
`dubber/llm.py:320`. `chat()` caches any non-empty content. `chat_json()` parses only after that, so a reply
like "Sure! Here is...", a reply cut off by max_tokens, or a refusal gets written to `~/.cache/dubber/llm`,
and `parse_json` raises the same `LLMError` on every rerun. It never goes to the fallback model, and the only
way out is to delete the cache by hand.
Scenario: the pinned model answers one translation window in prose. The translate stage crashes (see F2),
and every resume crashes the same way.
Fix: when `json_mode` is set, call `parse_json(content)` inside the per-model `try` before writing the cache.
On failure raise `LLMError(retryable=True)`, so the next model is tried and nothing is cached.
Test: `test_llm_non_json_answer_is_not_cached`.

### F2. One LLM failure kills the whole translate stage. CONFIRMED
`dubber/translate.py:117,120`. In `translate_all`, the calls to `translate_batch` have no `try`. Any
`LLMError` (the whole chain is down, or the parse failure from F1) propagates out of `run()`. `verify()` and
`shorten()` catch the same error and degrade to a flag; translation does not.
Fix: wrap both calls in `try/except LLMError: got = {}`, so a missing id gets flagged `verify_failed` as it
already would. Test: `test_translate_all_survives_llm_outage`.

### F3. The shorten loop can leave `tts.wav` speaking a different text from `translation`. CONFIRMED
`dubber/fit.py:238-245`. The order is: shorten, then `synth.synthesize(changed)`, then `translator.verify(s)`.
When the shortened line fails the judge, `verify()` re-translates it (`translate.py:183-189`). The new,
usually long, text replaces `s.translation`, but nothing synthesizes it. `fit_audio` then fits the old clip,
so the dub speaks the shortened line while `segments.json`, the `.srt` and the report show the
re-translation. The next run's tts stage sees a `tts_key` mismatch and re-synthesizes, so two consecutive runs
also produce different audio.
Fix: in `shorten_rounds`, verify **before** synthesizing, and only with `max_attempts=1`, so verify never
re-translates. If the shortened line fails, revert to the previous translation and let tempo/truncate handle
the length. Synthesize only the lines that survive.
Test: `test_shorten_then_verify_retranslation_desyncs_audio`.

### F4. A hand-edited translation (`verify.attempts == -1`) gets shortened, re-verified and re-translated. CONFIRMED
`dubber/fit.py:229-245`. `shorten_rounds` never checks `attempts == -1`. It shortens the human's text, sets
`passed=None`, and `verify()` increments attempts from -1 to 0, judges the line and re-translates it. The
architecture contract says such a translation counts as verified and must not be touched.
Fix: in `shorten_rounds`, skip any segment with `s.verify.attempts == -1` (it falls through to
stretch/truncate). Add the same guard at the top of `Translator.verify`.
Test: `test_shorten_never_touches_hand_edited_translation`.

### F5. When every TTS engine fails, the old audio stays referenced, and `tts_failed` is never cleared. CONFIRMED
`dubber/synth.py:96-98` (and `:84-86`). When synthesis of a changed translation fails on every engine,
`s.tts.wav`, `s.tts.fitted_wav`, `extra["tts_key"]` and `extra["fit_key"]` still point at the previous clip.
`fit.run` sees `tts.wav` exist with a `fit_key` mismatch, re-fits the **old** audio, and `mix` places it. The
user gets the old translation spoken under a `tts_failed` flag that claims "left silent". The same thing
happens inside `fit.shorten_rounds` when the re-synthesis fails: the new short text is paired with the old
long clip. Separately, a later success only clears `F_TTS_FALLBACK`, so `"tts_failed"` stays in the report
forever.
Fix: on total failure set `s.tts.wav = s.tts.fitted_wav = None` and pop `tts_key`/`fit_key`. On success add
`s.unflag_prefix("tts_failed")`. Also give it an `F_TTS_FAILED` constant in `schema.py`.
Test: `test_failed_resynthesis_does_not_keep_stale_audio`.

### F6. Deleting `segments.json` silently produces a dub with no voice clones and unknown genders, and mix crashes. CONFIRMED
`dubber.py:147-152,164`. With `state.json` intact, separate, transcribe and speakers count as up to date
because they hash only files. But they also write into the `Project` (`source.duration`, `source.language`,
`project.speakers`), so a fresh `Project` has `duration=None`, which crashes `mix.py:63`, has `language=None`
(the source language becomes "en"), and has no speakers (no ref clips, gender unknown, every line flagged
`no_ref_clip`).
Fix: when `segments.json` is missing, drop `separate`, `transcribe` and `speakers` from `state`. Better: make
`outputs_exist` for those stages also check the project fields they own (`source["duration"]`,
`source["language"]`, `project.speakers`).
Test: `test_missing_segments_json_reruns_project_writers`.

## MEDIUM

### F7. `--from translate|tts|fit` forces nothing. CONFIRMED
`dubber.py:157-170`. `forced` only bypasses the runner's skip, and incremental stages are never skipped
anyway. `translate.run` and the stages after it decide per segment from cache keys, so `--from translate`
does exactly what a plain run does. The prompt rewrite in this review, for example, will not re-translate an
existing work dir: `translate_key` (`schema.py:310`, `translate.py:88`) has no prompt version in it, and
neither do `tts_key` or `fit_key`.
Fix: set `opts.force = forced` (a new `Options.force: bool = False`) and have each incremental stage ignore
its per-segment key when it is set, still skipping hand edits. Also add `sha1(prompts.TRANSLATE_SYSTEM)` to
the translate context hash. Test: `test_from_translate_forces_retranslation`.

### F8. `--from A --only B` never forces B. CONFIRMED
`dubber.py:159-161`. `forced = forced or stage == a.from_stage` runs after the `--only` `continue`, so the
`--from` stage is never seen when it differs from `--only`. For example,
`--from transcribe --only speakers` reports "up to date".
Fix: swap the two lines (compute `forced` first). Test: `test_from_before_only_still_forces`.

### F9. Non-retryable HTTP errors are retried with the full backoff. CONFIRMED
`dubber/llm.py:311-327`. A non-retryable error does fall through to the next model, which is correct, but
`retryable` is never read. A 400 (context too long) or 401 (bad key) is retried for `1+len(backoff)` rounds
over the whole chain: 6 rounds x 6 models = 36 requests and 31 s of sleep per call, for an error that cannot
change. In a verify loop over 100 segments that is close to an hour.
Fix: keep a `dead` set. On `not e.retryable`, add the model and skip it in later rounds, and break out
when every model in the chain is dead.
The cache key correctly ignores `fallback`, but note that the cache file is keyed on the **pinned** model even
when a fallback answered (`llm.py:303,320`). An `auto` answer gets frozen as if the strong model gave it, and
nothing ever upgrades it. The model that answered is already stored as `"model": m`, so cache hits where it
differs from the pinned model could be treated as soft.
Test: `test_llm_non_retryable_error_not_retried_with_backoff`.

### F10. Division by zero in fit for a zero-length slot. CONFIRMED
`dubber/fit.py:201`. `tempo = raw / allowed`, where `allowed = duration + min(SPILL, gap)` is 0 when
`start == end` (hand edit, or a one-word segment with identical timestamps) and the next segment starts
exactly at `end`. Raises `ZeroDivisionError` and the whole fit stage dies.
Fix: `allowed = max(allowed, 0.05)` in `allowed_length`, or skip and flag segments with `allowed <= 0`.
Test: `test_fit_zero_allowed_length`.

### F11. A negative `start` crashes mix. CONFIRMED
`dubber/mix.py:318-320`. With `a = round(start*sr) < 0`, `voice[a:a+len(x)]` becomes a slice counted from the
end of the array (empty or short), and you get a broadcast `ValueError`. A negative start comes from hand
edits, or from a segmenter that pads words.
Fix: `if a < 0: x, a = x[-a:], 0` before the slice. Test: `test_mix_negative_start_does_not_crash`.

### F12. `segments.json` saved with a UTF-8 BOM fails to load. CONFIRMED
`dubber/schema.py:364`, the same for `translate.py:29` (glossary) and `config.py:407` (.env). Hand-editing
`segments.json` is the documented way to fix mistakes, and Windows Notepad's "UTF-8 with BOM" or PowerShell
5.1's `Set-Content -Encoding utf8` write a BOM. `json.loads` then raises "Unexpected UTF-8 BOM".
Fix: `encoding="utf-8-sig"` on those three reads (writes stay `utf-8`). CRLF is fine, because
`json`/`splitlines` handle it. Test: `test_segments_json_with_bom_loads`.

### F13. Prosody hints make `tts_key` unstable across segments. PLAUSIBLE
`dubber/synth.py:36-43,126`. `exaggeration` is the segment's loudness minus the **speaker's median** over all
segments, and the median is part of every segment's `tts_key`. Editing one segment's timing, re-segmenting,
or adding or removing one line shifts the median, which invalidates and re-synthesizes every segment of that
speaker. It defeats the per-segment incrementality.
Fix: quantize the hints (`round(x*10)/10`) before hashing, or hash only the translation, engine and ref, and
store the hints for information.

### F14. The last clip may spill past the video end and is then hard-cut without a fade. CONFIRMED
`dubber/fit.py:167-170`. When there is no next segment, the code allows `SPILL_MAX_S` with no regard for
`source.duration`. `mix.py:319` then truncates at `n` with no fade (a click) and no `truncated` flag.
Fix: `gap = min(gap, project.source["duration"] - seg.end)` when the duration is known.
Test: `test_last_clip_may_not_spill_past_video_end`.

### F15. `report.json` is missing required fields, and its timings mix runs. CONFIRMED (against `work/clip/report.json`)
`dubber/report.py:24-41` and `dubber.py:174`. The architecture requires "LLM calls made" and model ids. The
report has only `llm_model`: no LLM call count (translate and fit each build their own provider, so the
counts would need summing into some persisted per-stage store, the way `project.timings` works) and no TTS,
whisper or diarization model ids. `project.timings[stage]` is overwritten on every run, including no-op
incremental reruns (translate takes about 0.1 s on a resume), so after a resume `total_s` and
`seconds_per_video_minute` describe the resume, not the dub, while skipped stages keep their old times.
Fix: persist `llm_calls` per stage in `project.timings` or a sibling dict. Write stage timings only when work
was done, or report both `last_run_s` and `full_s`.

## LOW

* **F16 Single-object replies dropped** (CONFIRMED, `translate.py:103-105`). A one-id retry answered as
  `{"translations": {"id":0,...}}` or as a bare `{"id":0,"translation":...}` returns `{}`, and the line gets
  flagged. Fix: `if isinstance(items, dict) and "id" in items: items = [items]`.
  Test: `test_translate_accepts_single_object_reply`.
* **F17 Verify attempts accumulate** (CONFIRMED by reading, `fit.py:238`, `translate.py:159-166`). A shorten
  resets `passed` but not `attempts` (in the real run, segments 0, 2 and 4 show `attempts: 2` after one
  shorten), so a segment already at 3 attempts gets no retry. An LLM outage during verify increments
  `attempts` on every run and leaves `passed=None`. Fix: reset `attempts=0` whenever `translation` changes,
  and do not count failed calls.
* **F18 Clearing `text` does not un-dub** (CONFIRMED by reading, `translate.py:91`, `synth.py:135`,
  `mix.py:314`). Set `text: ""` on a speech segment and the old translation, wav and fitted clip are still
  placed. The same happens when a re-translation of an edited `text` returns nothing: the old translation,
  with `passed=True` from the old text, is still dubbed, though it is flagged. Fix: in translate, when
  `needs()` is false because `text` is empty, clear `translation` and `tts.*`.
* **F19 Hand edits survive a target-language change** (`translate.py:93`). `--to es` on a Hindi work dir
  keeps the Hindi hand edits. Fix: store the language with the hand edit (`extra["hand_lang"]`) or include
  `tgt` in the `-1` check.
* **F20 Re-segmentation wipes every hand edit** (PLAUSIBLE, `dubber.py:91`, `segment.py:92`). Any
  `transcript.json` mtime change (for example a re-transcribe with the same output) rebuilds
  `project.segments` from scratch. This one is in segment.py's territory; it is flagged here for the runner
  contract.
* **F21 Paths stored with backslashes** (`synth.py:82`, `fit.py:212`). `tts\00000.wav` does not resolve on
  the Linux demo Space, so moving a work dir re-synthesizes everything. Fix: `.as_posix()`. The
  `relative_to`/`is_relative_to` checks themselves are fine on Windows: `resolve()` returns the on-disk case
  (verified: `dubbing pipeline` becomes `Dubbing Pipeline`), and `PureWindowsPath` comparison is
  case-insensitive.
* **F22 `parse_json` edge cases** (PLAUSIBLE, `llm.py:212,224`). An unclosed `<think>` (the answer was cut
  off) or a `<thinking>` tag with braces inside it gets parsed as the answer. Fix:
  `re.compile(r"<(think|thinking)>.*?(</\1>|$)", re.S)`.
* **Mix length** is guaranteed by `_fit_length` plus the tolerance check (`mix.py:363-368`); I found no path
  that makes `mix.wav` differ. `mux` has no `-shortest`, so if the original audio stream is longer than the
  video stream, `source.duration` (the format duration) follows the audio, and so does the output. A long
  video holds about 5 full-length float32 stereo arrays at once, plus a librosa resample of the 192 kHz
  loudnorm output (PLAUSIBLE memory issue at 1 h or more).

## Real-run observation (`work/clip`)
Segment 4, "Great, I will book the flights tonight and we can talk more tomorrow", came out as
`ठीक, फ्लाइट बुक करूँगी, कल बात करेंगे।`. That is 38 chars against a budget of 50: "tonight" is dropped and the
style is borderline telegraphic. It passed at judge 0.70, exactly `JUDGE_MIN`, after fit's shorten. The
shorten target `len*allowed/raw*0.95` (`fit.py:233`) undershoots the budget. Fit took 141 s of the 323 s
total, with every segment going through 1 or 2 shorten rounds. The new judge rubric scores a lost time
reference at 0.6, so this line would now be retried.

## Prompt changes (`dubber/prompts.py`, owned by the reviewer)
Signatures, the JSON shapes and the test trigger phrases are unchanged. `tests/test_translate.py` passes.
* translate: explicit speaker-gender agreement (the gender column is the speaker's) and addressee agreement
  worked out from context; register kept consistent per speaker pair; concrete facts (names, numbers, times,
  places, negations, questions) must survive the budget; no merging or splitting of lines; one
  English->Spanish few-shot with a good line that is 4% over budget, a rejected telegraphic line and a
  rejected wrong-gender line.
* retry suffix: "close to max_chars, a little over is fine, telegraphic is not". The old "staying within
  max_chars" pushed models toward fragments.
* shorten: must keep facts, gender and register; return the shortest *fluent* version even if it is over
  the target; few-shot with good and rejected outputs.
* back-translation: literal, and must not smooth over fragments or gender errors, so the judge can see them.
  Previously a fluent back-translation could hide a telegraphic dub.
* judge: judges the target line itself; dropping fillers is not meaning loss; a lost concrete fact scores
  0.6, below `JUDGE_MIN`; wrong gender scores fluency 0.3; three calibrated few-shot verdicts, including a
  back-translation artefact that must score 1.0.
* Not verified against a live model: no LLM calls were made in this review. Existing work dirs keep their old
  translations until F7 is fixed.

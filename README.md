# dubber

Free, local AI video dubbing. Give it a video and a language, get the same video back with every
speaker talking in that language, in their own voice, over the original music. Songs are left alone.

```bash
python dubber.py input.mp4 --to hi --out dubbed.mp4
```

> **Demo video**: `python tests/make_demo_video.py` renders `output/demo_side_by_side_hi.mp4` (81 s):
> original on the left, Hindi dub on the right. The first minute is a two-person coffee-shop
> conversation recorded with both speakers' consent; the tail is the synthetic clip's two-speaker
> line and the 1919 song that passes through untouched. Generated files are not committed, so build it
> locally or watch the copy linked from the repository's releases page once it is uploaded there.

**About this project.** During an internship I worked on a production dubbing pipeline. This is a
from-scratch rebuild of the *idea* as a portfolio project: none of that company's code, data, prompts
or videos are used here. Every line was written for this repo, and every clip in it is either
synthetic, public domain, or recorded with the speakers' consent. It is non-commercial: some of the
models it can use carry non-commercial licences (listed below), and I have not tried to make money
from it.

It is also not "as good as the commercial products". It is a solid, honest pipeline that runs on a
gaming GPU for zero dollars, and the numbers below are the real ones.

---

## What it does, in one picture

```mermaid
flowchart LR
    V[video] --> A[ffmpeg<br/>extract audio]
    A --> D[Demucs<br/>vocals / no_vocals]
    D --> W[faster-whisper<br/>words + language]
    D --> P[pyannote<br/>who speaks when]
    D --> G[inaSpeechSegmenter + F0<br/>gender per speaker]
    P --> R[reference clip<br/>per speaker]
    W --> S[segments.json<br/>sentences, speakers, music]
    P --> S
    S --> T[LLM translate<br/>budgeted, in context]
    T --> X[verify: back-translate<br/>+ judge + embeddings]
    X --> C[TTS with voice cloning<br/>Chatterbox / XTTS / ...]
    R --> C
    C --> F[fit: shorten -> stretch<br/>-> trim, never overlap]
    F --> M[mix: voices over no_vocals<br/>songs = original, duck, -16 LUFS]
    D --> M
    M --> O[dubbed.mp4 + .srt]
```

Every stage writes to `.cache/work/<video>/`, and `segments.json` in there is the single source of truth.
If a crash happens, the next run picks up where it stopped. If the pipeline gets a name wrong or
mis-hears a line, open `segments.json`, fix the text or the translation, run again, and only that
segment is redone.

## Quick start (Windows)

You need Python 3.10, ffmpeg on `PATH` (a full build with `rubberband` and `loudnorm`), an NVIDIA GPU
with about 6 GB free (or `--cpu`, which is slow but works), and two free accounts: Hugging Face
(for the gated speaker-diarization model) and a FreeLLMAPI proxy (for translation).

```powershell
.\setup.ps1 -Engines chatterbox        # main venv + the primary TTS engine in its own venv
copy .env.example .env                 # then fill it in, see below
python tests\make_test_clip.py --single   # builds input\clip.mp4 and input\clip_single.mp4
python dubber.py input\clip_single.mp4 --to hi
```

Drop your own videos in `input/`; dubs land in `output/` as `<name>_<lang>.mp4` plus an `.srt`, and
every run writes `transcripts/<name>_<lang>.txt` with each line as
`Person 1 (female): [original]` followed by `Person 1 (female): [translation]`.

Prefer buttons? `python app.py` opens a small desktop window: pick the video, pick where to save,
pick the language, watch the eight stages light up, cancel if you must. It runs the same `dubber.py`.

`.env` needs three things, names only in `.env.example`:

| key | what |
|---|---|
| `DUBBER_LLM_BASE_URL` | your OpenAI-compatible `/v1`. I run [FreeLLMAPI](https://github.com/tashfeenahmed/freellmapi) locally; it spreads calls over free tiers of many providers. |
| `DUBBER_LLM_KEY` | the proxy's key |
| `HF_TOKEN` | a **fine-grained** Hugging Face token with only *"Read access to contents of all public gated repos you can access"*. First accept the conditions on [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1). |

Only text ever goes to the LLM, never audio or video. Free-tier providers may log prompts; don't dub
anything confidential through them.

Useful flags: `--keep-original-audio` (adds the original as a second track), `--glossary names.txt`
(one term per line, `Term` or `Term=Translation`), `--num-speakers 2`, `--source-lang en` (skip
detection), `--from translate` / `--only mix` (force or isolate a stage), `--cpu`.

## Why these models

I checked each model's licence on its own card on **2026-10-18** before picking it.

| job | model | licence | why |
|---|---|---|---|
| separation | Demucs `htdemucs` | MIT | still the best free vocals/accompaniment split; the `no_vocals` stem becomes the bed of the final mix |
| transcription | faster-whisper `large-v3` (GPU) / `distil-large-v3` (CPU) | MIT | word timestamps, 99 languages, int8/fp16 on CTranslate2 |
| who speaks when | pyannote.audio 4 + `speaker-diarization-community-1` | MIT code, CC-BY-4.0 weights (gated, free token) | measurably better speaker counting than the 3.1 pipeline; the `exclusive` output aligns cleanly with word timestamps |
| gender | inaSpeechSegmenter + median F0 (librosa pyin) | MIT | ina is French-trained, so it is cross-checked with pitch; disagreement lowers confidence and the pipeline then trusts the voice clone alone |
| meaning check | `paraphrase-multilingual-MiniLM-L12-v2` | Apache-2.0 | small, local, covers all target languages; cosine similarity between original and back-translation |
| translation | your LLM via FreeLLMAPI | n/a | one strong model pinned per run so style stays consistent (`nemotron-3-ultra-550b` on the free NVIDIA route at the time of writing); `auto:fast` for the cheap back-translation |
| **TTS, primary** | **Chatterbox Multilingual v3** | **MIT** | zero-shot cloning, 23 languages, best speaker similarity in my benchmark, the only free engine with Malay. Every output carries Resemble's PerTh watermark; I keep it. |
| TTS, fallback | XTTS-v2 via `coqui-tts` | **Coqui Public Model License (non-commercial)** | fast, good cloning for hi/ja/zh and 14 other languages |
| TTS, fallback | OpenVoice V2 + MeloTTS | MIT | light, fast, tone-colour transfer only; en es fr zh ja ko |
| TTS, bench only | F5-TTS | code MIT, weights **CC-BY-NC-4.0** | zh and en only; strong voice, but it mis-paces when the reference clip is in another language |
| TTS, last resort | Piper | **GPL-3.0** code, MIT voices | no cloning, runs on CPU in a second; this is what `--cpu` uses |

Engines that pin incompatible library versions live in their own virtualenv under `.envs/` and are
driven through a small subprocess runner, so one `pip` conflict never takes the pipeline down.
Which engine handles which language is a table in `dubber/config.py`.

## The two ideas that matter

**Duration budget.** A dubbed line has to fit the slot the original line occupied. Before translating
I compute a character budget from the slot length and a per-language speaking rate (measured from the
TTS itself: Hindi about 15 chars/s, Japanese 7.5, Mandarin 5.2). The translator gets the whole
transcript for context, the glossary, the speaker's gender, and that budget as a *target*. An early
version treated it as a hard limit and the Hindi came out telegraphic: nouns without verbs, meaning
technically there, flow gone. A fluent Hindi speaker caught it in review; the automatic judge had not.
The fix was to make fluency the first rule, let lines run up to ~20 % over, and absorb the rest with
up to 15 % tempo stretch in the fit stage. Only after that do we shorten by rewording, and a shortened
line has to pass the judge again or it is thrown away.

**Verification loop.** Each translation is back-translated to the source language by a cheap model,
then a stronger model judges three things: meaning preserved (0–1), fluency of the target text as
spoken dialogue (0–1, and telegraphic style scores 0.4), and a one-sentence issue. Embedding
similarity between the original and the back-translation is the third vote. A line passes at judge
≥ 0.7, fluency ≥ 0.7, similarity ≥ 0.75; otherwise it is re-translated with the judge's complaint
attached, up to three times, then flagged in `report.json`. Hand-edited translations are never
touched.

Timing fit, for the record: place each clip at its original start sample (so drift cannot
accumulate), allow it to spill at most 0.5 s into the silence after it but never into the next line,
stretch with rubberband up to 1.15× quietly and up to 1.25× with a flag, and if it still does not fit,
trim with a 40 ms fade and flag `truncated`. Shorter clips are simply padded; audio is never slowed
down. Music segments use the original audio with 50 ms crossfades and are never ducked.

## Numbers

Everything below was measured on this machine: RTX 3080 10 GB, Ryzen-class desktop, Windows 11,
models already downloaded. Nothing is estimated.

### TTS benchmark

Five dialogue lines translated into each language, cloned from the two test speakers, one engine at a
time (`benchmarks/tts_results.md` has the method and per-line data).

| engine | lang | load s | RTF | timing error mean / max | speaker sim | VRAM |
|---|---|---|---|---|---|---|
| chatterbox | hi | 24.2 | 0.81 | 1.86 / 5.30 s | 0.887 | 4.3 GB |
| chatterbox | ms | 21.3 | 0.95 | 0.66 / 2.18 s | 0.893 | 4.1 GB |
| chatterbox | ja | 21.8 | 1.15 | 0.46 / 0.79 s | 0.863 | 4.1 GB |
| chatterbox | zh | 25.0 | 0.68 | 2.63 / 8.47 s | 0.858 | 4.3 GB |
| xtts | hi | 26.9 | 0.21 | 2.35 / 2.75 s | 0.837 | 2.4 GB |
| xtts | ja | 23.4 | 0.37 | 0.50 / 1.22 s | 0.796 | 2.3 GB |
| xtts | zh | 23.5 | 0.35 | 0.48 / 0.71 s | 0.802 | 2.3 GB |
| openvoice | ja | 13.4 | 0.13 | 0.20 / 0.38 s | 0.723 | 0.9 GB |
| openvoice | zh | 17.5 | 0.15 | 0.73 / 1.13 s | 0.660 | 1.5 GB |
| f5 | zh | 18.2 | 0.99 | 1.24 / 1.68 s | 0.850 | 2.9 GB |
| piper | hi | 0.8 | 0.10 | 1.40 / 2.26 s | 0.637 | CPU |
| piper | ja | 1.0 | 0.09 | 0.25 / 0.62 s | 0.537 | CPU |
| piper | zh | 4.3 | 0.27 | 0.40 / 0.62 s | 0.548 | CPU |

RTF is synthesis seconds per second of audio (lower is faster). Timing error is how far the engine's
natural pace lands from the English slot *before* the fit stage. Speaker similarity is a resemblyzer
cosine against the reference clip; Piper does not clone, so its number is just "a generic voice of
the right gender". Chatterbox's big max errors are two babbles, where it ran several times longer than
the text; the pipeline detects those and re-synthesises once.

### A real conversation

The clip I actually care about is a 78 second coffee-shop conversation between two people who agreed
to be dubbed. English in, Hindi out, three times over as the pipeline changed:

| | first working run | after the speed work | final, with the accent fix |
|---|---|---|---|
| total time | 1705 s | about 590 s | 1392 s |
| seconds per minute of video | 1311 | about 454 | 1070 |
| lines that passed verification | 28 of 28 | 28 of 28 | 28 of 28 |
| mean judge / fluency / similarity | 0.98 / 1.00 / 0.95 | 0.98 / 1.00 / 0.91 | 0.98 / 1.00 / 0.91 |
| lines tempo-stretched / cut short | 1 / 1 | 1 / 1 | 10 / 2 |

The middle column is the honest best case: running LLM calls four at a time and keeping the TTS engine
warm between calls took almost three minutes off every minute of video. The last column is slower for
two reasons I measured rather than guessed. The free LLM route was simply slower that day (369 s against
260 s for the exact same calls), and the Hindi fine-tune, run with `cfg_weight=0` to stop the English
reference accent leaking through, speaks noticeably longer per character, so more lines had to be
reworded, stretched or cut. Better accent, worse timing. That trade-off is real and I'd rather show it
than hide it.

### The synthetic eval set

Two generated clips, one voice and two voices plus a song, into four languages, with the final code.
"Checked by" is honest: I read the Hindi myself; the other three only an LLM has judged.

| clip | lang | checked by | pass | judge | fluency | similarity | stretched | cut | s / min |
|---|---|---|---|---|---|---|---|---|---|
| two speakers + song, 44 s | hi | me | 5/5 | 0.91 | 1.00 | 0.87 | 1 | 0 | 614 |
| two speakers + song | ms | LLM only | 5/5 | 0.85 | 0.90 | 0.93 | 0 | 0 | 511 |
| two speakers + song | ja | LLM only | 5/5 | 1.00 | 1.00 | 0.87 | 0 | 0 | 235 |
| two speakers + song | zh | LLM only | 5/5 | 1.00 | 1.00 | 0.86 | 0 | 0 | 325 |
| one speaker, 24 s | hi | me | 3/3 | 0.95 | 1.00 | 0.96 | 2 | 1 | 1002 |
| one speaker | ms | LLM only | 3/3 | 0.90 | 0.93 | 0.93 | 0 | 0 | 835 |
| one speaker | ja | LLM only | 3/3 | 0.90 | 1.00 | 0.87 | 0 | 0 | 352 |
| one speaker | zh | LLM only | 3/3 | 1.00 | 1.00 | 0.86 | 0 | 0 | 441 |

Every run found both speakers with the right gender, left the song bit-identical (correlation 0.9997
after loudness normalisation), and placed every line within 150 ms of where the original was. The
seconds-per-minute figures on a 24 second clip are dominated by fixed costs, loading models and the
minimum three LLM round trips, and they swing with how busy the free LLM provider is that hour. On a
longer video they settle towards the TTS engine's real-time factor. The `--cpu` path (distil-whisper
plus Piper) does the 24 s clip in 396 s.

Where the time goes on a typical minute of video: separation and transcription under 15 s,
diarization and gender about 30 s, translation plus verification 2–6 minutes depending on the LLM
route, TTS 1–4 minutes, and the fit stage anywhere from nothing to several minutes depending on how
many lines ran long. Almost all of the waiting is on the LLM, not on this machine.

## Known limitations

- **Accent.** This is the big one, and it is still open. The reference clips are the speakers' own
  English, and a zero-shot cloner carries the reference's phonetics into the target language, so the
  first Hindi dubs sounded like an English speaker reading Hindi. Resemble's own advice for this case
  is `cfg_weight=0`, and their dedicated Hindi fine-tune helps further; both are in. A fluent listener
  still finds the result unconvincing. The path that removes the English audio from the equation
  entirely, synthesising from a native Hindi reference and then transferring only the speaker's timbre
  with OpenVoice, is prototyped and produces clearly better pronunciation at the cost of a weaker
  resemblance to the speaker. It is not the default yet; I want a human ear to pick the trade-off.
- **Short English lines into Hindi or Malay.** Those languages need more syllables for the same
  sentence. A 2.8 s English line sometimes has no fluent Hindi version that fits even at 1.25× tempo,
  and the pipeline then trims it with a fade and flags it, because telegraphic Hindi is worse.
- **Speech rate variance.** Chatterbox's pace swings by almost 2× between takes on short lines, and
  occasionally it babbles, turning eleven characters into seven seconds. Babbles are detected and
  re-rolled up to three times, keeping the shortest take; it catches most of it, not all.
- **Emotion.** Questions, statements and exclamations come out flatter than the original. The engine
  only exposes an intensity knob and a pacing knob, which the pipeline now drives from the original
  line's pitch rise, loudness swings and punctuation. It is a nudge, not acting.
- **Gender is a heuristic.** Pitch plus a French-trained classifier. A low female voice or a high
  tenor can flip it; when the two signals disagree the result is flagged and only the clone is
  trusted, so the voice is still right even if the label is not.
- **No lip sync.** Timing is per sentence, not per mouth movement.
- **Overlapping speech** is kept and both dubs are mixed, but Whisper only transcribes the dominant
  voice, so the quieter speaker's words are usually lost.
- **Speed.** Roughly 8–18 minutes per minute of video on this GPU with the free LLM routes I had,
  and the spread is the provider's, not the code's. Adding provider keys to FreeLLMAPI is the single
  biggest speed-up left.
- **Reference clips need 10 s of clean speech per speaker.** Under about 3 s the clone degrades
  noticeably; under 1 s the speaker is flagged `no_ref_clip`.
- **torchcodec** does not load against a static ffmpeg build, which is why the XTTS and F5
  environments pin torch 2.8.
- There is no hosted demo. The free ZeroGPU tier does not run the torch version Chatterbox pins, and
  I did not want to ship a Space that only works with the second-best engine.

## Ethics note

This tool clones voices. I only ran it on synthetic voices, a 1919 public-domain recording, and a
recording of two people who agreed to be dubbed for the demo. Please do the same: do not dub people
who have not consented, and do not pass dubbed speech off as something someone actually said.
Chatterbox stamps every clip with an imperceptible watermark that its detector can find; I leave it
in. The pipeline writes a `report.json` per run that lists every segment it had to stretch, truncate
or could not verify, so there is always a record of where the dub departs from the original.

## Fixing mistakes by hand

Open `.cache/work/<video>/segments.json`. You may change `speaker`, `gender`, `start`, `end`, `text`,
`type` (`speech` or `music`) and `translation`. If you write a `translation` yourself, set
`verify.attempts` to `-1` and it will be used as-is and never re-translated. Run `dubber.py` again
with the same arguments and only what you touched is redone; the stems, transcript and diarization
stay as they are. Notepad's UTF-8 with BOM is fine.

## Repository map

Three folders you touch, the code, and nothing else visible:

```
input/               put videos here (the synthetic test clips are built here too)
output/              dubbed .mp4 + .srt per run
transcripts/         per run: "Person N (gender): [line]" and its translation, in order
benchmarks/          TTS benchmark + end-to-end eval: scripts, sentences, tts_results.md, eval_results.md
dubber.py            CLI and stage runner (resume logic lives here)
app.py               desktop UI over dubber.py (tkinter: file pickers, stage progress, log, cancel)
dubber/
  schema.py          segments.json contract
  config.py          every tunable: speech rates, thresholds, engine table, model names, folders
  separate.py transcribe.py speakers.py segment.py music.py      stages 0-2
  translate.py prompts.py llm.py                                  stage 3
  synth.py tts/      stage 4 (one module per engine, subprocess runner in tts/base.py)
  fit.py mix.py      stage 5
  report.py          report.json
tests/               make_test_clip.py builds the clips; pytest -m "not slow" runs in seconds;
                     review_m3.md is the independent code review that shaped M3
setup.ps1            idempotent Windows setup
```

Generated and hidden: `.venv/` (main env), `.envs/<engine>/` (TTS engine envs), `.cache/work/<video>/`
(per-run working dirs), `.cache/models/` and `.cache/voices/` (downloaded weights and voices).

## What's next

Settle the accent question: make the native-reference-plus-timbre-transfer path the default for
cross-lingual dubs if listeners prefer it, and let people drop in a real recording of a native speaker
as the reference. Per-language syllable counting instead of characters per second. Better emotion
transfer than the current pitch, loudness and punctuation hints, probably an emotion classifier on the
original line or reference clips picked per mood. A hosted demo once an engine with the right accent
runs on the free GPU tier. And someone fluent in Malay, Japanese and Mandarin to tell me what the LLM
judge is missing.

## Licence

Code in this repository: MIT. The models it downloads keep their own licences, listed above; because
XTTS-v2 and F5-TTS weights are non-commercial and Piper is GPL, treat the pipeline as a whole as
non-commercial unless you remove those engines.

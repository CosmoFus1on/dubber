# TTS benchmark (2026-10-19, RTX 3080 10 GB, cuda)

5 fixed lines (first 5 of `tests/clip_truth.json`; translations in `bench/sentences.json`, made once by
`nemotron-3-ultra-550b` with `prompts.translate_messages`), references `work/clip/refs/SPEAKER_00.wav` (A, male)
and `SPEAKER_01.wav` (B, female). One `run_engine` call per (engine, language), engines sequential,
`gpu.unload()` between. Raw numbers and per-line durations: `bench/results.json`; audio: `bench/out/`.

* **load s**: wall time of the 1-sentence warm-up call (subprocess start, imports, model load, 1 line).
* **RTF**: synth seconds / trimmed audio seconds. Subprocess engines pay the load in both calls, so
  RTF = (5-line wall - warm-up wall) / audio of lines 2-5; Piper (in-process) = 5-line wall / audio.
* **timing err**: |duration after `librosa.effects.trim(top_db=35)` - slot| per line. Slots are the English
  timings, so this measures how well each engine's natural pace fits a dub slot *before* `fit.py` stretches.
* **speaker sim**: resemblyzer cosine, output vs the speaker's reference clip. Cross-language, so the ceiling
  is below the same-language ~0.9; Piper does not clone, its number is the distance of a generic voice.
* **VRAM peak**: device-wide `nvidia-smi memory.used` peak during the 5-line call minus idle (Piper runs on
  CPU, hence 0). 5-line batches, so these are lower bounds for long videos.

<!-- table -->
| engine | lang | load s (warm-up) | RTF | timing err mean / max ms | speaker sim | fails | VRAM peak MB |
|---|---|---|---|---|---|---|---|
| chatterbox | hi | 24.2 | 0.81 | 1857 / 5301 | 0.887 | 0/5 | 4306 |
| chatterbox | ms | 21.3 | 0.95 | 661 / 2182 | 0.893 | 0/5 | 4126 |
| chatterbox | ja | 21.8 | 1.15 | 459 / 794 | 0.863 | 0/5 | 4105 |
| chatterbox | zh | 25.0 | 0.68 | 2633 / 8470 | 0.858 | 0/5 | 4331 |
| xtts | hi | 26.9 | 0.21 | 2347 / 2753 | 0.837 | 0/5 | 2365 |
| xtts | ja | 23.4 | 0.37 | 497 / 1221 | 0.796 | 0/5 | 2295 |
| xtts | zh | 23.5 | 0.35 | 477 / 709 | 0.802 | 0/5 | 2257 |
| openvoice | ja | 13.4 | 0.13 | 200 / 383 | 0.723 | 0/5 | 907 |
| openvoice | zh | 17.5 | 0.15 | 729 / 1127 | 0.660 | 0/5 | 1503 |
| f5 | zh | 18.2 | 0.99 | 1244 / 1675 | 0.850 | 0/5 | 2939 |
| piper | hi | 0.8 | 0.10 | 1403 / 2264 | 0.637 | 0/5 | 0 |
| piper | ja | 1.0 | 0.09 | 246 / 615 | 0.537 | 0/5 | 0 |
| piper | zh | 4.3 | 0.27 | 400 / 615 | 0.548 | 0/5 | 0 |
<!-- /table -->

No engine failed outright (0/65 lines). Babble (a clip far longer than its text) is the main defect; in the
pipeline `synth._babbled` retries those once.

## Per engine

**chatterbox** (MIT; `envs/chatterbox`, torch 2.6). Best similarity everywhere (0.86-0.89) and the only engine
with Malay. Slowest (RTF 0.7-1.15) and the heaviest (~4.1-4.3 GB). Babbled twice: zh line 3 ran 11.5 s for a
3.0 s slot and hi line 5 7.6 s for 2.3 s; the rest sit within ~1 s. Stays primary.

**xtts** (XTTS-v2 weights: Coqui Public Model License, non-commercial; coqui-tts code MPL-2.0; `envs/xtts`,
torch 2.8 + transformers 4.57). Good cloning (0.80-0.84), fast (RTF 0.2-0.4), 2.3 GB. ja/zh timing is on par with
chatterbox (~0.5 s mean). Hindi runs consistently long (4.6-5.8 s for 2.3-3.6 s slots, +2.3 s on average): usable
with `speed` > 1 or `fit.py` stretching, but it is the slowest-paced Hindi here. No Malay. Best fallback for hi/ja/zh.

**openvoice** (OpenVoice V2 + MeloTTS, both MIT; `envs/openvoice`, torch 2.8, transformers 4.27.4). Lightest
cloner (0.9-1.5 GB) and fast (RTF 0.13-0.15). Japanese timing is the best of all engines (200 ms mean). It only
converts tone colour on top of Melo's base voice, so similarity is lower (0.72 ja, 0.66 zh) and it speaks
Chinese fast (all lines short, 0.7 s mean). No Hindi or Malay. Install went cleanly (MeloTTS from git, OpenVoice
`--no-deps`, `unidic download`, NLTK tagger for English); needed one workaround for an upstream bug
(`ToneColorConverter(enable_watermark=False)` raises), see `dubber/tts/openvoice.py`.

**f5** (F5-TTS v1 Base: code MIT, weights CC-BY-NC-4.0; `envs/f5`, torch 2.8, torchcodec removed). zh only in
this set (en also supported). Strong similarity (0.85), but every Chinese line is too short (1.1-2.2 s for
2.3-3.6 s slots): F5 sets output length from the ratio of reference audio to reference text, and the reference
transcript (auto-transcribed by its Whisper on `ref_text=""`) is English, which has a very different chars/s
from Chinese. A same-language reference clip, or passing an explicit duration, would fix that. RTF ~1.0
because the warm-up and main calls both run the Whisper reference transcription. 2.9 GB.

**piper** (code GPL-3.0, voices MIT; main venv, ONNX on CPU). No cloning, so similarity 0.54-0.64 is just a
generic voice of the right gender (voices: en ryan/lessac, hi rohan/priyamvada, zh chaowen/huayan,
ja hi_fi_captain male/female speaker ids). Near-instant load, RTF 0.09-0.27 on CPU, no VRAM. Japanese and Chinese
timing is decent (0.25-0.4 s); Hindi runs long (+1.4 s mean). The CPU/last-resort fallback it was meant to be.

## Licences

| engine | licence |
|---|---|
| chatterbox | Chatterbox Multilingual: MIT (outputs carry Resemble's PerTh watermark) |
| xtts | XTTS-v2 weights: Coqui Public Model License (non-commercial); coqui-tts code MPL-2.0 |
| openvoice | OpenVoice V2: MIT; MeloTTS: MIT |
| f5 | F5-TTS code MIT; F5TTS_v1_Base weights CC-BY-NC-4.0 |
| piper | piper-tts code GPL-3.0; rhasspy/piper-voices voices MIT |

## Reproduce

```
.\setup.ps1 -Engines chatterbox,xtts,openvoice,f5
.venv\Scripts\python.exe -X utf8 bench\smoke_tts.py xtts hi zh ja en      # per engine smoke test
.venv\Scripts\python.exe -X utf8 benchenchmark_tts.py                   # or: benchmark_tts.py xtts piper
```
`bench\make_sentences.py` rebuilds `sentences.json` (LLM calls, cached on disk); not needed for reruns.

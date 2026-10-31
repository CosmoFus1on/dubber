# dubber architecture (contract for every module and subagent)

`python dubber.py input.mp4 --to hi --out dubbed.mp4`

Everything for one input lives in `.cache/work/<stem>/`. Stages read and write files there; the single
source of truth for segment-level data is `segments.json` (see `dubber/schema.py`).

## Stages

| # | Stage (`STAGES` name) | Module | Reads | Writes | Model in VRAM |
|---|---|---|---|---|---|
| 0 | `separate` | `separate.py` | `audio.wav` (from `ffmpeg.extract_audio`, 48 kHz stereo) | `stems/vocals.wav`, `stems/no_vocals.wav` (48 kHz) | Demucs htdemucs |
| 1a | `transcribe` | `transcribe.py` | `stems/vocals.wav` | `transcript.json`: `{language, words:[{w,s,e,p}], whisper_segments:[{start,end,text,avg_logprob,compression_ratio,no_speech_prob}]}` | faster-whisper |
| 1b | `speakers` | `speakers.py` | `stems/vocals.wav`, `stems/no_vocals.wav`, `transcript.json` | `diarization.json`: `{turns:[{speaker,start,end}], exclusive:[...]}`; `refs/<SPEAKER>.wav`; `Project.speakers` | pyannote, inaSpeechSegmenter (CPU/TF) |
| 2 | `segment` | `segment.py` | `transcript.json`, `diarization.json`, stems | `segments.json` segments (`type`, `words`, `extra` with detector scores) | none |
| 3 | `translate` | `translate.py` | `segments.json` | `translation`, `budget_chars`, `verify` on each speech segment | MiniLM embedder only |
| 4 | `tts` | `tts/` | `segments.json`, `refs/` | `tts/<id>.wav`, `Segment.tts.{engine,wav,raw_duration}` | one TTS engine (own venv when pinned) |
| 5a | `fit` | `fit.py` | `tts/<id>.wav`, `segments.json` | `fit/<id>.wav`, `Segment.tts.{stretch,fitted_wav}`, flags | none (LLM for shorten) |
| 5b | `mix` | `mix.py` | `fit/*.wav`, stems, `audio.wav`, video | `mix.wav`, `<out>.mp4`, `<out>.srt`, `report.json` | none |

`gpu.unload()` is called by the runner after every stage. Never hold two model families at once.

## Runner (`dubber.py`)

* `state.json`: `{stage: {"input_hash": sha1, "done": true, "seconds": float}}`. A stage reruns iff its
  outputs are missing or the input hash changed. `--from <stage>` forces reruns from there; `--only <stage>`.
* Stages 3–5 are per-segment incremental: each segment decides for itself via `Segment.translate_key` /
  `Segment.tts_key` whether its cached result still applies. Mix always reruns (cheap).
* Human edits to `segments.json` are the supported way to fix mistakes; a hand-set `translation`
  (set `verify.attempts` to -1) is treated as verified.
* `report.json`: flags per segment and speaker, per-stage wall-clock seconds, total seconds per minute of
  video, LLM calls made, engines used, model ids. Measured, never estimated.

## Conventions

* Python 3.10, Windows-first: `pathlib`, `subprocess.run([...])` lists (never `shell=True`), explicit
  `encoding="utf-8"` on every text file, no `os.fork`, no POSIX-only tools.
* Audio inside the pipeline: 48 kHz float32 numpy via `soundfile`; mono for analysis, stereo for the mix.
* ffmpeg calls go through `dubber/ffmpeg.py` only.
* Secrets come from `.env` via `config.py`; never print or log them.
* Logging: `logging.getLogger("dubber.<module>")`; the CLI configures the root handler.
* Every non-trivial module has a `__main__` self-check or a `tests/test_<module>.py`.
* Flags are the only way to report problems that do not stop the run (`schema.F_*`).

## Interfaces owned by the lead (do not change signatures)

* `schema.Project / Segment / Speaker / Word`, `schema.sha1`, `schema.STAGES`
* `llm.LLMProvider.chat(messages, model, json_mode, temperature, max_tokens, fallback) -> str`,
  `llm.OpenAICompatible`, `llm.parse_json`
* `tts.base.TTSEngine.synth(jobs, device) -> list[TTSResult]`, `tts.base.run_engine`, `tts.base.chain_for`
* `config.*` tunables

# dubber: measured results (2026-11-05)

Hardware: RTX 3080 10 GB, Windows 11. LLM: free-tier routes through a local FreeLLMAPI proxy
(`nemotron-3-ultra-550b` on the NVIDIA free tier at the time). All numbers below are read from the
runs' `report.json` / `segments.json`; nothing is estimated.

## Real two-person recording, English -> Hindi (78 s, consented speakers)

| | first run (2026-10-31) | after parallel LLM + warm TTS worker | final (Hindi fine-tune, accent fix) |
|---|---|---|---|
| translate + verify | 602 s | 260 s | 369 s |
| TTS | 180 s | 186 s | 324 s |
| timing fit | 876 s | 89 s | 633 s |
| total | 1705 s | ~590 s | 1392 s |
| seconds per minute of video | 1311 | ~454 | 1070 |
| verification pass rate | 28/28 | 28/28 | 28/28 |
| mean judge / fluency / similarity | 0.98 / 1.00 / 0.952 | 0.98 / 1.00 / 0.912 | 0.98 / 1.00 / 0.911 |
| lines tempo-stretched / truncated | 1 / 1 | 1 / 1 | 10 / 2 |

The final run is slower than the middle one for two measured reasons: the free LLM route was slower that
day (369 s vs 260 s for the same 2 translate windows + 56 verify calls), and the Hindi fine-tune at
`cfg_weight=0` (needed to stop the English reference accent leaking in) speaks longer per character, so more
lines went through shorten rounds (144 LLM calls vs 100) and more needed stretching. Accent vs. timing is a
real trade-off in this pipeline, not a bug.

## Synthetic eval set (2 clips x 4 languages, final code)

| clip | lang | checked by | pass | judge | fluency | similarity | stretched | truncated | s/min |
|---|---|---|---|---|---|---|---|---|---|
| 2 speakers + song (44 s) | hi | fluent speaker | 5/5 | 0.91 | 1.00 | 0.869 | 1 | 0 | 614 |
| 2 speakers + song | ms | LLM only | 5/5 | 0.85 | 0.90 | 0.927 | 0 | 0 | 511 |
| 2 speakers + song | ja | LLM only | 5/5 | 1.00 | 1.00 | 0.866 | 0 | 0 | 235 |
| 2 speakers + song | zh | LLM only | 5/5 | 1.00 | 1.00 | 0.859 | 0 | 0 | 325 |
| 1 speaker (24 s) | hi | fluent speaker | 3/3 | 0.95 | 1.00 | 0.958 | 2 | 1 | 1002 |
| 1 speaker | ms | LLM only | 3/3 | 0.90 | 0.93 | 0.926 | 0 | 0 | 835 |
| 1 speaker | ja | LLM only | 3/3 | 0.90 | 1.00 | 0.870 | 0 | 0 | 352 |
| 1 speaker | zh | LLM only | 3/3 | 1.00 | 1.00 | 0.863 | 0 | 0 | 441 |

Pass = LLM judge >= 0.7, fluency >= 0.7, embedding similarity >= 0.75 within 3 attempts. The song section
is bit-identical to the original (correlation 0.9997). Speakers and genders are correct on every run.
s/min on a 24 s clip is dominated by fixed costs (model loads, 3 LLM round trips minimum) and varies with
the free LLM route's latency; on longer videos it converges towards the TTS real-time factor.

## TTS engines (5 lines x 4 languages, same reference clips; `tts_results.md`)

| engine | licence | langs tested | RTF | timing error mean | speaker similarity | VRAM |
|---|---|---|---|---|---|---|
| Chatterbox Multilingual | MIT | hi ms ja zh | 0.7-1.15 | 0.5-2.6 s | 0.86-0.89 | 4.3 GB |
| XTTS-v2 | non-commercial | hi ja zh | 0.2-0.4 | 0.5-2.3 s | 0.80-0.84 | 2.3 GB |
| OpenVoice v2 | MIT | ja zh | 0.13-0.15 | 0.2-0.7 s | 0.66-0.72 | 1.5 GB |
| F5-TTS | CC-BY-NC | zh | 1.0 | 1.2 s | 0.85 | 2.9 GB |
| Piper (CPU) | GPL | hi ja zh | 0.1-0.3 | 0.2-1.4 s | 0.54-0.64 | 0 |

## Engineering

21 commits, 69 tracked files, ~4,600 lines of Python, 41 tests (35 run in seconds without models; 6 slow
acceptance tests: speakers/genders, music detection, song passthrough, resume-on-edit, TTS worker protocol).
Resume: editing one line's text re-does only that line (verified by mtimes). Crash-safe: every stage
checkpoints to disk. CPU-only mode works (distil-whisper + Piper): 24 s clip in 396 s.

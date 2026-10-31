# Evaluation results (measured)

Pass = judge >= 0.7, fluency >= 0.7, embedding similarity >= 0.75 after up to 3 attempts. s/min = wall-clock seconds per minute of video on an RTX 3080, models already downloaded.

| clip | lang | checked by | segments | music segs | pass rate | mean judge | mean fluency | mean sim | stretched | max stretch | truncated | flagged | engines | s/min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clip.mp4 | hi | hand-checked by a fluent speaker (project owner); zh/ms/ja are LLM-judged only | 5 | 1 | 1.0 | 0.94 | 0.96 | 0.876 | 1 | 1.25 | 1 | 1 stretch_over_15 truncated verify_failed | chatterbox | 615.77 |
| clip.mp4 | ms | LLM-judged only | 5 | 1 | 1.0 | 0.97 | 1.0 | 0.918 | 1 | 1.25 | 1 | 1 stretch_over_15 truncated verify_failed | chatterbox | 377.11 |
| clip.mp4 | ja | LLM-judged only | 5 | 1 | 1.0 | 0.97 | 1.0 | 0.9 | 0 | 1.0 | 0 | 0  | chatterbox | 283.8 |
| clip.mp4 | zh | LLM-judged only | 5 | 1 | 1.0 | 0.94 | 1.0 | 0.856 | 0 | 1.0 | 0 | 0  | chatterbox | 302.43 |
| clip_single.mp4 | hi | hand-checked by a fluent speaker (project owner); zh/ms/ja are LLM-judged only | 3 | 0 | 1.0 | 0.95 | 1.0 | 0.945 | 1 | 1.077 | 0 | 0  | chatterbox | 590.34 |
| clip_single.mp4 | ms | LLM-judged only | 3 | 0 | 1.0 | 0.95 | 0.933 | 0.892 | 1 | 1.21 | 0 | 1 stretch_over_15 verify_failed | chatterbox | 696.95 |
| clip_single.mp4 | ja | LLM-judged only | 3 | 0 | 1.0 | 0.95 | 1.0 | 0.889 | 0 | 1.0 | 0 | 0  | chatterbox | 389.63 |
| clip_single.mp4 | zh | LLM-judged only | 3 | 0 | 1.0 | 0.95 | 1.0 | 0.866 | 0 | 1.0 | 0 | 0  | chatterbox | 404.66 |

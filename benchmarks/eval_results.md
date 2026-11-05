# Evaluation results (measured)

Pass = judge >= 0.7, fluency >= 0.7, embedding similarity >= 0.75 after up to 3 attempts. s/min = wall-clock seconds per minute of video on an RTX 3080, models already downloaded.

| clip | lang | checked by | segments | music segs | pass rate | mean judge | mean fluency | mean sim | stretched | max stretch | truncated | flagged | engines | s/min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clip.mp4 | hi | hand-checked by a fluent speaker (project owner); zh/ms/ja are LLM-judged only | 5 | 1 | 1.0 | 0.91 | 1.0 | 0.869 | 1 | 1.094 | 0 | 0  | chatterbox | 613.63 |
| clip.mp4 | ms | LLM-judged only | 5 | 1 | 1.0 | 0.85 | 0.9 | 0.927 | 0 | 1.0 | 0 | 0  | chatterbox | 511.03 |
| clip.mp4 | ja | LLM-judged only | 5 | 1 | 1.0 | 1.0 | 1.0 | 0.866 | 0 | 1.0 | 0 | 0  | chatterbox | 234.93 |
| clip.mp4 | zh | LLM-judged only | 5 | 1 | 1.0 | 1.0 | 1.0 | 0.859 | 0 | 1.0 | 0 | 0  | chatterbox | 324.94 |
| clip_single.mp4 | hi | hand-checked by a fluent speaker (project owner); zh/ms/ja are LLM-judged only | 3 | 0 | 1.0 | 0.95 | 1.0 | 0.958 | 2 | 1.25 | 1 | 1 stretch_over_15 truncated | chatterbox | 1001.67 |
| clip_single.mp4 | ms | LLM-judged only | 3 | 0 | 1.0 | 0.9 | 0.933 | 0.926 | 0 | 1.0 | 0 | 0  | chatterbox | 835.0 |
| clip_single.mp4 | ja | LLM-judged only | 3 | 0 | 1.0 | 0.9 | 1.0 | 0.87 | 0 | 1.0 | 0 | 0  | chatterbox | 351.82 |
| clip_single.mp4 | zh | LLM-judged only | 3 | 0 | 1.0 | 1.0 | 1.0 | 0.863 | 0 | 1.0 | 0 | 0  | chatterbox | 440.67 |

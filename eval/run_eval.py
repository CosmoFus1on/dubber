"""Evaluation harness: dub each clip into each language, collect measured numbers into eval/results.md.

Usage: python eval/run_eval.py [--langs hi,ms,ja,zh] [--clips tests/clip.mp4,...] [--skip-run]
Every number is read from the run's segments.json / report.json; nothing is estimated.
Human labels: a language is "hand-checked by a fluent speaker" only when eval/hand_checks.json says so.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval"


def run(clip: Path, lang: str) -> Path:
    work = ROOT / "work" / f"eval_{clip.stem}_{lang}"
    out = ROOT / "out" / f"eval_{clip.stem}_{lang}.mp4"
    cmd = [sys.executable, "-X", "utf8", "dubber.py", str(clip), "--to", lang, "--work", str(work), "--out", str(out)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(f"!! {clip.name} -> {lang} failed:\n{r.stderr[-1500:]}")
    return work


def collect(work: Path, clip: Path, lang: str, labels: dict) -> dict | None:
    seg_path, rep_path = work / "segments.json", work / "report.json"
    if not seg_path.exists() or not rep_path.exists():
        return None
    p = json.loads(seg_path.read_text(encoding="utf-8"))
    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    speech = [s for s in p["segments"] if s["type"] == "speech" and s.get("translation")]
    music = [s for s in p["segments"] if s["type"] == "music"]
    passed = [s for s in speech if s["verify"].get("passed") or s["verify"].get("attempts") == -1]
    judged = [s for s in speech if s["verify"].get("judge") is not None]
    stretch = [s["tts"].get("stretch") or 1.0 for s in speech]
    flagged = [s for s in p["segments"] if s["flags"]]
    mean = lambda xs: round(statistics.fmean(xs), 3) if xs else None  # noqa: E731
    return {
        "clip": clip.name, "lang": lang, "label": labels.get(lang, "LLM-judged only"),
        "segments": len(speech), "music_segments": len(music),
        "verify_pass_rate": round(len(passed) / len(speech), 3) if speech else None,
        "mean_judge": mean([s["verify"]["judge"] for s in judged]),
        "mean_fluency": mean([s["verify"]["fluency"] for s in judged if s["verify"].get("fluency") is not None]),
        "mean_sim": mean([s["verify"]["sim"] for s in judged]),
        "stretched": sum(1 for x in stretch if x > 1.0), "max_stretch": max(stretch) if stretch else None,
        "truncated": sum(1 for s in speech if "truncated" in s["flags"]),
        "flagged": len(flagged), "flags": sorted({f for s in flagged for f in s["flags"]}),
        "engines": rep.get("engines_used"), "llm_model": rep.get("llm_model"),
        "total_s": rep.get("total_s"), "s_per_video_min": rep.get("seconds_per_video_minute"),
        "stages": rep.get("stages"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="hi,ms,ja,zh")
    ap.add_argument("--clips", default="tests/clip.mp4,tests/clip_single.mp4")
    ap.add_argument("--skip-run", action="store_true", help="only collect from existing work dirs")
    a = ap.parse_args()
    labels_path = OUT / "hand_checks.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {}
    rows = []
    for clip in [ROOT / c for c in a.clips.split(",")]:
        for lang in a.langs.split(","):
            work = ROOT / "work" / f"eval_{clip.stem}_{lang}"
            if not a.skip_run:
                print(f"== {clip.name} -> {lang}")
                work = run(clip, lang)
            row = collect(work, clip, lang, labels)
            if row:
                rows.append(row)
                print(json.dumps({k: row[k] for k in ("clip", "lang", "verify_pass_rate", "mean_fluency", "stretched", "flagged", "s_per_video_min")}))
    (OUT / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["# Evaluation results (measured)", "",
             "Pass = judge >= 0.7, fluency >= 0.7, embedding similarity >= 0.75 after up to 3 attempts. "
             "s/min = wall-clock seconds per minute of video on an RTX 3080, models already downloaded.", "",
             "| clip | lang | checked by | segments | music segs | pass rate | mean judge | mean fluency | mean sim | stretched | max stretch | truncated | flagged | engines | s/min |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['clip']} | {r['lang']} | {r['label']} | {r['segments']} | {r['music_segments']} | {r['verify_pass_rate']} | "
                     f"{r['mean_judge']} | {r['mean_fluency']} | {r['mean_sim']} | {r['stretched']} | {r['max_stretch']} | "
                     f"{r['truncated']} | {r['flagged']} {' '.join(r['flags'])} | {','.join(r['engines'] or [])} | {r['s_per_video_min']} |")
    (OUT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT / "results.md")


if __name__ == "__main__":
    main()

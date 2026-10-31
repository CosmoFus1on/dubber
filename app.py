"""Local Gradio UI for dubber: upload a video, pick a language, download the dub.

    python app.py            ->  http://127.0.0.1:7860
Runs the same dubber.py pipeline in a subprocess, so every log line and report.json is where the CLI puts them.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import gradio as gr

from dubber import config
from dubber.tts.chatterbox import LANGS

ROOT = Path(__file__).resolve().parent
LANG_CHOICES = sorted(LANGS)


def dub(video: str, lang: str, keep_original: bool, max_seconds: float):
    if not video:
        raise gr.Error("upload a video first")
    src = config.INPUT_DIR / Path(video).name
    config.INPUT_DIR.mkdir(exist_ok=True)
    if max_seconds and max_seconds > 0:  # demo-style cap: trim before dubbing
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", video, "-t", str(max_seconds),
                        "-c", "copy", str(src)], check=True)
    elif Path(video).resolve() != src.resolve():
        shutil.copyfile(video, src)
    out = config.OUTPUT_DIR / f"{src.stem}_{lang}.mp4"
    cmd = [sys.executable, "-X", "utf8", str(ROOT / "dubber.py"), str(src), "--to", lang, "--out", str(out)]
    if keep_original:
        cmd.append("--keep-original-audio")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise gr.Error("pipeline failed:\n" + r.stderr[-1500:])
    rep = json.loads((config.WORK_DIR / src.stem / "report.json").read_text(encoding="utf-8"))
    flagged = rep.get("segments_flagged", [])
    summary = (f"{rep['counts']['speech']} lines, {rep['counts']['speakers']} speaker(s), "
               f"{rep['counts']['music']} music segment(s), {len(flagged)} flagged, "
               f"{rep['total_s']:.0f} s for {rep['duration_s']:.0f} s of video\n\n")
    summary += "\n".join(f"[{s['start']:.1f}-{s['end']:.1f}s] {', '.join(s['flags'])}: {s['translation']}" for s in flagged)
    return str(out), str(out.with_suffix(".srt")), summary


with gr.Blocks(title="dubber") as ui:
    gr.Markdown("## dubber: free, local AI video dubbing\nVoices are cloned from the speakers in the video. "
                "Only dub people who agreed to it. Text (never audio) goes to the configured LLM proxy.")
    with gr.Row():
        video = gr.Video(label="video")
        with gr.Column():
            lang = gr.Dropdown(LANG_CHOICES, value="hi", label="target language")
            keep = gr.Checkbox(False, label="keep original audio as a second track")
            cap = gr.Number(0, label="trim to N seconds first (0 = whole video)")
            btn = gr.Button("Dub", variant="primary")
    with gr.Row():
        result = gr.Video(label="dubbed")
        srt = gr.File(label="subtitles (.srt)")
    report = gr.Textbox(label="report", lines=8)
    btn.click(dub, [video, lang, keep, cap], [result, srt, report])

if __name__ == "__main__":
    ui.launch(server_name="127.0.0.1", server_port=7860, show_error=True)

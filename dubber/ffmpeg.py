"""Thin ffmpeg/ffprobe wrappers. All ffmpeg calls in the pipeline go through here."""
from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path

log = logging.getLogger("dubber.ffmpeg")


def run(args: list[str]) -> subprocess.CompletedProcess:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *map(str, args)]
    log.debug("%s", cmd)
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({p.returncode}): {p.stderr.strip()}")
    return p


def probe(path) -> dict:
    p = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
                        str(path)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {p.stderr.strip()}")
    return json.loads(p.stdout)


def duration(path) -> float:
    return float(probe(path)["format"]["duration"])


def extract_audio(video, wav, sr: int = 48000, channels: int = 2) -> Path:
    wav = Path(wav)
    wav.parent.mkdir(parents=True, exist_ok=True)
    run(["-i", str(video), "-vn", "-ac", str(channels), "-ar", str(sr), "-c:a", "pcm_s16le", str(wav)])
    return wav


def _atempo_chain(tempo: float) -> str:
    parts = []
    while tempo > 2.0:
        parts.append(2.0)
        tempo /= 2.0
    while tempo < 0.5:
        parts.append(0.5)
        tempo /= 0.5
    parts.append(tempo)
    return ",".join(f"atempo={t:.6f}" for t in parts)


def rubberband(src, dst, tempo: float) -> Path:
    dst = Path(dst)
    try:
        run(["-i", str(src), "-af", f"rubberband=tempo={tempo:.6f}", str(dst)])
    except RuntimeError as e:
        if "rubberband" not in str(e).lower():
            raise
        log.warning("rubberband filter unavailable, falling back to atempo")
        run(["-i", str(src), "-af", _atempo_chain(tempo), str(dst)])
    return dst


def loudnorm(src, dst, lufs: float = -16.0, two_pass: bool = True) -> Path:
    dst = Path(dst)
    base = f"loudnorm=I={lufs}:TP=-1.5:LRA=11"
    if not two_pass:
        run(["-i", str(src), "-af", base, str(dst)])
        return dst
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(src), "-af", base + ":print_format=json",
                        "-f", "null", "-"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg loudnorm pass 1 failed: {p.stderr.strip()[-500:]}")
    m = re.findall(r"\{[^{}]*\}", p.stderr, re.S)
    if not m:
        raise RuntimeError("loudnorm pass 1 printed no JSON")
    d = json.loads(m[-1])
    f = (f"{base}:measured_I={d['input_i']}:measured_TP={d['input_tp']}:measured_LRA={d['input_lra']}"
         f":measured_thresh={d['input_thresh']}:offset={d['target_offset']}:linear=true")
    run(["-i", str(src), "-af", f, str(dst)])
    return dst


def duck(background, voice, dst, threshold: float = 0.03, ratio: float = 4, attack: float = 20,
         release: float = 300, makeup: float = 1) -> Path:
    dst = Path(dst)
    fc = (f"[0:a][1:a]sidechaincompress=threshold={threshold}:ratio={ratio}:attack={attack}:release={release}"
          f":makeup={makeup}[o]")
    run(["-i", str(background), "-i", str(voice), "-filter_complex", fc, "-map", "[o]", str(dst)])
    return dst


def mux(video, audio_wav, out, keep_original_audio: bool = False, srt=None) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    args = ["-i", str(video), "-i", str(audio_wav)]
    if srt:
        args += ["-i", str(srt)]
    args += ["-map", "0:v:0", "-map", "1:a:0"]
    if keep_original_audio:
        args += ["-map", "0:a:0"]
    if srt:
        args += ["-map", "2:0"]
    args += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-metadata:s:a:0", "title=Dubbed"]
    if keep_original_audio:
        args += ["-metadata:s:a:1", "title=Original"]
    if srt:
        args += ["-c:s", "mov_text"]
    run([*args, str(out)])
    return out


def _ts(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments, path, key: str = "translation") -> Path:
    path = Path(path)
    out, n = [], 1
    for s in segments:
        text = s.text if s.type == "music" else getattr(s, key, None)
        if not text or not text.strip():
            continue
        out.append(f"{n}\n{_ts(s.start)} --> {_ts(s.end)}\n{text.strip()}\n")
        n += 1
    path.write_text("\n".join(out), encoding="utf-8")
    return path


def side_by_side(original_video, dubbed_video, out, label_left: str = "Original", label_right: str = "Dubbed") -> Path:
    out = Path(out)
    dt = "fontcolor=white:fontsize=28:box=1:boxcolor=black@0.5:x=10:y=10"
    fc = (f"[0:v]drawtext=text='{label_left}':{dt}[l];[1:v]drawtext=text='{label_right}':{dt}[r];"
          f"[l][r]hstack=inputs=2[v]")
    run(["-i", str(original_video), "-i", str(dubbed_video), "-filter_complex", fc, "-map", "[v]", "-map", "1:a:0",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", str(out)])
    return out

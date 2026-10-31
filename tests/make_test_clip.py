"""Build input/clip.mp4 (~45 s, two speakers, pause, song) and, with --single, input/clip_single.mp4.

Dialogue comes from Piper voices in .cache/voices/ (git-ignored; setup.ps1 or this script downloads them).
Ground truth timings are written next to the clip as clip_truth.json / clip_single_truth.json.

    .venv\\Scripts\\python.exe -X utf8 tests\\make_test_clip.py [--single] [--fake-song]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from dubber import config  # noqa: E402
sys.path.insert(0, str(HERE.parent))
from dubber import audio, ffmpeg  # noqa: E402

SR = 48000
VOICES = config.VOICES_DIR
CLIPS = config.INPUT_DIR
MALE, FEMALE = "en_US-ryan-medium", "en_US-lessac-medium"
SONG_ITEM = "christmasbells1919"          # Edison blue amberol 1919, public domain
SONG_URL = f"https://archive.org/download/{SONG_ITEM}/{SONG_ITEM}_64kb.mp3"
FONT = "C\\:/Windows/Fonts/arial.ttf"

A_LINES = ["Hey, I was thinking we should finally plan that trip to Portugal.",
           "Maybe in the first week of May, when the weather is nice.",
           "I have always wanted to ride the old yellow trams through the city."]
B_LINES = ["That sounds wonderful, but I need to check my work schedule first.",
           "Do you want to start in Lisbon or in Porto?",
           "I also heard the food in the north is incredible."]
A_LINE3 = "Let's start in Lisbon, and then take the train north."
B_CLOSE = "Great, I will book the flights tonight, and we can talk more tomorrow."
SINGLE = ["Good morning everyone, and welcome to this short update about our travel plans.",
          "We will leave on the second of May and return ten days later.",
          "The hotel in Lisbon is booked, and the train tickets are almost ready.",
          "Please send me your passport details before the end of this week.",
          "If anything changes, call me on my phone and I will fix the plan right away.",
          "Thank you all for listening, and I look forward to seeing you at the airport."]
LYRICS = ["Silver bells are ringing, ringing all the night",
          "Over hills and meadows, in the pale moonlight",
          "Sing a song of Christmas, sing it loud and clear"]


def piper(text: str, voice: str, wav: Path) -> None:
    subprocess.run([sys.executable, "-m", "piper", "-m", str(VOICES / f"{voice}.onnx"), "-f", str(wav)],
                   input=text, text=True, encoding="utf-8", check=True, capture_output=True)


def say(text: str, voice: str, tmp: Path) -> np.ndarray:
    w = tmp / f"say_{abs(hash((text, voice)))}.wav"
    piper(text, voice, w)
    x, _ = audio.read(w, sr=SR, mono=True)
    return audio.peak_normalize(x, -3.0) * 0.9


def real_song(tmp: Path) -> np.ndarray:
    mp3 = VOICES / f"{SONG_ITEM}.mp3"
    if not mp3.exists():
        req = urllib.request.Request(SONG_URL, headers={"User-Agent": "Mozilla/5.0 (dubber test clip builder)"})
        with urllib.request.urlopen(req, timeout=60) as r:
            mp3.write_bytes(r.read())
    ex = tmp / "song.wav"
    ffmpeg.run(["-ss", "40", "-t", "12", "-i", str(mp3), "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(ex)])
    x, _ = audio.read(ex, mono=True)
    assert len(x) > 11 * SR, "song excerpt too short"
    return x


def fake_song(tmp: Path) -> np.ndarray:
    """Chord bed (3 sine partials) + lyric lines pitched up 5 semitones and slowed."""
    bed = tmp / "bed.wav"
    ffmpeg.run(["-f", "lavfi", "-i", "aevalsrc=0.3*sin(2*PI*220*t)+0.25*sin(2*PI*277.2*t)+0.2*sin(2*PI*329.6*t):d=12:s=48000",
                "-ac", "1", str(bed)])
    bg, _ = audio.read(bed, mono=True)
    voc = np.zeros_like(bg)
    t = 0
    for i, line in enumerate(LYRICS):
        raw = tmp / f"lyr{i}.wav"
        sung = tmp / f"lyr{i}_s.wav"
        piper(line, FEMALE, raw)
        ffmpeg.run(["-i", str(raw), "-af", "rubberband=pitch=1.33:tempo=0.8", "-ar", str(SR), str(sung)])
        x, _ = audio.read(sung, mono=True)
        x = x[: len(voc) - t]
        voc[t:t + len(x)] += x
        t += len(x) + int(0.2 * SR)
        if t >= len(voc):
            break
    return np.clip(0.5 * bg + audio.peak_normalize(voc, -6.0), -1, 1)


def build(single: bool, force_fake_song: bool) -> tuple[Path, dict, str]:
    tmp_ctx = tempfile.TemporaryDirectory(prefix="dubber_clip_")
    tmp = Path(tmp_ctx.name)
    parts: list[np.ndarray] = []
    t = 0.0
    lines, turns = [], {"A": [], "B": []}
    labels: list[tuple[float, float, str]] = []
    truth_extra: dict = {}
    song_source = None

    def add_silence(sec: float) -> None:
        nonlocal t
        parts.append(np.zeros(int(round(sec * SR)), dtype=np.float32))
        t += sec

    def add_speech(spk: str, text: str, voice: str) -> None:
        nonlocal t
        x = say(text, voice, tmp)
        lines.append({"speaker": spk, "text": text, "start": round(t, 3), "end": round(t + len(x) / SR, 3)})
        turns[spk].append([lines[-1]["start"], lines[-1]["end"]])
        labels.append((t, t + len(x) / SR, "Speaker A (male)" if spk == "A" else "Speaker B (female)"))
        parts.append(x)
        t += len(x) / SR

    add_silence(1.0)
    if single:
        for i, s in enumerate(SINGLE):
            add_speech("A", s, MALE)
            if i < len(SINGLE) - 1:
                add_silence(0.4)
    else:
        for s in A_LINES:
            add_speech("A", s, MALE)
            add_silence(0.4)
        add_silence(0.1)
        for i, s in enumerate(B_LINES):
            add_speech("B", s, FEMALE)
            add_silence(0.4 if i == 0 else 0.0)
        p0 = t
        add_silence(3.0)
        truth_extra["pause"] = [round(p0, 3), round(t, 3)]
        labels.append((p0, t, "(pause)"))
        add_speech("A", A_LINE3, MALE)
        add_silence(0.8)
        try:
            if force_fake_song:
                raise RuntimeError("forced fallback")
            song = real_song(tmp)
            song_source = "internet-archive:" + SONG_ITEM
        except Exception as e:  # network or ffmpeg failure -> synthetic fallback
            print(f"song download failed ({e}); using synthetic fallback", file=sys.stderr)
            song = fake_song(tmp)
            song_source = "synthetic-fallback"
        song = song[: 12 * SR]
        song = song * (10 ** (-20 / 20) / max(np.sqrt(np.mean(song ** 2)), 1e-6))
        truth_extra["song"] = [round(t, 3), round(t + len(song) / SR, 3)]
        labels.append((t, t + len(song) / SR, "(song)"))
        parts.append(song.astype(np.float32))
        t += len(song) / SR
        add_silence(0.8)
        add_speech("B", B_CLOSE, FEMALE)
    add_silence(1.0)

    dialogue = np.concatenate(parts)
    n = len(dialogue)
    # quiet chord bed (-24 dBFS rms) under everything so separation has work to do
    tt = np.arange(n) / SR
    bed = sum(a * np.sin(2 * np.pi * f * tt) for f, a in ((196.0, 1.0), (246.9, 0.8), (293.7, 0.6)))
    bed = bed * (1 + 0.2 * np.sin(2 * np.pi * 0.25 * tt))
    bed = (bed * (10 ** (-24 / 20) / np.sqrt(np.mean(bed ** 2)))).astype(np.float32)
    mix = np.clip(dialogue + bed, -1, 1)

    stem = "clip_single" if single else "clip"
    CLIPS.mkdir(exist_ok=True)
    wav = CLIPS / f"{stem}_audio.wav"
    audio.write(wav, np.stack([mix, mix], axis=1), SR)
    dur = n / SR
    dt = ",".join(f"drawtext=fontfile='{FONT}':text='{txt}':fontcolor=white:fontsize=28:box=1:boxcolor=black@0.6"
                  f":x=(w-text_w)/2:y=h-60:enable='between(t,{a:.3f},{b:.3f})'" for a, b, txt in labels)
    out = CLIPS / f"{stem}.mp4"
    ffmpeg.run(["-f", "lavfi", "-i", f"testsrc2=size=640x360:rate=25:duration={dur:.3f}", "-i", str(wav),
                "-vf", dt, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                "-ac", "2", "-t", f"{dur:.3f}", str(out)])
    wav.unlink()
    truth = {"speakers": [{"id": "A", "gender": "male", "turns": turns["A"]}] +
                         ([] if single else [{"id": "B", "gender": "female", "turns": turns["B"]}]),
             **truth_extra, "lines": lines, "duration": round(dur, 3), "song_source": song_source}
    (HERE / f"{stem}_truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    tmp_ctx.cleanup()
    return out, truth, song_source or "n/a"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true", help="also build clip_single.mp4")
    ap.add_argument("--fake-song", action="store_true", help="force the synthetic song fallback")
    a = ap.parse_args()
    VOICES.mkdir(exist_ok=True)
    for v in (MALE, FEMALE):
        if not (VOICES / f"{v}.onnx").exists():
            subprocess.run([sys.executable, "-m", "piper.download_voices", "--download-dir", str(VOICES), v],
                           check=True)
    for single in ([False, True] if a.single else [False]):
        out, truth, src = build(single, a.fake_song)
        print(f"{out} {truth['duration']}s song={src}")

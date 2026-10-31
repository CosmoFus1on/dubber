"""Build the README demo video: original (left) vs dubbed (right), dubbed audio.

Part 1: the consented two-person recording (input/demo.mp4 vs output/demo_<lang>.mp4), first MAX_DEMO_S.
Part 2: the synthetic clip's two-speaker line + the public-domain song that passes through untouched.
Usage: python tests/make_demo_video.py [--lang hi]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from dubber import config, ffmpeg  # noqa: E402

MAX_DEMO_S = 60.0
SONG_PART = (22.0, 43.0)   # clip.mp4: speaker A line, the 1919 song, speaker B line
FONT = Path("C:/Windows/Fonts/arial.ttf")   # ffmpeg's drawtext has no fontconfig on Windows; name a font file
_font = "fontfile='" + FONT.as_posix().replace(":", "\\:") + "':" if FONT.exists() else ""
DT = _font + "fontcolor=white:fontsize=26:box=1:boxcolor=black@0.55:x=12:y=12"


def part(orig: Path, dub: Path, out: Path, start: float, length: float, right_label: str) -> Path:
    fc = (f"[0:v]scale=640:360,drawtext=text='Original':{DT}[l];"
          f"[1:v]scale=640:360,drawtext=text='{right_label}':{DT}[r];[l][r]hstack=inputs=2[v]")
    ffmpeg.run(["-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(orig),
                "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(dub),
                "-filter_complex", fc, "-map", "[v]", "-map", "1:a:0", "-r", "25",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", str(out)])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="hi")
    a = ap.parse_args()
    demo, demo_dub = config.INPUT_DIR / "demo.mp4", config.OUTPUT_DIR / f"demo_{a.lang}.mp4"
    clip, clip_dub = config.INPUT_DIR / "clip.mp4", config.OUTPUT_DIR / f"clip_{a.lang}.mp4"
    for p in (demo, demo_dub, clip, clip_dub):
        if not p.exists():
            sys.exit(f"missing {p}; dub it first: python dubber.py {p.name} --to {a.lang}")
    out = config.OUTPUT_DIR / f"demo_side_by_side_{a.lang}.mp4"
    with tempfile.TemporaryDirectory(prefix="dubber_demo_") as td:
        td = Path(td)
        p1 = part(demo, demo_dub, td / "p1.mp4", 0.0, min(MAX_DEMO_S, ffmpeg.duration(demo)), f"Dubbed ({a.lang})")
        p2 = part(clip, clip_dub, td / "p2.mp4", SONG_PART[0], SONG_PART[1] - SONG_PART[0], f"Dubbed ({a.lang}), song untouched")
        lst = td / "list.txt"
        lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in (p1, p2)), encoding="utf-8")
        ffmpeg.run(["-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)])
    print(f"wrote {out} ({ffmpeg.duration(out):.1f}s)")


if __name__ == "__main__":
    main()

"""Timing fit + mix on synthetic material (no models, no LLM):
- clip longer than slot -> stretched <= 1.25, flagged correctly, truncated when still too long
- clip shorter than slot -> untouched
- mix: output length == video length, each clip starts within 150 ms of segment.start, music region
  is the original audio (corr > 0.999), no clip overlaps the next segment.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from dubber import audio, config, ffmpeg, fit, mix
from dubber.schema import F_STRETCH_15, F_TRUNCATED, MUSIC, Project, Segment, Speaker, TTSInfo

SR = 48000


def tone(seconds, freq=440.0, sr=SR, amp=0.3):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * math.pi * freq * t)).astype(np.float32)


def make_env(tmp_path, dur=12.0):
    work = tmp_path / "work"
    (work / "stems").mkdir(parents=True)
    (work / "tts").mkdir()
    video = tmp_path / "in.mp4"
    ffmpeg.run(["-f", "lavfi", "-i", f"testsrc2=size=160x90:rate=25:duration={dur}",
                "-f", "lavfi", "-i", f"sine=frequency=110:duration={dur}:sample_rate={SR}",
                "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(video)])
    # original audio = low hum everywhere + a loud 880 Hz "song" from 8..11 s
    hum = tone(dur, 110, amp=0.05)
    orig = hum.copy()
    orig[8 * SR:11 * SR] += tone(3.0, 880, amp=0.4)
    audio.write(work / "audio.wav", np.stack([orig, orig], 1), SR)
    # the "no_vocals" stem lacks the song entirely, so passthrough is only right if the mix uses the original there
    audio.write(work / "stems" / "no_vocals.wav", np.stack([hum, hum], 1), SR)
    audio.write(work / "stems" / "vocals.wav", np.zeros((int(dur * SR), 2), np.float32), SR)
    p = Project(source={"video": str(video), "duration": dur, "sample_rate": SR, "language": "en"}, target_language="hi")
    p.speakers["SPEAKER_00"] = Speaker(gender="male", gender_confidence=0.9, ref_clip=None)
    segs = [  # id, start, end, raw tts length
        (0, 1.0, 2.0, 1.1),   # slightly long: fits within spill (gap to next is 1.0 s, spill capped 0.25)
        (1, 3.0, 4.0, 1.4),   # needs 1.12x -> within 1.15, no flag
        (2, 5.0, 6.0, 1.5),   # needs 1.2x -> stretch_over_15
        (3, 6.5, 7.0, 1.2),   # needs 1.6x -> capped 1.25 + truncated
        (4, 11.2, 11.8, 0.3), # short: untouched
    ]
    for i, s, e, raw in segs:
        seg = Segment(id=i, speaker="SPEAKER_00", gender="male", start=s, end=e, text=f"t{i}", translation=f"x{i}",
                      tts=TTSInfo(engine="fake", wav=f"tts/{i:05d}.wav", raw_duration=raw))
        audio.write(work / seg.tts.wav, tone(raw, 440 + 100 * i), SR)
        p.segments.append(seg)
    p.segments.append(Segment(id=5, speaker="SPEAKER_00", gender="male", start=8.0, end=11.0, text="la la", type=MUSIC))
    opts = config.Options(video=video, target="hi", out=tmp_path / "out" / "dubbed.mp4", work=work, device="cpu", cpu=True)
    return p, work, opts


def test_fit_and_mix(tmp_path, monkeypatch):
    p, work, opts = make_env(tmp_path)
    monkeypatch.setattr(fit, "default_provider", lambda: None)
    monkeypatch.setattr(fit, "shorten", lambda *a, **k: None)
    fit.run(p, work, opts)
    by = {s.id: s for s in p.segments}
    assert by[0].tts.stretch == 1.0 and not by[0].flags
    assert 1.1 <= by[1].tts.stretch <= 1.15 and F_STRETCH_15 not in by[1].flags
    assert 1.15 < by[2].tts.stretch <= 1.25 and F_STRETCH_15 in by[2].flags and F_TRUNCATED not in by[2].flags
    assert by[3].tts.stretch == config.STRETCH_HARD and F_TRUNCATED in by[3].flags
    assert by[4].tts.stretch == 1.0
    for s in p.speech():
        assert audio.duration(work / s.tts.fitted_wav) <= fit.allowed_length(p, s) + 0.01
    # fit is incremental: a second run has nothing to do
    keys = {s.id: s.extra["fit_key"] for s in p.speech()}
    fit.run(p, work, opts)
    assert keys == {s.id: s.extra["fit_key"] for s in p.speech()}

    mix.run(p, work, opts)
    out = opts.out
    assert out.exists() and out.with_suffix(".srt").exists()
    assert abs(audio.duration(work / "mix.wav") - 12.0) <= config.LENGTH_TOLERANCE_S
    assert abs(ffmpeg.duration(out) - ffmpeg.duration(opts.video)) < 0.1

    # placement: energy of each clip's own frequency should rise within 150 ms of segment.start
    m, sr = audio.read(work / "mix.wav", sr=SR, mono=True)
    voice, _ = audio.read(work / "mix" / "voice.wav", sr=SR, mono=True)
    for s in p.speech():
        a = int(s.start * sr)
        before = np.abs(voice[max(0, a - int(0.15 * sr)):a]).max() if a > 0 else 0.0
        after = np.abs(voice[a:a + int(0.15 * sr)]).max()
        assert after > 0.1 and before < 0.01, f"segment {s.id} not placed at its start"
        nxt = p.next_start_after(s)
        if nxt is not None:
            b = int(nxt * sr)
            assert np.abs(voice[b - int(0.02 * sr):b]).max() < 0.01 or by[s.id].flags, f"segment {s.id} overlaps next"
    # music passthrough: in 8.2..10.8 s the mix should correlate with the ORIGINAL (880 Hz), not the halved stem
    o, _ = audio.read(work / "audio.wav", sr=SR, mono=True)
    a, b = int(8.2 * sr), int(10.8 * sr)
    seg_m, seg_o = m[a:b], o[a:b]
    corr = float(np.dot(seg_m, seg_o) / (np.linalg.norm(seg_m) * np.linalg.norm(seg_o) + 1e-9))
    assert corr > 0.999, corr

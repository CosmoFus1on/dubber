import pytest

from dubber import audio, ffmpeg


@pytest.fixture()
def clip(tmp_path):
    p = tmp_path / "t.mp4"
    ffmpeg.run(["-f", "lavfi", "-i", "testsrc2=size=160x90:rate=25:duration=3", "-f", "lavfi",
                "-i", "sine=frequency=440:duration=3:sample_rate=48000", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-t", "3", str(p)])
    return p


def test_extract_mux_roundtrip(clip, tmp_path):
    wav = ffmpeg.extract_audio(clip, tmp_path / "a.wav")
    assert abs(audio.duration(wav) - 3.0) < 0.1
    x, sr = audio.read(wav)
    assert sr == 48000 and x.shape[1] == 2
    out = ffmpeg.mux(clip, wav, tmp_path / "o.mp4", keep_original_audio=True)
    info = ffmpeg.probe(out)
    assert [s["codec_type"] for s in info["streams"]].count("audio") == 2
    assert abs(ffmpeg.duration(out) - ffmpeg.duration(clip)) < 0.05


def test_tempo_and_loudnorm(clip, tmp_path):
    wav = ffmpeg.extract_audio(clip, tmp_path / "a.wav")
    fast = ffmpeg.rubberband(wav, tmp_path / "f.wav", 1.5)
    assert abs(audio.duration(fast) - 2.0) < 0.1
    ffmpeg.loudnorm(wav, tmp_path / "n.wav")
    assert (tmp_path / "n.wav").exists()

from dubber.schema import Segment, TTSInfo, Verify
from dubber.segment import carry_over


def seg(i, start, text, **kw):
    return Segment(id=i, speaker="A", gender="male", start=start, end=start + 1, text=text, **kw)


def test_resegmentation_keeps_translations_for_unchanged_lines():
    old = [seg(0, 0.0, "hello there", translation="नमस्ते", verify=Verify(passed=True, attempts=1),
               tts=TTSInfo(engine="x", wav="tts/00000.wav"), flags=["overlap"]),
           seg(1, 2.0, "bye", translation="अलविदा")]
    new = [seg(0, 0.02, "hello there"), seg(1, 2.0, "bye now"), seg(2, 5.0, "new line")]
    assert carry_over(old, new) == 1
    assert new[0].translation == "नमस्ते" and new[0].verify.passed and new[0].tts.wav == "tts/00000.wav"
    assert new[0].flags == ["overlap"]
    assert new[1].translation is None and new[2].translation is None

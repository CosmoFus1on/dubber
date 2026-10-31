from dubber.report import transcript_text
from dubber.schema import MUSIC, Project, Segment, Speaker


def test_transcript_format():
    p = Project(source={"video": "x.mp4", "duration": 10, "sample_rate": 48000, "language": "en"}, target_language="hi")
    p.speakers = {"SPEAKER_01": Speaker(gender="male"), "SPEAKER_00": Speaker(gender="female")}
    p.segments = [
        Segment(id=0, speaker="SPEAKER_01", gender="male", start=0, end=1, text="Hi there.", translation="नमस्ते।"),
        Segment(id=1, speaker="SPEAKER_00", gender="female", start=1, end=2, text="Hello!", translation="हैलो!"),
        Segment(id=2, speaker="MUSIC", gender="unknown", start=2, end=4, type=MUSIC, text="la la"),
        Segment(id=3, speaker="SPEAKER_01", gender="male", start=4, end=5, text="Bye.", translation=None),
    ]
    assert transcript_text(p) == (
        "Person 1 (male): [Hi there.]\nPerson 1 (male): [नमस्ते।]\n\n"
        "Person 2 (female): [Hello!]\nPerson 2 (female): [हैलो!]\n\n"
        "Music: [la la]\n\n"
        "Person 1 (male): [Bye.]\nPerson 1 (male): []\n"
    )

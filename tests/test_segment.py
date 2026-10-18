from dubber import config
from dubber.schema import Speaker
from dubber.segment import build


def words(spec):
    return [{"w": " " + w, "s": s, "e": e, "p": 1.0} for w, s, e in spec]


TURNS = [{"speaker": "A", "start": 0, "end": 100}]
SP = {"A": Speaker(gender="male"), "B": Speaker(gender="female")}


def test_pause_split():
    gap = config.MERGE_PAUSE_S + 0.2
    t = {"words": words([("hello", 0, 0.4), ("there", 0.5, 0.9), ("again", 0.9 + gap, 1.5 + gap)])}
    segs = build(t, TURNS, SP)
    assert [s.text for s in segs] == ["hello there", "again"]
    assert segs[0].gender == "male" and [s.id for s in segs] == [0, 1]


def test_cap_split_prefers_sentence_end():
    ws = [(f"w{i}.", i, i + 0.9) if i == 5 else (f"w{i}", i, i + 0.9) for i in range(16)]
    segs = build({"words": words(ws)}, TURNS, SP)
    assert segs[0].text.endswith("w5.")
    assert all(s.end - s.start <= config.SEGMENT_CAP_S for s in segs)


def test_speaker_change():
    turns = [{"speaker": "A", "start": 0, "end": 1.0}, {"speaker": "B", "start": 1.0, "end": 5}]
    t = {"words": words([("hi", 0, 0.4), ("you", 0.45, 0.9), ("hello", 1.05, 1.5)])}
    segs = build(t, turns, SP)
    assert [(s.speaker, s.gender) for s in segs] == [("A", "male"), ("B", "female")]


def test_sentence_split():
    # 5 s sentence, short gap, next sentence: splits at the full stop (gap >= SENTENCE_GAP_S, length >= SENTENCE_MIN_S)
    ws = [(f"a{i}", i, i + 0.9) for i in range(5)] + [("end.", 5, 5.5), ("next", 5.8, 6.2), ("one", 6.3, 6.7)]
    segs = build({"words": words(ws)}, TURNS, SP)
    assert [s.text.split()[-1] for s in segs] == ["end.", "one"]
    # same punctuation but the segment is still short: no split
    short = [("hi.", 0, 0.4), ("there", 0.8, 1.2)]
    assert len(build({"words": words(short)}, TURNS, SP)) == 1


def test_overlap_flag():
    from dubber.segment import flag_overlaps
    t = {"words": words([("hi", 0, 1.0), ("you", 1.1, 2.0), ("yes", 3.0, 4.0)])}
    exc = [{"speaker": "A", "start": 0, "end": 2.0}, {"speaker": "B", "start": 2.5, "end": 5}]
    segs = build(t, exc, SP)
    flag_overlaps(segs, exc + [{"speaker": "B", "start": 1.2, "end": 2.0}])
    assert segs[0].flags == ["overlap"] and segs[0].extra["overlaps_with"] == "B" and not segs[1].flags

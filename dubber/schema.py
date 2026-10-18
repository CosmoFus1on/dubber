"""segments.json contract. Every stage reads and writes through this module.

Human-editable fields on a segment: speaker, gender, start, end, text, type, translation.
A hand-edited translation (verify.attempts == -1) counts as verified.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

STAGES = ["separate", "transcribe", "speakers", "segment", "translate", "tts", "fit", "mix"]
SPEECH, MUSIC = "speech", "music"
GENDERS = ("male", "female", "unknown")

# flag names (also aggregated into report.json)
F_LOW_GENDER = "low_gender_confidence"
F_VERIFY_FAILED = "verify_failed"
F_STRETCH_15 = "stretch_over_15"
F_TRUNCATED = "truncated"
F_NO_REF = "no_ref_clip"
F_OVERLAP = "overlap"
F_TTS_FALLBACK = "tts_fallback:"  # + engine name
F_TTS_FAILED = "tts_failed"


def sha1(*parts: Any) -> str:
    """Stable content hash. Paths hash their bytes; everything else hashes as sorted JSON."""
    h = hashlib.sha1()
    for p in parts:
        if isinstance(p, Path):
            h.update(p.read_bytes() if p.exists() else b"<missing>")
        else:
            h.update(json.dumps(p, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
    return h.hexdigest()


@dataclass
class Word:
    w: str
    s: float
    e: float
    p: float = 1.0  # whisper word probability


@dataclass
class Verify:
    back_translation: Optional[str] = None
    judge: Optional[float] = None
    fluency: Optional[float] = None
    sim: Optional[float] = None
    passed: Optional[bool] = None
    attempts: int = 0
    issue: Optional[str] = None


@dataclass
class TTSInfo:
    engine: Optional[str] = None
    wav: Optional[str] = None          # raw synthesis, relative to work dir
    raw_duration: Optional[float] = None
    stretch: Optional[float] = None    # tempo factor applied in fit (1.0 = none)
    fitted_wav: Optional[str] = None   # after fit, relative to work dir
    shorten_rounds: int = 0


@dataclass
class Segment:
    id: int
    speaker: str
    gender: str
    start: float
    end: float
    type: str = SPEECH
    text: str = ""
    words: list[Word] = field(default_factory=list)
    translation: Optional[str] = None
    budget_chars: Optional[int] = None
    verify: Verify = field(default_factory=Verify)
    tts: TTSInfo = field(default_factory=TTSInfo)
    flags: list[str] = field(default_factory=list)
    # whisper stats and detector scores land here (avg_logprob, compression_ratio, no_speech_prob, music_score ...)
    extra: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def flag(self, name: str) -> None:
        if name not in self.flags:
            self.flags.append(name)

    def unflag_prefix(self, prefix: str) -> None:
        self.flags = [f for f in self.flags if not f.startswith(prefix)]

    # cache keys: the only thing downstream stages look at to decide "redo this segment?"
    def translate_key(self, context_hash: str, lang: str, model: str) -> str:
        return sha1(self.text, context_hash, lang, model, self.budget_chars)

    def tts_key(self, engine: str, ref_hash: str, hints: dict) -> str:
        return sha1(self.translation, engine, ref_hash, hints)

    @staticmethod
    def from_dict(d: dict) -> "Segment":
        d = dict(d)
        d["words"] = [Word(**w) for w in d.get("words", [])]
        d["verify"] = Verify(**d.get("verify", {}))
        d["tts"] = TTSInfo(**d.get("tts", {}))
        return Segment(**d)


@dataclass
class Speaker:
    gender: str = "unknown"
    gender_confidence: float = 0.0
    f0_median: Optional[float] = None
    ina_male_ratio: Optional[float] = None
    ref_clip: Optional[str] = None     # relative to work dir
    flags: list[str] = field(default_factory=list)


@dataclass
class Project:
    source: dict                       # {video, duration, sample_rate, language}
    target_language: str
    version: int = 1
    speakers: dict[str, Speaker] = field(default_factory=dict)
    segments: list[Segment] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)  # stage -> wall-clock seconds

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Project":
        return Project(
            source=d["source"],
            target_language=d["target_language"],
            version=d.get("version", 1),
            speakers={k: Speaker(**v) for k, v in d.get("speakers", {}).items()},
            segments=[Segment.from_dict(s) for s in d.get("segments", [])],
            timings=d.get("timings", {}),
        )

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)  # atomic rename on Windows and POSIX

    @staticmethod
    def load(path: Path) -> "Project":
        # utf-8-sig: Notepad and PowerShell 5.1 write a BOM when a human edits the file
        return Project.from_dict(json.loads(path.read_text(encoding="utf-8-sig")))

    def speech(self) -> list[Segment]:
        return [s for s in self.segments if s.type == SPEECH]

    def sorted_segments(self) -> list[Segment]:
        return sorted(self.segments, key=lambda s: (s.start, s.id))

    def next_start_after(self, seg: Segment) -> Optional[float]:
        """Start of the next segment that begins at or after seg.end (any speaker); None if last."""
        later = [s.start for s in self.segments if s.id != seg.id and s.start >= seg.end]
        return min(later) if later else None


if __name__ == "__main__":  # self-check
    p = Project(source={"video": "x.mp4", "duration": 1.0, "sample_rate": 48000, "language": "en"}, target_language="hi")
    p.segments.append(Segment(id=0, speaker="SPEAKER_00", gender="male", start=0.0, end=0.5, text="hi", words=[Word("hi", 0, 0.5)]))
    p.segments.append(Segment(id=1, speaker="SPEAKER_01", gender="female", start=0.6, end=1.0, text="yo"))
    tmp = Path("_schema_selfcheck.json")
    p.save(tmp)
    q = Project.load(tmp)
    tmp.unlink()
    assert q.to_dict() == p.to_dict()
    assert q.next_start_after(q.segments[0]) == 0.6 and q.next_start_after(q.segments[1]) is None
    k1 = q.segments[0].translate_key("ctx", "hi", "m")
    q.segments[0].text = "hello"
    assert q.segments[0].translate_key("ctx", "hi", "m") != k1
    print("schema ok")

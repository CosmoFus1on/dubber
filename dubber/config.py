"""All tunables in one place. Env overrides come from .env (loaded here) or the process environment."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent

# ---- folders. Visible: input/, output/, benchmarks/. Everything generated hides under .cache/ and .envs/ ----
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
BENCH_DIR = ROOT / "benchmarks"
TRANSCRIPTS_DIR = ROOT / "transcripts"   # human-readable "Person N (gender): line / translation" per run
CACHE_ROOT = ROOT / ".cache"
WORK_DIR = CACHE_ROOT / "work"          # per-video working dirs: .cache/work/<stem>/
MODELS_DIR = CACHE_ROOT / "models"      # engine-downloaded weights/voices
VOICES_DIR = CACHE_ROOT / "voices"      # Piper voices used to build the test clips


def load_env(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines). Values are never logged."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            if v:  # an empty value must not shadow a real one (and "Bearer " breaks huggingface_hub)
                os.environ.setdefault(k.strip(), v)


load_env()

# ---- LLM (FreeLLMAPI or any OpenAI-compatible /v1) ----
LLM_BASE_URL = os.environ.get("DUBBER_LLM_BASE_URL", "http://localhost:31415/v1")
LLM_KEY = os.environ.get("DUBBER_LLM_KEY", "")
# One strong open-weight model pinned for translation so style stays consistent across a video.
# Chosen 2026-10-18 from what the proxy could actually route (Groq/OpenRouter/Google keys configured).
LLM_MODEL = os.environ.get("DUBBER_LLM_MODEL") or "nemotron-3-ultra-550b"
LLM_FALLBACK_CHAIN = ["nemotron-3-super-120b", "qwen3.8-27b", "gemma-4-31b-it", "gemini-2.5-flash", "auto"]
LLM_FAST = os.environ.get("DUBBER_LLM_FAST") or "auto:fast"   # judge + back-translation
LLM_FAST_FALLBACK = ["nemotron-3-super-120b", "auto", "qwen3.8-27b"]
LLM_TEMPERATURE = 0.3
LLM_BACKOFF_S = [1, 2, 4, 8, 16]
LLM_TIMEOUT_S = 120
LLM_PARALLEL = 4                # concurrent translate windows / verify calls (free tiers throttle above this)
CACHE_DIR = Path(os.environ.get("DUBBER_CACHE", Path.home() / ".cache" / "dubber"))

# ---- models ----
WHISPER_GPU = "large-v3"
WHISPER_CPU = "distil-large-v3"
DEMUCS_MODEL = "htdemucs"
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
EMBEDDER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ---- segmentation ----
MERGE_PAUSE_S = 0.6
SEGMENT_CAP_S = 12.0
MUSIC_VOTES_NEEDED = 2          # of 3 detectors
MUSIC_HOP_S = 0.1               # vote grid
MUSIC_SUSTAIN_S = 0.5           # stem levels are power-averaged over this window before thresholding
MUSIC_MIN_S = 2.0               # music runs shorter than this are dropped (after bridging)
MUSIC_BRIDGE_S = 1.0            # gaps shorter than this between music runs are bridged
MUSIC_BED_FLOOR_DB = -35.0      # energy detector: bed (no_vocals) must be at least this loud
MUSIC_BED_MARGIN_DB = 12.0      # ... and no more than this below the vocals stem
MUSIC_VOCALS_MIN_DB = -45.0     # whisper detector: vocals energy with no words = singing Whisper skipped
MUSIC_BED_RISE_DB = 3.0         # ... or the bed this far above its median level under transcribed words
MUSIC_LOGPROB_MAX = -0.8        # whisper segment stats that mean "not confident speech"
MUSIC_COMPRESSION_MIN = 2.4
MUSIC_NO_SPEECH_MIN = 0.6
MUSIC_NGRAM_REPEATS = 3         # a word 3-gram repeated this often in one whisper segment (chorus)
MUSIC_SEGMENT_FRAC = 0.6        # a speech segment more than this inside music regions becomes music
SENTENCE_GAP_S = 0.15           # split after sentence punctuation when the gap is at least this (whisper reports real 0.4 s pauses as 0.20-0.28 s)
SENTENCE_MIN_S = 4.0            # ... and the running segment is already this long
OVERLAP_MIN_S = 0.3             # another speaker's turn overlapping a segment by more than this flags it

# ---- speakers ----
REF_CLIP_MIN_S = 10.0
REF_CLIP_MAX_S = 30.0
GENDER_CONF_MIN = 0.6
F0_MALE_MAX_HZ = 165.0          # ponytail: single F0 threshold; a logistic fit on (F0, ina ratio) is the upgrade

# ---- translation budget: characters per second of speech, per target language ----
# ponytail: chars/s table; per-language syllable counting is the upgrade path.
CPS = {
    "en": 15, "es": 16, "de": 14, "fr": 15, "it": 15, "pt": 15, "nl": 14, "pl": 13, "ru": 13, "tr": 14,
    "sv": 14, "da": 14, "no": 14, "fi": 13, "el": 13, "he": 11, "ar": 12, "hi": 15, "ms": 15, "sw": 14,
    "ja": 7, "zh": 5.5, "ko": 9,
}
CPS_DEFAULT = 13
BUDGET_FACTOR = 1.0             # budget = what fits the slot at natural pace; the +15 % tempo headroom absorbs overshoot
PRESHORTEN_OVER = 1.3           # translation longer than budget * this is shortened by text before any TTS
FLUENCY_MIN = 0.7               # judge's native-fluency score of the target text
TRANSLATE_WINDOW = 20           # segments per LLM request
CONTEXT_FULL_MAX = 600          # above this, send +/- CONTEXT_HALF segments instead of the full transcript
CONTEXT_HALF = 40

# ---- verification ----
JUDGE_MIN = 0.7
SIM_MIN = 0.75
VERIFY_MAX_ATTEMPTS = 3

# ---- timing fit ----
SHORTEN_ROUNDS = 2
STRETCH_SOFT = 1.15
STRETCH_HARD = 1.25
SPILL_MAX_S = 0.5               # a clip may run this far into the silence after its slot, never into the next segment
BABBLE_RATIO = 1.6              # TTS clip longer than text-length-implies * this -> re-synthesize once
FADE_MS = 40

# ---- mix ----
DUCK_DB = -6.0
TARGET_LUFS = -16.0
MUSIC_XFADE_MS = 50
LENGTH_TOLERANCE_S = 0.020
MIX_SR = 48000

# ---- TTS ----
# Per-language engine preference. Only engines that list the language are tried; "piper" never clones.
TTS_FALLBACK = {
    "hi": ["chatterbox", "xtts", "piper"],
    "ms": ["chatterbox"],
    "zh": ["chatterbox", "xtts", "openvoice", "f5", "piper"],
    "ja": ["chatterbox", "xtts", "openvoice", "piper"],
    "en": ["chatterbox", "xtts", "openvoice", "f5", "piper"],
}
TTS_DEFAULT_CHAIN = ["chatterbox", "xtts", "openvoice", "piper"]
TTS_CPU_CHAIN = ["piper", "chatterbox"]
ENGINE_VENVS = ROOT / ".envs"   # .envs/<engine>/Scripts/python.exe when an engine needs its own pins

# ---- language aliases (whisper/pyannote codes vs engine codes) ----
LANG_ALIASES = {"zh-cn": "zh", "zh-tw": "zh", "cmn": "zh", "pt-br": "pt", "jw": "jv"}


@dataclass
class Options:
    """Per-run options; built by the CLI and passed to every stage's run(project, work, opts)."""
    video: Path
    target: str                       # normalized 2-letter target language
    out: Path
    work: Path                        # .cache/work/<stem>
    device: str = "cuda"              # "cuda" or "cpu"
    cpu: bool = False
    keep_original_audio: bool = False
    glossary: Optional[Path] = None   # text file, one term per line ("Term" or "Term=Translation")
    num_speakers: Optional[int] = None
    demo_fast: bool = False           # hosted demo: 1 verify attempt, no shorten rounds
    force: bool = False               # set by the runner for --from: incremental stages ignore per-segment caches


def norm_lang(code: str) -> str:
    code = (code or "").lower().replace("_", "-")
    return LANG_ALIASES.get(code, code.split("-")[0])


def cps(lang: str) -> float:
    return CPS.get(norm_lang(lang), CPS_DEFAULT)

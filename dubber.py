"""dubber CLI and stage runner:  python dubber.py input.mp4 --to hi"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import time
from pathlib import Path

from dubber import config, gpu, llm, report
from dubber.schema import STAGES, Project, sha1

log = logging.getLogger("dubber.runner")

MODULES = {"separate": "dubber.separate", "transcribe": "dubber.transcribe", "speakers": "dubber.speakers",
           "segment": "dubber.segment", "translate": "dubber.translate", "tts": "dubber.synth",
           "fit": "dubber.fit", "mix": "dubber.mix"}


def _stat(path: Path):
    """(size, mtime_ns) of a file, or None if missing."""
    try:
        st = path.stat()
        return [st.st_size, st.st_mtime_ns]
    except OSError:
        return None


def input_hash(stage: str, work: Path, opts, device: str) -> str:
    v, stems, tr = opts.video, work / "stems", work / "transcript.json"
    if stage == "separate":
        return sha1(str(v.resolve()), _stat(v))
    if stage == "transcribe":
        from dubber.transcribe import model_name
        return sha1(_stat(stems / "vocals.wav"), model_name(device))
    if stage == "speakers":
        return sha1(_stat(stems / "vocals.wav"), _stat(stems / "no_vocals.wav"), _stat(tr), opts.num_speakers)
    if stage == "segment":
        return sha1(_stat(tr), _stat(work / "diarization.json"))
    return "incremental"


def outputs_exist(stage: str, work: Path, project: Project) -> bool:
    if stage == "separate":
        return all((work / "stems" / f"{n}.wav").exists() for n in ("vocals", "no_vocals"))
    if stage == "transcribe":
        return (work / "transcript.json").exists()
    if stage == "speakers":
        return (work / "diarization.json").exists()
    if stage == "segment":
        return bool(project.segments)
    return False  # translate/tts/fit/mix decide per segment themselves and always run


def setup_logging(verbose: bool) -> None:
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, force=True,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="dubber", description="Free local AI video dubbing")
    ap.add_argument("input", type=Path)
    ap.add_argument("--to", required=True, help="target language code, e.g. hi")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--work", type=Path)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--from", dest="from_stage", choices=STAGES, help="force this stage and all later ones")
    ap.add_argument("--only", choices=STAGES, help="run just this stage (combine with --from to force it)")
    ap.add_argument("--keep-original-audio", action="store_true")
    ap.add_argument("--glossary", type=Path)
    ap.add_argument("--num-speakers", type=int)
    ap.add_argument("--source-lang")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    setup_logging(a.verbose)
    video = a.input.resolve()
    lang = config.norm_lang(a.to)
    work = (a.work or Path("work") / video.stem).resolve()
    out = a.out or Path("out") / f"{video.stem}_{lang}.mp4"
    work.mkdir(parents=True, exist_ok=True)
    device = gpu.device(a.cpu)
    opts = config.Options(video=video, target=lang, out=out, work=work, device=device, cpu=a.cpu,
                          keep_original_audio=a.keep_original_audio, glossary=a.glossary,
                          num_speakers=a.num_speakers)

    seg_path, state_path = work / "segments.json", work / "state.json"
    if seg_path.exists():
        project = Project.load(seg_path)
        project.target_language = lang
    else:
        project = Project(source={"video": str(video), "duration": None, "sample_rate": 48000, "language": None},
                          target_language=lang)
    if a.source_lang:
        project.source["language"] = config.norm_lang(a.source_lang)
    state = json.loads(state_path.read_text(encoding="utf-8-sig")) if state_path.exists() else {}
    if not seg_path.exists():
        # a missing segments.json is a reset: the stages that write into the Project must run again
        for st in ("separate", "transcribe", "speakers"):
            state.pop(st, None)

    forced = False
    for stage in STAGES:
        forced = forced or stage == a.from_stage
        if a.only and stage != a.only:
            continue
        opts.force = forced
        h = input_hash(stage, work, opts, device)
        st = state.get(stage, {})
        if not forced and st.get("input_hash") == h and st.get("done") and outputs_exist(stage, work, project):
            log.info("[%s] up to date, skipping", stage)
            continue
        log.info("[%s] running", stage)
        t0 = time.perf_counter()
        llm_before = llm.STATS["calls"]
        state[stage] = {"input_hash": h, "done": False, "seconds": 0.0}
        importlib.import_module(MODULES[stage]).run(project, work, opts)
        secs = round(time.perf_counter() - t0, 2)
        # hash again: a stage may legitimately change its own inputs (e.g. separate sets source.duration)
        state[stage] = {"input_hash": input_hash(stage, work, opts, device), "done": True, "seconds": secs,
                        "llm_calls": llm.STATS["calls"] - llm_before}
        project.timings[stage] = secs
        project.save(seg_path)
        state_path.write_text(json.dumps(state, indent=1), encoding="utf-8")
        gpu.unload()
        log.info("[%s] done in %.1fs", stage, secs)

    project.save(seg_path)
    report.write(project, work, opts, state)
    flagged = sum(1 for s in project.segments if s.flags)
    print(f"output: {opts.out}")
    print(f"flagged segments: {flagged}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

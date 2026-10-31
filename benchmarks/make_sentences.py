"""Build benchmarks/sentences.json once: first 5 lines of tests/clip_truth.json + LLM translations (hi, ms, ja, zh).

    .venv\Scripts\python.exe -X utf8 bench\make_sentences.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dubber import config, prompts  # noqa: E402
from dubber.llm import default_provider  # noqa: E402

GENDER = {"A": "male", "B": "female"}
truth = json.loads((ROOT / "tests" / "clip_truth.json").read_text(encoding="utf-8"))
lines = truth["lines"][:5]
sents = [{"id": i, "speaker": l["speaker"], "gender": GENDER[l["speaker"]], "slot": round(l["end"] - l["start"], 3),
          "text": {"en": l["text"]}} for i, l in enumerate(lines)]
llm = default_provider()
ctx = [f"{s['id']} | {s['speaker']} | {s['gender']} | {s['text']['en']}" for s in sents]
for tgt in ["hi", "ms", "ja", "zh"]:
    req = [{"id": s["id"], "speaker": s["speaker"], "gender": s["gender"],
            "max_chars": int(s["slot"] * config.cps(tgt)), "text": s["text"]["en"]} for s in sents]
    data = llm.chat_json(prompts.translate_messages("en", tgt, {}, ctx, req), config.LLM_MODEL,
                         fallback=config.LLM_FALLBACK_CHAIN)
    items = data.get("translations", data) if isinstance(data, dict) else data
    got = {int(it["id"]): str(it["translation"]).strip() for it in items}
    for s in sents:
        s["text"][tgt] = got[s["id"]]
out = ROOT / "bench" / "sentences.json"
out.write_text(json.dumps({"model": config.LLM_MODEL, "sentences": sents}, ensure_ascii=False, indent=1), encoding="utf-8")
print(out.read_text(encoding="utf-8"))

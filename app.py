"""Desktop UI for dubber (tkinter, no extra dependency):  python app.py

Pick a video, pick where to save, pick the language, watch the stages and the log, cancel if needed.
Runs dubber.py as a subprocess so the CLI, the UI and the tests all share one code path.
"""
from __future__ import annotations

import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from dubber import config
from dubber.prompts import LANG_NAMES
from dubber.schema import STAGES
from dubber.tts.chatterbox import LANGS

ROOT = Path(__file__).resolve().parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
LANG_LABELS = {f"{LANG_NAMES.get(c, c)} ({c})": c for c in sorted(LANGS, key=lambda c: LANG_NAMES.get(c, c))}
STAGE_RE = re.compile(r"\[(\w+)\] (running|done in|up to date)")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("dubber")
        self.geometry("860x620")
        self.proc: subprocess.Popen | None = None
        self.q: queue.Queue[str] = queue.Queue()
        self.video = tk.StringVar()
        self.out = tk.StringVar()
        self.lang = tk.StringVar(value="Hindi (hi)")
        self.keep = tk.BooleanVar(value=False)
        self.build()
        self.after(100, self.pump)

    def build(self):
        f = ttk.Frame(self, padding=10)
        f.pack(fill="x")
        f.columnconfigure(1, weight=1)
        ttk.Label(f, text="Video").grid(row=0, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.video).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(f, text="Browse...", command=self.pick_video).grid(row=0, column=2)
        ttk.Label(f, text="Save as").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(f, textvariable=self.out).grid(row=1, column=1, sticky="ew", padx=6)
        ttk.Button(f, text="Choose...", command=self.pick_out).grid(row=1, column=2)
        ttk.Label(f, text="Dub into").grid(row=2, column=0, sticky="w")
        ttk.Combobox(f, textvariable=self.lang, values=list(LANG_LABELS), state="readonly", width=28).grid(row=2, column=1, sticky="w", padx=6)
        ttk.Checkbutton(f, text="keep original audio as a second track", variable=self.keep).grid(row=3, column=1, sticky="w", padx=6)
        b = ttk.Frame(f)
        b.grid(row=4, column=1, sticky="w", padx=6, pady=8)
        self.start_btn = ttk.Button(b, text="Dub", command=self.start)
        self.start_btn.pack(side="left")
        self.cancel_btn = ttk.Button(b, text="Cancel", command=self.cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=6)

        p = ttk.LabelFrame(self, text="Progress", padding=10)
        p.pack(fill="x", padx=10)
        self.stage_labels = {}
        for i, st in enumerate(STAGES):
            lbl = ttk.Label(p, text=f"  {st}  ", relief="groove", anchor="center", width=12)
            lbl.grid(row=0, column=i, padx=2)
            self.stage_labels[st] = lbl
        self.bar = ttk.Progressbar(p, maximum=len(STAGES), length=820)
        self.bar.grid(row=1, column=0, columnspan=len(STAGES), pady=(8, 0), sticky="ew")
        self.status = ttk.Label(p, text="idle")
        self.status.grid(row=2, column=0, columnspan=len(STAGES), sticky="w", pady=(6, 0))

        self.log = tk.Text(self, height=18, wrap="word", state="disabled", font=("Consolas", 9))
        self.log.pack(fill="both", expand=True, padx=10, pady=10)

    # ---- pickers ----
    def pick_video(self):
        p = filedialog.askopenfilename(title="Video to dub", initialdir=config.INPUT_DIR,
                                       filetypes=[("Video", "*.mp4 *.mkv *.mov *.webm *.avi"), ("All", "*.*")])
        if p:
            self.video.set(p)
            if not self.out.get():
                self.out.set(str(config.OUTPUT_DIR / f"{Path(p).stem}_{self.code()}.mp4"))

    def pick_out(self):
        p = filedialog.asksaveasfilename(title="Save dubbed video as", initialdir=config.OUTPUT_DIR,
                                         initialfile=f"{Path(self.video.get() or 'dubbed').stem}_{self.code()}.mp4",
                                         defaultextension=".mp4", filetypes=[("MP4", "*.mp4")])
        if p:
            self.out.set(p)

    def code(self) -> str:
        return LANG_LABELS.get(self.lang.get(), "hi")

    # ---- run ----
    def start(self):
        if self.proc:
            return
        video, out = self.video.get().strip(), self.out.get().strip()
        if not video or not Path(video).exists():
            messagebox.showerror("dubber", "pick a video first")
            return
        if not out:
            out = str(config.OUTPUT_DIR / f"{Path(video).stem}_{self.code()}.mp4")
            self.out.set(out)
        cmd = [str(PY if PY.exists() else sys.executable), "-X", "utf8", str(ROOT / "dubber.py"), video,
               "--to", self.code(), "--out", out]
        if self.keep.get():
            cmd.append("--keep-original-audio")
        for lbl in self.stage_labels.values():
            lbl.configure(background="")
        self.bar["value"] = 0
        self.clear_log()
        self.append(" ".join(cmd) + "\n")
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace", bufsize=1)
        self.start_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.status.configure(text="starting...")
        threading.Thread(target=self.reader, daemon=True).start()

    def reader(self):
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            self.q.put(line)
        self.q.put(f"__exit__ {self.proc.wait()}")

    def pump(self):
        try:
            while True:
                line = self.q.get_nowait()
                if line.startswith("__exit__"):
                    self.finish(int(line.split()[1]))
                else:
                    self.handle(line)
        except queue.Empty:
            pass
        self.after(100, self.pump)

    def handle(self, line: str):
        m = STAGE_RE.search(line)
        if m:
            st, what = m.group(1), m.group(2)
            i = STAGES.index(st) if st in STAGES else -1
            if what == "running":
                self.stage_labels[st].configure(background="#ffe08a")
                self.status.configure(text=f"{st}...")
            else:
                self.stage_labels[st].configure(background="#9be39b")
                self.bar["value"] = i + 1
        if "translated" in line or "engine " in line or "fit:" in line or "placed" in line:
            self.status.configure(text=line.split(": ", 2)[-1].strip()[:110])
        if not any(x in line for x in ("httpx", "triton", "Redirects", "flop_counter", "Loading weights")):
            self.append(line)

    def finish(self, code: int):
        self.proc = None
        self.start_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        if code == 0:
            self.status.configure(text=f"done: {self.out.get()}")
            messagebox.showinfo("dubber", f"Dubbed video saved to\n{self.out.get()}\n\n"
                                f"Transcript in {config.TRANSCRIPTS_DIR}")
        elif code == -1 or code == 1 and "cancel" in self.status.cget("text"):
            self.status.configure(text="cancelled")
        else:
            self.status.configure(text=f"failed (exit {code}), see log")

    def cancel(self):
        if self.proc:
            self.status.configure(text="cancelling...")
            subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"], capture_output=True)

    # ---- log ----
    def append(self, text: str):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")


if __name__ == "__main__":
    App().mainloop()

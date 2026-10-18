# dubber setup (Windows PowerShell 5.1). Idempotent: safe to re-run.
#   .\setup.ps1                       main venv, GPU torch
#   .\setup.ps1 -Cpu                  CPU-only torch
#   .\setup.ps1 -Engines chatterbox,xtts,openvoice,f5   also create envs\<engine> venvs
param(
    [switch]$Cpu,
    [string[]]$Engines = @()
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$Py310 = Join-Path $env:LOCALAPPDATA "Programs\Python\Python310\python.exe"
$Venv = Join-Path $PSScriptRoot ".venv"
$VenvPy = Join-Path $Venv "Scripts\python.exe"

function Run-Native {
    # run a native command, fail loudly on non-zero exit
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "failed: $Exe $($Arguments -join ' ')" }
}

# ---- engine venvs: one obvious place to adjust per-engine installs ----
# Each value is a list of argument lists for envs\<engine>\Scripts\python.exe (one call per entry).
# torch 2.8.0 for xtts/openvoice/f5: torch 2.9+ routes torchaudio through torchcodec, which needs a shared
# FFmpeg build (the usual winget/gyan.dev build is static). Piper runs in the main venv (requirements.txt).
$Cu128 = "https://download.pytorch.org/whl/cu128"
$EngineSpecs = @{
    "chatterbox" = @(, @("-m", "pip", "install", "chatterbox-tts", "--extra-index-url", "https://download.pytorch.org/whl/cu126"))  # pins torch 2.6.0
    "xtts"       = @(
        @("-m", "pip", "install", "torch==2.8.0", "torchaudio==2.8.0", "--index-url", $Cu128),
        @("-m", "pip", "install", "coqui-tts[ja,zh,ko]==0.27.5", "transformers>=4.57,<5"))  # 0.27.5 breaks on transformers 5
    "openvoice"  = @(
        @("-m", "pip", "install", "torch==2.8.0", "torchaudio==2.8.0", "--index-url", $Cu128),
        @("-m", "pip", "install", "git+https://github.com/myshell-ai/MeloTTS.git"),
        @("-m", "pip", "install", "--no-deps", "git+https://github.com/myshell-ai/OpenVoice.git"),  # its own pins are stale; Melo brings what it needs
        @("-m", "unidic", "download"),
        @("-c", "import nltk, sys; [nltk.download(r, download_dir=sys.prefix + '/nltk_data', quiet=True) for r in ('averaged_perceptron_tagger_eng', 'cmudict')]"))
    "f5"         = @(
        @("-m", "pip", "install", "torch==2.8.0", "torchaudio==2.8.0", "--index-url", $Cu128),
        @("-m", "pip", "install", "f5-tts"),
        @("-m", "pip", "uninstall", "-y", "torchcodec"))  # f5-tts pulls a torchcodec built for newer torch; transformers then falls back to the ffmpeg CLI
}

# ---- main venv ----
if (-not (Test-Path $VenvPy)) {
    if (-not (Test-Path $Py310)) { throw "Python 3.10 not found at $Py310" }
    Write-Host "Creating .venv"
    Run-Native $Py310 @("-m", "venv", $Venv)
}
Run-Native $VenvPy @("-m", "pip", "install", "--upgrade", "pip")

if ($Cpu) {
    Run-Native $VenvPy @("-m", "pip", "install", "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu")
} else {
    Run-Native $VenvPy @("-m", "pip", "install", "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu128")
}
Run-Native $VenvPy @("-m", "pip", "install", "-r", "requirements.txt")
Run-Native $VenvPy @("-m", "pip", "install", "nvidia-cudnn-cu12")
Run-Native $VenvPy @("-m", "pip", "install", "--no-deps", "resemblyzer")  # bench only; its deps would pull the py2 `typing` backport and a webrtcvad sdist

# ---- Piper test voices ----
$Voices = Join-Path $PSScriptRoot "tests\voices"
New-Item -ItemType Directory -Force -Path $Voices | Out-Null
Run-Native $VenvPy @("-m", "piper.download_voices", "--download-dir", $Voices, "en_US-ryan-medium", "en_US-lessac-medium")

# ---- engine venvs ----
foreach ($e in $Engines) {
    if (-not $EngineSpecs.ContainsKey($e)) { Write-Warning "unknown engine '$e'"; continue }
    $dir = Join-Path $PSScriptRoot "envs\$e"
    $epy = Join-Path $dir "Scripts\python.exe"
    if (-not (Test-Path $epy)) {
        Write-Host "Creating envs\$e"
        Run-Native $Py310 @("-m", "venv", $dir)
    }
    Run-Native $epy @("-m", "pip", "install", "--upgrade", "pip")
    foreach ($spec in $EngineSpecs[$e]) {
        Run-Native $epy $spec
    }
}

# ---- smoke test ----
Write-Host "CUDA smoke test:"
Run-Native $VenvPy @("-X", "utf8", "-c", "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); x = torch.randn(64, 64, device='cuda' if torch.cuda.is_available() else 'cpu'); print('matmul ok', float((x @ x).sum()) == float((x @ x).sum()))")

Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Copy .env.example to .env and fill in the values."
Write-Host "  2. Accept the model terms at https://huggingface.co/pyannote/speaker-diarization-community-1"
Write-Host "  3. Create a fine-grained Hugging Face token with only 'Read access to contents of all public gated repos you can access' and put it in .env as HF_TOKEN."
Write-Host "  4. Try it:  .venv\Scripts\python.exe -X utf8 tests\make_test_clip.py --single"
Write-Host "              .venv\Scripts\python.exe -X utf8 dubber.py tests\clip_single.mp4 --to hi"

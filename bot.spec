# PyInstaller spec file for the standalone Telegram Voice Bot
#
# Build with:
#   cd standalone/
#   pip install pyinstaller
#   pyinstaller bot.spec
#
# Output: dist/telegram-voice-bot/  (one-dir mode — starts instantly)
# The user runs:  dist/telegram-voice-bot/telegram-voice-bot(.exe)
#
# NOTE: Model weights (Whisper, Kokoro) are NOT bundled — they are
# downloaded automatically on first run and cached in the same folder
# as the executable.

import sys
import os
import site
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

def _find_package_dir(pkg_name):
    """Find the directory of an installed package."""
    for sp in site.getsitepackages():
        candidate = os.path.join(sp, pkg_name)
        if os.path.isdir(candidate):
            return candidate
    return None

# ---- Collect all library components ----

hidden_imports = []
binaries_list = []
datas_list = []

# PyAV: Cython extensions + bundled FFmpeg shared libs (libopus, libogg, libavcodec...)
hidden_imports += collect_submodules("av")
binaries_list  += collect_dynamic_libs("av")

# ctranslate2: faster-whisper inference backend
hidden_imports += collect_submodules("ctranslate2")
binaries_list  += collect_dynamic_libs("ctranslate2")
datas_list     += collect_data_files("ctranslate2")

# faster-whisper: VAD ONNX asset (assets/silero_vad_v6.onnx)
hidden_imports += collect_submodules("faster_whisper")
datas_list     += collect_data_files("faster_whisper")

# onnxruntime: libonnxruntime.so + provider shared libs (used by kokoro-onnx)
hidden_imports += collect_submodules("onnxruntime")
binaries_list  += collect_dynamic_libs("onnxruntime")
datas_list     += collect_data_files("onnxruntime")

# kokoro-onnx (explicitly include config.json by direct path)
hidden_imports += collect_submodules("kokoro_onnx")
datas_list     += collect_data_files("kokoro_onnx")
_kokoro_dir = _find_package_dir("kokoro_onnx")
if _kokoro_dir:
    datas_list.append((os.path.join(_kokoro_dir, "config.json"), "kokoro_onnx"))

# language_tags, csvw, segments — data-heavy packages missed by collect_data_files
for _pkg in ("language_tags", "csvw", "segments"):
    _pkg_dir = _find_package_dir(_pkg)
    if _pkg_dir:
        datas_list.append((_pkg_dir, _pkg))

# espeakng-loader: bundled libespeak-ng.so + phoneme data (dependency of kokoro-onnx)
hidden_imports += collect_submodules("espeakng_loader")
binaries_list  += collect_dynamic_libs("espeakng_loader")
datas_list     += collect_data_files("espeakng_loader")

# phonemizer (dependency of kokoro-onnx for text normalization)
hidden_imports += collect_submodules("phonemizer")
datas_list     += collect_data_files("phonemizer")

# python-telegram-bot
hidden_imports += collect_submodules("telegram")

# httpx / httpcore / anyio async backends (used by python-telegram-bot internally)
hidden_imports += [
    "anyio._backends._asyncio",
    "httpx",
    "httpcore",
    "httpcore._async.http11",
    "httpcore._async.http2",
    "httpcore._sync.http11",
    "httpcore._sync.http2",
]

# huggingface_hub (used only during first-run wizard for model download)
hidden_imports += collect_submodules("huggingface_hub")

# ---- Analysis ----

a = Analysis(
    ["bot.py"],
    pathex=[],
    binaries=binaries_list,
    datas=datas_list,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude PyTorch — kokoro-onnx uses onnxruntime, not torch
        "torch",
        "torchvision",
        "torchaudio",
        # Exclude GUI and unneeded heavy libs
        "matplotlib",
        "scipy",
        "PIL",
        "IPython",
        "notebook",
        "pytest",
        "tkinter",
        "_tkinter",
        "wx",
        "PyQt5",
        "PySide2",
        "PySide6",
        "PyQt6",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,   # one-file: bundle everything into the exe
    a.zipfiles,
    a.datas,
    exclude_binaries=False,
    name="telegram-voice-bot",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,    # don't UPX — breaks DLLs inside the single-file bundle
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

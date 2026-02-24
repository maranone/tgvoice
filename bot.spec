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
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

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

# kokoro-onnx
hidden_imports += collect_submodules("kokoro_onnx")

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
    [],
    exclude_binaries=True,  # use COLLECT (one-dir mode for fast startup)
    name="telegram-voice-bot",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # needs a terminal for the first-run config wizard
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[
        # Don't UPX-compress shared libraries — breaks RPATH / load paths
        "libav*.so*",
        "libctranslate2*.so*",
        "libonnxruntime*.so*",
        "libespeak*.so*",
        "*.dll",
    ],
    name="telegram-voice-bot",
)

"""
Standalone Telegram Voice Bot
==============================
Single-file version. Dependencies (installed via pip):
  python-telegram-bot, faster-whisper, kokoro-onnx, av, huggingface-hub, numpy

Runtime deps (NOT bundled):
  Claude CLI  -- https://claude.ai/cli
  Codex CLI   -- https://github.com/openai/codex

First run: interactive config wizard saves bot_config.json next to this file.
Subsequent runs: config loaded from JSON, models loaded from cache.
"""

# ============================================================
# SECTION 0: IMPORTS
# All imports are explicit and static so PyInstaller can detect them.
# ============================================================

import asyncio
import gc
import io
import json
import logging
import os
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional

import av
import av.audio.resampler  # explicit Cython submodule — required for PyInstaller
import numpy as np
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ============================================================
# SECTION 1: CONFIG
# ============================================================

# Trivial digit substitution — prevents plain copy/paste of tokens,
# not intended as real security. Digits shift: 0→5, 1→6, ..., 4→9, 5→0, ...
_ENC = str.maketrans("0123456789", "5678901234")
_DEC = str.maketrans("5678901234", "0123456789")

def _obfuscate(s: str) -> str:
    return s.translate(_ENC)

def _deobfuscate(s: str) -> str:
    return s.translate(_DEC)

# Locate base directory: next to .exe when frozen, next to script otherwise
if getattr(sys, "frozen", False):
    _BASE_DIR = Path(sys.executable).parent
else:
    _BASE_DIR = Path(__file__).parent

CONFIG_FILE = _BASE_DIR / "bot_config.json"

# Global config dict — populated by load_config() at startup
CONFIG: dict = {}

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ============================================================
# SECTION 2: FIRST-RUN CONFIG WIZARD
# ============================================================

def _cuda_available() -> bool:
    try:
        import ctypes
        ctypes.CDLL("libcuda.so.1")
        return True
    except Exception:
        return False


def _validate_token(token: str) -> bool:
    """Basic format check: digits:alphanumeric string."""
    parts = token.split(":")
    return len(parts) == 2 and parts[0].isdigit() and len(parts[1]) > 20


def _prompt(question: str, default: str = "") -> str:
    """Prompt user, returning default on empty input."""
    suffix = f" [{default}]" if default else ""
    answer = input(f"{question}{suffix}: ").strip()
    return answer if answer else default


def _download_kokoro_models(models_dir: Path) -> None:
    """Download kokoro-onnx model files from Hugging Face Hub."""
    from huggingface_hub import hf_hub_download

    models_dir.mkdir(parents=True, exist_ok=True)
    repo = "thewh1teagle/kokoro-onnx-v1.0"

    for filename, label in [
        ("kokoro-v1.0.onnx", "Kokoro ONNX model (~310MB)"),
        ("voices-v1.0.bin", "Kokoro voices file"),
    ]:
        dest = models_dir / filename
        if dest.exists():
            print(f"  {filename} already cached, skipping.")
        else:
            print(f"  Downloading {label}...")
            hf_hub_download(repo_id=repo, filename=filename, local_dir=str(models_dir))
            print(f"  Done.")


def run_config_wizard() -> dict:
    """Interactive first-run setup. Saves config to JSON and downloads models."""
    print()
    print("=" * 60)
    print("  TELEGRAM VOICE BOT — First Run Setup")
    print("=" * 60)
    print()

    cfg: dict = {}

    # --- Telegram ---
    print("[ Telegram ]")
    while True:
        token = input("  Bot token (from @BotFather): ").strip()
        if _validate_token(token):
            cfg["telegram_token"] = _obfuscate(token)
            break
        print("  Invalid token format, please try again.")

    owner = input("  Your chat ID for access control (leave blank = no restriction): ").strip()
    cfg["owner_chat_id"] = owner or None

    # --- LLM Backend ---
    print()
    print("[ LLM Backend ]")
    print("  1) Claude CLI")
    print("  2) Codex CLI")
    choice = _prompt("  Choice", "1")
    cfg["llm_backend"] = "codex" if choice == "2" else "claude"

    home = str(Path.home())
    if cfg["llm_backend"] == "claude":
        cfg["claude_cli_path"] = _prompt("  Claude CLI path", f"{home}/.local/bin/claude")
        cfg["claude_cli_timeout"] = int(_prompt("  Claude timeout (seconds)", "120"))
    else:
        cfg["codex_cli_path"] = _prompt("  Codex CLI path", "/usr/local/bin/codex")
        cfg["codex_cli_timeout"] = int(_prompt("  Codex timeout (seconds)", "120"))
        cfg["codex_workdir"] = _prompt("  Codex working directory", home)

    # --- Whisper ---
    print()
    print("[ Speech Recognition — Whisper ]")
    print("  tiny (~75MB) / base (~150MB) / small (~250MB) / medium (~800MB)")
    print("  large-v3 (~1.5GB) / large-v3-turbo (~800MB, recommended)")
    cfg["whisper_model"] = _prompt("  Model", "large-v3-turbo")
    has_cuda = _cuda_available()
    cfg["whisper_device"] = "cuda" if has_cuda else "cpu"
    cfg["whisper_compute_type"] = "float16" if has_cuda else "int8"
    if has_cuda:
        print("  CUDA detected — will use GPU acceleration.")
    else:
        print("  No CUDA detected — will use CPU (slower).")

    # --- TTS ---
    print()
    print("[ Text-to-Speech — Kokoro ONNX ]")
    print("  Spanish voices: ef_dora (female), em_alex (male), em_santa (male)")
    print("  English voices: af_heart, af_bella, am_adam ...")
    cfg["tts_voice"] = _prompt("  Voice", "ef_dora")
    cfg["tts_lang"] = _prompt("  Language code (es / en / fr ...)", "es")

    # --- Paths ---
    data_dir = _BASE_DIR / "data"
    models_dir = _BASE_DIR / "models"
    cfg["data_dir"] = str(data_dir)
    cfg["models_dir"] = str(models_dir)
    cfg["audio_temp_dir"] = str(data_dir / "audio_temp")
    cfg["conversation_file"] = str(data_dir / "conversation.md")
    cfg["max_conversation_chars"] = 12000

    # Save config
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"\nConfig saved to: {CONFIG_FILE}")

    # Download models
    print()
    print("[ Downloading models — this only happens once ]")
    _download_kokoro_models(models_dir)

    # Whisper downloads itself when WhisperModel() is first instantiated
    print(f"\n  Pre-loading Whisper {cfg['whisper_model']} (downloads if needed)...")
    WhisperModel(
        cfg["whisper_model"],
        device=cfg["whisper_device"],
        compute_type=cfg["whisper_compute_type"],
    )
    print("  Whisper ready.")

    print()
    print("Setup complete. Starting bot...")
    print()
    return cfg


def load_config() -> dict:
    """Load config from JSON or run first-run wizard."""
    global CONFIG
    if CONFIG_FILE.exists():
        CONFIG = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    else:
        CONFIG = run_config_wizard()

    # Ensure runtime directories exist
    for key in ("data_dir", "audio_temp_dir"):
        Path(CONFIG[key]).mkdir(parents=True, exist_ok=True)

    return CONFIG


# ============================================================
# SECTION 3: CONVERSATION STORE
# ============================================================

def conversation_get_context() -> str:
    path = Path(CONFIG["conversation_file"])
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    max_chars = CONFIG.get("max_conversation_chars", 12000)
    if len(text) > max_chars:
        text = text[-max_chars:]
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]
    return text


def conversation_append(role: str, text: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    line = f"**{role} [{timestamp}]:** {text}\n"
    with open(CONFIG["conversation_file"], "a", encoding="utf-8") as f:
        f.write(line)


def conversation_clear() -> None:
    Path(CONFIG["conversation_file"]).write_text("", encoding="utf-8")


# ============================================================
# SECTION 4: AUDIO UTILS (PyAV — no ffmpeg binary needed)
# ============================================================

def numpy_to_ogg_bytes(audio_np: np.ndarray, sample_rate: int = 24000) -> bytes:
    """Encode a float32 numpy audio array to OGG Opus bytes using PyAV.

    kokoro-onnx returns float32 arrays at 24kHz mono.
    libopus natively supports 24kHz so no resampling is needed.
    """
    if audio_np.ndim == 1:
        audio_np = audio_np.reshape(1, -1)  # (1, samples) for mono 'flt' format

    buf = io.BytesIO()
    with av.open(buf, mode="w", format="ogg") as output:
        stream = output.add_stream("libopus", rate=sample_rate)
        stream.layout = "mono"
        stream.format = "flt"

        frame = av.AudioFrame.from_ndarray(audio_np, format="flt", layout="mono")
        frame.sample_rate = sample_rate
        frame.pts = 0

        for packet in stream.encode(frame):
            output.mux(packet)
        for packet in stream.encode(None):  # flush encoder
            output.mux(packet)

    buf.seek(0)
    return buf.read()


def cleanup(*paths: str) -> None:
    """Delete temporary files, silently ignoring missing ones."""
    for p in paths:
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass


# ============================================================
# SECTION 5: WHISPER ENGINE
# ============================================================

_whisper_executor = ThreadPoolExecutor(max_workers=1)


@lru_cache(maxsize=1)
def get_whisper_model() -> WhisperModel:
    """Singleton Whisper model. Loads (and downloads if needed) once at startup."""
    cfg = CONFIG
    logger.info(
        "Loading Whisper %s on %s (%s)...",
        cfg["whisper_model"], cfg["whisper_device"], cfg["whisper_compute_type"],
    )
    model = WhisperModel(
        cfg["whisper_model"],
        device=cfg["whisper_device"],
        compute_type=cfg["whisper_compute_type"],
    )
    logger.info("Whisper model loaded.")
    return model


def _transcribe_sync(ogg_path: str) -> str:
    """Synchronous transcription. faster-whisper accepts OGG directly via PyAV."""
    model = get_whisper_model()
    segments, _ = model.transcribe(
        ogg_path,
        language="es",
        beam_size=5,
        best_of=5,
        vad_filter=True,
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


async def whisper_transcribe(ogg_path: str) -> str:
    """Non-blocking transcription via thread executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_whisper_executor, _transcribe_sync, ogg_path)


# ============================================================
# SECTION 6: TTS ENGINE (kokoro-onnx, fully local)
# ============================================================

_tts_executor = ThreadPoolExecutor(max_workers=1)


@lru_cache(maxsize=1)
def get_kokoro() -> Kokoro:
    """Singleton Kokoro ONNX model."""
    models_dir = Path(CONFIG["models_dir"])
    onnx_path = str(models_dir / "kokoro-v1.0.onnx")
    voices_path = str(models_dir / "voices-v1.0.bin")
    logger.info("Loading Kokoro ONNX model from %s ...", onnx_path)
    model = Kokoro(onnx_path, voices_path)
    logger.info("Kokoro model loaded.")
    return model


def _tts_sync(text: str) -> bytes:
    """Synchronous TTS: text → OGG Opus bytes (no disk I/O)."""
    kokoro = get_kokoro()
    samples, sample_rate = kokoro.create(
        text,
        voice=CONFIG.get("tts_voice", "ef_dora"),
        speed=1.0,
        lang=CONFIG.get("tts_lang", "es"),
    )
    return numpy_to_ogg_bytes(samples, sample_rate)


async def text_to_ogg_bytes(text: str) -> bytes:
    """Async TTS via thread executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_tts_executor, _tts_sync, text)


# ============================================================
# SECTION 7: LLM CLIENT (subprocess to Claude CLI / Codex CLI)
# ============================================================

def _build_prompt(user_message: str) -> str:
    conv_file = CONFIG["conversation_file"]
    return (
        f"Read {conv_file} for conversation history, then respond to: {user_message}"
    )


def _run_claude(prompt: str) -> str:
    cli = CONFIG.get("claude_cli_path", "/home/waz/.local/bin/claude")
    timeout = CONFIG.get("claude_cli_timeout", 120)
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    env.pop("CLAUDE_CODE", None)
    result = subprocess.run(
        [cli, "--print", prompt, "--output-format", "text"],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=str(Path.home()),
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude exited {result.returncode}: {result.stderr.strip()}")
    return result.stdout.strip()


def _run_codex(prompt: str) -> str:
    cli = CONFIG.get("codex_cli_path", "/usr/local/bin/codex")
    timeout = CONFIG.get("codex_cli_timeout", 120)
    workdir = CONFIG.get("codex_workdir", str(Path.home()))
    output_file = str(
        Path(CONFIG["audio_temp_dir"]) / f"codex_out_{uuid.uuid4().hex}.txt"
    )
    try:
        result = subprocess.run(
            [
                cli, "exec",
                "--skip-git-repo-check",
                "--full-auto",
                "-C", workdir,
                "-o", output_file,
                prompt,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(f"codex exited {result.returncode}: {result.stderr.strip()}")
        with open(output_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    finally:
        try:
            os.unlink(output_file)
        except FileNotFoundError:
            pass


def llm_get_response(user_message: str) -> str:
    prompt = _build_prompt(user_message)
    try:
        backend = CONFIG.get("llm_backend", "claude")
        if backend == "codex":
            return _run_codex(prompt)
        else:
            return _run_claude(prompt)
    except subprocess.TimeoutExpired:
        return "Lo siento, la solicitud tardó demasiado tiempo. Por favor, inténtalo de nuevo."
    except Exception as e:
        return f"Lo siento, ocurrió un error al procesar tu mensaje: {e}"


# ============================================================
# SECTION 8: TELEGRAM HANDLERS
# ============================================================

def is_authorized(update: Update) -> bool:
    owner = CONFIG.get("owner_chat_id")
    if not owner:
        return True
    return str(update.effective_chat.id) == str(owner)


async def send_processing_message(update: Update) -> None:
    await update.message.reply_text("Procesando tu mensaje...")


async def process_and_reply(update: Update, user_text: str) -> None:
    """Call LLM, save to conversation history, send text + voice reply."""
    reply_text = llm_get_response(user_text)

    conversation_append("User", user_text)
    conversation_append("Assistant", reply_text)

    display = f"*Dijiste:* {user_text}\n\n*Respuesta:* {reply_text}"
    await update.message.reply_text(display, parse_mode="Markdown")

    try:
        ogg_bytes = await text_to_ogg_bytes(reply_text)
        buf = io.BytesIO(ogg_bytes)
        buf.name = "voice.ogg"  # telegram-bot uses filename for MIME detection
        await update.message.reply_voice(voice=buf)
    except Exception as e:
        logger.error("TTS/voice send failed: %s", e)
        await update.message.reply_text("(No se pudo generar el audio de respuesta.)")


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return

    await send_processing_message(update)

    ogg_path = str(Path(CONFIG["audio_temp_dir"]) / f"{uuid.uuid4().hex}.ogg")
    try:
        voice_file = await update.message.voice.get_file()
        await voice_file.download_to_drive(ogg_path)

        # Pass OGG directly — faster-whisper decodes it internally via PyAV
        user_text = await whisper_transcribe(ogg_path)
        if not user_text:
            await update.message.reply_text(
                "No pude entender el audio. Por favor, intenta de nuevo."
            )
            return

        logger.info("Transcribed: %s", user_text)
        await process_and_reply(update, user_text)

    except Exception as e:
        logger.exception("Error handling voice message")
        await update.message.reply_text(f"Ocurrió un error: {e}")
    finally:
        cleanup(ogg_path)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return

    user_text = update.message.text.strip()
    if not user_text:
        return

    await send_processing_message(update)

    try:
        await process_and_reply(update, user_text)
    except Exception as e:
        logger.exception("Error handling text message")
        await update.message.reply_text(f"Ocurrió un error: {e}")


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    await update.message.reply_text(
        "¡Hola! Soy tu asistente de voz personal.\n\n"
        "Puedes enviarme mensajes de voz o texto.\n\n"
        "Comandos disponibles:\n"
        "/context — Ver el historial de conversación\n"
        "/clear — Borrar el historial\n"
        "/status — Ver configuración actual"
    )


async def cmd_context(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    ctx = conversation_get_context()
    if not ctx.strip():
        await update.message.reply_text("No hay historial de conversación todavía.")
        return
    for i in range(0, len(ctx), 4096):
        await update.message.reply_text(ctx[i : i + 4096])


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    conversation_clear()
    await update.message.reply_text("Historial de conversación borrado.")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    conv_path = Path(CONFIG["conversation_file"])
    size = conv_path.stat().st_size if conv_path.exists() else 0
    backend = CONFIG.get("llm_backend", "claude")
    cli_path = CONFIG.get("claude_cli_path" if backend == "claude" else "codex_cli_path", "?")
    await update.message.reply_text(
        f"*Estado del bot*\n\n"
        f"Whisper: `{CONFIG['whisper_model']}` ({CONFIG['whisper_device']})\n"
        f"TTS voz: `{CONFIG.get('tts_voice', '?')}`\n"
        f"Backend LLM: `{backend}` (`{cli_path}`)\n"
        f"Historial: `{size}` bytes",
        parse_mode="Markdown",
    )


# ============================================================
# SECTION 9: MAIN ENTRYPOINT
# ============================================================

def main() -> None:
    load_config()  # runs wizard on first run, loads JSON otherwise

    logger.info("Pre-loading Whisper model %s...", CONFIG["whisper_model"])
    get_whisper_model()

    logger.info("Pre-loading Kokoro ONNX model...")
    get_kokoro()

    app = ApplicationBuilder().token(_deobfuscate(CONFIG["telegram_token"])).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("context", cmd_context))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Bot started. Polling...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()

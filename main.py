"""
Voice assistant: WebM → ffmpeg → Groq STT → RAG (FAISS + fastembed) → Groq chat → Kokoro TTS.

Requires Python 3.12, espeak-ng (Kokoro), ffmpeg (STT), optional tesseract (PDF OCR),
GROQ_LLM_API_KEY in .env. RAG index files live under voice_testing/data/rag/.

Run from voice_testing/:
  export PATH="/opt/homebrew/bin:$PATH"   # macOS: espeak-ng, ffmpeg, tesseract
  .venv/bin/uvicorn main:app --reload --host 0.0.0.0 --port 8765
Then open http://127.0.0.1:8765/
"""
from __future__ import annotations

import base64
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from groq import Groq
from pydantic import BaseModel, Field

from rag.ingest import ingest_pdf_bytes, retrieve_context
from rag.store import get_rag_store

from kokoro_tts import synthesize_wav_bytes

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
load_dotenv(ROOT / ".env")

RAG_DATA_DIR = ROOT / "data" / "rag"
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
MAX_UPLOAD_MB = int(os.getenv("RAG_MAX_UPLOAD_MB", "25"))

GROQ_KEY = os.getenv("GROQ_LLM_API_KEY", "").strip()
STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")
CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "llama-3.1-8b-instant")

SYSTEM_PROMPT = """You are a helpful voice assistant. Keep answers concise and natural for spoken dialogue (short sentences, no markdown, no bullet lists unless very brief)."""

app = FastAPI(title="Voice Kokoro + Groq")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_groq() -> Groq:
    if not GROQ_KEY:
        raise HTTPException(
            status_code=500,
            detail="GROQ_LLM_API_KEY is not set. Add it to voice_testing/.env",
        )
    return Groq(api_key=GROQ_KEY)


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _ffmpeg_to_wav16k_mono(data: bytes, src_suffix: str) -> bytes | None:
    """Decode browser WebM/Opus (etc.) to 16 kHz mono PCM WAV — Groq STT accepts this reliably."""
    ff = shutil.which("ffmpeg")
    if not ff:
        return None
    src_suffix = src_suffix if src_suffix.startswith(".") else f".{src_suffix}"
    if src_suffix == ".":
        src_suffix = ".webm"
    src_fd, src_path = tempfile.mkstemp(suffix=src_suffix, prefix="stt_in_")
    dst_fd, dst_path = tempfile.mkstemp(suffix=".wav", prefix="stt_out_")
    os.close(src_fd)
    os.close(dst_fd)
    try:
        Path(src_path).write_bytes(data)
        subprocess.run(
            [
                ff,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                src_path,
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                dst_path,
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        out = Path(dst_path).read_bytes()
        return out if len(out) > 44 else None
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    finally:
        for p in (src_path, dst_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def transcribe_audio_bytes(data: bytes, filename: str) -> str:
    if len(data) < 400:
        raise ValueError(
            f"Audio too small ({len(data)} bytes). Record a bit longer, or check the microphone."
        )

    suffix = Path(filename).suffix.lower() or ".webm"
    converted = _ffmpeg_to_wav16k_mono(data, suffix)

    if converted is not None:
        payload, upload_suffix = converted, ".wav"
    elif ffmpeg_available():
        raise ValueError(
            "ffmpeg could not decode this recording (often too short or an incomplete WebM). "
            "Try holding record for at least one second."
        )
    else:
        if suffix in (".webm", ".weba", ".ogg", ""):
            raise ValueError(
                "Browser WebM needs ffmpeg for reliable Groq STT. Install: brew install ffmpeg "
                "(Linux: apt install ffmpeg)."
            )
        payload, upload_suffix = data, suffix

    client = get_groq()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=upload_suffix, prefix="stt_", delete=False
        ) as tmp:
            tmp.write(payload)
            tmp_path = tmp.name
        with open(tmp_path, "rb") as handle:
            tr = client.audio.transcriptions.create(file=handle, model=STT_MODEL)
        return (tr.text or "").strip()
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def groq_chat(user_text: str, rag_context: str | None = None) -> str:
    client = get_groq()
    system = SYSTEM_PROMPT
    if rag_context:
        system += (
            "\n\n## Retrieved from user-uploaded PDFs\n"
            "Use these excerpts when they help answer the question. "
            "If they are irrelevant or insufficient, say so briefly and answer from general knowledge.\n\n"
            + rag_context
        )
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        temperature=0.6,
        max_tokens=512,
    )
    msg = completion.choices[0].message
    return (msg.content or "").strip()


@app.get("/api/health")
def health():
    rag = get_rag_store(RAG_DATA_DIR).stats()
    return {
        "ok": True,
        "groq_configured": bool(GROQ_KEY),
        "ffmpeg": ffmpeg_available(),
        "stt_model": STT_MODEL,
        "chat_model": CHAT_MODEL,
        "rag_chunks": rag["chunks"],
        "rag_model": rag["model"],
    }


@app.post("/api/voice-chat")
async def voice_chat(audio: UploadFile = File(...)):
    """
    Multipart form field name: `audio` (e.g. webm from browser MediaRecorder).
    Returns JSON: transcript, reply, audio_wav_base64 (PCM WAV @ 24kHz).
    """
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio body")

    fname = (audio.filename or "").strip() or "speech.webm"

    try:
        transcript = transcribe_audio_bytes(raw, fname)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Speech-to-text failed: {e!s}") from e

    if not transcript:
        raise HTTPException(status_code=400, detail="No speech detected; try again closer to the mic")

    rag_block, rag_hits = retrieve_context(transcript, RAG_DATA_DIR, top_k=RAG_TOP_K)
    rag_context = rag_block if rag_block else None

    try:
        reply = groq_chat(transcript, rag_context=rag_context)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Groq chat failed: {e!s}") from e

    if not reply:
        reply = "I did not have an answer for that."

    try:
        wav = synthesize_wav_bytes(reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {e!s}") from e

    return {
        "transcript": transcript,
        "reply": reply,
        "audio_wav_base64": base64.b64encode(wav).decode("ascii"),
        "sample_rate": 24000,
        "rag_used": bool(rag_context),
        "rag_hits": rag_hits,
    }


class TextChatBody(BaseModel):
    text: str = Field(..., min_length=1)


@app.post("/api/text-chat")
async def text_chat(body: TextChatBody):
    """JSON body {\"text\": \"...\"} -> Groq reply + Kokoro WAV (testing without mic)."""
    text = body.text.strip()
    rag_block, rag_hits = retrieve_context(text, RAG_DATA_DIR, top_k=RAG_TOP_K)
    rag_context = rag_block if rag_block else None
    try:
        reply = groq_chat(text, rag_context=rag_context)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Groq chat failed: {e!s}") from e
    wav = synthesize_wav_bytes(reply)
    return {
        "reply": reply,
        "audio_wav_base64": base64.b64encode(wav).decode("ascii"),
        "sample_rate": 24000,
        "rag_used": bool(rag_context),
        "rag_hits": rag_hits,
    }


@app.post("/api/rag/upload")
async def rag_upload(
    file: UploadFile = File(...),
    replace: bool = Query(False, description="If true, delete existing FAISS index before ingesting"),
):
    """
    Upload a PDF: extract text (PyMuPDF + Tesseract OCR on sparse pages), chunk, embed with fastembed, index in FAISS on disk.
    Query param `replace=true` clears the existing index first.
    """
    name = (file.filename or "document.pdf").strip()
    if not name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported")
    raw = await file.read()
    max_b = MAX_UPLOAD_MB * 1024 * 1024
    if len(raw) > max_b:
        raise HTTPException(
            status_code=400,
            detail=f"PDF too large (max {MAX_UPLOAD_MB} MB)",
        )
    if len(raw) < 100:
        raise HTTPException(status_code=400, detail="Empty or corrupt PDF upload")
    try:
        out = ingest_pdf_bytes(raw, source_name=name, data_dir=RAG_DATA_DIR, replace=replace)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e!s}") from e
    return out


@app.delete("/api/rag/clear")
async def rag_clear():
    """Remove local FAISS index and chunk metadata."""
    get_rag_store(RAG_DATA_DIR).clear()
    return {"ok": True, "message": "RAG index cleared"}


@app.get("/api/rag/status")
def rag_status():
    return get_rag_store(RAG_DATA_DIR).stats()


@app.get("/")
def root_index():
    index = STATIC / "index.html"
    if index.is_file():
        return FileResponse(index)
    return {"message": "Add static/index.html for the UI", "api": "/api/voice-chat"}

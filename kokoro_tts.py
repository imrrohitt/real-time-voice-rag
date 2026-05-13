"""Kokoro TTS helpers (lazy pipeline, WAV bytes)."""
from __future__ import annotations

import io
import shutil
from typing import Optional

import numpy as np
import soundfile as sf
from kokoro import KPipeline

SAMPLE_RATE = 24000
_PIPELINE: Optional[KPipeline] = None


def espeak_available() -> bool:
    return shutil.which("espeak-ng") is not None or shutil.which("espeak") is not None


def get_pipeline(lang_code: str = "a") -> KPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        if not espeak_available():
            raise RuntimeError(
                "espeak-ng is required for Kokoro G2P. Install: brew install espeak-ng (macOS) "
                "or apt-get install -y espeak-ng (Linux)."
            )
        _PIPELINE = KPipeline(lang_code=lang_code, repo_id="hexgrad/Kokoro-82M")
    return _PIPELINE


def synthesize_wav_bytes(text: str, voice: str = "af_heart", lang_code: str = "a") -> bytes:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty text for TTS")
    pipeline = get_pipeline(lang_code=lang_code)
    chunks: list[np.ndarray] = []
    for _gs, _ps, audio in pipeline(text, voice=voice):
        chunks.append(np.asarray(audio, dtype=np.float32).flatten())
    if not chunks:
        raise RuntimeError("Kokoro returned no audio segments")
    combined = np.concatenate(chunks)
    peak = float(np.max(np.abs(combined))) or 1.0
    pcm = (combined / peak * 0.99 * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    sf.write(buf, pcm, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()

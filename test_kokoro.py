"""
Local smoke test for hexgrad/Kokoro-82M via the `kokoro` package.
https://huggingface.co/hexgrad/Kokoro-82M

Run from voice_testing/: python test_kokoro.py

System dependency (G2P): install espeak-ng — on macOS: brew install espeak-ng
(Linux/Colab: apt-get install -y espeak-ng)
"""
from __future__ import annotations

import os
import shutil
import sys

import soundfile as sf

from kokoro import KPipeline

SAMPLE_RATE = 24000
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def _espeak_on_path() -> bool:
    return shutil.which("espeak-ng") is not None or shutil.which("espeak") is not None


def main() -> int:
    if not _espeak_on_path():
        print(
            "espeak-ng not found in PATH. Kokoro's pipeline needs it for phonemization.\n"
            "  macOS: brew install espeak-ng\n"
            "  Debian/Ubuntu: sudo apt-get install -y espeak-ng\n",
            file=sys.stderr,
        )
        return 1

    print("Loading Kokoro pipeline (lang_code='a', American English)...", flush=True)
    pipeline = KPipeline(lang_code="a")
    text = """
Happy Birthday Riyaz

Wishing you lots of happiness, success, and growth in the year ahead. It’s great working with you and seeing your dedication and enthusiasm every day.
Keep learning, keep growing, and keep doing great work. 
Have an amazing birthday and a wonderful year ahead.
""".strip()

    print("Synthesizing...", flush=True)
    last_path = ""
    for i, (gs, ps, audio) in enumerate(pipeline(text, voice="af_heart")):
        print(i, gs, ps, flush=True)
        last_path = os.path.join(OUT_DIR, f"kokoro_{i}.wav")
        sf.write(last_path, audio, SAMPLE_RATE)

    if not last_path:
        print("No audio segments returned.", file=sys.stderr)
        return 1
    print(f"OK: wrote segment WAV(s) under {OUT_DIR} (e.g. {last_path})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

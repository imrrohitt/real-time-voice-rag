# Voice testing — Groq + Kokoro + RAG

A small **FastAPI** demo that runs a **voice assistant** end-to-end in the browser:

1. **Speech-to-text** — browser audio (WebM) is normalized with **ffmpeg**, then sent to **Groq** Whisper.
2. **Retrieval (optional)** — the transcript is embedded with **fastembed** and matched against a local **FAISS** index built from uploaded PDFs.
3. **LLM** — **Groq** chat completions answer using general instructions plus retrieved context when available.
4. **Text-to-speech** — replies are synthesized with **Kokoro** ([hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)) and returned as WAV.

The UI is a single page: `static/index.html` (hold-to-record, PDF upload for RAG, automatic playback of the reply).

---

## Requirements

| Requirement | Why |
|-------------|-----|
| **Python 3.12** | `kokoro>=0.9.2` does not support Python 3.13 yet. |
| **espeak-ng** | Kokoro’s G2P pipeline expects `espeak-ng` on `PATH`. |
| **ffmpeg** | Converts browser WebM to 16 kHz mono WAV before Groq STT (raw WebM is often rejected). |
| **Tesseract** (optional) | OCR for PDF pages with little embedded text (`brew install tesseract`). |

---

## Quick start

```bash
cd voice_testing

# Python 3.12 venv (example with Homebrew Python)
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# macOS: ensure CLI tools are on PATH (adjust if not using Homebrew)
export PATH="/opt/homebrew/bin:$PATH"

# Secrets — copy the example file and add your Groq key
cp .env.example .env
# Edit .env: set GROQ_LLM_API_KEY=...

uvicorn main:app --reload --host 0.0.0.0 --port 8765
```

Open **http://127.0.0.1:8765/** in the browser.

If you serve `static/index.html` from another port (e.g. Live Server), the page defaults API calls to `http://127.0.0.1:8765`, or you can override with `?api=http://127.0.0.1:8765`.

---

## Environment variables

Create **`voice_testing/.env`** (see `.gitignore` — do not commit secrets).

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_LLM_API_KEY` | **Yes** | — | Groq API key (STT + chat). |
| `GROQ_STT_MODEL` | No | `whisper-large-v3-turbo` | Groq transcription model. |
| `GROQ_CHAT_MODEL` | No | `llama-3.1-8b-instant` | Groq chat model. |
| `RAG_EMBED_MODEL` | No | `BAAI/bge-small-en-v1.5` | fastembed model (changing it invalidates existing index dimensions — clear/reindex). |
| `RAG_TOP_K` | No | `5` | Number of chunks retrieved for each query. |
| `RAG_EMBED_BATCH` | No | `24` | Embedding batch size during PDF ingest. |
| `RAG_MAX_UPLOAD_MB` | No | `25` | Max PDF upload size. |

---

## Project layout

```
voice_testing/
├── main.py              # FastAPI app, routes, Groq + STT + RAG wiring
├── kokoro_tts.py        # Lazy Kokoro pipeline → WAV bytes
├── test_kokoro.py       # Standalone Kokoro smoke test
├── requirements.txt
├── .env                 # local secrets (gitignored)
├── static/
│   └── index.html       # Voice UI + PDF RAG upload
├── rag/
│   ├── store.py         # FAISS + chunks.json persistence
│   ├── ingest.py        # PDF ingest + retrieve_context()
│   ├── chunking.py      # Text chunking for indexing
│   └── pdf_extract.py   # PyMuPDF + Tesseract OCR fallback
└── data/rag/            # gitignored — vectors.faiss + chunks.json
```

---

## HTTP API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serves `static/index.html`. |
| `GET` | `/api/health` | JSON: Groq configured, ffmpeg, STT/chat models, RAG chunk count. |
| `POST` | `/api/voice-chat` | Multipart field **`audio`** (e.g. WebM). Returns `transcript`, `reply`, `audio_wav_base64`, `sample_rate`, `rag_used`, `rag_hits`. |
| `POST` | `/api/text-chat` | JSON `{"text":"..."}` — same RAG + TTS path without a microphone. |
| `POST` | `/api/rag/upload` | Multipart **`file`** (PDF). Query `replace=true` wipes the index first. |
| `GET` | `/api/rag/status` | RAG stats (`chunks`, `model`, `data_dir`). |
| `DELETE` | `/api/rag/clear` | Deletes local FAISS index and chunk metadata. |

Interactive docs: **http://127.0.0.1:8765/docs**

---

## RAG (FAISS + fastembed)

1. Use the UI **“RAG — upload PDF”** or `POST /api/rag/upload` with a `.pdf` file.
2. Text is extracted with **PyMuPDF**; pages with very little text are run through **Tesseract** if it is installed.
3. Text is split into overlapping chunks (`rag/chunking.py`), embedded with **fastembed**, and appended to a **FAISS** `IndexFlatIP` index (CPU, L2-normalized vectors for cosine-style search).
4. On each voice or text query, the top-`RAG_TOP_K` chunks are concatenated (length-capped) and injected into the Groq **system** prompt.

Index files live under **`data/rag/`** (`vectors.faiss`, `chunks.json`). They are safe to delete; the app will recreate them on the next ingest.

---

## Speech pipeline notes

- **Microphone**: Browsers require a secure context or localhost; allow the site when prompted.
- **Autoplay**: The UI resumes an `AudioContext` on record and prefers **Web Audio** playback for the reply so audio can start right after the network response.
- **Groq STT errors**: If you see “invalid media file”, ensure **ffmpeg** is installed and on `PATH` on the machine running uvicorn.

---

## Kokoro-only smoke test

```bash
source .venv/bin/activate
export PATH="/opt/homebrew/bin:$PATH"
python test_kokoro.py
```

Writes `kokoro_*.wav` in this directory.

---

## License and upstream models

- **Kokoro-82M**: Apache 2.0 — see [model card](https://huggingface.co/hexgrad/Kokoro-82M).
- **Groq**: Usage subject to [Groq](https://groq.com) terms and quotas.
- **fastembed / BGE**: Check the embedding model card on Hugging Face for license details.

---

## Troubleshooting

| Symptom | Things to check |
|---------|------------------|
| `kokoro` install fails on 3.13 | Use **Python 3.12** for the venv. |
| Kokoro / G2P errors | `which espeak-ng` — install espeak-ng. |
| STT 400 / invalid media | `which ffmpeg` — install ffmpeg; record ≥ ~1 s. |
| Empty PDF text | Install **Tesseract** for OCR; some PDFs are image-only. |
| RAG dimension mismatch | You changed `RAG_EMBED_MODEL`; call **`DELETE /api/rag/clear`** or upload with **`replace=true`**, then re-upload PDFs. |

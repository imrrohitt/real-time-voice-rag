"""FAISS + JSON file store; fastembed embeddings; CPU only."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from fastembed import TextEmbedding

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

_store: "RAGStore | None" = None
_store_lock = threading.Lock()


def get_rag_store(data_dir: Path) -> "RAGStore":
    global _store
    with _store_lock:
        if _store is None or _store.data_dir != data_dir.resolve():
            _store = RAGStore(data_dir)
            _store.load()
        return _store


class RAGStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.data_dir / "vectors.faiss"
        self.meta_path = self.data_dir / "chunks.json"
        self.model_name = os.environ.get("RAG_EMBED_MODEL", DEFAULT_MODEL)
        self._embedder: TextEmbedding | None = None
        self._mutex = threading.Lock()
        self.index: faiss.Index | None = None
        self.chunks: list[dict[str, Any]] = []

    def _embedder_lazy(self) -> TextEmbedding:
        if self._embedder is None:
            self._embedder = TextEmbedding(self.model_name)
        return self._embedder

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        model = self._embedder_lazy()
        vecs = np.stack([np.asarray(e, dtype=np.float32) for e in model.embed(texts)])
        faiss.normalize_L2(vecs)
        return vecs

    def load(self) -> None:
        with self._mutex:
            if self.index_path.is_file() and self.meta_path.is_file():
                self.index = faiss.read_index(str(self.index_path))
                self.chunks = json.loads(self.meta_path.read_text(encoding="utf-8"))
            else:
                self.index = None
                self.chunks = []

    def _save_unlocked(self) -> None:
        if self.index is None:
            return
        faiss.write_index(self.index, str(self.index_path))
        self.meta_path.write_text(json.dumps(self.chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    def clear(self) -> None:
        with self._mutex:
            self.index = None
            self.chunks = []
            for p in (self.index_path, self.meta_path):
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass

    def stats(self) -> dict[str, Any]:
        with self._mutex:
            n = int(self.index.ntotal) if self.index is not None else 0
            return {
                "chunks": n,
                "model": self.model_name,
                "data_dir": str(self.data_dir),
            }

    def add_text_chunks(self, texts: list[str], source: str) -> int:
        texts = [t.strip() for t in texts if t and t.strip()]
        if not texts:
            return 0
        with self._mutex:
            batch_size = int(os.environ.get("RAG_EMBED_BATCH", "24"))
            for i in range(0, len(texts), batch_size):
                sub = texts[i : i + batch_size]
                vecs = self._embed_batch(sub)
                dim = vecs.shape[1]
                if self.index is None:
                    self.index = faiss.IndexFlatIP(dim)
                    self.chunks = []
                elif self.index.d != dim:
                    raise ValueError(
                        f"Embedding dimension mismatch: index has {self.index.d}, new vectors {dim}"
                    )
                self.index.add(vecs)
                for t in sub:
                    self.chunks.append({"text": t, "source": source})
            self._save_unlocked()
        return len(texts)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float, str]]:
        """Return list of (text, score, source)."""
        q = (query or "").strip()
        if not q or self.index is None or self.index.ntotal == 0:
            return []
        with self._mutex:
            qv = self._embed_batch([q])
            scores, idx = self.index.search(qv, min(top_k, int(self.index.ntotal)))
        out: list[tuple[str, float, str]] = []
        for rank, j in enumerate(idx[0]):
            if j < 0 or j >= len(self.chunks):
                continue
            row = self.chunks[int(j)]
            out.append((row["text"], float(scores[0][rank]), row.get("source", "")))
        return out

"""Persistent local vector store built on numpy.

Documents are chunked, embedded via an OpenAI-compatible embeddings endpoint,
and searched with cosine similarity. State is persisted to disk so it survives
restarts. Retrieval is vectorized over a normalized matrix for speed.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class RAGStore:
    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.index_path = os.path.join(data_dir, "index.json")
        self.vectors_path = os.path.join(data_dir, "vectors.npz")
        self._lock = threading.RLock()
        self.documents: dict[str, dict] = {}  # doc_id -> metadata
        self.vectors: dict[tuple[str, int], np.ndarray] = {}  # (doc_id, chunk_idx) -> vec
        self._keys: list[tuple[str, int]] = []
        self._matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.dim: int | None = None
        self._load()
        self._rebuild_matrix()

    # ---- persistence ---------------------------------------------------
    def _load(self):
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "r", encoding="utf-8") as fh:
                    self.documents = json.load(fh)
            except Exception:
                self.documents = {}
        if os.path.exists(self.vectors_path):
            if os.path.getsize(self.vectors_path) == 0:
                self.vectors = {}
                return
            try:
                with np.load(self.vectors_path, allow_pickle=False) as data:
                    doc_ids = [str(k) for k in data["doc_ids"].tolist()]
                    idxs = [int(k) for k in data["idxs"].tolist()]
                    matrix = data["matrix"]
                    self.vectors = {
                        (did, idx): np.asarray(row, dtype=np.float32)
                        for did, idx, row in zip(doc_ids, idxs, matrix)
                    }
            except (KeyError, ValueError, TypeError, OSError, EOFError):
                self._load_legacy()

    def _load_legacy(self):
        """Read older npz layouts (object keys, or list-of-tuples keys) and re-save."""
        if not os.path.exists(self.vectors_path) or os.path.getsize(self.vectors_path) == 0:
            self.vectors = {}
            return
        try:
            with np.load(self.vectors_path, allow_pickle=True) as data:
                self.vectors = {}
                if "matrix" in data.files:
                    keys = [tuple(k) for k in data["keys"].tolist()]
                    matrix = data["matrix"]
                    for key, row in zip(keys, matrix):
                        did, idx = key
                        self.vectors[(str(did), int(idx))] = np.asarray(row, dtype=np.float32)
                elif "keys" in data.files and "vectors" in data.files:
                    keys = data["keys"]
                    arrs = data["vectors"]
                    for key, vec in zip(keys, arrs):
                        did, idx = key
                        self.vectors[(str(did), int(idx))] = np.asarray(vec, dtype=np.float32)
                self._save()
        except Exception:
            self.vectors = {}

    def _save(self):
        """Atomically persist vectors and document index to disk."""
        keys = list(self.vectors.keys())
        matrix = (
            np.vstack([self.vectors[k] for k in keys])
            if keys
            else np.zeros((0, 0), dtype=np.float32)
        )
        # Atomic write for npz
        fd_v, tmp_v = tempfile.mkstemp(dir=self.data_dir, suffix=".npz")
        try:
            with os.fdopen(fd_v, "wb") as fh:
                np.savez(
                    fh,
                    doc_ids=np.array([k[0] for k in keys]),
                    idxs=np.array([k[1] for k in keys], dtype=np.int64),
                    matrix=matrix,
                )
            os.replace(tmp_v, self.vectors_path)
        finally:
            if os.path.exists(tmp_v):
                try:
                    os.unlink(tmp_v)
                except OSError:
                    pass

        # Atomic write for json
        fd_i, tmp_i = tempfile.mkstemp(dir=self.data_dir, suffix=".json")
        try:
            with os.fdopen(fd_i, "w", encoding="utf-8") as fh:
                json.dump(self.documents, fh, ensure_ascii=False, indent=2)
            os.replace(tmp_i, self.index_path)
        finally:
            if os.path.exists(tmp_i):
                try:
                    os.unlink(tmp_i)
                except OSError:
                    pass

    def _rebuild_matrix(self):
        """Pre-normalized stacked vectors plus aligned flat keys for fast search."""
        keys = list(self.vectors.keys())
        if keys:
            first_vec = self.vectors[keys[0]]
            self.dim = int(first_vec.shape[0])
            matrix = np.vstack([self.vectors[k] for k in keys]).astype(np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix = matrix / norms
        else:
            self.dim = None
            matrix = np.zeros((0, 0), dtype=np.float32)
        self._keys = keys
        self._matrix = matrix

    # ---- public API ----------------------------------------------------
    def list_documents(self) -> list[dict]:
        with self._lock:
            out = []
            for did, meta in self.documents.items():
                n_chunks = sum(1 for (d, _) in self.vectors if d == did)
                entry = dict(meta)
                entry["id"] = did
                entry["chunks"] = n_chunks
                out.append(entry)
            out.sort(key=lambda d: d.get("added_at", 0), reverse=True)
            return out

    def add_document(self, name: str, chunks: list[str], vectors: list[list[float]]) -> dict:
        if not chunks:
            raise ValueError("Cannot add document with 0 chunks")
        if len(chunks) != len(vectors):
            raise ValueError("Mismatch between number of chunks and vectors")

        incoming_dim = len(vectors[0])
        with self._lock:
            if self.dim is not None and self.vectors and incoming_dim != self.dim:
                raise ValueError(
                    f"Embedding dimension mismatch: index uses {self.dim}-dimensional vectors, "
                    f"but new document provided {incoming_dim}-dimensional vectors. "
                    f"Please use the matching embedding model or remove existing documents."
                )

            doc_id = uuid.uuid4().hex
            self.documents[doc_id] = {
                "name": name,
                "added_at": time.time(),
                "num_chunks": len(chunks),
                "body": chunks,
            }
            for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
                self.vectors[(doc_id, i)] = np.asarray(vec, dtype=np.float32)
            self._rebuild_matrix()
            self._save()
            return {"id": doc_id, "name": name, "chunks": len(chunks)}

    def remove_document(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id not in self.documents:
                return False
            del self.documents[doc_id]
            keys = [(d, i) for (d, i) in self.vectors if d == doc_id]
            for k in keys:
                del self.vectors[k]
            self._rebuild_matrix()
            self._save()
            return True

    def clear(self) -> None:
        """Wipe all documents and vectors."""
        with self._lock:
            self.documents.clear()
            self.vectors.clear()
            self._rebuild_matrix()
            self._save()

    def retrieve(
        self, query_vector: list[float], top_k: int = 5
    ) -> list[dict]:
        """Return top_k chunks ranked by cosine similarity to the query."""
        with self._lock:
            if not self._keys or self._matrix.shape[0] == 0:
                return []
            q = np.asarray(query_vector, dtype=np.float32)
            if self.dim is not None and q.shape[0] != self.dim:
                raise ValueError(
                    f"Query vector dimension ({q.shape[0]}) does not match index dimension ({self.dim}). "
                    f"Make sure you are using the same embedding model as the indexed documents."
                )
            q_norm = float(np.linalg.norm(q))
            if q_norm == 0:
                return []
            scores = self._matrix @ (q / q_norm)
            k = min(int(top_k), len(scores))
            order = np.argsort(-scores)[:k]

            out = []
            for pos in order:
                sim = float(scores[pos])
                did, idx = self._keys[int(pos)]
                meta = self.documents.get(did, {})
                out.append(
                    {
                        "source": meta.get("name", did),
                        "chunk": idx,
                        "score": round(sim, 4),
                        "text": self.chunk_text(did, idx),
                    }
                )
            return out

    def chunk_text(self, doc_id: str, idx: int) -> str:
        with self._lock:
            body = self.documents.get(doc_id, {}).get("body", [])
            try:
                return body[idx]
            except (IndexError, KeyError, TypeError):
                return ""

    def count_chunks(self) -> int:
        with self._lock:
            return len(self.vectors)
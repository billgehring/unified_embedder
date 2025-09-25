"""Lightweight retrieval engine for the shared RAG service.

This module mirrors the retrieval diagnostics used in the unified embedder
project's ``test_retrieval.py`` while adapting the logic for a production FastAPI
service.  It provides:

* Shared FastEmbed dense and sparse embedders with warm-up and thread safety.
* Direct Qdrant queries for dense and sparse vectors with Reciprocal Rank
  Fusion (RRF) score blending.
* Simple data structures for returning ranked documents alongside diagnostics.

The goal is to keep latency low (sub-second on commodity hardware) while
remaining faithful to the conventions exercised by ``test_retrieval.sh`` in the
unified embedder repository.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


try:  # Optional heavy dependency – handled gracefully when missing.
    from haystack_integrations.components.embedders.fastembed import (
        FastembedTextEmbedder,
        FastembedSparseTextEmbedder,
    )
except ImportError:  # pragma: no cover - informative runtime guard.
    FastembedTextEmbedder = None  # type: ignore[assignment]
    FastembedSparseTextEmbedder = None  # type: ignore[assignment]


try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qdrant_models
except ImportError:  # pragma: no cover - runtime guard for missing dependency.
    QdrantClient = None  # type: ignore[assignment]
    qdrant_models = None  # type: ignore[assignment]


@dataclass
class RetrievedDocument:
    """Normalized representation of a retrieved document."""

    id: str
    score: float
    content: str
    metadata: Dict[str, Any]
    retrieval_methods: List[str] = field(default_factory=list)
    raw_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class RetrievalReport:
    """Aggregate output from a retrieval call."""

    documents: List[RetrievedDocument]
    search_time_ms: float
    diagnostics: Dict[str, Any]


class EmbeddingService:
    """Thread-safe wrapper around FastEmbed dense and sparse embedders."""

    def __init__(self, dense_model: str, sparse_model: str):
        if FastembedTextEmbedder is None:
            raise ImportError(
                "FastembedTextEmbedder is unavailable. Install haystack-integrations=="  # noqa: E501
                ">=0.1.0 and fastembed-haystack to enable retrieval."
            )

        self._dense_model_name = dense_model
        self._sparse_model_name = sparse_model
        self._dense_embedder: Optional[FastembedTextEmbedder] = None
        self._sparse_embedder: Optional[FastembedSparseTextEmbedder] = None
        self._dense_lock = threading.Lock()
        self._sparse_lock = threading.Lock()

    def embed_dense(self, text: str) -> List[float]:
        """Return a dense embedding for ``text``."""

        if not text:
            raise ValueError("Text for dense embedding must be non-empty")

        with self._dense_lock:
            if self._dense_embedder is None:
                logger.info("Loading dense FastEmbed model: %s", self._dense_model_name)
                self._dense_embedder = FastembedTextEmbedder(model=self._dense_model_name)
                self._dense_embedder.warm_up()

            result = self._dense_embedder.run(text=text)
        return result["embedding"]

    def embed_sparse(self, text: str) -> Optional[qdrant_models.SparseVector]:
        """Return a Qdrant ``SparseVector`` for ``text`` if the sparse model is available."""

        if FastembedSparseTextEmbedder is None or qdrant_models is None:
            return None
        if not text:
            raise ValueError("Text for sparse embedding must be non-empty")

        with self._sparse_lock:
            if self._sparse_embedder is None:
                logger.info("Loading sparse FastEmbed model: %s", self._sparse_model_name)
                self._sparse_embedder = FastembedSparseTextEmbedder(model=self._sparse_model_name)
                self._sparse_embedder.warm_up()

            sparse_embedding = self._sparse_embedder.run(text=text)["sparse_embedding"]

        if not sparse_embedding.indices:
            return None

        return qdrant_models.SparseVector(
            indices=sparse_embedding.indices,
            values=sparse_embedding.values,
        )


class RetrievalEngine:
    """Dense + sparse retrieval with Reciprocal Rank Fusion."""

    def __init__(
        self,
        *,
        collection_name: str,
        qdrant_url: str,
        embedder_service: EmbeddingService,
        api_key: Optional[str] = None,
        dense_weight: float = 0.65,
        sparse_weight: float = 0.35,
        dense_limit: int = 12,
        sparse_limit: int = 12,
        rrf_k: int = 60,
    ):
        if QdrantClient is None:
            raise ImportError("qdrant-client is required for retrieval")

        self.collection_name = collection_name
        self._client = QdrantClient(
            url=qdrant_url,
            api_key=api_key,
            timeout=60,
            prefer_grpc=True,
        )
        self._embedder_service = embedder_service
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight
        self._dense_limit = dense_limit
        self._sparse_limit = sparse_limit
        self._rrf_k = rrf_k

    def get_collection_health(self) -> Dict[str, Any]:
        """Return lightweight collection diagnostics."""

        try:
            info = self._client.get_collection(self.collection_name)
            return {
                "points": info.points_count,
                "vectors": list((info.config.params.vectors or {}).keys()),
                "sparse": list((info.config.params.sparse_vectors or {}).keys()),
            }
        except Exception as exc:  # pragma: no cover - defensive, runtime only.
            logger.warning("Collection health check failed: %s", exc)
            return {"error": str(exc)}

    def search(self, query: str, limit: int) -> RetrievalReport:
        """Execute dense+sparse retrieval with RRF fusion."""

        if not query or not query.strip():
            raise ValueError("Query text must be provided")

        limit = max(1, limit)
        start_time = time.perf_counter()

        dense_hits, dense_time = self._search_dense(query, self._dense_limit)
        sparse_hits, sparse_time = self._search_sparse(query, self._sparse_limit)

        documents = self._fuse_hits(
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            final_limit=limit,
        )

        total_time = (time.perf_counter() - start_time) * 1000

        diagnostics = {
            "dense_hits": len(dense_hits),
            "sparse_hits": len(sparse_hits),
            "dense_time_ms": dense_time,
            "sparse_time_ms": sparse_time,
            "rrf_k": self._rrf_k,
        }

        return RetrievalReport(
            documents=documents,
            search_time_ms=total_time,
            diagnostics=diagnostics,
        )

    # --- Internal helpers -------------------------------------------------

    def _search_dense(self, query: str, limit: int) -> Tuple[List[Any], float]:
        """Return dense hits and latency."""

        try:
            start = time.perf_counter()
            query_vector = self._embedder_service.embed_dense(query)
            response = self._client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using="text-dense",
                limit=limit,
                with_payload=True,
            )
            duration = (time.perf_counter() - start) * 1000
            return list(response.points), duration
        except Exception as exc:
            logger.warning("Dense search failed: %s", exc)
            return [], 0.0

    def _search_sparse(self, query: str, limit: int) -> Tuple[List[Any], float]:
        """Return sparse hits and latency."""

        sparse_vector = None
        try:
            sparse_vector = self._embedder_service.embed_sparse(query)
        except Exception as exc:
            logger.debug("Sparse embedding unavailable: %s", exc)

        if sparse_vector is None:
            return [], 0.0

        try:
            start = time.perf_counter()
            response = self._client.query_points(
                collection_name=self.collection_name,
                query=sparse_vector,
                using="text-sparse",
                limit=limit,
                with_payload=True,
            )
            duration = (time.perf_counter() - start) * 1000
            return list(response.points), duration
        except Exception as exc:
            logger.warning("Sparse search failed: %s", exc)
            return [], 0.0

    def _fuse_hits(
        self,
        *,
        dense_hits: List[Any],
        sparse_hits: List[Any],
        final_limit: int,
    ) -> List[RetrievedDocument]:
        """Apply Reciprocal Rank Fusion to dense and sparse hits."""

        if not dense_hits and not sparse_hits:
            return []

        fused: Dict[str, Dict[str, Any]] = {}

        def add_hits(hits: List[Any], method: str, weight: float) -> None:
            if weight <= 0:
                return
            for rank, hit in enumerate(hits):
                doc_id = str(hit.id)
                payload = hit.payload or {}
                entry = fused.setdefault(
                    doc_id,
                    {
                        "rrf_score": 0.0,
                        "raw_scores": {},
                        "methods": set(),
                        "content": self._extract_content(payload),
                        "metadata": self._extract_metadata(payload),
                    },
                )

                entry["rrf_score"] += weight / (self._rrf_k + rank)
                entry["raw_scores"][method] = float(hit.score)
                entry["methods"].add(method)

        add_hits(dense_hits, "dense", self._dense_weight)
        add_hits(sparse_hits, "sparse", self._sparse_weight)

        documents = [
            RetrievedDocument(
                id=doc_id,
                score=values["rrf_score"],
                content=values["content"],
                metadata=values["metadata"],
                retrieval_methods=sorted(values["methods"]),
                raw_scores=values["raw_scores"],
            )
            for doc_id, values in fused.items()
        ]

        documents.sort(key=lambda doc: doc.score, reverse=True)
        return documents[:final_limit]

    @staticmethod
    def _extract_content(payload: Dict[str, Any]) -> str:
        content = payload.get("content") or payload.get("text") or ""
        if isinstance(content, list):
            content = "\n".join(str(item) for item in content if item)
        return str(content)

    @staticmethod
    def _extract_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
        if "meta" in payload and isinstance(payload["meta"], dict):
            return payload["meta"]
        return {k: v for k, v in payload.items() if k not in {"content", "text"}}

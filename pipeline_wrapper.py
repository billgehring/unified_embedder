"""Pipeline management for the shared RAG service.

The previous iteration of this project delegated to a legacy Haystack pipeline
that has since diverged from the unified embedder repository.  This module
introduces a leaner manager that:

* Re-uses the dense/sparse retrieval stack validated by
  ``unified_embedder/test_retrieval.py`` via :mod:`retrieval_engine`.
* Preserves basic conversation tracking with lightweight in-memory storage.
* Offers optional OpenAI completion while providing an offline-friendly
  fallback so functional testing does not require network access.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import settings
from models import (
    ConversationResetResponse,
    QueryResponse,
    SourceDocument,
)
from retrieval_engine import EmbeddingService, RetrievalEngine, RetrievedDocument


logger = logging.getLogger(__name__)


try:  # Optional dependency – the service works without OpenAI.
    from openai import OpenAI
except ImportError:  # pragma: no cover - harmless when OpenAI SDK missing.
    OpenAI = None  # type: ignore[assignment]


@dataclass
class PipelineStats:
    """Operational statistics for observability endpoints."""

    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    average_latency_ms: float = 0.0
    last_error: Optional[str] = None
    last_query_ms: Optional[float] = None


class AnswerSynthesizer:
    """Combine retrieved context into a final answer."""

    def __init__(self, *, model: str, temperature: float, max_tokens: int, api_key: Optional[str]):
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._api_key = api_key
        self._client: Optional[OpenAI] = None
        self._client_lock = asyncio.Lock()

    async def generate(
        self,
        *,
        query: str,
        documents: List[RetrievedDocument],
        conversation_history: List[Dict[str, str]],
    ) -> Dict[str, Optional[float]]:
        """Return dict with ``text`` and ``confidence``."""

        # Prefer deterministic fallback; initialize OpenAI lazily if available.
        await self._ensure_client()
        if self._client:
            try:
                return await self._call_openai(query=query, documents=documents, history=conversation_history)
            except Exception as exc:  # pragma: no cover - depends on external API.
                logger.warning("OpenAI generation failed: %s", exc)

        return self._fallback_answer(query=query, documents=documents)

    async def _ensure_client(self) -> None:
        if self._client or not self._api_key or OpenAI is None:
            return
        async with self._client_lock:
            if self._client is None and self._api_key and OpenAI is not None:  # double-check after lock
                try:
                    self._client = OpenAI(api_key=self._api_key)
                    logger.info("OpenAI client initialised for model %s", self._model)
                except Exception as exc:  # pragma: no cover - depends on SDK availability.
                    logger.warning("Unable to initialise OpenAI client: %s", exc)

    async def _call_openai(
        self,
        *,
        query: str,
        documents: List[RetrievedDocument],
        history: List[Dict[str, str]],
    ) -> Dict[str, Optional[float]]:
        assert self._client is not None  # Guarded by _ensure_client

        loop = asyncio.get_running_loop()
        prompt = self._build_prompt(query=query, documents=documents, history=history)

        def _request() -> str:
            response = self._client.responses.create(
                model=self._model,
                input=prompt,
                temperature=self._temperature,
                max_output_tokens=self._max_tokens,
            )
            try:
                return response.output[0].content[0].text  # type: ignore[index]
            except Exception:  # pragma: no cover - defensive
                return str(response)

        text = await loop.run_in_executor(None, _request)
        confidence = self._derive_confidence(documents)
        return {"text": text, "confidence": confidence}

    @staticmethod
    def _derive_confidence(documents: List[RetrievedDocument]) -> Optional[float]:
        if not documents:
            return None
        dense_score = documents[0].raw_scores.get("dense", documents[0].score)
        normalized = (dense_score + 1.0) / 2.0  # Cosine similarity -> [0, 1]
        return max(0.0, min(1.0, normalized))

    @staticmethod
    def _build_prompt(
        *,
        query: str,
        documents: List[RetrievedDocument],
        history: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        history_messages = [
            {"role": item.get("role", "user"), "content": item.get("content", "")} for item in history
        ]
        context_blocks = []
        for idx, doc in enumerate(documents[:5], start=1):
            snippet = doc.content.strip().replace("\n", " ")
            if len(snippet) > 500:
                snippet = f"{snippet[:500]}..."
            context_blocks.append(f"[{idx}] score={doc.score:.4f} source={doc.metadata.get('filename', 'unknown')}\n{snippet}")

        system_prompt = (
            "You are an educational tutor. Answer the user's question using the provided retrieved context. "
            "Cite the source numbers when possible and do not fabricate information."
        )
        user_prompt = (
            f"Question: {query}\n\n"
            f"Context:\n{chr(10).join(context_blocks) if context_blocks else 'No context available.'}\n\n"
            "Respond with a concise, accurate explanation."
        )

        return [
            {"role": "system", "content": system_prompt},
            *history_messages,
            {"role": "user", "content": user_prompt},
        ]

    def _fallback_answer(self, *, query: str, documents: List[RetrievedDocument]) -> Dict[str, Optional[float]]:
        if not documents:
            return {
                "text": (
                    "I could not find relevant context for this question. Please run the embedder "
                    "pipeline to populate the Qdrant collection and try again."
                ),
                "confidence": None,
            }

        bullet_lines = []
        for doc in documents[:3]:
            snippet = doc.content.strip().replace("\n", " ")
            if len(snippet) > 220:
                snippet = f"{snippet[:220]}..."
            bullet_lines.append(f"• {snippet}")

        answer_text = (
            f"I could not contact the language model, so here are the top matches for '{query}':\n"
            + "\n".join(bullet_lines)
        )
        return {"text": answer_text, "confidence": self._derive_confidence(documents)}


class ProjectPipeline:
    """Encapsulates retrieval, answer synthesis, and conversation memory."""

    def __init__(self, project: str, config: Dict[str, any], embedder_service: EmbeddingService):
        self.project = project
        self.config = config
        self.retrieval = RetrievalEngine(
            collection_name=config["qdrant_collection"],
            qdrant_url=config["qdrant_url"],
            api_key=config.get("qdrant_api_key"),
            embedder_service=embedder_service,
        )
        self.answer = AnswerSynthesizer(
            model=config["llm_model"],
            temperature=config["llm_temperature"],
            max_tokens=config["llm_max_tokens"],
            api_key=settings.OPENAI_API_KEY,
        )
        self._history: Dict[str, List[Dict[str, str]]] = {}
        self._history_lock = asyncio.Lock()

    async def query(
        self,
        *,
        text: str,
        max_results: int,
        conversation_id: Optional[str],
    ) -> Dict[str, any]:
        conversation_id = conversation_id or self._create_conversation_id()

        retrieval_report = await self._run_retrieval(text=text, limit=max_results)
        answer_start = time.perf_counter()
        history = await self._get_history(conversation_id)
        answer_payload = await self.answer.generate(
            query=text,
            documents=retrieval_report["documents"],
            conversation_history=history,
        )
        answer_time_ms = (time.perf_counter() - answer_start) * 1000

        await self._append_history(
            conversation_id,
            question=text,
            answer=answer_payload["text"],
        )

        return {
            "conversation_id": conversation_id,
            "documents": retrieval_report["documents"],
            "diagnostics": retrieval_report["diagnostics"],
            "retrieval_time_ms": retrieval_report["search_time_ms"],
            "answer_time_ms": answer_time_ms,
            "answer": answer_payload["text"],
            "confidence": answer_payload.get("confidence"),
        }

    async def reset(self, conversation_id: Optional[str]) -> None:
        async with self._history_lock:
            if conversation_id:
                self._history.pop(conversation_id, None)
            else:
                self._history.clear()

    async def _run_retrieval(self, *, text: str, limit: int) -> Dict[str, any]:
        loop = asyncio.get_running_loop()
        report = await loop.run_in_executor(None, self.retrieval.search, text, limit)
        return {
            "documents": report.documents,
            "diagnostics": report.diagnostics,
            "search_time_ms": report.search_time_ms,
        }

    async def _get_history(self, conversation_id: str) -> List[Dict[str, str]]:
        async with self._history_lock:
            return list(self._history.get(conversation_id, []))

    async def _append_history(self, conversation_id: str, *, question: str, answer: str) -> None:
        async with self._history_lock:
            history = self._history.setdefault(conversation_id, [])
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})

    @staticmethod
    def _create_conversation_id() -> str:
        return uuid.uuid4().hex[:12]


class PipelineManager:
    """Factory and cache for :class:`ProjectPipeline` instances."""

    def __init__(self) -> None:
        self._pipelines: Dict[str, ProjectPipeline] = {}
        self._stats: Dict[str, PipelineStats] = {project: PipelineStats() for project in settings.SUPPORTED_PROJECTS}
        self._embedder_service = EmbeddingService(
            dense_model=settings.DENSE_MODEL,
            sparse_model=settings.SPARSE_MODEL,
        )
        self.initialized = False

    async def initialize(self) -> None:
        logger.info("Initialising pipeline manager for projects: %s", settings.SUPPORTED_PROJECTS)
        await self._verify_qdrant()
        self.initialized = True

    async def cleanup(self) -> None:  # pragma: no cover - nothing persistent yet.
        self._pipelines.clear()

    async def run_query(
        self,
        *,
        project: str,
        text: str,
        max_results: int,
        conversation_id: Optional[str],
        include_sources: bool,
        include_methods: bool,
    ) -> QueryResponse:
        start = time.perf_counter()
        stats = self._stats.setdefault(project, PipelineStats())
        stats.total_queries += 1

        pipeline = self._get_or_create_pipeline(project)

        try:
            result = await pipeline.query(
                text=text,
                max_results=max_results,
                conversation_id=conversation_id,
            )
            total_time_ms = (time.perf_counter() - start) * 1000

            stats.successful_queries += 1
            stats.last_query_ms = total_time_ms
            stats.average_latency_ms = self._update_running_average(
                stats.average_latency_ms,
                total_time_ms,
                stats.successful_queries,
            )

            sources = self._format_sources(
                documents=result["documents"],
                include_sources=include_sources,
                include_methods=include_methods,
            )

            return QueryResponse(
                answer=result["answer"],
                conversation_id=result["conversation_id"],
                project=project,
                source_documents=sources,
                confidence=result["confidence"],
                processing_time_ms=int(total_time_ms),
            )
        except Exception as exc:
            stats.failed_queries += 1
            stats.last_error = str(exc)
            raise

    async def reset_conversation(self, project: str, conversation_id: Optional[str]) -> ConversationResetResponse:
        pipeline = self._get_or_create_pipeline(project)
        await pipeline.reset(conversation_id)
        return ConversationResetResponse(
            success=True,
            message="Conversation memory reset successfully",
            conversation_id=conversation_id,
            project=project,
        )

    def get_stats(self) -> Dict[str, PipelineStats]:
        return self._stats

    def get_collection_health(self, project: str) -> Dict[str, any]:
        pipeline = self._get_or_create_pipeline(project)
        return pipeline.retrieval.get_collection_health()

    async def _verify_qdrant(self) -> None:
        import httpx

        url = f"{settings.QDRANT_URL.rstrip('/')}/healthz"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
                if response.status_code != 200:
                    logger.warning("Qdrant health endpoint returned %s", response.status_code)
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("Qdrant health check failed: %s", exc)

    def _get_or_create_pipeline(self, project: str) -> ProjectPipeline:
        key = project.lower()
        if key not in self._pipelines:
            config = settings.get_pipeline_config(project)
            logger.info(
                "Creating pipeline for project=%s collection=%s",
                project,
                config["qdrant_collection"],
            )
            self._pipelines[key] = ProjectPipeline(
                project=project,
                config=config,
                embedder_service=self._embedder_service,
            )
        return self._pipelines[key]

    @staticmethod
    def _format_sources(
        *,
        documents: List[RetrievedDocument],
        include_sources: bool,
        include_methods: bool,
    ) -> Optional[List[SourceDocument]]:
        if not include_sources or not documents:
            return None

        formatted: List[SourceDocument] = []
        for rank, doc in enumerate(documents, start=1):
            preview = doc.content.strip().replace("\n", " ")
            if len(preview) > 320:
                preview = f"{preview[:320]}..."

            metadata = dict(doc.metadata)
            metadata.setdefault("scores", doc.raw_scores)

            formatted.append(
                SourceDocument(
                    rank=rank,
                    id=doc.id,
                    score=doc.score,
                    content_preview=preview,
                    metadata=metadata,
                    retrieval_method="+".join(doc.retrieval_methods) if (include_methods and doc.retrieval_methods) else None,
                )
            )

        return formatted

    @staticmethod
    def _update_running_average(previous_avg: float, new_value: float, count: int) -> float:
        if count <= 1:
            return new_value
        return previous_avg + (new_value - previous_avg) / count


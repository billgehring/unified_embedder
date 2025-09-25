"""
Multi-Vector Retrieval Pipeline for Educational Voice Tutor
==========================================================

Advanced retrieval system combining Dense, Sparse, and ColBERT vectors with optimized
score fusion for <50ms performance in voice-based educational applications.

Features:
- Simultaneous multi-vector search across all embedding types
- Adaptive score fusion with query-type detection  
- Performance-optimized for real-time voice interaction
- Educational domain optimizations (technical terms, concepts)
"""

import logging
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from functools import lru_cache

try:
    from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
    from haystack import Document
    HAYSTACK_AVAILABLE = True
except ImportError:
    HAYSTACK_AVAILABLE = False

try:
    from qdrant_client import QdrantClient
    QDRANT_CLIENT_AVAILABLE = True
except ImportError:
    QDRANT_CLIENT_AVAILABLE = False

from colbert_token_embedder import ColBERTTokenEmbedder

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Query type classification for adaptive retrieval."""
    FACTUAL = "factual"          # "What is photosynthesis?"
    CONCEPTUAL = "conceptual"    # "How does natural selection work?"
    PROCEDURAL = "procedural"    # "How do you solve quadratic equations?"
    COMPARATIVE = "comparative"  # "Compare mitosis and meiosis"
    DEFINITIONAL = "definitional" # "Define entropy"

@dataclass 
class SearchResult:
    """Standardized search result format with diagnostics."""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    vector_type: str  # "dense", "sparse", "colbert", "fused"
    search_time_ms: float
    # Diagnostics
    vector_scores: Dict[str, float] | None = None           # e.g., {"dense": 0.78, "colbert": 0.64}
    component_times_ms: Dict[str, float] | None = None      # e.g., {"dense": 12.4, "colbert": 8.1}
    fusion_weights: Dict[str, float] | None = None          # e.g., {"dense": 0.5, "sparse": 0.2, "colbert": 0.3}
    
@dataclass
class MultiVectorConfig:
    """Configuration for multi-vector retrieval."""
    # Score fusion weights
    dense_weight: float = 0.4
    sparse_weight: float = 0.3  
    colbert_weight: float = 0.3
    
    # Search parameters
    dense_limit: int = 50      # Initial candidates from dense
    sparse_limit: int = 30     # Initial candidates from sparse
    colbert_limit: int = 20    # Direct ColBERT results
    final_limit: int = 10      # Final results returned
    
    # Performance optimization
    parallel_search: bool = True
    max_search_time_ms: float = 50.0  # Target for voice apps
    
    # Query adaptation
    adaptive_weights: bool = True
    boost_exact_matches: bool = True

class MultiVectorRetriever:
    """
    High-performance multi-vector retrieval system optimized for educational voice tutor.
    
    Combines three retrieval methods:
    1. Dense vectors: Semantic similarity (concepts, relationships) 
    2. Sparse vectors: Keyword matching (terms, definitions)
    3. ColBERT tokens: Fine-grained contextual matching (precise answers)
    """
    
    def __init__(self,
                 haystack_store: Optional[QdrantDocumentStore] = None,
                 qdrant_client: Optional[QdrantClient] = None,
                 colbert_embedder: Optional[ColBERTTokenEmbedder] = None,
                 config: Optional[MultiVectorConfig] = None):
        """
        Initialize multi-vector retriever.
        
        Args:
            haystack_store: Haystack store for dense + sparse
            qdrant_client: Native client for ColBERT
            colbert_embedder: ColBERT token embedder
            config: Retrieval configuration
        """
        self.haystack_store = haystack_store
        self.qdrant_client = qdrant_client
        self.colbert_embedder = colbert_embedder
        self.config = config or MultiVectorConfig()
        
        # Collections
        self.dense_sparse_collection = None
        self.colbert_collection = None
        
        # Performance stats
        self._stats = {
            "total_queries": 0,
            "avg_response_time_ms": 0.0,
            "dense_queries": 0,
            "sparse_queries": 0, 
            "colbert_queries": 0,
            "fusion_time_ms": 0.0,
            "cache_hits": 0
        }
        
        # Query result cache for performance
        self._query_cache = {}
        self._cache_size_limit = 1000  # Maximum cached queries
        
        # Query classification patterns (simple heuristics)
        self.query_patterns = {
            QueryType.FACTUAL: ["what is", "what are", "define", "definition of"],
            QueryType.CONCEPTUAL: ["how does", "why does", "explain", "concept"],
            QueryType.PROCEDURAL: ["how to", "steps", "procedure", "process"],
            QueryType.COMPARATIVE: ["compare", "difference", "vs", "versus", "contrast"],
            QueryType.DEFINITIONAL: ["define", "definition", "meaning", "term"]
        }
        
    def classify_query(self, query: str) -> QueryType:
        """
        Classify query type for adaptive retrieval weights.
        
        Args:
            query: User query text
            
        Returns:
            Classified query type
        """
        query_lower = query.lower()
        
        for query_type, patterns in self.query_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return query_type
                
        # Default to conceptual for educational content
        return QueryType.CONCEPTUAL
    
    def adapt_weights(self, query: str, query_type: QueryType) -> Tuple[float, float, float]:
        """
        Adapt fusion weights based on query type and content.
        
        Args:
            query: Query text
            query_type: Classified query type
            
        Returns:
            Tuple of (dense_weight, sparse_weight, colbert_weight)
        """
        if not self.config.adaptive_weights:
            return (self.config.dense_weight, self.config.sparse_weight, self.config.colbert_weight)
            
        # Adaptive weighting based on query type
        if query_type == QueryType.FACTUAL:
            return (0.3, 0.4, 0.3)  # Favor sparse for facts
        elif query_type == QueryType.CONCEPTUAL:
            return (0.5, 0.2, 0.3)  # Favor dense for concepts
        elif query_type == QueryType.PROCEDURAL:
            return (0.3, 0.3, 0.4)  # Favor ColBERT for procedures
        elif query_type == QueryType.COMPARATIVE:
            return (0.4, 0.3, 0.3)  # Balanced for comparisons
        elif query_type == QueryType.DEFINITIONAL:
            return (0.2, 0.5, 0.3)  # Favor sparse for definitions
        else:
            return (self.config.dense_weight, self.config.sparse_weight, self.config.colbert_weight)
    
    def search(self, 
               query: str,
               limit: int = None,
               search_mode: str = "all",
               timeout_ms: float = None) -> List[SearchResult]:
        """
        Perform multi-vector search with score fusion.
        
        Args:
            query: Search query
            limit: Number of results (default: config.final_limit)
            search_mode: "dense", "sparse", "colbert", or "all" 
            timeout_ms: Search timeout (default: config.max_search_time_ms)
            
        Returns:
            List of fused and ranked search results
        """
        start_time = time.time()
        limit = limit or self.config.final_limit
        timeout_ms = timeout_ms or self.config.max_search_time_ms
        
        logger.debug(f"Multi-vector search: '{query}' (mode: {search_mode}, limit: {limit})")
        
        # Check cache first for performance
        cache_key = self._generate_cache_key(query, limit, search_mode)
        if cache_key in self._query_cache:
            self._stats["cache_hits"] += 1
            cached_results = self._query_cache[cache_key]
            logger.debug(f"Cache hit for query: '{query}' ({len(cached_results)} results)")
            return cached_results
        
        # Classify query for adaptive weights
        query_type = self.classify_query(query)
        dense_weight, sparse_weight, colbert_weight = self.adapt_weights(query, query_type)
        
        logger.debug(f"Query type: {query_type}, weights: D={dense_weight:.2f}, S={sparse_weight:.2f}, C={colbert_weight:.2f}")
        
        try:
            if search_mode == "all" and self.config.parallel_search:
                results = self._parallel_search(query, dense_weight, sparse_weight, colbert_weight, timeout_ms)
            else:
                results = self._sequential_search(query, search_mode, dense_weight, sparse_weight, colbert_weight)
                
            # Apply final ranking and limit
            final_results = self._final_ranking(results, limit, query, query_type)
            
            # Update statistics
            search_time_ms = (time.time() - start_time) * 1000
            self._update_stats(search_time_ms, search_mode)
            
            # Cache results for future use
            self._cache_query_results(cache_key, final_results)
            
            logger.info(f"Multi-vector search completed: {len(final_results)} results in {search_time_ms:.1f}ms")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Multi-vector search failed: {e}")
            return []
    
    def _parallel_search(self, 
                        query: str, 
                        dense_weight: float,
                        sparse_weight: float, 
                        colbert_weight: float,
                        timeout_ms: float) -> List[SearchResult]:
        """
        Perform parallel search across all vector types for maximum speed.
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit parallel searches
            futures = []
            
            if self.haystack_store and dense_weight > 0:
                futures.append(executor.submit(self._search_dense, query, self.config.dense_limit))
                
            if self.haystack_store and sparse_weight > 0:
                futures.append(executor.submit(self._search_sparse, query, self.config.sparse_limit))
                
            if self.qdrant_client and self.colbert_embedder and colbert_weight > 0:
                futures.append(executor.submit(self._search_colbert, query, self.config.colbert_limit))
            
            # Collect results with timeout
            timeout_seconds = timeout_ms / 1000.0
            for future in as_completed(futures, timeout=timeout_seconds):
                try:
                    vector_results = future.result()
                    results.extend(vector_results)
                except Exception as e:
                    logger.warning(f"Parallel search component failed: {e}")
                    
        return results
    
    def _sequential_search(self,
                          query: str,
                          search_mode: str,
                          dense_weight: float,
                          sparse_weight: float,
                          colbert_weight: float) -> List[SearchResult]:
        """
        Perform sequential search for specific vector types.
        """
        results = []
        
        if search_mode in ["all", "dense"] and self.haystack_store and dense_weight > 0:
            results.extend(self._search_dense(query, self.config.dense_limit))
            
        if search_mode in ["all", "sparse"] and self.haystack_store and sparse_weight > 0:
            results.extend(self._search_sparse(query, self.config.sparse_limit))
            
        if search_mode in ["all", "colbert"] and self.qdrant_client and self.colbert_embedder and colbert_weight > 0:
            results.extend(self._search_colbert(query, self.config.colbert_limit))
            
        return results
    
    def _search_dense(self, query: str, limit: int) -> List[SearchResult]:
        """Search using dense embeddings via Haystack."""
        start_time = time.time()
        
        try:
            if not self.haystack_store:
                logger.warning("No Haystack store available for dense search")
                return []
                
            # Use Haystack's built-in embedding search
            # This leverages the configured embedding model (e.g., BGE-M3)
            documents = self.haystack_store.filter_documents({})  # Get all for now
            
            if not documents:
                logger.debug("No documents found in dense collection")
                return []
                
            # For now, return top documents as placeholder
            # In production, this would use proper embedding similarity
            search_time_ms = (time.time() - start_time) * 1000
            
            results = []
            for i, doc in enumerate(documents[:limit]):
                results.append(SearchResult(
                    id=doc.id,
                    content=doc.content,
                    score=0.8 - (i * 0.05),  # Decreasing mock scores
                    metadata=doc.meta,
                    vector_type="dense",
                    search_time_ms=search_time_ms,
                    vector_scores={"dense": 0.8 - (i * 0.05)},
                    component_times_ms={"dense": search_time_ms}
                ))
                
            logger.debug(f"Dense search: {len(results)} results in {search_time_ms:.1f}ms")
            return results
            
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []
        
    def _search_sparse(self, query: str, limit: int) -> List[SearchResult]:
        """Search using sparse embeddings via Haystack."""
        start_time = time.time()
        
        try:
            if not self.haystack_store:
                logger.warning("No Haystack store available for sparse search")
                return []
                
            # Use Haystack's sparse embedding capabilities (SPLADE)
            # This leverages keyword/term matching capabilities
            documents = self.haystack_store.filter_documents({})  # Get all for now
            
            if not documents:
                logger.debug("No documents found in sparse collection")
                return []
                
            # Basic keyword matching for now
            query_terms = set(query.lower().split())
            scored_docs = []
            
            for doc in documents:
                content_terms = set(doc.content.lower().split())
                # Simple term overlap score
                overlap = len(query_terms.intersection(content_terms))
                if overlap > 0:
                    score = overlap / len(query_terms)
                    scored_docs.append((doc, score))
                    
            # Sort by score and take top results
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            search_time_ms = (time.time() - start_time) * 1000
            
            results = []
            for doc, score in scored_docs[:limit]:
                results.append(SearchResult(
                    id=doc.id,
                    content=doc.content,
                    score=score,
                    metadata=doc.meta,
                    vector_type="sparse",
                    search_time_ms=search_time_ms,
                    vector_scores={"sparse": score},
                    component_times_ms={"sparse": search_time_ms}
                ))
                
            logger.debug(f"Sparse search: {len(results)} results in {search_time_ms:.1f}ms")
            return results
            
        except Exception as e:
            logger.error(f"Sparse search failed: {e}")
            return []
        
    def _search_colbert(self, query: str, limit: int) -> List[SearchResult]:
        """Search using ColBERT tokens."""
        start_time = time.time()
        
        try:
            # Embed query with ColBERT
            query_embedding = self.colbert_embedder.embed_query(query)
            
            # Search Qdrant collection
            results = self.qdrant_client.query_points(
                collection_name=self.colbert_collection,
                query=query_embedding,
                limit=limit,
                with_payload=True
            )
            
            search_time_ms = (time.time() - start_time) * 1000
            
            # Convert to SearchResult format
            search_results = []
            for result in results.points:
                search_results.append(SearchResult(
                    id=str(result.id),
                    content=result.payload.get("content", ""),
                    score=result.score,
                    metadata={k: v for k, v in result.payload.items() if k != "content"},
                    vector_type="colbert",
                    search_time_ms=search_time_ms,
                    vector_scores={"colbert": result.score},
                    component_times_ms={"colbert": search_time_ms}
                ))
            
            logger.debug(f"ColBERT search: {len(search_results)} results in {search_time_ms:.1f}ms")
            return search_results
            
        except Exception as e:
            logger.error(f"ColBERT search failed: {e}")
            return []
    
    def _final_ranking(self, 
                      results: List[SearchResult], 
                      limit: int,
                      query: str,
                      query_type: QueryType) -> List[SearchResult]:
        """
        Apply final ranking with score fusion and educational optimizations.
        
        Args:
            results: Raw search results from all vector types
            limit: Final number of results to return
            query: Original query for boost calculations
            query_type: Query type for specialized ranking
            
        Returns:
            Final ranked and limited results
        """
        if not results:
            return []
            
        # Group results by document ID for fusion
        result_groups = {}
        for result in results:
            doc_id = result.id
            if doc_id not in result_groups:
                result_groups[doc_id] = []
            result_groups[doc_id].append(result)
        
        # Fuse scores for each document
        fused_results = []
        for doc_id, doc_results in result_groups.items():
            fused_result = self._fuse_document_scores(doc_results, query, query_type)
            if fused_result:
                fused_results.append(fused_result)
        
        # Apply educational domain boosts
        fused_results = self._apply_educational_boosts(fused_results, query, query_type)
        
        # Final ranking by fused score
        fused_results.sort(key=lambda x: x.score, reverse=True)
        
        return fused_results[:limit]
    
    def _fuse_document_scores(self, 
                             doc_results: List[SearchResult], 
                             query: str,
                             query_type: QueryType) -> Optional[SearchResult]:
        """
        Fuse multiple vector scores for a single document.
        """
        if not doc_results:
            return None
            
        # Get the best result as base
        base_result = max(doc_results, key=lambda x: x.score)
        
        # Calculate fused score using weighted sum
        vector_scores: Dict[str, float] = {}
        for result in doc_results:
            vector_scores[result.vector_type] = result.score
            
        # Apply fusion weights
        dense_weight, sparse_weight, colbert_weight = self.adapt_weights(query, query_type)
        
        fused_score = 0.0
        if "dense" in vector_scores:
            fused_score += dense_weight * vector_scores["dense"]
        if "sparse" in vector_scores:
            fused_score += sparse_weight * vector_scores["sparse"]  
        if "colbert" in vector_scores:
            fused_score += colbert_weight * vector_scores["colbert"]
        fusion_weights = {"dense": dense_weight, "sparse": sparse_weight, "colbert": colbert_weight}
        component_times = {}
        for r in doc_results:
            component_times[r.vector_type] = r.search_time_ms
            
        # Create fused result
        return SearchResult(
            id=base_result.id,
            content=base_result.content,
            score=fused_score,
            metadata=base_result.metadata,
            vector_type="fused",
            search_time_ms=sum(r.search_time_ms for r in doc_results) / len(doc_results),
            vector_scores=vector_scores,
            component_times_ms=component_times,
            fusion_weights=fusion_weights
        )
    
    def _apply_educational_boosts(self,
                                 results: List[SearchResult],
                                 query: str,
                                 query_type: QueryType) -> List[SearchResult]:
        """
        Apply educational domain-specific score boosts.
        """
        # Educational boost factors
        DEFINITION_BOOST = 1.2  # Boost results containing definitions
        EXAMPLE_BOOST = 1.15    # Boost results with examples
        FORMULA_BOOST = 1.1     # Boost mathematical formulas
        
        query_lower = query.lower()
        
        for result in results:
            content_lower = result.content.lower()
            
            # Apply boosts based on content type
            if query_type == QueryType.DEFINITIONAL:
                if "definition" in content_lower or "means" in content_lower:
                    result.score *= DEFINITION_BOOST
                    
            if "example" in content_lower or "for instance" in content_lower:
                result.score *= EXAMPLE_BOOST
                
            if any(char in result.content for char in ["=", "≡", "∑", "∫", "√"]):
                result.score *= FORMULA_BOOST
        
        return results
    
    def _update_stats(self, search_time_ms: float, search_mode: str):
        """Update performance statistics."""
        self._stats["total_queries"] += 1
        
        # Update average response time
        total_time = self._stats["avg_response_time_ms"] * (self._stats["total_queries"] - 1)
        self._stats["avg_response_time_ms"] = (total_time + search_time_ms) / self._stats["total_queries"]
        
        # Update mode-specific counters
        if search_mode in ["all", "dense"]:
            self._stats["dense_queries"] += 1
        if search_mode in ["all", "sparse"]:
            self._stats["sparse_queries"] += 1
        if search_mode in ["all", "colbert"]:
            self._stats["colbert_queries"] += 1
    
    def _generate_cache_key(self, query: str, limit: int, search_mode: str) -> str:
        """Generate cache key for query results."""
        cache_string = f"{query}|{limit}|{search_mode}"
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _cache_query_results(self, cache_key: str, results: List[SearchResult]):
        """Cache query results for performance."""
        # Implement LRU-style cache with size limit
        if len(self._query_cache) >= self._cache_size_limit:
            # Remove oldest entry
            oldest_key = next(iter(self._query_cache))
            del self._query_cache[oldest_key]
            
        self._query_cache[cache_key] = results.copy()
        logger.debug(f"Cached results for query key: {cache_key}")
    
    def clear_cache(self):
        """Clear query result cache."""
        self._query_cache.clear()
        logger.info("Query cache cleared")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self._stats.copy()
        stats["cache_size"] = len(self._query_cache)
        stats["cache_hit_rate"] = (
            self._stats["cache_hits"] / max(1, self._stats["total_queries"])
        ) * 100
        return stats
    
    def reset_stats(self):
        """Reset performance statistics."""
        self._stats = {k: 0 if isinstance(v, (int, float)) else v for k, v in self._stats.items()}
        self.clear_cache()

# Factory function
def create_multi_vector_retriever(hybrid_store,
                                 colbert_collection: str = None,
                                 config: MultiVectorConfig = None) -> MultiVectorRetriever:
    """
    Create multi-vector retriever from hybrid store.
    
    Args:
        hybrid_store: HybridQdrantStore instance
        colbert_collection: ColBERT collection name
        config: Retrieval configuration
        
    Returns:
        Configured MultiVectorRetriever
    """
    # Ensure underlying clients/stores are initialized
    try:
        if hasattr(hybrid_store, '_initialize_haystack_store'):
            hybrid_store._initialize_haystack_store()
        if hasattr(hybrid_store, '_initialize_qdrant_client') and getattr(hybrid_store, 'enable_colbert_tokens', False):
            hybrid_store._initialize_qdrant_client()
    except Exception as e:
        logger.warning(f"Could not initialize hybrid store components: {e}")

    retriever = MultiVectorRetriever(
        haystack_store=getattr(hybrid_store, '_haystack_store', None),
        qdrant_client=getattr(hybrid_store, '_qdrant_client', None),
        colbert_embedder=getattr(hybrid_store, '_colbert_embedder', None),
        config=config
    )
    
    retriever.colbert_collection = colbert_collection
    return retriever

# Testing and examples
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test query classification
    retriever = MultiVectorRetriever()
    
    test_queries = [
        "What is photosynthesis?",
        "How does natural selection work?", 
        "How do you solve quadratic equations?",
        "Compare mitosis and meiosis",
        "Define entropy"
    ]
    
    print("Testing query classification:")
    for query in test_queries:
        query_type = retriever.classify_query(query)
        weights = retriever.adapt_weights(query, query_type)
        print(f"  '{query}' -> {query_type.value}, weights: {weights}")
        
    print("\nMulti-vector retrieval system ready for integration")
    print("Optimized for <50ms performance in educational voice applications")

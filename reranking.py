"""
Advanced Reranking Module for Unified Embedder
=============================================

This module provides reranking capabilities to improve retrieval quality
using state-of-the-art reranking models like bge-reranker-v2-m3.
"""

import logging
import traceback
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import sys

try:
    from sentence_transformers import CrossEncoder
    from haystack import Document
    from haystack.utils import Secret
    from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
    from haystack_integrations.components.embedders.fastembed import FastembedTextEmbedder
    from qdrant_client import QdrantClient
    from query_expansion import integrate_query_expansion_into_retrieval
except ImportError as e:
    print(f"Required libraries not installed: {e}")
    sys.exit(1)

# Configure logging
logger = logging.getLogger(__name__)

class AdvancedReranker:
    """Advanced reranking with multiple strategies and model support."""
    
    def __init__(self, 
                 model_name: str = "BAAI/bge-reranker-v2-m3",
                 max_length: int = 1024,
                 device: str = "cpu"):
        """
        Initialize the reranker.
        
        Args:
            model_name: Reranking model name (supports bge-reranker series)
            max_length: Maximum sequence length for reranker
            device: Device to run on (cpu/cuda)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        self.reranker = None
        
        # Performance tracking
        self.stats = {
            "queries_processed": 0,
            "documents_reranked": 0,
            "avg_improvement": 0.0,
            "cache_hits": 0
        }
        
    def _initialize_reranker(self):
        """Lazy initialization of reranker model."""
        if self.reranker is None:
            try:
                logger.info(f"Loading reranker model: {self.model_name}")
                self.reranker = CrossEncoder(
                    self.model_name,
                    max_length=self.max_length,
                    device=self.device
                )
                logger.info(f"Reranker model loaded successfully on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load reranker model {self.model_name}: {e}")
                logger.error(traceback.format_exc())
                # Fallback to a simpler model
                try:
                    fallback_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
                    logger.warning(f"Falling back to {fallback_model}")
                    self.reranker = CrossEncoder(
                        fallback_model,
                        max_length=self.max_length,
                        device=self.device
                    )
                    self.model_name = fallback_model
                except Exception as e2:
                    logger.error(f"Failed to load fallback reranker: {e2}")
                    raise e2
    
    def rerank_documents(self, 
                        query: str, 
                        documents: List[Dict[str, Any]], 
                        top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Rerank documents using the reranking model.
        
        Args:
            query: The search query
            documents: List of document dictionaries with 'content' and 'score' keys
            top_k: Number of top documents to return (None = return all)
            
        Returns:
            Reranked list of documents with updated scores
        """
        if not documents:
            return documents
            
        self._initialize_reranker()
        self.stats["queries_processed"] += 1
        self.stats["documents_reranked"] += len(documents)
        
        try:
            # Prepare query-document pairs for reranking
            pairs = []
            for doc in documents:
                content = doc.get('content', '')
                if not content and 'content_preview' in doc:
                    content = doc['content_preview']
                
                # Truncate content if too long for reranker
                if len(content) > self.max_length - len(query) - 10:  # Leave room for special tokens
                    content = content[:self.max_length - len(query) - 10] + "..."
                
                pairs.append([query, content])
            
            # Get reranking scores
            logger.debug(f"Reranking {len(pairs)} document pairs for query: {query[:50]}...")
            rerank_scores = self.reranker.predict(pairs)
            
            # Update document scores and sort
            reranked_docs = []
            for doc, new_score in zip(documents, rerank_scores):
                doc_copy = doc.copy()
                original_score = doc_copy.get('score', 0.0)
                doc_copy['original_score'] = original_score
                doc_copy['rerank_score'] = float(new_score)
                doc_copy['score'] = float(new_score)  # Update primary score
                doc_copy['reranked'] = True
                reranked_docs.append(doc_copy)
            
            # Sort by reranking score (descending)
            reranked_docs.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            # Apply top_k limit if specified
            if top_k is not None:
                reranked_docs = reranked_docs[:top_k]
            
            # Calculate improvement metric
            if len(documents) > 1:
                original_top_score = max(doc.get('score', 0) for doc in documents)
                new_top_score = max(doc.get('rerank_score', 0) for doc in reranked_docs)
                improvement = (new_top_score - original_top_score) / max(original_top_score, 1e-6)
                self.stats["avg_improvement"] = (self.stats["avg_improvement"] * (self.stats["queries_processed"] - 1) + improvement) / self.stats["queries_processed"]
            
            logger.info(f"Reranked {len(documents)} documents, returning top {len(reranked_docs)}")
            
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            logger.error(traceback.format_exc())
            # Return original documents on error
            return documents
    
    def rerank_with_metadata_boost(self, 
                                 query: str, 
                                 documents: List[Dict[str, Any]], 
                                 metadata_boost_fields: List[str] = None,
                                 boost_factor: float = 1.2,
                                 top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Rerank documents with additional metadata-based boosting.
        
        Args:
            query: The search query
            documents: List of document dictionaries
            metadata_boost_fields: Fields to boost (e.g., ['filename', 'title'])
            boost_factor: Multiplicative boost for metadata matches
            top_k: Number of top documents to return
            
        Returns:
            Reranked and boosted list of documents
        """
        if metadata_boost_fields is None:
            metadata_boost_fields = ['filename', 'title', 'source']
        
        # First apply standard reranking
        reranked_docs = self.rerank_documents(query, documents, top_k=None)
        
        # Then apply metadata boosting
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        for doc in reranked_docs:
            boost_applied = False
            
            # Check various metadata fields for query term matches
            payload = doc.get('payload', {})
            meta = payload.get('meta', {}) if payload else doc.get('meta', {})
            
            for field in metadata_boost_fields:
                field_value = meta.get(field, '') or doc.get(field, '')
                if field_value and isinstance(field_value, str):
                    field_lower = field_value.lower()
                    
                    # Check for exact query match
                    if query_lower in field_lower:
                        doc['score'] *= boost_factor
                        doc['metadata_boost'] = f"{field}: exact_match"
                        boost_applied = True
                        break
                    
                    # Check for term matches
                    field_terms = set(field_lower.split())
                    overlap = query_terms.intersection(field_terms)
                    if overlap:
                        # Boost proportional to overlap
                        overlap_ratio = len(overlap) / len(query_terms)
                        doc['score'] *= (1 + (boost_factor - 1) * overlap_ratio)
                        doc['metadata_boost'] = f"{field}: {len(overlap)}/{len(query_terms)} terms"
                        boost_applied = True
                        break
            
            if not boost_applied:
                doc['metadata_boost'] = "none"
        
        # Re-sort after boosting
        reranked_docs.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply top_k limit
        if top_k is not None:
            reranked_docs = reranked_docs[:top_k]
        
        return reranked_docs
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reranker performance statistics."""
        return self.stats.copy()

class HybridRetrievalPipeline:
    """Complete hybrid retrieval pipeline with reranking."""
    
    def __init__(self,
                 qdrant_url: str = "http://localhost:6333",
                 qdrant_api_key: Optional[str] = None,
                 collection_name: str = "unified_embeddings",
                 embedding_model: str = "BAAI/bge-m3",
                 reranker_model: str = "BAAI/bge-reranker-v2-m3"):
        """
        Initialize the hybrid retrieval pipeline.
        
        Args:
            qdrant_url: Qdrant server URL
            qdrant_api_key: Qdrant API key
            collection_name: Collection name
            embedding_model: Dense embedding model
            reranker_model: Reranking model
        """
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # Initialize components
        self.client = None
        self.embedder = None
        self.reranker = AdvancedReranker(model_name=reranker_model)
        
        # Performance tracking
        self.pipeline_stats = {
            "queries_processed": 0,
            "total_retrieval_time": 0.0,
            "total_rerank_time": 0.0
        }
    
    def connect(self) -> bool:
        """Connect to Qdrant and initialize embedder."""
        try:
            # Connect to Qdrant
            self.client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key if self.qdrant_api_key else None,
                timeout=60  # Extended timeout for reranking operations
            )
            
            # Test connection
            collections = self.client.get_collections()
            logger.info(f"Connected to Qdrant. Available collections: {[c.name for c in collections.collections]}")
            
            # Initialize embedder
            self.embedder = FastembedTextEmbedder(model=self.embedding_model)
            self.embedder.warm_up()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    def search_and_rerank(self,
                         query: str,
                         initial_k: int = 20,
                         final_k: int = 5,
                         use_metadata_boost: bool = True,
                         boost_factor: float = 1.2,
                         use_query_expansion: bool = True) -> List[Dict[str, Any]]:
        """
        Perform hybrid search with reranking.
        
        Args:
            query: Search query
            initial_k: Number of documents to retrieve initially
            final_k: Number of final documents to return after reranking
            use_metadata_boost: Whether to apply metadata boosting
            boost_factor: Factor for metadata boosting
            use_query_expansion: Whether to use query expansion
            
        Returns:
            List of reranked documents
        """
        import time
        start_time = time.time()
        
        try:
            # Apply query expansion if enabled
            search_queries = [query]
            if use_query_expansion:
                try:
                    expanded_queries = integrate_query_expansion_into_retrieval(query)
                    search_queries = expanded_queries[:3]  # Use top 3 variants
                    logger.info(f"Using query expansion: {len(search_queries)} variants")
                except Exception as e:
                    logger.warning(f"Query expansion failed: {e}")
            
            # Generate query embeddings (use original query for embedding)
            query_result = self.embedder.run(text=query)
            query_embedding = query_result["embedding"]
            
            # Retrieve initial candidates
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                using="text-dense",
                limit=initial_k,
                with_payload=True
            )
            
            retrieval_time = time.time() - start_time
            
            # Convert to document format
            documents = []
            for hit in search_result.points:
                doc = {
                    'id': str(hit.id),
                    'score': hit.score,
                    'payload': hit.payload,
                    'content': hit.payload.get('content', ''),
                }
                
                # Extract metadata
                if 'meta' in hit.payload and hit.payload['meta']:
                    doc['meta'] = hit.payload['meta']
                
                documents.append(doc)
            
            # Apply reranking
            rerank_start = time.time()
            
            if use_metadata_boost:
                reranked_docs = self.reranker.rerank_with_metadata_boost(
                    query=query,
                    documents=documents,
                    boost_factor=boost_factor,
                    top_k=final_k
                )
            else:
                reranked_docs = self.reranker.rerank_documents(
                    query=query,
                    documents=documents,
                    top_k=final_k
                )
            
            rerank_time = time.time() - rerank_start
            total_time = time.time() - start_time
            
            # Update stats
            self.pipeline_stats["queries_processed"] += 1
            self.pipeline_stats["total_retrieval_time"] += retrieval_time
            self.pipeline_stats["total_rerank_time"] += rerank_time
            
            logger.info(f"Search completed: {initial_k} retrieved -> {len(reranked_docs)} reranked in {total_time:.3f}s")
            logger.debug(f"Breakdown: retrieval={retrieval_time:.3f}s, rerank={rerank_time:.3f}s")
            
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error in search_and_rerank: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics."""
        stats = self.pipeline_stats.copy()
        stats["reranker_stats"] = self.reranker.get_stats()
        
        # Calculate averages
        if stats["queries_processed"] > 0:
            stats["avg_retrieval_time"] = stats["total_retrieval_time"] / stats["queries_processed"]
            stats["avg_rerank_time"] = stats["total_rerank_time"] / stats["queries_processed"]
        
        return stats

def test_reranking():
    """Test function for reranking capabilities."""
    logging.basicConfig(level=logging.INFO)
    
    # Test with sample data
    sample_docs = [
        {
            'content': 'This is about machine learning algorithms and neural networks.',
            'score': 0.5,
            'meta': {'filename': 'ml_basics.pdf'}
        },
        {
            'content': 'Psychology and cognitive science research methods.',
            'score': 0.7,
            'meta': {'filename': 'psychology_methods.pdf'}
        },
        {
            'content': 'Machine learning applications in psychology and behavior.',
            'score': 0.6,
            'meta': {'filename': 'ml_psychology.pdf'}
        }
    ]
    
    # Test reranker
    reranker = AdvancedReranker()
    query = "machine learning psychology"
    
    print(f"Testing reranking for query: '{query}'")
    print("Original order:")
    for i, doc in enumerate(sample_docs):
        print(f"  {i+1}. {doc['meta']['filename']} (score: {doc['score']:.3f})")
    
    reranked = reranker.rerank_with_metadata_boost(query, sample_docs)
    
    print("\nReranked order:")
    for i, doc in enumerate(reranked):
        print(f"  {i+1}. {doc['meta']['filename']} (score: {doc['score']:.3f}, original: {doc.get('original_score', 'N/A'):.3f})")
        if 'metadata_boost' in doc:
            print(f"      Metadata boost: {doc['metadata_boost']}")
    
    print(f"\nReranker stats: {reranker.get_stats()}")

if __name__ == "__main__":
    test_reranking()
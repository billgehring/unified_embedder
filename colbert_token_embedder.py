"""
Native ColBERT Token Embedder using Qdrant's Multivector Support
===============================================================

Optimized ColBERT integration using Qdrant's native multivector capabilities
and FastEmbed's LateInteractionTextEmbedding for educational voice tutor.

Based on Qdrant's official ColBERT documentation and optimized for <50ms retrieval.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Generator, Tuple
import numpy as np
from pathlib import Path

try:
    from fastembed import LateInteractionTextEmbedding
    from qdrant_client import models
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False
    print("FastEmbed not available. Install with: uv add fastembed")

logger = logging.getLogger(__name__)

class ColBERTTokenEmbedder:
    """
    Production-ready ColBERT token embedder using FastEmbed and Qdrant multivectors.
    
    Features:
    - Native Qdrant multivector integration  
    - Efficient batch processing
    - Memory-optimized for large document collections
    - Separate document/query embedding methods
    - Performance monitoring and optimization
    """
    
    def __init__(self, 
                 model_name: str = "colbert-ir/colbertv2.0",
                 cache_dir: Optional[str] = None,
                 max_length: int = 512):
        """
        Initialize ColBERT token embedder with FastEmbed backend.
        
        Args:
            model_name: ColBERT model to use (default: official ColBERT v2.0)
            cache_dir: Directory to cache model files
            max_length: Maximum token length for documents
        """
        if not FASTEMBED_AVAILABLE:
            raise ImportError("FastEmbed is required for ColBERT token embedding")
            
        self.model_name = model_name
        self.max_length = max_length
        self.cache_dir = cache_dir
        self._model = None
        self._stats = {
            "documents_processed": 0,
            "tokens_generated": 0,
            "embedding_time": 0.0,
            "batch_count": 0
        }
        
    def _initialize_model(self):
        """Lazy initialization of the FastEmbed model."""
        if self._model is None:
            logger.info(f"Loading ColBERT model: {self.model_name}")
            start_time = time.time()
            
            try:
                self._model = LateInteractionTextEmbedding(
                    model_name=self.model_name,
                    cache_dir=self.cache_dir
                )
                
                load_time = time.time() - start_time
                logger.info(f"ColBERT model loaded in {load_time:.2f}s")
                
                # Get model info
                supported_models = LateInteractionTextEmbedding.list_supported_models()
                model_info = next((m for m in supported_models if m['model'] == self.model_name), None)
                if model_info:
                    logger.info(f"Model info: {model_info['dim']} dims, {model_info['size_in_GB']:.2f}GB")
                    
            except Exception as e:
                logger.error(f"Failed to load ColBERT model {self.model_name}: {e}")
                raise
    
    def get_vector_dimension(self) -> int:
        """Get the vector dimension for this ColBERT model."""
        self._initialize_model()
        
        # Get dimension from model info
        supported_models = LateInteractionTextEmbedding.list_supported_models()
        model_info = next((m for m in supported_models if m['model'] == self.model_name), None)
        
        if model_info:
            return model_info['dim']
        else:
            # Fallback: embed a test sentence
            test_embedding = list(self._model.embed(["test"]))[0]
            return test_embedding.shape[1]
    
    def embed_documents(self, 
                       texts: List[str], 
                       batch_size: int = 32) -> List[np.ndarray]:
        """
        Embed documents using ColBERT late interaction.
        
        Args:
            texts: List of document texts to embed
            batch_size: Batch size for processing (memory optimization)
            
        Returns:
            List of token embedding matrices (shape: [num_tokens, dim])
        """
        self._initialize_model()
        
        logger.info(f"Embedding {len(texts)} documents with ColBERT (batch_size={batch_size})")
        start_time = time.time()
        
        embeddings = []
        total_tokens = 0
        
        # Process in batches to manage memory
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_start = time.time()
            
            # Generate embeddings for batch
            batch_embeddings = list(self._model.embed(batch_texts))
            embeddings.extend(batch_embeddings)
            
            # Update statistics
            batch_tokens = sum(emb.shape[0] for emb in batch_embeddings)
            total_tokens += batch_tokens
            
            batch_time = time.time() - batch_start
            logger.debug(f"Batch {i//batch_size + 1}: {len(batch_texts)} docs, "
                        f"{batch_tokens} tokens, {batch_time:.2f}s")
            
            self._stats["batch_count"] += 1
        
        # Update statistics
        total_time = time.time() - start_time
        self._stats["documents_processed"] += len(texts)
        self._stats["tokens_generated"] += total_tokens
        self._stats["embedding_time"] += total_time
        
        logger.info(f"ColBERT embedding completed: {len(texts)} docs, "
                   f"{total_tokens} tokens, {total_time:.2f}s "
                   f"({len(texts)/total_time:.1f} docs/s)")
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query using ColBERT query processing.
        
        Args:
            query: Query text to embed
            
        Returns:
            Query token embedding matrix (shape: [num_tokens, dim])
        """
        self._initialize_model()
        
        start_time = time.time()
        query_embedding = list(self._model.query_embed([query]))[0]
        embed_time = time.time() - start_time
        
        logger.debug(f"Query embedded: {query_embedding.shape[0]} tokens, {embed_time*1000:.1f}ms")
        
        return query_embedding
    
    def create_qdrant_vectors_config(self) -> Dict[str, Any]:
        """
        Create Qdrant collection configuration for ColBERT multivectors.
        
        Returns:
            Dictionary with vectors_config for QdrantClient.create_collection()
        """
        dimension = self.get_vector_dimension()
        
        return {
            "size": dimension,
            "distance": models.Distance.COSINE,  # ColBERT works well with cosine
            "multivector_config": models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM  # Late interaction scoring
            ),
            "hnsw_config": {
                "m": 16,  # Balanced performance
                "ef_construct": 200,  # Higher for better quality
                "full_scan_threshold": 10000,  # Exact search for small datasets
            },
            "quantization_config": {
                "scalar": {
                    "type": "int8",  # 8-bit quantization for memory efficiency
                    "always_ram": True
                }
            }
        }
    
    def convert_to_qdrant_points(self, 
                                documents: List[Dict[str, Any]], 
                                embeddings: List[np.ndarray]) -> List[models.PointStruct]:
        """
        Convert documents and embeddings to Qdrant points format.
        
        Args:
            documents: List of document metadata dictionaries
            embeddings: List of ColBERT token embedding matrices
            
        Returns:
            List of PointStruct objects ready for Qdrant upload
        """
        if len(documents) != len(embeddings):
            raise ValueError(f"Mismatch: {len(documents)} docs vs {len(embeddings)} embeddings")
        
        points = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            point = models.PointStruct(
                id=doc.get('id', i),  # Use document ID if available, otherwise index
                payload=doc,  # Store all document metadata
                vector=embedding  # ColBERT token matrix
            )
            points.append(point)
        
        logger.debug(f"Created {len(points)} Qdrant points with ColBERT vectors")
        return points
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring and optimization."""
        stats = self._stats.copy()
        
        if stats["documents_processed"] > 0:
            stats["avg_docs_per_second"] = stats["documents_processed"] / max(stats["embedding_time"], 0.001)
            stats["avg_tokens_per_doc"] = stats["tokens_generated"] / stats["documents_processed"]
        
        if stats["batch_count"] > 0:
            stats["avg_batch_time"] = stats["embedding_time"] / stats["batch_count"]
            
        return stats
    
    def reset_stats(self):
        """Reset performance statistics."""
        self._stats = {
            "documents_processed": 0,
            "tokens_generated": 0, 
            "embedding_time": 0.0,
            "batch_count": 0
        }

# Utility functions for integration
def get_supported_colbert_models() -> List[Dict[str, Any]]:
    """Get list of supported ColBERT models from FastEmbed."""
    if not FASTEMBED_AVAILABLE:
        return []
    
    try:
        return LateInteractionTextEmbedding.list_supported_models()
    except Exception as e:
        logger.error(f"Failed to list ColBERT models: {e}")
        return []

def estimate_colbert_storage(num_documents: int, avg_tokens_per_doc: int = 300) -> Dict[str, str]:
    """
    Estimate storage requirements for ColBERT tokens.
    
    Args:
        num_documents: Number of document chunks
        avg_tokens_per_doc: Average tokens per document
        
    Returns:
        Storage estimates in human-readable format
    """
    # ColBERT v2.0 uses 128-dimensional vectors
    bytes_per_token = 128 * 4  # 4 bytes per float32
    total_bytes = num_documents * avg_tokens_per_doc * bytes_per_token
    
    # Convert to human-readable format
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if total_bytes < 1024:
            size_str = f"{total_bytes:.1f} {unit}"
            break
        total_bytes /= 1024
    
    return {
        "colbert_tokens": size_str,
        "avg_tokens_per_doc": str(avg_tokens_per_doc),
        "total_documents": str(num_documents)
    }

# Performance testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test basic functionality
    if FASTEMBED_AVAILABLE:
        print("Testing ColBERT Token Embedder...")
        
        embedder = ColBERTTokenEmbedder()
        print(f"Supported models: {len(get_supported_colbert_models())}")
        
        # Test with small dataset
        test_docs = [
            "Machine learning is a fascinating field of study.",
            "Natural language processing enables computers to understand text.",
            "Educational technology transforms how students learn."
        ]
        
        embeddings = embedder.embed_documents(test_docs, batch_size=2)
        print(f"Generated {len(embeddings)} embeddings")
        
        for i, emb in enumerate(embeddings):
            print(f"  Doc {i}: {emb.shape[0]} tokens, {emb.shape[1]} dims")
        
        # Test query embedding
        query_emb = embedder.embed_query("What is machine learning?")
        print(f"Query: {query_emb.shape[0]} tokens, {query_emb.shape[1]} dims")
        
        # Performance stats
        stats = embedder.get_performance_stats()
        print(f"Performance: {stats}")
        
        # Storage estimates
        estimates = estimate_colbert_storage(10000, 350)
        print(f"Storage for 10k docs: {estimates}")
        
    else:
        print("FastEmbed not available - install with: uv add fastembed")
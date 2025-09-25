"""
Qdrant Schema Design for Multi-Vector ColBERT Token Storage
==========================================================

Optimized for educational voice tutor with <50ms retrieval requirements.
Supports Dense + Sparse + ColBERT tokens with efficient named vector indexing.
"""

from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class VectorConfig:
    """Configuration for different vector types in Qdrant collection"""
    name: str
    size: int
    distance: str
    on_disk: bool = False
    quantization: Optional[str] = None

class ColBERTQdrantSchema:
    """
    Multi-vector schema design for optimal educational retrieval performance.
    
    Architecture:
    - Dense vectors: Semantic similarity (BGE-M3, 1024 dims)
    - Sparse vectors: Keyword matching (SPLADE, variable dims) 
    - ColBERT tokens: Fine-grained contextual matching (128 dims per token)
    """
    
    # Vector configurations for different embedding types
    VECTOR_CONFIGS = {
        "dense": VectorConfig(
            name="text-dense",
            size=1024,  # BGE-M3 embedding dimension
            distance="Cosine",
            on_disk=False,  # Keep in memory for speed
            quantization=None  # Full precision for accuracy
        ),
        
        "sparse": VectorConfig(
            name="text-sparse", 
            size=0,  # Variable size, handled by sparse vectors config
            distance="Dot",
            on_disk=False,
            quantization=None
        ),
        
        "colbert": VectorConfig(
            name="colbert-tokens",
            size=128,  # ColBERT token dimension (compressed)
            distance="Dot",  # Inner product for normalized vectors
            on_disk=True,  # Can be on disk due to efficient indexing
            quantization="scalar"  # 8-bit quantization for memory efficiency
        )
    }
    
    @classmethod
    def get_collection_config(cls, enable_colbert_tokens: bool = False) -> Dict[str, Any]:
        """
        Generate Qdrant collection configuration for multi-vector storage.
        
        Args:
            enable_colbert_tokens: Whether to include ColBERT token vectors
            
        Returns:
            Dictionary with collection configuration for QdrantDocumentStore
        """
        config = {
            "vectors_config": {
                cls.VECTOR_CONFIGS["dense"].name: {
                    "size": cls.VECTOR_CONFIGS["dense"].size,
                    "distance": cls.VECTOR_CONFIGS["dense"].distance
                }
            },
            "sparse_vectors_config": {
                cls.VECTOR_CONFIGS["sparse"].name: {
                    "index": {"on_disk": cls.VECTOR_CONFIGS["sparse"].on_disk},
                    "modifier": "idf"
                }
            }
        }
        
        # Add ColBERT token configuration if enabled
        if enable_colbert_tokens:
            config["vectors_config"][cls.VECTOR_CONFIGS["colbert"].name] = {
                "size": cls.VECTOR_CONFIGS["colbert"].size,
                "distance": cls.VECTOR_CONFIGS["colbert"].distance,
                "hnsw_config": {
                    "m": 16,  # Balanced speed/accuracy
                    "ef_construct": 128,  # Higher for better quality
                    "full_scan_threshold": 10000,  # Switch to exact search for small datasets
                },
                "quantization_config": {
                    "scalar": {
                        "type": "int8",  # 8-bit quantization
                        "always_ram": True  # Keep quantized vectors in RAM
                    }
                } if cls.VECTOR_CONFIGS["colbert"].quantization else None,
                "on_disk": cls.VECTOR_CONFIGS["colbert"].on_disk
            }
            
        logger.info(f"Generated Qdrant schema with ColBERT tokens: {enable_colbert_tokens}")
        return config
    
    @classmethod 
    def get_storage_estimates(cls, num_documents: int, avg_tokens_per_doc: int = 300) -> Dict[str, str]:
        """
        Calculate storage requirements for different vector types.
        
        Args:
            num_documents: Number of document chunks
            avg_tokens_per_doc: Average tokens per document chunk
            
        Returns:
            Storage estimates for each vector type
        """
        # Dense vector storage (1024 dims * 4 bytes)
        dense_size = num_documents * 1024 * 4
        
        # Sparse vector storage (estimated ~200 non-zero terms)
        sparse_size = num_documents * 200 * 8  # 4 bytes value + 4 bytes index
        
        # ColBERT token storage (avg_tokens * 128 dims * 4 bytes)  
        colbert_size = num_documents * avg_tokens_per_doc * 128 * 4
        
        def format_bytes(bytes_val):
            for unit in ['B', 'KB', 'MB', 'GB']:
                if bytes_val < 1024:
                    return f"{bytes_val:.1f} {unit}"
                bytes_val /= 1024
            return f"{bytes_val:.1f} TB"
        
        return {
            "dense_vectors": format_bytes(dense_size),
            "sparse_vectors": format_bytes(sparse_size), 
            "colbert_tokens": format_bytes(colbert_size),
            "total_without_colbert": format_bytes(dense_size + sparse_size),
            "total_with_colbert": format_bytes(dense_size + sparse_size + colbert_size),
            "colbert_overhead": f"{colbert_size / (dense_size + sparse_size):.1f}x increase"
        }
    
    @classmethod
    def validate_schema_compatibility(cls) -> bool:
        """
        Validate that the schema is compatible with current Qdrant version.
        
        Returns:
            True if schema is valid and supported
        """
        try:
            # Check if required Qdrant features are available
            from qdrant_client import QdrantClient
            from qdrant_client.http import models
            
            # Validate vector dimensions are within limits
            max_dense_dim = cls.VECTOR_CONFIGS["dense"].size
            max_colbert_dim = cls.VECTOR_CONFIGS["colbert"].size
            
            if max_dense_dim > 65536 or max_colbert_dim > 65536:
                logger.error("Vector dimensions exceed Qdrant limits")
                return False
                
            logger.info("Schema compatibility validation passed")
            return True
            
        except ImportError as e:
            logger.error(f"Qdrant client not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return False

# Performance optimization constants
COLBERT_BATCH_SIZE = 32  # Optimal batch size for ColBERT token generation
COLBERT_MAX_TOKENS = 512  # Maximum tokens per document chunk
RETRIEVAL_TIMEOUT_MS = 50  # Target retrieval time for voice applications

# Usage example and testing
if __name__ == "__main__":
    # Example usage for educational tutor scenario
    schema = ColBERTQdrantSchema()
    
    # Calculate storage for typical course (10,000 chunks)
    estimates = schema.get_storage_estimates(num_documents=10000)
    
    print("Storage Estimates for 10,000 Document Chunks:")
    for vector_type, size in estimates.items():
        print(f"  {vector_type}: {size}")
        
    # Generate configurations
    config_without_colbert = schema.get_collection_config(enable_colbert_tokens=False)
    config_with_colbert = schema.get_collection_config(enable_colbert_tokens=True)
    
    print(f"\nCollection config keys: {list(config_with_colbert.keys())}")
    print(f"Vector types: {list(config_with_colbert['vectors_config'].keys())}")
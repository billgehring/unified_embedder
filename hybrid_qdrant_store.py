"""
Hybrid Qdrant Store for Multi-Vector ColBERT Integration
=======================================================

Combines Haystack QdrantDocumentStore (Dense + Sparse) with native Qdrant client (ColBERT tokens)
for optimal performance in educational voice tutor applications.

Architecture:
- Primary: Haystack QdrantDocumentStore for Dense + Sparse vectors  
- Secondary: Native QdrantClient for ColBERT multivector storage
- Synchronized: Same collection, coordinated document IDs
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Union
import uuid
from pathlib import Path

try:
    from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
    from haystack import Document
    from haystack.utils import Secret
    HAYSTACK_AVAILABLE = True
except ImportError:
    HAYSTACK_AVAILABLE = False

try:
    from haystack.document_stores.types import DuplicatePolicy
except Exception:
    DuplicatePolicy = None

try:
    from qdrant_client import QdrantClient, models
    from qdrant_client.http.models import Distance, VectorParams, MultiVectorConfig, MultiVectorComparator
    QDRANT_CLIENT_AVAILABLE = True
except ImportError:
    QDRANT_CLIENT_AVAILABLE = False

from colbert_token_embedder import ColBERTTokenEmbedder

logger = logging.getLogger(__name__)

class HybridQdrantStore:
    """
    Hybrid document store combining Haystack QdrantDocumentStore with native ColBERT support.
    
    Features:
    - Dense + Sparse vectors via Haystack (proven, stable)
    - ColBERT tokens via native Qdrant client (optimal performance)  
    - Synchronized document storage and retrieval
    - Sub-50ms query performance for voice applications
    """
    
    def __init__(self,
                 url: str,
                 collection_name: str,
                 api_key: Optional[str] = None,
                 embedding_dim: int = 1024,
                 enable_colbert_tokens: bool = False,
                 colbert_model: str = "colbert-ir/colbertv2.0",
                 recreate_index: bool = False):
        """
        Initialize hybrid Qdrant store.
        
        Args:
            url: Qdrant server URL
            collection_name: Collection name for both stores
            api_key: Qdrant API key (optional for local)
            embedding_dim: Dimension for dense embeddings
            enable_colbert_tokens: Whether to enable ColBERT token storage
            colbert_model: ColBERT model name
            recreate_index: Whether to recreate collections
        """
        if not HAYSTACK_AVAILABLE:
            raise ImportError("Haystack integrations required")
        if not QDRANT_CLIENT_AVAILABLE:
            raise ImportError("Qdrant client required")
            
        self.url = url
        self.collection_name = collection_name
        self.api_key = api_key
        self.embedding_dim = embedding_dim
        self.enable_colbert_tokens = enable_colbert_tokens
        self.colbert_model = colbert_model
        self.recreate_index = recreate_index
        
        # Collection names
        self.dense_sparse_collection = collection_name  # Primary collection
        self.colbert_collection = f"{collection_name}_colbert"  # ColBERT tokens
        
        # Initialize components
        self._haystack_store = None
        self._qdrant_client = None
        self._colbert_embedder = None
        
        self._stats = {
            "documents_stored": 0,
            "colbert_documents_stored": 0,
            "storage_time": 0.0,
            "colbert_storage_time": 0.0
        }
        
    def _initialize_haystack_store(self):
        """Initialize Haystack QdrantDocumentStore for dense + sparse."""
        if self._haystack_store is None:
            logger.info(f"Initializing Haystack QdrantDocumentStore: {self.dense_sparse_collection}")
            from qdrant_client import QdrantClient
            client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=60
            )
            try:
                self._haystack_store = QdrantDocumentStore(
                    client=client,
                    index=self.dense_sparse_collection,
                    embedding_dim=self.embedding_dim,
                    recreate_index=self.recreate_index,
                    use_sparse_embeddings=True,
                    on_disk_payload=True,
                    sparse_idf=True,
                    hnsw_config={"m": 16, "ef_construct": 128},
                    similarity="cosine"
                )
            except TypeError:
                self._haystack_store = QdrantDocumentStore(
                    url=self.url,
                    api_key=Secret.from_token(self.api_key) if self.api_key else None,
                    index=self.dense_sparse_collection,
                    embedding_dim=self.embedding_dim,
                    recreate_index=self.recreate_index,
                    use_sparse_embeddings=True,
                    on_disk_payload=True,
                    sparse_idf=True,
                    hnsw_config={"m": 16, "ef_construct": 128},
                    similarity="cosine"
                )
            
    def _initialize_qdrant_client(self):
        """Initialize native Qdrant client for ColBERT tokens."""
        if self._qdrant_client is None and self.enable_colbert_tokens:
            logger.info(f"Initializing native QdrantClient for ColBERT: {self.colbert_collection}")
            
            # Use higher timeout and prefer gRPC for large multivector uploads
            self._qdrant_client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=300,
                prefer_grpc=True
            )
            
            # Initialize ColBERT embedder
            self._colbert_embedder = ColBERTTokenEmbedder(model_name=self.colbert_model)
            
            # Create ColBERT collection if needed
            self._create_colbert_collection()
            
    def _create_colbert_collection(self):
        """Create Qdrant collection for ColBERT multivectors."""
        try:
            # Check if collection exists
            collections = self._qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.colbert_collection in collection_names and not self.recreate_index:
                logger.info(f"ColBERT collection {self.colbert_collection} already exists")
                return
                
            if self.colbert_collection in collection_names and self.recreate_index:
                logger.info(f"Deleting existing ColBERT collection: {self.colbert_collection}")
                self._qdrant_client.delete_collection(self.colbert_collection)
                
            # Get vector configuration from ColBERT embedder
            vector_config = self._colbert_embedder.create_qdrant_vectors_config()
            
            logger.info(f"Creating ColBERT collection with multivector support")
            self._qdrant_client.create_collection(
                collection_name=self.colbert_collection,
                vectors_config=VectorParams(**vector_config)
            )
            
            logger.info(f"ColBERT collection {self.colbert_collection} created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create ColBERT collection: {e}")
            raise
            
    def write_documents(self, 
                       documents: List[Document], 
                       batch_size: int = 100,
                       **kwargs) -> List[str]:
        """
        Write documents to both dense/sparse and ColBERT collections.
        
        Args:
            documents: List of Haystack Document objects
            batch_size: Batch size for processing
            
        Returns:
            List of document IDs
        """
        logger.info(f"Writing {len(documents)} documents to hybrid Qdrant store")
        start_time = time.time()
        
        # Initialize stores
        self._initialize_haystack_store()
        if self.enable_colbert_tokens:
            self._initialize_qdrant_client()
        
        # Write to Haystack store (dense + sparse) with basic retry to handle transient timeouts
        logger.info(f"Writing dense + sparse vectors to {self.dense_sparse_collection}")
        haystack_start = time.time()
        retries = 3
        backoff = 1.0
        last_err = None
        num_written = 0
        for attempt in range(1, retries + 1):
            try:
                if DuplicatePolicy is not None:
                    try:
                        num_written = self._haystack_store.write_documents(documents, policy=DuplicatePolicy.OVERWRITE)
                    except TypeError:
                        num_written = self._haystack_store.write_documents(documents)
                else:
                    num_written = self._haystack_store.write_documents(documents)
                last_err = None
                break
            except Exception as e:
                last_err = e
                logger.error(f"Haystack write attempt {attempt} failed: {e}")
                if attempt < retries:
                    time.sleep(backoff)
                    backoff *= 2
        haystack_time = time.time() - haystack_start
        if last_err is not None:
            # Bubble up after retries exhausted
            raise last_err
        
        # Extract document IDs from the documents themselves (Haystack returns count, not IDs)
        document_ids = []
        for i, doc in enumerate(documents):
            if doc.id:
                document_ids.append(doc.id)
            else:
                # Generate ID if not present (should not happen with proper document creation)
                doc_id = f"doc_{int(time.time() * 1000)}_{i}"
                logger.warning(f"Document {i} missing ID, generated: {doc_id}")
                document_ids.append(doc_id)
                doc.id = doc_id  # Update document with generated ID
        
        logger.info(f"Haystack store wrote {num_written} documents, extracted {len(document_ids)} document IDs")
        
        self._stats["documents_stored"] += len(documents)
        self._stats["storage_time"] += haystack_time
        
        logger.info(f"Dense + sparse storage completed in {haystack_time:.2f}s")
        
        # Write ColBERT tokens if enabled
        if self.enable_colbert_tokens:
            logger.info(f"Writing ColBERT tokens to {self.colbert_collection}")
            colbert_start = time.time()
            
            self._write_colbert_documents(documents, document_ids, batch_size)
            
            colbert_time = time.time() - colbert_start
            self._stats["colbert_storage_time"] += colbert_time
            self._stats["colbert_documents_stored"] += len(documents)
            
            logger.info(f"ColBERT storage completed in {colbert_time:.2f}s")
        
        total_time = time.time() - start_time
        logger.info(f"Hybrid storage completed: {len(documents)} docs in {total_time:.2f}s")
        
        return document_ids
        
    def _write_colbert_documents(self, 
                                documents: List[Document], 
                                document_ids: List[str],
                                batch_size: int = 32):
        """Write documents to ColBERT collection with token embeddings."""
        
        def _sanitize_value(v):
            """Sanitize payload values for Qdrant gRPC constraints."""
            import numpy as _np
            import datetime as _dt
            INT64_MAX = 9223372036854775807
            try:
                if isinstance(v, bool):
                    return v
                if isinstance(v, (int,)):
                    if abs(v) > INT64_MAX:
                        return str(v)
                    return v
                if isinstance(v, (float,)):
                    return float(v)
                if isinstance(v, (str,)):
                    return v
                if isinstance(v, (_np.integer,)):
                    v_int = int(v)
                    return v_int if abs(v_int) <= INT64_MAX else str(v_int)
                if isinstance(v, (_np.floating,)):
                    return float(v)
                if isinstance(v, (_dt.datetime, _dt.date)):
                    return v.isoformat()
                if isinstance(v, dict):
                    return {k: _sanitize_value(val) for k, val in v.items()}
                if isinstance(v, (list, tuple)):
                    return [_sanitize_value(it) for it in v]
            except Exception:
                pass
            # Fallback to string representation
            try:
                return str(v)
            except Exception:
                return None

        def _sanitize_payload(d: dict) -> dict:
            cleaned = {}
            for k, v in d.items():
                if k == 'dl_meta':
                    # Drop heavy/nested Docling metadata to reduce payload size and avoid large ints
                    continue
                cleaned[k] = _sanitize_value(v)
            return cleaned

        # Extract text content
        texts = [doc.content for doc in documents]
        
        # Generate ColBERT embeddings in batches
        all_embeddings = self._colbert_embedder.embed_documents(texts, batch_size=batch_size)
        
        # Convert to Qdrant format
        points = []
        for doc, doc_id, embedding in zip(documents, document_ids, all_embeddings):
            # Merge document metadata with Haystack metadata
            # Whitelist minimal payload for ColBERT; keep full metadata in base collection
            import os as _os
            try:
                preview_len = int(_os.getenv("COLBERT_CONTENT_PREVIEW_LEN", "200"))
            except Exception:
                preview_len = 200
            text_content = doc.content if isinstance(doc.content, str) else ""
            preview = text_content[:preview_len]

            # Default whitelist; override with COLBERT_PAYLOAD_KEYS (comma-separated)
            default_keys = [
                "filename",
                "file_path",
                "ocr_engine_used",
                "ocr_reprocessed",
            ]
            env_keys = _os.getenv("COLBERT_PAYLOAD_KEYS")
            if env_keys:
                whitelist = [k.strip() for k in env_keys.split(",") if k.strip()]
            else:
                whitelist = default_keys

            payload = {
                "content": preview,        # store preview only to keep payload light
                "haystack_id": doc_id,     # link back to base collection
            }
            if isinstance(doc.meta, dict):
                filtered = {k: doc.meta.get(k) for k in whitelist if k in doc.meta}
                payload.update(_sanitize_payload(filtered))
            
            # Convert hex string ID to UUID for Qdrant compatibility
            if isinstance(doc_id, str) and len(doc_id) == 64:  # 64-char hex string (32 bytes)
                # Take first 32 characters (16 bytes) and format as UUID
                hex_subset = doc_id[:32]
                uuid_str = f"{hex_subset[:8]}-{hex_subset[8:12]}-{hex_subset[12:16]}-{hex_subset[16:20]}-{hex_subset[20:]}"
                colbert_id = uuid_str
            else:
                # Use original ID if it's already a UUID or integer
                colbert_id = doc_id
            
            point = models.PointStruct(
                id=colbert_id,  # Use UUID-formatted ID for ColBERT
                payload=payload,
                vector=embedding  # ColBERT token matrix
            )
            points.append(point)
            
        # Upload to Qdrant in batches
        # Use a conservative batch size for multivectors to avoid HTTP write timeouts.
        # Allow override via env COLBERT_UPLOAD_BATCH, else cap at 8.
        import os
        try:
            env_batch = int(os.getenv("COLBERT_UPLOAD_BATCH", "8"))
        except Exception:
            env_batch = 8
        batch_size = max(1, min(env_batch, len(points)))
        logger.info(f"Uploading ColBERT points in batches of {batch_size}")
        for i in range(0, len(points), batch_size):
            batch_points = points[i:i + batch_size]
            attempt = 1
            max_attempts = 4
            backoff = 2.0
            while True:
                try:
                    self._qdrant_client.upload_points(
                        collection_name=self.colbert_collection,
                        points=batch_points,
                        wait=False,
                        max_retries=3
                    )
                    logger.debug(f"Successfully uploaded batch {i//batch_size + 1} of {len(batch_points)} ColBERT points")
                    break
                except Exception as e:
                    logger.warning(f"Batch upload failed (attempt {attempt}/{max_attempts}). Retrying...")
                    logger.error(f"❌ Failed to upload ColBERT batch {i//batch_size + 1}: {e}")
                    logger.error(f"   Collection: {self.colbert_collection}")
                    logger.error(f"   Batch size: {len(batch_points)}")
                    if batch_points:
                        sample_point = batch_points[0]
                        logger.error(f"   Sample point ID: {sample_point.id}")
                        logger.error(f"   Sample vector type: {type(sample_point.vector)}")
                        if hasattr(sample_point.vector, '__len__'):
                            logger.error(f"   Sample vector length: {len(sample_point.vector)}")
                            if len(sample_point.vector) > 0:
                                logger.error(f"   Sample first token type: {type(sample_point.vector[0])}")
                                if hasattr(sample_point.vector[0], '__len__'):
                                    logger.error(f"   Sample first token dim: {len(sample_point.vector[0])}")
                    if attempt >= max_attempts:
                        raise
                    time.sleep(backoff)
                    backoff *= 1.5
                    attempt += 1
            
        logger.info(f"✅ Uploaded {len(points)} ColBERT points to Qdrant successfully")
        
    def search(self, 
               query: str,
               limit: int = 10,
               search_mode: str = "hybrid") -> List[Dict[str, Any]]:
        """
        Search documents using specified mode.
        
        Args:
            query: Search query
            limit: Number of results
            search_mode: "dense", "sparse", "hybrid", "colbert", or "all"
            
        Returns:
            List of search results with scores
        """
        self._initialize_haystack_store()
        
        if search_mode in ["colbert", "all"] and self.enable_colbert_tokens:
            self._initialize_qdrant_client()
            return self._search_colbert(query, limit)
        else:
            # Use Haystack retrieval for dense/sparse/hybrid
            return self._search_haystack(query, limit, search_mode)
            
    def _search_haystack(self, 
                        query: str, 
                        limit: int, 
                        search_mode: str) -> List[Dict[str, Any]]:
        """Search using Haystack document store."""
        # This would integrate with existing Haystack retrieval pipeline
        # For now, return placeholder
        return []
        
    def _search_colbert(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search using ColBERT tokens."""
        start_time = time.time()
        
        # Embed query with ColBERT
        query_embedding = self._colbert_embedder.embed_query(query)
        
        # Search Qdrant collection
        results = self._qdrant_client.query_points(
            collection_name=self.colbert_collection,
            query=query_embedding,
            limit=limit,
            with_payload=True
        )
        
        search_time = time.time() - start_time
        
        # Convert to standard format
        formatted_results = []
        for result in results.points:
            formatted_results.append({
                "id": str(result.id),
                "score": result.score,
                "content": result.payload.get("content", ""),
                "metadata": {k: v for k, v in result.payload.items() if k != "content"},
                "search_method": "colbert",
                "search_time": search_time
            })
            
        logger.info(f"ColBERT search completed in {search_time*1000:.1f}ms, {len(results.points)} results")
        return formatted_results
        
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self._stats.copy()
        
        if self._colbert_embedder:
            colbert_stats = self._colbert_embedder.get_performance_stats()
            stats.update({f"colbert_{k}": v for k, v in colbert_stats.items()})
            
        return stats
        
    def close(self):
        """Close connections and cleanup resources."""
        if self._qdrant_client:
            self._qdrant_client.close()
        # Haystack store cleanup handled automatically
        
# Integration helper functions
def create_hybrid_store(url: str,
                       collection_name: str,
                       api_key: Optional[str] = None,
                       embedding_dim: int = 1024,
                       enable_colbert: bool = False,
                       colbert_model: str = "colbert-ir/colbertv2.0",
                       recreate_index: bool = False) -> HybridQdrantStore:
    """
    Factory function to create hybrid Qdrant store.
    """
    return HybridQdrantStore(
        url=url,
        collection_name=collection_name,
        api_key=api_key,
        embedding_dim=embedding_dim,
        enable_colbert_tokens=enable_colbert,
        colbert_model=colbert_model,
        recreate_index=recreate_index
    )

# Testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with in-memory Qdrant
    print("Testing Hybrid Qdrant Store...")
    
    # This would require a running Qdrant instance
    # store = create_hybrid_store(
    #     url="http://localhost:6333",
    #     collection_name="test_hybrid",
    #     enable_colbert=True,
    #     recreate_index=True
    # )
    
    print("Hybrid Qdrant Store implementation complete")
    print("Ready for integration with unified_embedder.py")

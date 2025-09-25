"""
ColBERT Integration Module for Unified Embedder
==============================================

This module provides integration with ColBERT (Contextualized Late Interaction over BERT)
for improved retrieval through late interaction modeling. ColBERT offers better retrieval
quality than dense embeddings while being more efficient than cross-encoders.
"""

import logging
import traceback
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import numpy as np
import time

try:
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModel
    from sklearn.metrics.pairwise import cosine_similarity
    import faiss
except ImportError as e:
    print(f"Required libraries not installed: {e}")
    print("Please run: uv add torch sentence-transformers transformers faiss-cpu")

# Configure logging
logger = logging.getLogger(__name__)

class ColBERTEmbedder:
    """ColBERT-style embedder for late interaction retrieval."""
    
    def __init__(self, 
                 model_name: str = "colbert-ir/colbertv2.0",
                 max_length: int = 512,
                 device: str = "cpu",
                 dim: int = 128):
        """
        Initialize ColBERT embedder.
        
        Args:
            model_name: ColBERT model name or regular transformer model
            max_length: Maximum sequence length
            device: Device to use (cpu/cuda)
            dim: Embedding dimension (ColBERT uses compressed representations)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        self.dim = dim
        
        # Lazy-loaded components
        self.model = None
        self.tokenizer = None
        
        # Statistics
        self.stats = {
            "documents_encoded": 0,
            "queries_encoded": 0,
            "total_tokens_processed": 0,
            "avg_doc_length": 0.0
        }
    
    def _initialize_model(self):
        """Initialize the ColBERT model and tokenizer."""
        if self.model is None:
            try:
                logger.info(f"Loading ColBERT model: {self.model_name}")
                
                # Try to load as ColBERT model first, fallback to regular transformer
                try:
                    # For actual ColBERT models (if available)
                    from colbert import Indexer, Searcher
                    logger.info("Using native ColBERT implementation")
                    # This would require the official ColBERT library
                except ImportError:
                    # Fallback to transformer-based implementation
                    logger.info("Using transformer-based ColBERT implementation")
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    self.model = AutoModel.from_pretrained(self.model_name)
                    self.model.to(self.device)
                    self.model.eval()
                
                logger.info(f"ColBERT model loaded on {self.device}")
                
            except Exception as e:
                logger.error(f"Failed to load ColBERT model {self.model_name}: {e}")
                # Fallback to a working model
                fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
                logger.warning(f"Falling back to {fallback_model}")
                self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                self.model = AutoModel.from_pretrained(fallback_model)
                self.model.to(self.device)
                self.model.eval()
                self.model_name = fallback_model
    
    def _tokenize_and_encode(self, texts: List[str], is_query: bool = False) -> torch.Tensor:
        """Tokenize and encode texts with ColBERT-style processing."""
        self._initialize_model()
        
        # Add special tokens for queries vs documents
        if is_query:
            texts = [f"[Q] {text}" for text in texts]
        else:
            texts = [f"[D] {text}" for text in texts]
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(self.device)
        
        # Get contextualized embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use last hidden state
            embeddings = outputs.last_hidden_state
        
        return embeddings, inputs['attention_mask']
    
    def encode_documents(self, documents: List[str]) -> List[np.ndarray]:
        """
        Encode documents using ColBERT-style late interaction.
        
        Args:
            documents: List of document texts
            
        Returns:
            List of token-level embeddings for each document
        """
        logger.info(f"Encoding {len(documents)} documents with ColBERT")
        self.stats["documents_encoded"] += len(documents)
        
        # Process in batches to manage memory
        batch_size = 8
        all_doc_embeddings = []
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            
            # Get contextualized embeddings
            embeddings, attention_mask = self._tokenize_and_encode(batch_docs, is_query=False)
            
            # Process each document in the batch
            for j, doc_emb in enumerate(embeddings):
                # Mask out padding tokens
                mask = attention_mask[j].bool()
                valid_embeddings = doc_emb[mask]  # [seq_len, hidden_dim]
                
                # Optional: Apply linear projection to reduce dimensionality
                if self.dim < valid_embeddings.shape[-1]:
                    # Simple linear projection (in practice, this would be learned)
                    projected = valid_embeddings[:, :self.dim]
                else:
                    projected = valid_embeddings
                
                # Normalize embeddings (important for ColBERT)
                projected = torch.nn.functional.normalize(projected, dim=-1)
                
                all_doc_embeddings.append(projected.cpu().numpy())
                self.stats["total_tokens_processed"] += len(projected)
        
        # Update average document length
        if len(documents) > 0:
            avg_length = sum(len(emb) for emb in all_doc_embeddings) / len(all_doc_embeddings)
            self.stats["avg_doc_length"] = (self.stats["avg_doc_length"] * (self.stats["documents_encoded"] - len(documents)) + 
                                          avg_length * len(documents)) / self.stats["documents_encoded"]
        
        logger.info(f"Encoded documents: avg tokens per doc = {avg_length:.1f}")
        return all_doc_embeddings
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a query using ColBERT-style processing.
        
        Args:
            query: Query text
            
        Returns:
            Token-level embeddings for the query
        """
        self.stats["queries_encoded"] += 1
        
        # Get contextualized embeddings
        embeddings, attention_mask = self._tokenize_and_encode([query], is_query=True)
        
        # Process query embedding
        query_emb = embeddings[0]  # First (and only) query
        mask = attention_mask[0].bool()
        valid_embeddings = query_emb[mask]
        
        # Apply dimensionality reduction if needed
        if self.dim < valid_embeddings.shape[-1]:
            projected = valid_embeddings[:, :self.dim]
        else:
            projected = valid_embeddings
        
        # Normalize
        projected = torch.nn.functional.normalize(projected, dim=-1)
        
        return projected.cpu().numpy()

class ColBERTRetriever:
    """ColBERT-based retrieval system with late interaction scoring."""
    
    def __init__(self, 
                 embedder: ColBERTEmbedder,
                 use_faiss: bool = True):
        """
        Initialize ColBERT retriever.
        
        Args:
            embedder: ColBERT embedder instance
            use_faiss: Whether to use FAISS for efficient search
        """
        self.embedder = embedder
        self.use_faiss = use_faiss
        
        # Document storage
        self.doc_embeddings = []
        self.doc_metadata = []
        
        # FAISS index (if used)
        self.faiss_index = None
        self.token_to_doc_mapping = []  # Maps token indices to document indices
        
        # Statistics
        self.retrieval_stats = {
            "searches_performed": 0,
            "avg_search_time": 0.0,
            "total_documents_indexed": 0
        }
    
    def index_documents(self, documents: List[Dict[str, Any]]):
        """
        Index documents for ColBERT retrieval.
        
        Args:
            documents: List of document dictionaries with 'content' and metadata
        """
        logger.info(f"Indexing {len(documents)} documents for ColBERT retrieval")
        
        # Extract text content
        doc_texts = []
        for doc in documents:
            content = doc.get('content', '') or doc.get('text', '')
            doc_texts.append(content)
        
        # Encode documents
        self.doc_embeddings = self.embedder.encode_documents(doc_texts)
        self.doc_metadata = documents
        
        # Build FAISS index if requested
        if self.use_faiss:
            self._build_faiss_index()
        
        self.retrieval_stats["total_documents_indexed"] = len(documents)
        logger.info(f"Indexing completed: {len(documents)} documents indexed")
    
    def _build_faiss_index(self):
        """Build FAISS index for efficient token-level search."""
        if not self.doc_embeddings:
            return
            
        logger.info("Building FAISS index for ColBERT tokens")
        
        # Flatten all token embeddings
        all_tokens = []
        self.token_to_doc_mapping = []
        
        for doc_idx, doc_emb in enumerate(self.doc_embeddings):
            all_tokens.append(doc_emb)
            # Map each token to its document
            self.token_to_doc_mapping.extend([doc_idx] * len(doc_emb))
        
        # Stack all tokens
        if all_tokens:
            token_matrix = np.vstack(all_tokens)
            
            # Create FAISS index
            dim = token_matrix.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)  # Inner product for normalized vectors
            self.faiss_index.add(token_matrix.astype('float32'))
            
            logger.info(f"FAISS index built: {len(token_matrix)} tokens indexed")
    
    def _compute_late_interaction_score(self, query_emb: np.ndarray, 
                                      doc_emb: np.ndarray) -> float:
        """
        Compute ColBERT late interaction score between query and document.
        
        Args:
            query_emb: Query token embeddings [query_len, dim]
            doc_emb: Document token embeddings [doc_len, dim]
            
        Returns:
            Late interaction score
        """
        # Compute similarity matrix between all query and document tokens
        similarity_matrix = np.dot(query_emb, doc_emb.T)  # [query_len, doc_len]
        
        # MaxSim operation: for each query token, find max similarity with any doc token
        max_similarities = np.max(similarity_matrix, axis=1)  # [query_len]
        
        # Sum across query tokens (ColBERT scoring)
        score = np.sum(max_similarities)
        
        return float(score)
    
    def search(self, query: str, k: int = 10, use_faiss: bool = None) -> List[Dict[str, Any]]:
        """
        Search documents using ColBERT late interaction.
        
        Args:
            query: Search query
            k: Number of results to return
            use_faiss: Whether to use FAISS (overrides instance setting)
            
        Returns:
            List of search results with scores
        """
        if not self.doc_embeddings:
            logger.warning("No documents indexed for search")
            return []
        
        start_time = time.time()
        self.retrieval_stats["searches_performed"] += 1
        
        # Encode query
        query_emb = self.embedder.encode_query(query)
        
        # Choose search method
        if (use_faiss if use_faiss is not None else self.use_faiss) and self.faiss_index:
            results = self._search_with_faiss(query_emb, k)
        else:
            results = self._search_exhaustive(query_emb, k)
        
        # Update timing statistics
        search_time = time.time() - start_time
        self.retrieval_stats["avg_search_time"] = (
            (self.retrieval_stats["avg_search_time"] * (self.retrieval_stats["searches_performed"] - 1) + 
             search_time) / self.retrieval_stats["searches_performed"]
        )
        
        logger.info(f"ColBERT search completed in {search_time:.3f}s, returned {len(results)} results")
        return results
    
    def _search_exhaustive(self, query_emb: np.ndarray, k: int) -> List[Dict[str, Any]]:
        """Exhaustive search over all documents."""
        scores = []
        
        for doc_idx, doc_emb in enumerate(self.doc_embeddings):
            score = self._compute_late_interaction_score(query_emb, doc_emb)
            scores.append((score, doc_idx))
        
        # Sort by score (descending)
        scores.sort(reverse=True)
        
        # Return top k results
        results = []
        for score, doc_idx in scores[:k]:
            result = self.doc_metadata[doc_idx].copy()
            result['score'] = score
            result['colbert_score'] = score
            result['search_method'] = 'exhaustive'
            results.append(result)
        
        return results
    
    def _search_with_faiss(self, query_emb: np.ndarray, k: int) -> List[Dict[str, Any]]:
        """FAISS-accelerated search."""
        if self.faiss_index is None:
            return self._search_exhaustive(query_emb, k)
        
        # Search for top tokens for each query token
        candidates_per_query_token = min(100, len(self.token_to_doc_mapping))
        
        doc_scores = {}
        
        for query_token in query_emb:
            # Find most similar document tokens
            token_scores, token_indices = self.faiss_index.search(
                query_token.reshape(1, -1).astype('float32'), 
                candidates_per_query_token
            )
            
            # Aggregate scores by document
            for score, token_idx in zip(token_scores[0], token_indices[0]):
                doc_idx = self.token_to_doc_mapping[token_idx]
                if doc_idx not in doc_scores:
                    doc_scores[doc_idx] = 0
                doc_scores[doc_idx] = max(doc_scores[doc_idx], score)  # MaxSim
        
        # Sum scores across query tokens for each document
        final_scores = list(doc_scores.items())
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        results = []
        for doc_idx, score in final_scores[:k]:
            result = self.doc_metadata[doc_idx].copy()
            result['score'] = float(score)
            result['colbert_score'] = float(score)
            result['search_method'] = 'faiss'
            results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        return {
            **self.retrieval_stats,
            "embedder_stats": self.embedder.stats
        }

class HybridColBERTRetrieval:
    """Hybrid retrieval combining dense embeddings with ColBERT."""
    
    def __init__(self, 
                 dense_embedder: Any,
                 colbert_embedder: ColBERTEmbedder,
                 alpha: float = 0.7):
        """
        Initialize hybrid retrieval.
        
        Args:
            dense_embedder: Dense embedding model (e.g., BGE)
            colbert_embedder: ColBERT embedder
            alpha: Weight for dense scores vs ColBERT scores
        """
        self.dense_embedder = dense_embedder
        self.colbert_retriever = ColBERTRetriever(colbert_embedder)
        self.alpha = alpha
    
    def index_documents(self, documents: List[Dict[str, Any]]):
        """Index documents for both dense and ColBERT retrieval."""
        logger.info("Indexing documents for hybrid dense + ColBERT retrieval")
        
        # Index for ColBERT
        self.colbert_retriever.index_documents(documents)
        
        # Note: Dense indexing would typically be handled by the main pipeline
        logger.info("Hybrid indexing completed")
    
    def search_hybrid(self, query: str, k: int = 10, 
                     dense_results: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining dense and ColBERT results.
        
        Args:
            query: Search query
            k: Number of results
            dense_results: Pre-computed dense retrieval results
            
        Returns:
            Hybrid search results
        """
        # Get ColBERT results
        colbert_results = self.colbert_retriever.search(query, k=k*2)  # Get more for fusion
        
        if dense_results is None:
            # If no dense results provided, return ColBERT results
            return colbert_results[:k]
        
        # Combine results using score fusion
        result_map = {}
        
        # Add dense results
        for i, result in enumerate(dense_results):
            doc_id = result.get('id', str(i))
            dense_score = result.get('score', 0.0)
            
            result_map[doc_id] = {
                **result,
                'dense_score': dense_score,
                'colbert_score': 0.0,
                'hybrid_score': self.alpha * dense_score
            }
        
        # Add ColBERT results
        for result in colbert_results:
            doc_id = result.get('id', result.get('path', str(hash(result.get('content', '')))))
            colbert_score = result.get('colbert_score', 0.0)
            
            if doc_id in result_map:
                # Update existing result
                result_map[doc_id]['colbert_score'] = colbert_score
                result_map[doc_id]['hybrid_score'] += (1 - self.alpha) * colbert_score
            else:
                # Add new result
                result_map[doc_id] = {
                    **result,
                    'dense_score': 0.0,
                    'colbert_score': colbert_score,
                    'hybrid_score': (1 - self.alpha) * colbert_score
                }
        
        # Sort by hybrid score
        hybrid_results = list(result_map.values())
        hybrid_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        # Mark as hybrid results
        for result in hybrid_results:
            result['search_method'] = 'hybrid_dense_colbert'
        
        return hybrid_results[:k]

def integrate_colbert_into_pipeline(documents: List[Dict[str, Any]], 
                                  config: Dict[str, Any] = None) -> ColBERTRetriever:
    """
    Integrate ColBERT into the main embedding pipeline.
    
    Args:
        documents: Documents to index
        config: ColBERT configuration
        
    Returns:
        Configured ColBERT retriever
    """
    if config is None:
        config = {
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2',  # Fallback model
            'max_length': 512,
            'dim': 128,
            'use_faiss': True
        }
    
    logger.info("Integrating ColBERT into embedding pipeline")
    
    # Initialize ColBERT components
    embedder = ColBERTEmbedder(
        model_name=config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2'),
        max_length=config.get('max_length', 512),
        dim=config.get('dim', 128)
    )
    
    retriever = ColBERTRetriever(
        embedder=embedder,
        use_faiss=config.get('use_faiss', True)
    )
    
    # Index documents
    retriever.index_documents(documents)
    
    logger.info("ColBERT integration completed")
    return retriever

def test_colbert():
    """Test ColBERT functionality."""
    logging.basicConfig(level=logging.INFO)
    
    # Sample documents
    documents = [
        {
            'id': '1',
            'content': 'Machine learning algorithms are used for pattern recognition and data analysis.',
            'meta': {'title': 'ML Basics'}
        },
        {
            'id': '2', 
            'content': 'Deep learning neural networks can process complex data representations.',
            'meta': {'title': 'Deep Learning'}
        },
        {
            'id': '3',
            'content': 'Natural language processing enables computers to understand human language.',
            'meta': {'title': 'NLP Overview'}
        }
    ]
    
    print("Testing ColBERT Integration")
    print("=" * 50)
    
    # Initialize and test
    colbert_retriever = integrate_colbert_into_pipeline(documents)
    
    # Test search
    query = "machine learning neural networks"
    results = colbert_retriever.search(query, k=2)
    
    print(f"\nQuery: '{query}'")
    print(f"Results:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result.get('meta', {}).get('title', 'Unknown')} (score: {result['score']:.3f})")
        print(f"     Method: {result.get('search_method', 'unknown')}")
    
    print(f"\nStats: {colbert_retriever.get_stats()}")

if __name__ == "__main__":
    test_colbert()
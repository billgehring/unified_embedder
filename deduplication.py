"""
Document Deduplication Module for Unified Embedder
================================================

This module provides document deduplication capabilities using content hashing,
semantic similarity, and metadata comparison to identify and handle duplicate documents.
"""

import hashlib
import logging
import re
from typing import List, Dict, Any, Set, Tuple, Optional
from pathlib import Path
import json
from collections import defaultdict
import traceback

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Required libraries not installed: {e}")
    print("Please run: uv add numpy scikit-learn sentence-transformers")

# Configure logging
logger = logging.getLogger(__name__)

class DocumentDeduplicator:
    """Advanced document deduplication with multiple strategies."""
    
    def __init__(self, 
                 similarity_threshold: float = 0.95,
                 content_hash_method: str = "sha256",
                 use_semantic_similarity: bool = True,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the deduplicator.
        
        Args:
            similarity_threshold: Threshold for semantic similarity (0-1)
            content_hash_method: Hash method for content hashing
            use_semantic_similarity: Whether to use embedding-based similarity
            embedding_model: Model for semantic similarity computation
        """
        self.similarity_threshold = similarity_threshold
        self.content_hash_method = content_hash_method
        self.use_semantic_similarity = use_semantic_similarity
        self.embedding_model_name = embedding_model
        
        # Lazy-loaded components
        self.embedding_model = None
        self.tfidf_vectorizer = None
        
        # Deduplication statistics
        self.stats = {
            "total_documents": 0,
            "exact_duplicates": 0,
            "near_duplicates": 0,
            "unique_documents": 0,
            "hash_collisions": 0,
            "semantic_duplicates": 0
        }
        
        # Storage for hashes and embeddings
        self.content_hashes = set()
        self.fuzzy_hashes = {}
        self.document_embeddings = {}
    
    def _initialize_models(self):
        """Lazy initialization of models."""
        if self.use_semantic_similarity and self.embedding_model is None:
            try:
                logger.info(f"Loading embedding model for deduplication: {self.embedding_model_name}")
                self.embedding_model = SentenceTransformer(self.embedding_model_name, device="cpu")
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                self.use_semantic_similarity = False
        
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
    
    def _normalize_content(self, content: str) -> str:
        """Normalize content for consistent hashing."""
        if not content:
            return ""
            
        # Convert to lowercase
        normalized = content.lower()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Remove common formatting artifacts
        normalized = re.sub(r'[^\w\s\.,;:!?\'"()-]', '', normalized)
        
        # Remove page numbers and common headers/footers
        normalized = re.sub(r'\bpage\s+\d+\b', '', normalized)
        normalized = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', '', normalized)  # dates
        
        return normalized.strip()
    
    def _compute_content_hash(self, content: str) -> str:
        """Compute content hash for exact duplicate detection."""
        normalized_content = self._normalize_content(content)
        
        if self.content_hash_method == "sha256":
            return hashlib.sha256(normalized_content.encode('utf-8')).hexdigest()
        elif self.content_hash_method == "md5":
            return hashlib.md5(normalized_content.encode('utf-8')).hexdigest()
        else:
            raise ValueError(f"Unsupported hash method: {self.content_hash_method}")
    
    def _compute_fuzzy_hash(self, content: str) -> str:
        """Compute fuzzy hash for near-duplicate detection."""
        # Simple fuzzy hash based on content structure
        normalized = self._normalize_content(content)
        
        # Extract key features: word count, character count, first/last words
        words = normalized.split()
        word_count = len(words)
        char_count = len(normalized)
        
        # Create structural fingerprint
        first_words = " ".join(words[:10]) if words else ""
        last_words = " ".join(words[-10:]) if words else ""
        
        # Create fuzzy hash from structural features
        fuzzy_content = f"{word_count}:{char_count}:{first_words}:{last_words}"
        return hashlib.md5(fuzzy_content.encode('utf-8')).hexdigest()[:16]
    
    def _compute_semantic_similarity(self, content1: str, content2: str) -> float:
        """Compute semantic similarity between two documents."""
        if not self.use_semantic_similarity:
            return 0.0
            
        self._initialize_models()
        
        try:
            # Get embeddings
            embeddings = self.embedding_model.encode([content1, content2])
            
            # Compute cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Error computing semantic similarity: {e}")
            return 0.0
    
    def _compute_tfidf_similarity(self, content1: str, content2: str) -> float:
        """Compute TF-IDF based similarity (faster than embeddings)."""
        try:
            # Fit and transform on both documents
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([content1, content2])
            
            # Compute cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Error computing TF-IDF similarity: {e}")
            return 0.0
    
    def is_duplicate(self, document: Dict[str, Any], existing_docs: List[Dict[str, Any]]) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        Check if document is a duplicate of any existing documents.
        
        Args:
            document: Document to check
            existing_docs: List of existing documents
            
        Returns:
            Tuple of (is_duplicate, original_document, duplicate_type)
        """
        content = document.get('text', '') or document.get('content', '')
        if not content:
            return False, None, "no_content"
        
        # Compute hashes for current document
        content_hash = self._compute_content_hash(content)
        fuzzy_hash = self._compute_fuzzy_hash(content)
        
        # Check existing documents
        for existing_doc in existing_docs:
            existing_content = existing_doc.get('text', '') or existing_doc.get('content', '')
            if not existing_content:
                continue
            
            # 1. Exact duplicate check (content hash)
            existing_hash = self._compute_content_hash(existing_content)
            if content_hash == existing_hash:
                return True, existing_doc, "exact_duplicate"
            
            # 2. Near duplicate check (fuzzy hash)
            existing_fuzzy = self._compute_fuzzy_hash(existing_content)
            if fuzzy_hash == existing_fuzzy:
                # Double-check with TF-IDF similarity
                tfidf_sim = self._compute_tfidf_similarity(content, existing_content)
                if tfidf_sim > self.similarity_threshold:
                    return True, existing_doc, "near_duplicate"
            
            # 3. Semantic similarity check (if enabled)
            if self.use_semantic_similarity:
                semantic_sim = self._compute_semantic_similarity(content, existing_content)
                if semantic_sim > self.similarity_threshold:
                    return True, existing_doc, "semantic_duplicate"
        
        return False, None, "unique"
    
    def deduplicate_documents(self, documents: List[Dict[str, Any]], 
                            keep_metadata: bool = True) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Deduplicate a list of documents.
        
        Args:
            documents: List of documents to deduplicate
            keep_metadata: Whether to merge metadata from duplicates
            
        Returns:
            Tuple of (unique_documents, duplicate_info)
        """
        logger.info(f"Starting deduplication of {len(documents)} documents")
        
        self.stats["total_documents"] = len(documents)
        unique_docs = []
        duplicate_info = []
        
        for i, doc in enumerate(documents):
            if i % 100 == 0:
                logger.info(f"Processing document {i+1}/{len(documents)}")
            
            is_dup, original_doc, dup_type = self.is_duplicate(doc, unique_docs)
            
            if is_dup:
                # Handle duplicate
                duplicate_info.append({
                    'duplicate_doc': doc,
                    'original_doc': original_doc,
                    'duplicate_type': dup_type,
                    'duplicate_index': i
                })
                
                # Update stats
                if dup_type == "exact_duplicate":
                    self.stats["exact_duplicates"] += 1
                elif dup_type == "near_duplicate":
                    self.stats["near_duplicates"] += 1
                elif dup_type == "semantic_duplicate":
                    self.stats["semantic_duplicates"] += 1
                
                # Optionally merge metadata
                if keep_metadata and original_doc:
                    original_meta = original_doc.get('meta', {})
                    duplicate_meta = doc.get('meta', {})
                    
                    # Merge paths and sources
                    if 'duplicate_sources' not in original_meta:
                        original_meta['duplicate_sources'] = []
                    
                    duplicate_source = {
                        'path': doc.get('path', ''),
                        'meta': duplicate_meta,
                        'duplicate_type': dup_type
                    }
                    original_meta['duplicate_sources'].append(duplicate_source)
                    original_doc['meta'] = original_meta
                
                logger.debug(f"Found {dup_type}: {doc.get('path', 'unknown')} -> {original_doc.get('path', 'unknown')}")
                
            else:
                # Unique document
                unique_docs.append(doc)
        
        self.stats["unique_documents"] = len(unique_docs)
        
        logger.info(f"Deduplication completed:")
        logger.info(f"  Total documents: {self.stats['total_documents']}")
        logger.info(f"  Unique documents: {self.stats['unique_documents']}")
        logger.info(f"  Exact duplicates: {self.stats['exact_duplicates']}")
        logger.info(f"  Near duplicates: {self.stats['near_duplicates']}")
        logger.info(f"  Semantic duplicates: {self.stats['semantic_duplicates']}")
        
        return unique_docs, duplicate_info
    
    def deduplicate_by_path_patterns(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate documents based on path patterns (e.g., backup files, versions).
        
        Args:
            documents: List of documents
            
        Returns:
            Filtered list without obvious path-based duplicates
        """
        logger.info("Filtering documents by path patterns")
        
        # Group documents by normalized filename
        filename_groups = defaultdict(list)
        
        for doc in documents:
            path = doc.get('path', '')
            if not path:
                continue
                
            # Extract base filename without version/backup indicators
            filename = Path(path).name
            
            # Remove common version/backup patterns
            base_name = re.sub(r'[-_\s](v\d+|\d+|copy|backup|final|draft)(\.[^.]+)?$', '', filename, flags=re.IGNORECASE)
            base_name = re.sub(r'\s*\(\d+\)(\.[^.]+)?$', '', base_name)  # Remove (1), (2), etc.
            
            filename_groups[base_name].append(doc)
        
        # For each group, keep the "best" version
        filtered_docs = []
        for base_name, group_docs in filename_groups.items():
            if len(group_docs) == 1:
                filtered_docs.extend(group_docs)
            else:
                # Choose the best document from the group
                best_doc = self._choose_best_document(group_docs)
                filtered_docs.append(best_doc)
                
                # Add info about filtered duplicates
                for doc in group_docs:
                    if doc != best_doc:
                        if 'meta' not in best_doc:
                            best_doc['meta'] = {}
                        if 'path_duplicates' not in best_doc['meta']:
                            best_doc['meta']['path_duplicates'] = []
                        best_doc['meta']['path_duplicates'].append(doc.get('path', ''))
        
        logger.info(f"Path-based filtering: {len(documents)} -> {len(filtered_docs)} documents")
        return filtered_docs
    
    def _choose_best_document(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Choose the best document from a list of candidates."""
        if len(candidates) == 1:
            return candidates[0]
        
        # Scoring criteria (higher is better)
        def score_document(doc):
            score = 0
            path = doc.get('path', '').lower()
            
            # Prefer non-backup files
            if 'backup' not in path and 'copy' not in path:
                score += 10
            
            # Prefer files without version numbers in path
            if not re.search(r'[-_\s]v?\d+[-_\s]', path):
                score += 5
            
            # Prefer longer content
            content = doc.get('text', '') or doc.get('content', '')
            score += len(content) / 1000  # Length bonus
            
            # Prefer files with more metadata
            meta = doc.get('meta', {})
            score += len(meta)
            
            # Prefer more recent files (if timestamp available)
            if 'created_at' in meta:
                try:
                    # Simple heuristic: more recent = higher score
                    score += 1
                except:
                    pass
            
            return score
        
        # Return document with highest score
        return max(candidates, key=score_document)
    
    def export_deduplication_report(self, duplicate_info: List[Dict[str, Any]], 
                                  output_path: str = "deduplication_report.json"):
        """Export detailed deduplication report."""
        report = {
            'statistics': self.stats,
            'duplicates': [],
            'generated_at': str(Path().cwd()),
            'configuration': {
                'similarity_threshold': self.similarity_threshold,
                'content_hash_method': self.content_hash_method,
                'use_semantic_similarity': self.use_semantic_similarity,
                'embedding_model': self.embedding_model_name
            }
        }
        
        for dup_info in duplicate_info:
            report['duplicates'].append({
                'duplicate_type': dup_info['duplicate_type'],
                'duplicate_path': dup_info['duplicate_doc'].get('path', ''),
                'original_path': dup_info['original_doc'].get('path', ''),
                'duplicate_index': dup_info['duplicate_index']
            })
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Deduplication report exported to: {output_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics."""
        return self.stats.copy()

def integrate_deduplication_into_pipeline(documents: List[Dict[str, Any]], 
                                        config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Integrate deduplication into the main embedding pipeline.
    
    Args:
        documents: List of documents from unified_embedder
        config: Deduplication configuration
        
    Returns:
        Deduplicated list of documents
    """
    if config is None:
        config = {
            'similarity_threshold': 0.95,
            'use_semantic_similarity': True,
            'use_path_filtering': True,
            'export_report': True
        }
    
    logger.info("Starting integrated deduplication process")
    
    # Initialize deduplicator
    deduplicator = DocumentDeduplicator(
        similarity_threshold=config.get('similarity_threshold', 0.95),
        use_semantic_similarity=config.get('use_semantic_similarity', True)
    )
    
    # Step 1: Path-based filtering (quick wins)
    if config.get('use_path_filtering', True):
        documents = deduplicator.deduplicate_by_path_patterns(documents)
    
    # Step 2: Content-based deduplication
    unique_docs, duplicate_info = deduplicator.deduplicate_documents(documents)
    
    # Step 3: Export report if requested
    if config.get('export_report', True):
        deduplicator.export_deduplication_report(duplicate_info)
    
    logger.info(f"Deduplication complete: {len(documents)} -> {len(unique_docs)} documents")
    return unique_docs

def test_deduplication():
    """Test function for deduplication capabilities."""
    logging.basicConfig(level=logging.INFO)
    
    # Create test documents with duplicates
    test_docs = [
        {
            'path': '/test/doc1.pdf',
            'text': 'This is a sample document about machine learning and artificial intelligence.',
            'meta': {'source': 'test1'}
        },
        {
            'path': '/test/doc1_copy.pdf', 
            'text': 'This is a sample document about machine learning and artificial intelligence.',
            'meta': {'source': 'test1_copy'}
        },
        {
            'path': '/test/doc2.pdf',
            'text': 'This is a sample document about machine learning and AI.',  # Near duplicate
            'meta': {'source': 'test2'}
        },
        {
            'path': '/test/doc3.pdf',
            'text': 'Completely different content about biology and chemistry.',
            'meta': {'source': 'test3'}
        }
    ]
    
    print(f"Testing deduplication with {len(test_docs)} documents")
    
    # Test deduplication
    deduplicator = DocumentDeduplicator(similarity_threshold=0.8)
    unique_docs, duplicate_info = deduplicator.deduplicate_documents(test_docs)
    
    print(f"\nResults:")
    print(f"  Unique documents: {len(unique_docs)}")
    print(f"  Duplicates found: {len(duplicate_info)}")
    print(f"  Stats: {deduplicator.get_stats()}")
    
    print(f"\nUnique documents:")
    for doc in unique_docs:
        print(f"  - {doc['path']}")
        if 'duplicate_sources' in doc.get('meta', {}):
            print(f"    Merged from: {[src['path'] for src in doc['meta']['duplicate_sources']]}")
    
    print(f"\nDuplicate info:")
    for dup in duplicate_info:
        print(f"  - {dup['duplicate_doc']['path']} -> {dup['original_doc']['path']} ({dup['duplicate_type']})")

if __name__ == "__main__":
    test_deduplication()
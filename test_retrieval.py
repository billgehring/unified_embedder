#!/usr/bin/env python3
"""
Test Retrieval Script for Unified Embedder Document Stores

This script tests retrieval functionality for document stores created by unified_embedder.py.
It can query Qdrant collections and perform similarity searches.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger("test_retrieval")

try:
    from haystack import Document
    from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
    from haystack_integrations.components.embedders.fastembed import (
        FastembedTextEmbedder,
        FastembedSparseTextEmbedder
    )
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from haystack.utils import Secret
    from reranking import HybridRetrievalPipeline
    
    # Import our new ColBERT components
    from hybrid_qdrant_store import HybridQdrantStore, create_hybrid_store
    from multi_vector_retrieval import (
        MultiVectorRetriever, 
        MultiVectorConfig, 
        create_multi_vector_retriever,
        QueryType
    )
    from colbert_token_embedder import ColBERTTokenEmbedder
except ImportError as e:
    logger.error(f"Required libraries not installed: {e}")
    logger.error("Please run: uv add haystack-ai qdrant-haystack fastembed-haystack qdrant-client sentence-transformers")
    sys.exit(1)


class DocumentStoreRetriever:
    """Test retrieval functionality for document stores."""
    
    def __init__(self, 
                 qdrant_url: str = "http://localhost:6333",
                 qdrant_api_key: Optional[str] = None,
                 collection_name: str = "test_fixes"):
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.collection_name = collection_name
        self.client = None
        self.document_store = None
        
    def connect_to_qdrant(self) -> bool:
        """Connect to Qdrant and verify collection exists."""
        try:
            # Direct client connection
            self.client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key if self.qdrant_api_key else None,
                timeout=60  # Extended timeout for large collections
            )
            
            # Test connection
            collections = self.client.get_collections()
            logger.info(f"Connected to Qdrant. Available collections: {[c.name for c in collections.collections]}")
            
            # Check if our collection exists
            collection_exists = any(c.name == self.collection_name for c in collections.collections)
            if not collection_exists:
                logger.error(f"Collection '{self.collection_name}' not found!")
                logger.info(f"Available collections: {[c.name for c in collections.collections]}")
                return False
                
            # Get collection info
            collection_info = self.client.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' info:")
            logger.info(f"  - Points count: {collection_info.points_count}")
            logger.info(f"  - Vectors config: {collection_info.config.params.vectors}")
            
            # Initialize Haystack document store with proper configuration
            self.document_store = QdrantDocumentStore(
                url=self.qdrant_url,
                api_key=Secret.from_token(self.qdrant_api_key) if self.qdrant_api_key else None,
                index=self.collection_name,
                return_embedding=True,
                use_sparse_embeddings=True,  # Enable sparse embeddings support
                embedding_dim=1024,  # Match the vector dimension
                recreate_index=False  # Don't recreate existing collection
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            return False
    
    def get_sample_documents(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get sample documents from the collection."""
        try:
            # Use scroll to get documents with payload and vectors
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=True
            )
            
            points = scroll_result[0]  # First element is the points list
            logger.info(f"Retrieved {len(points)} sample documents")
            
            samples = []
            for point in points:
                sample = {
                    'id': str(point.id),
                    'payload': point.payload,
                    'has_vectors': bool(point.vector)
                }
                
                # Check for content
                if 'content' in point.payload:
                    content = point.payload['content']
                    sample['content_preview'] = content[:200] + "..." if len(content) > 200 else content
                    sample['content_length'] = len(content)
                
                samples.append(sample)
                
            return samples
            
        except Exception as e:
            logger.error(f"Failed to get sample documents: {e}")
            return []
    
    def search_similar(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Perform similarity search using query text."""
        try:
            # Initialize embedder (same model as unified_embedder.py default)
            embedder = FastembedTextEmbedder(model="BAAI/bge-m3")
            embedder.warm_up()
            
            # Generate query embedding
            query_result = embedder.run(text=query)
            query_embedding = query_result["embedding"]
            
            logger.info(f"Generated query embedding with dimension: {len(query_embedding)}")
            
            # Search using Qdrant client - use query_points with proper vector specification
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                using="text-dense",  # Specify which vector to use
                limit=limit,
                with_payload=True
            )
            
            results = []
            for hit in search_result.points:
                result = {
                    'id': str(hit.id),
                    'score': hit.score,
                    'payload': hit.payload,
                    'vector_type': 'dense',
                    'search_method': 'dense'
                }
                
                # Extract content preview and metadata
                if 'content' in hit.payload:
                    content = hit.payload['content']
                    result['content_preview'] = content[:300] + "..." if len(content) > 300 else content
                    result['content_length'] = len(content)
                
                # Extract useful metadata
                if 'meta' in hit.payload and hit.payload['meta']:
                    meta = hit.payload['meta']
                    if 'filename' in meta:
                        result['filename'] = meta['filename']
                    if 'file_path' in meta:
                        result['file_path'] = meta['file_path']
                    if 'processed_by' in meta:
                        result['processed_by'] = meta['processed_by']
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {e}")
            return []
    
    def test_haystack_retrieval(self, query: str, top_k: int = 3):
        """Test retrieval using Haystack document store directly."""
        try:
            # Use document store's filter_documents method
            all_docs = self.document_store.filter_documents()
            logger.info(f"Total documents in store: {len(all_docs)}")
            
            if not all_docs:
                logger.warning("No documents found in document store")
                return
                
            # Show sample documents
            for i, doc in enumerate(all_docs[:3]):
                logger.info(f"Sample doc {i+1}:")
                logger.info(f"  Content: {doc.content[:100]}...")
                logger.info(f"  Meta keys: {list(doc.meta.keys()) if doc.meta else 'None'}")
                
        except Exception as e:
            logger.error(f"Failed to test Haystack retrieval: {e}")
    
    def inspect_document_content(self, limit: int = 2) -> None:
        """Inspect raw document content to debug content issues."""
        try:
            logger.info("=== Document Content Inspection ===")
            
            # Get raw documents
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False  # Don't need vectors for content inspection
            )
            
            points = scroll_result[0]
            logger.info(f"Inspecting {len(points)} documents from '{self.collection_name}'")
            
            for i, point in enumerate(points, 1):
                logger.info(f"\n--- Document {i} ---")
                logger.info(f"ID: {point.id}")
                
                if point.payload:
                    # Check content field
                    if 'content' in point.payload:
                        content = point.payload['content']
                        content_len = len(content)
                        logger.info(f"Content length: {content_len} chars")
                        
                        # Show first 500 chars of content
                        preview = content[:500] if len(content) > 500 else content
                        logger.info(f"Content preview:\n{repr(preview)}")
                        
                        # Check if it's mostly XML
                        if content.strip().startswith('<?xml'):
                            logger.warning("⚠️  Content appears to be XML - possible processing issue!")
                    
                    # Check metadata
                    if 'meta' in point.payload:
                        meta = point.payload['meta']
                        if isinstance(meta, dict):
                            logger.info("Metadata keys:")
                            for key, value in list(meta.items())[:10]:  # First 10 keys
                                if isinstance(value, str) and len(value) < 100:
                                    logger.info(f"  {key}: {value}")
                                else:
                                    logger.info(f"  {key}: {type(value)} (len={len(str(value))})")
                            
                            if len(meta) > 10:
                                logger.info(f"  ... and {len(meta) - 10} more keys")
                    
                else:
                    logger.warning("No payload data!")
        
        except Exception as e:
            logger.error(f"Error inspecting content: {e}")
    
    def diagnose_collection_issues(self):
        """Diagnose potential issues with the collection."""
        try:
            # Get detailed collection info
            collection_info = self.client.get_collection(self.collection_name)
            
            logger.info("=== Collection Diagnosis ===")
            logger.info(f"Collection: {self.collection_name}")
            logger.info(f"Status: {collection_info.status}")
            logger.info(f"Points count: {collection_info.points_count}")
            logger.info(f"Config: {collection_info.config}")
            
            # Check for empty collection
            if collection_info.points_count == 0:
                logger.warning("Collection is empty - no documents were successfully indexed")
                return
            
            # Try to get a single point to check data integrity
            try:
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=1,
                    with_payload=True,
                    with_vectors=True
                )
                
                if scroll_result[0]:
                    point = scroll_result[0][0]
                    logger.info("Sample point structure:")
                    logger.info(f"  ID: {point.id}")
                    logger.info(f"  Payload keys: {list(point.payload.keys()) if point.payload else 'None'}")
                    logger.info(f"  Has vector: {bool(point.vector)}")
                    
                    if point.vector:
                        if isinstance(point.vector, dict):
                            logger.info(f"  Vector types: {list(point.vector.keys())}")
                            for vec_name, vec_data in point.vector.items():
                                if hasattr(vec_data, '__len__'):
                                    logger.info(f"    {vec_name}: length {len(vec_data)}")
                        else:
                            logger.info(f"  Vector length: {len(point.vector)}")
                else:
                    logger.warning("Could not retrieve any points despite non-zero count")
                    
            except Exception as e:
                logger.error(f"Error retrieving sample point: {e}")
                logger.error("This might indicate data corruption or format issues")
                
        except Exception as e:
            logger.error(f"Failed to diagnose collection: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test retrieval from unified embedder document stores")
    parser.add_argument("--qdrant_url", default="http://localhost:6333", help="Qdrant server URL")
    parser.add_argument("--qdrant_api_key", help="Qdrant API key")
    parser.add_argument("--collection", default="test_fixes", help="Collection name to test")
    parser.add_argument("--query", default="memory psychology lecture", help="Test query for similarity search")
    parser.add_argument("--limit", type=int, default=3, help="Number of results to return")
    parser.add_argument("--samples", type=int, default=5, help="Number of sample documents to show")
    parser.add_argument("--diagnose", action="store_true", help="Run collection diagnostics")
    parser.add_argument("--inspect", action="store_true", help="Inspect raw document content for debugging")
    parser.add_argument("--print_components", action="store_true", help="Print per-result diagnostic components (vector scores, weights, times)")
    parser.add_argument("--json_out", type=str, help="Write full result list (with diagnostics) to JSON file")
    parser.add_argument("--rerank", action="store_true", help="Use reranking for improved retrieval quality")
    parser.add_argument("--initial_k", type=int, default=20, help="Initial number of documents to retrieve for reranking")
    parser.add_argument("--reranker_model", default="BAAI/bge-reranker-v2-m3", help="Reranking model to use")
    parser.add_argument("--expand_query", action="store_true", help="Use query expansion for better keyword matching")
    parser.add_argument("--colbert", action="store_true", help="Test ColBERT token-based retrieval")
    parser.add_argument("--colbert_model", default="colbert-ir/colbertv2.0", help="ColBERT model to use")
    parser.add_argument("--multi_vector", action="store_true", help="Test multi-vector retrieval with all embedding types")
    parser.add_argument("--search_mode", default="all", choices=["dense", "sparse", "colbert", "all"], help="Vector search mode")
    parser.add_argument("--adaptive_weights", action="store_true", help="Use adaptive query type-based weights")
    parser.add_argument("--show_timing", action="store_true", help="Show detailed timing information")
    
    args = parser.parse_args()
    
    logger.info(f"Testing retrieval from collection: {args.collection}")
    logger.info(f"Qdrant URL: {args.qdrant_url}")
    
    # Initialize retriever
    retriever = DocumentStoreRetriever(
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        collection_name=args.collection
    )
    
    # Connect to Qdrant
    if not retriever.connect_to_qdrant():
        logger.error("Failed to connect to Qdrant. Exiting.")
        sys.exit(1)
    
    # Run diagnostics if requested
    if args.diagnose:
        retriever.diagnose_collection_issues()
        return
    
    # Inspect content if requested
    if args.inspect:
        retriever.inspect_document_content()
        return
    
    # Get sample documents
    logger.info(f"\n=== Getting {args.samples} sample documents ===")
    samples = retriever.get_sample_documents(args.samples)
    for i, sample in enumerate(samples, 1):
        logger.info(f"\nSample {i}:")
        logger.info(f"  ID: {sample['id']}")
        logger.info(f"  Has vectors: {sample['has_vectors']}")
        if 'content_preview' in sample:
            logger.info(f"  Content ({sample['content_length']} chars): {sample['content_preview']}")
        if 'filename' in sample.get('payload', {}):
            logger.info(f"  Filename: {sample['payload']['filename']}")
    
    # Test ColBERT token retrieval if requested
    if args.colbert or args.multi_vector:
        logger.info(f"\n=== Testing ColBERT Token Retrieval ===")
        logger.info(f"Query: '{args.query}'")
        logger.info(f"ColBERT model: {args.colbert_model}")
        logger.info(f"Search mode: {args.search_mode if not args.multi_vector else 'multi-vector'}")
        
        try:
            # Try to find ColBERT collection
            colbert_collection = f"{args.collection}_colbert"
            
            # Create hybrid store to test ColBERT tokens
            hybrid_store = create_hybrid_store(
                url=args.qdrant_url,
                collection_name=args.collection,
                api_key=args.qdrant_api_key,
                enable_colbert=True,
                colbert_model=args.colbert_model
            )
            
            # Create multi-vector retriever configuration
            config = MultiVectorConfig(
                adaptive_weights=args.adaptive_weights,
                final_limit=args.limit,
                max_search_time_ms=100.0 if args.show_timing else 50.0
            )
            
            # Create retriever
            multi_retriever = create_multi_vector_retriever(
                hybrid_store=hybrid_store,
                colbert_collection=colbert_collection,
                config=config
            )
            
            # Perform search
            search_mode = "colbert" if args.colbert and not args.multi_vector else args.search_mode
            start_time = time.time()
            
            search_results = multi_retriever.search(
                query=args.query,
                limit=args.limit,
                search_mode=search_mode
            )
            
            search_time_ms = (time.time() - start_time) * 1000
            
            # Convert to standard format for display, including diagnostics
            results = []
            for search_result in search_results:
                converted_result = {
                    'id': search_result.id,
                    'score': search_result.score,
                    'content_preview': search_result.content[:300] if search_result.content else '',
                    'content_length': len(search_result.content) if search_result.content else 0,
                    'vector_type': search_result.vector_type,
                    'search_time_ms': search_result.search_time_ms,
                    'search_method': f'{search_result.vector_type}_token_retrieval'
                }
                # Diagnostics from multi-vector retriever
                if getattr(search_result, 'vector_scores', None):
                    converted_result['vector_scores'] = search_result.vector_scores
                if getattr(search_result, 'fusion_weights', None):
                    converted_result['fusion_weights'] = search_result.fusion_weights
                if getattr(search_result, 'component_times_ms', None):
                    converted_result['component_times_ms'] = search_result.component_times_ms
                
                # Add metadata info
                if search_result.metadata:
                    if 'filename' in search_result.metadata:
                        converted_result['filename'] = search_result.metadata['filename']
                    if 'file_path' in search_result.metadata:
                        converted_result['file_path'] = search_result.metadata['file_path']
                    converted_result['payload'] = {'meta': search_result.metadata}
                        
                results.append(converted_result)
            
            # Show performance stats
            perf_stats = multi_retriever.get_performance_stats()
            logger.info(f"\n📊 Performance Stats:")
            logger.info(f"  Search time: {search_time_ms:.1f}ms")
            logger.info(f"  Average response time: {perf_stats.get('avg_response_time_ms', 0):.1f}ms")
            logger.info(f"  Cache hit rate: {perf_stats.get('cache_hit_rate', 0):.1f}%")
            logger.info(f"  Query classification: {multi_retriever.classify_query(args.query).value}")
            
            if args.show_timing:
                logger.info(f"\n⏱️  Detailed Timing:")
                for key, value in perf_stats.items():
                    if 'time' in key or 'queries' in key:
                        logger.info(f"  {key}: {value}")
            
        except ImportError as e:
            logger.error(f"Multi-vector retrieval not available: {e}")
            results = retriever.search_similar(args.query, args.limit)
        except Exception as e:
            logger.error(f"ColBERT token retrieval failed: {e}")
            logger.error(f"Falling back to standard similarity search")
            results = retriever.search_similar(args.query, args.limit)
    
    # Test similarity search
    elif args.rerank:
        logger.info(f"\n=== Testing hybrid retrieval with reranking ===")
        logger.info(f"Query: '{args.query}'")
        logger.info(f"Reranker model: {args.reranker_model}")
        
        # Initialize hybrid pipeline
        pipeline = HybridRetrievalPipeline(
            qdrant_url=args.qdrant_url,
            qdrant_api_key=args.qdrant_api_key,
            collection_name=args.collection,
            reranker_model=args.reranker_model
        )
        
        if pipeline.connect():
            results = pipeline.search_and_rerank(
                query=args.query,
                initial_k=args.initial_k,
                final_k=args.limit,
                use_metadata_boost=True,
                use_query_expansion=args.expand_query
            )
            
            # Add reranking-specific info to results
            for result in results:
                if 'reranked' in result:
                    result['retrieval_method'] = 'hybrid_reranked'
                    
            logger.info(f"Pipeline stats: {pipeline.get_pipeline_stats()}")
        else:
            logger.error("Failed to initialize hybrid pipeline, falling back to standard search")
            results = retriever.search_similar(args.query, args.limit)
    else:
        logger.info(f"\n=== Testing similarity search ===")
        logger.info(f"Query: '{args.query}'")
        results = retriever.search_similar(args.query, args.limit)
    
    if results:
        logger.info(f"Found {len(results)} similar documents:")
        for i, result in enumerate(results, 1):
            # Show enhanced scoring info for all result types
            score_info = f"score: {result['score']:.4f}"
            if 'original_score' in result:
                score_info += f" (original: {result['original_score']:.4f})"
            if 'metadata_boost' in result:
                score_info += f" [boost: {result['metadata_boost']}]"
            if 'vector_type' in result:
                score_info += f" [{result['vector_type']}]"
            if 'search_time_ms' in result:
                score_info += f" ({result['search_time_ms']:.1f}ms)"
            
            logger.info(f"\n--- Result {i} ({score_info}) ---")
            
            # Show search method
            if 'search_method' in result:
                logger.info(f"  🔍 Method: {result['search_method']}")
            
            # Show filename/file path
            if 'filename' in result:
                logger.info(f"  📄 File: {result['filename']}")
            elif 'file_path' in result:
                logger.info(f"  📂 Path: {result['file_path']}")
            
            # Show processing method
            if 'processed_by' in result:
                logger.info(f"  🔧 Processed by: {result['processed_by']}")
            
            # Show content info
            if 'content_length' in result:
                logger.info(f"  📊 Content length: {result['content_length']} chars")
            
            # Show content preview with better formatting
            if 'content_preview' in result:
                content = result['content_preview']
                if content.strip():
                    # Clean up whitespace and show meaningful content
                    content = ' '.join(content.split())  # Normalize whitespace
                    if len(content) > 200:
                        content = content[:200] + "..."
                    logger.info(f"  📝 Content: {content}")
                else:
                    logger.warning(f"  ⚠️  Empty or whitespace-only content!")

            # Show diagnostic scoring breakdown when requested
            if args.print_components:
                if 'vector_scores' in result and isinstance(result['vector_scores'], dict):
                    logger.info(f"  📈 Vector scores: {result['vector_scores']}")
                if 'fusion_weights' in result and isinstance(result['fusion_weights'], dict):
                    logger.info(f"  ⚖️  Fusion weights: {result['fusion_weights']}")
                if 'component_times_ms' in result and isinstance(result['component_times_ms'], dict):
                    logger.info(f"  ⏱️  Component times (ms): {result['component_times_ms']}")
            
            # Show key metadata if available
            if 'payload' in result and 'meta' in result['payload']:
                meta = result['payload']['meta']
                if isinstance(meta, dict):
                    interesting_keys = ['file_type', 'source', 'created_at', 'size']
                    for key in interesting_keys:
                        if key in meta and meta[key]:
                            logger.info(f"  🏷️  {key}: {meta[key]}")
    else:
        logger.warning("No search results found")
    
    # Test Haystack retrieval
    logger.info(f"\n=== Testing Haystack document store ===")
    retriever.test_haystack_retrieval(args.query)
    
    # Optional: write results JSON for downstream analysis
    if args.json_out:
        try:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(results or [], f, indent=2, ensure_ascii=False)
            logger.info(f"\n📝 Wrote results to {args.json_out}")
        except Exception as e:
            logger.error(f"Failed to write JSON results to {args.json_out}: {e}")

    logger.info("\nRetrieval testing completed!")
    logger.info("\n💡 Tips for testing:")
    logger.info("  • Use --multi_vector for comprehensive retrieval testing")
    logger.info("  • Use --colbert for ColBERT-only token retrieval")
    logger.info("  • Use --adaptive_weights for query-type adaptive scoring")
    logger.info("  • Use --show_timing for performance analysis")
    logger.info("  • Use --search_mode [dense|sparse|colbert] for specific vector types")


if __name__ == "__main__":
    main()

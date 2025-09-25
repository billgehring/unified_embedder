#!/usr/bin/env python3
"""
Monitored Retrieval Wrapper
==========================

Integration wrapper that combines multi-vector retrieval with 
real-time performance monitoring for production voice tutor applications.

Usage:
    from monitored_retrieval import MonitoredRetrieval
    
    retriever = MonitoredRetrieval(collection_name="my_collection")
    results = retriever.search("What is photosynthesis?")
"""

import logging
from typing import List, Dict, Any, Optional
import time
from pathlib import Path

from multi_vector_retrieval import (
    MultiVectorRetriever, 
    MultiVectorConfig, 
    SearchResult,
    QueryType
)
from performance_monitor import PerformanceMonitor, console_alert_callback, log_alert_callback
from hybrid_qdrant_store import create_hybrid_store

logger = logging.getLogger(__name__)

class MonitoredRetrieval:
    """
    Production-ready retrieval system with integrated performance monitoring.
    
    Combines multi-vector retrieval with real-time monitoring and alerting
    for educational voice tutor applications.
    """
    
    def __init__(self,
                 collection_name: str,
                 qdrant_url: str = "http://localhost:6333",
                 qdrant_api_key: Optional[str] = None,
                 colbert_model: str = "colbert-ir/colbertv2.0",
                 enable_colbert: bool = True,
                 alert_threshold_ms: float = 50.0,
                 enable_console_alerts: bool = True,
                 enable_log_alerts: bool = True):
        """
        Initialize monitored retrieval system.
        
        Args:
            collection_name: Qdrant collection name
            qdrant_url: Qdrant server URL  
            qdrant_api_key: Qdrant API key
            colbert_model: ColBERT model name
            enable_colbert: Enable ColBERT token retrieval
            alert_threshold_ms: Alert threshold for slow queries
            enable_console_alerts: Show alerts in console
            enable_log_alerts: Log alerts to file
        """
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.colbert_model = colbert_model
        self.enable_colbert = enable_colbert
        
        # Initialize performance monitor
        self.monitor = PerformanceMonitor(
            alert_threshold_ms=alert_threshold_ms,
            history_window_minutes=60,  # Keep 1 hour of history
            trend_window_minutes=10     # 10-minute trend analysis
        )
        
        # Setup alert callbacks
        if enable_console_alerts:
            self.monitor.add_alert_callback(console_alert_callback)
        if enable_log_alerts:
            self.monitor.add_alert_callback(log_alert_callback)
        
        # Initialize retrieval components
        self.hybrid_store = None
        self.retriever = None
        self._initialized = False
        
        logger.info(f"MonitoredRetrieval initialized for collection: {collection_name}")
    
    def initialize(self) -> bool:
        """Initialize retrieval components."""
        if self._initialized:
            return True
            
        try:
            logger.info("Initializing retrieval components...")
            
            # Create hybrid store
            self.hybrid_store = create_hybrid_store(
                url=self.qdrant_url,
                collection_name=self.collection_name,
                api_key=self.qdrant_api_key,
                enable_colbert=self.enable_colbert,
                colbert_model=self.colbert_model
            )
            
            # Create multi-vector retriever with optimized config
            config = MultiVectorConfig(
                adaptive_weights=True,
                parallel_search=True,
                max_search_time_ms=45.0,  # Slightly under 50ms target
                final_limit=10,
                dense_limit=30,
                sparse_limit=20,
                colbert_limit=15
            )
            
            # Import the factory function
            from multi_vector_retrieval import create_multi_vector_retriever
            
            colbert_collection = f"{self.collection_name}_colbert" if self.enable_colbert else None
            self.retriever = create_multi_vector_retriever(
                hybrid_store=self.hybrid_store,
                colbert_collection=colbert_collection,
                config=config
            )
            
            # Start background monitoring
            self.monitor.start_monitoring(interval_seconds=30)
            
            self._initialized = True
            logger.info("Retrieval components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize retrieval components: {e}")
            return False
    
    def search(self,
              query: str,
              limit: int = 10,
              search_mode: str = "all",
              timeout_ms: Optional[float] = None) -> List[SearchResult]:
        """
        Perform monitored search with performance tracking.
        
        Args:
            query: Search query text
            limit: Maximum number of results
            search_mode: Search mode (dense, sparse, colbert, all)
            timeout_ms: Query timeout in milliseconds
            
        Returns:
            List of search results
        """
        if not self._initialized and not self.initialize():
            logger.error("Cannot search - initialization failed")
            return []
        
        start_time = time.time()
        
        try:
            # Classify query for monitoring
            query_type = self.retriever.classify_query(query)
            
            # Perform search
            results = self.retriever.search(
                query=query,
                limit=limit,
                search_mode=search_mode,
                timeout_ms=timeout_ms
            )
            
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            
            # Check for cache hit (simplified detection)
            retriever_stats = self.retriever.get_performance_stats()
            previous_cache_hits = getattr(self, '_last_cache_hits', 0)
            current_cache_hits = retriever_stats.get('cache_hits', 0)
            cache_hit = current_cache_hits > previous_cache_hits
            self._last_cache_hits = current_cache_hits
            
            # Record performance metrics
            self.monitor.record_query(
                query=query,
                query_type=query_type.value,
                response_time_ms=response_time_ms,
                results_count=len(results),
                cache_hit=cache_hit
            )
            
            logger.debug(f"Search completed: {len(results)} results in {response_time_ms:.1f}ms")
            return results
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Search failed after {response_time_ms:.1f}ms: {e}")
            
            # Record failed query
            self.monitor.record_query(
                query=query,
                query_type="unknown",
                response_time_ms=response_time_ms,
                results_count=0,
                cache_hit=False
            )
            return []
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return self.monitor.get_current_stats()
    
    def get_trend_analysis(self) -> Dict[str, Any]:
        """Get performance trend analysis.""" 
        return self.monitor.get_trend_analysis()
    
    def get_query_patterns(self) -> Dict[str, Any]:
        """Get query type performance patterns."""
        return self.monitor.get_query_pattern_analysis()
    
    def export_performance_data(self, output_file: str):
        """Export performance monitoring data."""
        self.monitor.export_metrics(output_file)
    
    def is_voice_ready(self) -> bool:
        """Check if performance meets voice tutor requirements."""
        stats = self.get_performance_stats()
        if "error" in stats:
            return False
            
        # Voice ready criteria:
        # - 95% of queries under 50ms
        # - Average response time under 40ms  
        voice_ready_percent = stats.get("voice_ready_percent", 0)
        avg_response_time = stats.get("avg_response_time_ms", 100)
        
        return voice_ready_percent >= 95.0 and avg_response_time <= 40.0
    
    def shutdown(self):
        """Gracefully shutdown monitoring and connections."""
        logger.info("Shutting down monitored retrieval...")
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        # Close connections
        if self.hybrid_store:
            self.hybrid_store.close()
            
        logger.info("Shutdown complete")

# Example usage and testing
if __name__ == "__main__":
    import argparse
    import random
    
    parser = argparse.ArgumentParser(description="Test monitored retrieval system")
    parser.add_argument("--collection", required=True, help="Collection name to test")
    parser.add_argument("--qdrant_url", default="http://localhost:6333", help="Qdrant URL")
    parser.add_argument("--queries", type=int, default=20, help="Number of test queries")
    parser.add_argument("--export", help="Export performance data to file")
    
    args = parser.parse_args()
    
    # Sample educational queries
    test_queries = [
        "What is photosynthesis?",
        "How does memory formation work?", 
        "Define classical conditioning",
        "Compare DNA and RNA structure",
        "Steps to solve quadratic equations",
        "What is cognitive dissonance?",
        "How do neural networks learn?",
        "Explain natural selection process",
        "What are the stages of mitosis?",
        "Define statistical significance"
    ]
    
    # Initialize monitored retrieval
    retriever = MonitoredRetrieval(
        collection_name=args.collection,
        qdrant_url=args.qdrant_url,
        enable_console_alerts=True,
        enable_log_alerts=True
    )
    
    if not retriever.initialize():
        print("❌ Failed to initialize retrieval system")
        sys.exit(1)
    
    print(f"🚀 Testing monitored retrieval with {args.queries} queries...")
    print(f"📊 Collection: {args.collection}")
    
    # Run test queries
    try:
        for i in range(args.queries):
            query = random.choice(test_queries)
            
            print(f"\nQuery {i+1}/{args.queries}: {query}")
            results = retriever.search(query, limit=5)
            
            print(f"  ✅ Found {len(results)} results")
            if results:
                print(f"  📝 Top result: {results[0].content[:100]}...")
            
            # Small delay between queries
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n⏹️  Testing interrupted by user")
    
    # Show performance summary
    print("\n" + "="*60)
    print("📈 PERFORMANCE SUMMARY")
    print("="*60)
    
    stats = retriever.get_performance_stats()
    if "error" not in stats:
        print(f"Total queries: {stats['total_queries']}")
        print(f"Average response time: {stats['avg_response_time_ms']:.1f}ms")
        print(f"95th percentile: {stats['p95_response_time_ms']:.1f}ms")
        print(f"Voice-ready queries: {stats['voice_ready_percent']:.1f}%")
        print(f"Cache hit rate: {stats['cache_hit_rate']:.1f}%")
        
        if retriever.is_voice_ready():
            print("✅ VOICE TUTOR READY")
        else:
            print("⚠️  OPTIMIZATION NEEDED")
    
    # Show trends
    trends = retriever.get_trend_analysis()
    if "message" not in trends:
        print(f"\n📊 Performance trend: {trends['trend_direction']}")
        print(f"Change: {trends['percent_change']:+.1f}%")
    
    # Export data if requested
    if args.export:
        retriever.export_performance_data(args.export)
        print(f"\n💾 Performance data exported to: {args.export}")
    
    # Cleanup
    retriever.shutdown()
    print("\n✅ Test completed!")
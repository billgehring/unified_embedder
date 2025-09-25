#!/usr/bin/env python3
"""
Performance Benchmarking and Monitoring Tools
============================================

Comprehensive benchmarking suite for the unified embedder ColBERT integration,
designed for educational voice tutor applications with <50ms retrieval targets.

Features:
- Multi-vector retrieval performance testing
- Storage efficiency analysis  
- Query response time profiling
- Educational domain query simulations
- Real-time monitoring capabilities
- Performance regression detection
"""

import argparse
import logging
import time
import json
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import concurrent.futures
import threading
import psutil
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger("performance_benchmark")

try:
    from hybrid_qdrant_store import HybridQdrantStore, create_hybrid_store
    from multi_vector_retrieval import (
        MultiVectorRetriever, 
        MultiVectorConfig, 
        create_multi_vector_retriever,
        QueryType
    )
    from colbert_token_embedder import ColBERTTokenEmbedder
    from qdrant_client import QdrantClient
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Required components not available: {e}")
    COMPONENTS_AVAILABLE = False

@dataclass
class PerformanceMetrics:
    """Performance metrics for a single test run."""
    query: str
    query_type: str
    search_mode: str
    response_time_ms: float
    results_count: int
    vector_search_time_ms: float
    fusion_time_ms: float
    cache_hit: bool
    memory_usage_mb: float
    cpu_usage_percent: float
    timestamp: str

@dataclass
class BenchmarkSummary:
    """Summary of benchmark results."""
    total_queries: int
    avg_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    cache_hit_rate: float
    queries_under_50ms: int
    queries_over_100ms: int
    memory_peak_mb: float
    cpu_peak_percent: float
    storage_overhead_factor: float

class EducationalQueryGenerator:
    """Generate realistic educational domain queries for testing."""
    
    # Educational query patterns categorized by type
    FACTUAL_QUERIES = [
        "What is photosynthesis?",
        "Define classical conditioning",
        "What are the stages of mitosis?",
        "What is the quadratic formula?",
        "Define cognitive dissonance",
        "What is Newton's first law?",
        "What are the parts of a neuron?",
        "What is the periodic table?",
        "Define operant conditioning",
        "What is cellular respiration?"
    ]
    
    CONCEPTUAL_QUERIES = [
        "How does natural selection work?",
        "Explain the water cycle process",
        "How do neural networks learn?",
        "Explain Freud's theory of personality",
        "How does memory formation work?",
        "Explain the greenhouse effect",
        "How do antibodies work?",
        "Explain supply and demand",
        "How does gene expression work?",
        "Explain cognitive development stages"
    ]
    
    PROCEDURAL_QUERIES = [
        "How do you solve quadratic equations?",
        "Steps to conduct a t-test",
        "How to calculate standard deviation?",
        "Steps in the scientific method",
        "How to balance chemical equations?",
        "How to find the derivative of x^2?",
        "Steps to perform CPR",
        "How to write a thesis statement?",
        "How to conduct a literature review?",
        "How to solve linear systems?"
    ]
    
    COMPARATIVE_QUERIES = [
        "Compare mitosis and meiosis",
        "Difference between classical and operant conditioning",
        "Compare DNA and RNA structure",
        "Prokaryotic vs eukaryotic cells",
        "Compare correlation and causation",
        "Difference between anxiety and depression",
        "Compare arteries and veins",
        "Intrinsic vs extrinsic motivation",
        "Compare covalent and ionic bonds",
        "Difference between weather and climate"
    ]
    
    DEFINITIONAL_QUERIES = [
        "Define statistical significance",
        "What does correlation mean?",
        "Define homeostasis in biology",
        "Meaning of cognitive load",
        "Define entropy in physics",
        "What is neuroplasticity?",
        "Define independent variable",
        "What is cultural relativism?",
        "Define ecosystem biodiversity",
        "What is social learning theory?"
    ]
    
    def __init__(self):
        self.all_queries = {
            QueryType.FACTUAL: self.FACTUAL_QUERIES,
            QueryType.CONCEPTUAL: self.CONCEPTUAL_QUERIES,
            QueryType.PROCEDURAL: self.PROCEDURAL_QUERIES,
            QueryType.COMPARATIVE: self.COMPARATIVE_QUERIES,
            QueryType.DEFINITIONAL: self.DEFINITIONAL_QUERIES
        }
    
    def generate_test_queries(self, count_per_type: int = 5) -> List[Tuple[str, QueryType]]:
        """Generate balanced set of educational queries."""
        queries = []
        for query_type, query_list in self.all_queries.items():
            selected_queries = query_list[:count_per_type]
            for query in selected_queries:
                queries.append((query, query_type))
        return queries
    
    def get_random_query(self, query_type: Optional[QueryType] = None) -> Tuple[str, QueryType]:
        """Get a random query, optionally of specific type."""
        import random
        if query_type:
            query_list = self.all_queries[query_type]
            query = random.choice(query_list)
            return (query, query_type)
        else:
            query_type = random.choice(list(self.all_queries.keys()))
            query = random.choice(self.all_queries[query_type])
            return (query, query_type)

class PerformanceBenchmark:
    """Comprehensive performance benchmarking for ColBERT integration."""
    
    def __init__(self,
                 qdrant_url: str = "http://localhost:6333",
                 qdrant_api_key: Optional[str] = None,
                 collection_name: str = "test_benchmark",
                 colbert_model: str = "colbert-ir/colbertv2.0"):
        """
        Initialize performance benchmark.
        
        Args:
            qdrant_url: Qdrant server URL
            qdrant_api_key: Qdrant API key
            collection_name: Collection name to benchmark
            colbert_model: ColBERT model for testing
        """
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.collection_name = collection_name
        self.colbert_model = colbert_model
        
        self.query_generator = EducationalQueryGenerator()
        self.metrics_history: List[PerformanceMetrics] = []
        
        # System monitoring
        self.process = psutil.Process()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024
        
        # Initialize components
        self.hybrid_store = None
        self.retriever = None
        
    def initialize_components(self) -> bool:
        """Initialize hybrid store and retriever components."""
        try:
            logger.info("Initializing benchmark components...")
            
            # Create hybrid store
            self.hybrid_store = create_hybrid_store(
                url=self.qdrant_url,
                collection_name=self.collection_name,
                api_key=self.qdrant_api_key,
                enable_colbert=True,
                colbert_model=self.colbert_model
            )
            
            # Create multi-vector retriever
            config = MultiVectorConfig(
                adaptive_weights=True,
                parallel_search=True,
                max_search_time_ms=50.0,  # Target for voice apps
                final_limit=10
            )
            
            colbert_collection = f"{self.collection_name}_colbert"
            self.retriever = create_multi_vector_retriever(
                hybrid_store=self.hybrid_store,
                colbert_collection=colbert_collection,
                config=config
            )
            
            logger.info("Components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            return False
    
    def get_system_metrics(self) -> Tuple[float, float]:
        """Get current memory and CPU usage."""
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        cpu_percent = self.process.cpu_percent()
        return memory_mb, cpu_percent
    
    def benchmark_single_query(self,
                              query: str,
                              query_type: QueryType,
                              search_mode: str = "all") -> PerformanceMetrics:
        """Benchmark a single query execution."""
        start_time = time.time()
        memory_start, cpu_start = self.get_system_metrics()
        
        # Execute query
        results = self.retriever.search(
            query=query,
            search_mode=search_mode,
            limit=10
        )
        
        end_time = time.time()
        memory_end, cpu_end = self.get_system_metrics()
        
        # Get retriever stats
        stats = self.retriever.get_performance_stats()
        
        # Create metrics
        metrics = PerformanceMetrics(
            query=query,
            query_type=query_type.value,
            search_mode=search_mode,
            response_time_ms=(end_time - start_time) * 1000,
            results_count=len(results),
            vector_search_time_ms=stats.get("avg_response_time_ms", 0),
            fusion_time_ms=stats.get("fusion_time_ms", 0),
            cache_hit=stats.get("cache_hits", 0) > 0,
            memory_usage_mb=memory_end - memory_start,
            cpu_usage_percent=max(cpu_end, cpu_start),
            timestamp=datetime.now().isoformat()
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def run_latency_benchmark(self, 
                             query_count: int = 50,
                             search_modes: List[str] = None) -> Dict[str, Any]:
        """Run comprehensive latency benchmark."""
        if not search_modes:
            search_modes = ["dense", "sparse", "colbert", "all"]
            
        logger.info(f"Running latency benchmark with {query_count} queries")
        logger.info(f"Search modes: {search_modes}")
        
        # Generate test queries
        queries_per_type = max(1, query_count // 5)  # Distribute across 5 query types
        test_queries = self.query_generator.generate_test_queries(queries_per_type)
        
        # Extend if needed
        while len(test_queries) < query_count:
            test_queries.extend(self.query_generator.generate_test_queries(1))
        
        test_queries = test_queries[:query_count]
        
        results_by_mode = {}
        
        for search_mode in search_modes:
            logger.info(f"Testing search mode: {search_mode}")
            mode_metrics = []
            
            for query, query_type in test_queries:
                metrics = self.benchmark_single_query(query, query_type, search_mode)
                mode_metrics.append(metrics)
                
                # Log slow queries
                if metrics.response_time_ms > 100:
                    logger.warning(f"Slow query ({metrics.response_time_ms:.1f}ms): {query}")
            
            results_by_mode[search_mode] = mode_metrics
            
        logger.info("Latency benchmark completed")
        return results_by_mode
    
    def run_concurrent_benchmark(self,
                                concurrent_users: int = 10,
                                queries_per_user: int = 5) -> Dict[str, Any]:
        """Simulate concurrent user queries for voice tutor scenario."""
        logger.info(f"Running concurrent benchmark: {concurrent_users} users, {queries_per_user} queries each")
        
        def user_session(user_id: int) -> List[PerformanceMetrics]:
            """Simulate a single user session."""
            session_metrics = []
            for _ in range(queries_per_user):
                query, query_type = self.query_generator.get_random_query()
                metrics = self.benchmark_single_query(query, query_type, "all")
                session_metrics.append(metrics)
                
                # Small delay between queries (realistic user behavior)
                time.sleep(0.1)
                
            return session_metrics
        
        # Run concurrent sessions
        concurrent_metrics = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(user_session, i) for i in range(concurrent_users)]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    session_metrics = future.result()
                    concurrent_metrics.extend(session_metrics)
                except Exception as e:
                    logger.error(f"Concurrent session failed: {e}")
        
        return {"concurrent_metrics": concurrent_metrics}
    
    def analyze_storage_efficiency(self) -> Dict[str, Any]:
        """Analyze storage efficiency and overhead of ColBERT tokens."""
        try:
            client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key, timeout=60)
            
            # Get primary collection info
            primary_collection = client.get_collection(self.collection_name)
            primary_size = primary_collection.points_count
            
            # Get ColBERT collection info
            colbert_collection = f"{self.collection_name}_colbert"
            try:
                colbert_collection_info = client.get_collection(colbert_collection)
                colbert_size = colbert_collection_info.points_count
            except:
                logger.warning("ColBERT collection not found")
                colbert_size = 0
            
            # Estimate storage overhead
            # ColBERT typically uses ~27x more storage for token matrices
            estimated_overhead = 27.0 if colbert_size > 0 else 1.0
            
            return {
                "primary_collection_points": primary_size,
                "colbert_collection_points": colbert_size,
                "storage_overhead_factor": estimated_overhead,
                "collections_synchronized": primary_size == colbert_size
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze storage: {e}")
            return {}
    
    def generate_benchmark_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        all_metrics = []
        
        # Collect all metrics
        for mode_results in results.values():
            if isinstance(mode_results, list):
                all_metrics.extend(mode_results)
            elif "concurrent_metrics" in mode_results:
                all_metrics.extend(mode_results["concurrent_metrics"])
        
        if not all_metrics:
            return {"error": "No metrics collected"}
        
        # Calculate summary statistics
        response_times = [m.response_time_ms for m in all_metrics]
        cache_hits = sum(1 for m in all_metrics if m.cache_hit)
        under_50ms = sum(1 for rt in response_times if rt < 50)
        over_100ms = sum(1 for rt in response_times if rt > 100)
        
        memory_usage = [m.memory_usage_mb for m in all_metrics]
        cpu_usage = [m.cpu_usage_percent for m in all_metrics]
        
        # Storage analysis
        storage_info = self.analyze_storage_efficiency()
        
        summary = BenchmarkSummary(
            total_queries=len(all_metrics),
            avg_response_time_ms=statistics.mean(response_times),
            p95_response_time_ms=statistics.quantiles(response_times, n=20)[18],  # 95th percentile
            p99_response_time_ms=statistics.quantiles(response_times, n=100)[98],  # 99th percentile
            cache_hit_rate=(cache_hits / len(all_metrics)) * 100,
            queries_under_50ms=under_50ms,
            queries_over_100ms=over_100ms,
            memory_peak_mb=max(memory_usage) if memory_usage else 0,
            cpu_peak_percent=max(cpu_usage) if cpu_usage else 0,
            storage_overhead_factor=storage_info.get("storage_overhead_factor", 1.0)
        )
        
        # Performance analysis
        voice_ready = (summary.p95_response_time_ms < 50 and 
                      (under_50ms / len(all_metrics)) > 0.95)
        
        report = {
            "summary": asdict(summary),
            "voice_tutor_ready": voice_ready,
            "storage_analysis": storage_info,
            "recommendations": self._generate_recommendations(summary),
            "detailed_metrics": [asdict(m) for m in all_metrics[-20:]]  # Last 20 for brevity
        }
        
        return report
    
    def _generate_recommendations(self, summary: BenchmarkSummary) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        if summary.avg_response_time_ms > 50:
            recommendations.append(
                f"Average response time ({summary.avg_response_time_ms:.1f}ms) exceeds 50ms target. "
                "Consider enabling query caching or optimizing vector search parameters."
            )
        
        if summary.queries_over_100ms > 0:
            recommendations.append(
                f"{summary.queries_over_100ms} queries exceeded 100ms. "
                "These may need query optimization or timeout handling."
            )
        
        if summary.cache_hit_rate < 20:
            recommendations.append(
                f"Low cache hit rate ({summary.cache_hit_rate:.1f}%). "
                "Consider increasing cache size or implementing query normalization."
            )
        
        if summary.storage_overhead_factor > 30:
            recommendations.append(
                f"High storage overhead ({summary.storage_overhead_factor:.1f}x). "
                "Monitor disk usage and consider ColBERT parameter tuning."
            )
        
        if summary.memory_peak_mb > 2000:
            recommendations.append(
                f"High memory usage ({summary.memory_peak_mb:.1f}MB). "
                "Consider batch size optimization or memory-efficient embedding models."
            )
        
        if not recommendations:
            recommendations.append("Performance looks good! System meets voice tutor requirements.")
        
        return recommendations
    
    def save_benchmark_results(self, results: Dict[str, Any], output_file: str):
        """Save benchmark results to JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Performance benchmarking for ColBERT integration")
    parser.add_argument("--qdrant_url", default="http://localhost:6333", help="Qdrant server URL")
    parser.add_argument("--qdrant_api_key", help="Qdrant API key")
    parser.add_argument("--collection", required=True, help="Collection name to benchmark")
    parser.add_argument("--colbert_model", default="colbert-ir/colbertv2.0", help="ColBERT model")
    
    # Benchmark parameters
    parser.add_argument("--query_count", type=int, default=50, help="Number of queries for latency test")
    parser.add_argument("--concurrent_users", type=int, default=10, help="Concurrent users for load test")
    parser.add_argument("--queries_per_user", type=int, default=5, help="Queries per concurrent user")
    parser.add_argument("--output", default="./logs/benchmark_results.json", help="Output file for results")
    
    # Test selection
    parser.add_argument("--skip_latency", action="store_true", help="Skip latency benchmark")
    parser.add_argument("--skip_concurrent", action="store_true", help="Skip concurrent benchmark")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark (fewer queries)")
    
    args = parser.parse_args()
    
    if not COMPONENTS_AVAILABLE:
        logger.error("Required components not available. Cannot run benchmark.")
        sys.exit(1)
    
    # Adjust for quick mode
    if args.quick:
        args.query_count = min(10, args.query_count)
        args.concurrent_users = min(3, args.concurrent_users)
        args.queries_per_user = min(2, args.queries_per_user)
        logger.info("Quick mode enabled - reduced test parameters")
    
    logger.info("Starting ColBERT Performance Benchmark")
    logger.info(f"Collection: {args.collection}")
    logger.info(f"ColBERT model: {args.colbert_model}")
    
    # Initialize benchmark
    benchmark = PerformanceBenchmark(
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        collection_name=args.collection,
        colbert_model=args.colbert_model
    )
    
    if not benchmark.initialize_components():
        logger.error("Failed to initialize benchmark components")
        sys.exit(1)
    
    results = {}
    
    # Run latency benchmark
    if not args.skip_latency:
        logger.info("🚀 Starting latency benchmark...")
        latency_results = benchmark.run_latency_benchmark(
            query_count=args.query_count,
            search_modes=["dense", "sparse", "colbert", "all"]
        )
        results.update(latency_results)
    
    # Run concurrent benchmark
    if not args.skip_concurrent:
        logger.info("👥 Starting concurrent user benchmark...")
        concurrent_results = benchmark.run_concurrent_benchmark(
            concurrent_users=args.concurrent_users,
            queries_per_user=args.queries_per_user
        )
        results.update(concurrent_results)
    
    # Generate comprehensive report
    logger.info("📊 Generating benchmark report...")
    report = benchmark.generate_benchmark_report(results)
    
    # Save results
    benchmark.save_benchmark_results(report, args.output)
    
    # Display summary
    logger.info("\n" + "="*60)
    logger.info("🎯 BENCHMARK SUMMARY")
    logger.info("="*60)
    
    if "summary" in report:
        summary = report["summary"]
        logger.info(f"Total queries: {summary['total_queries']}")
        logger.info(f"Average response time: {summary['avg_response_time_ms']:.1f}ms")
        logger.info(f"95th percentile: {summary['p95_response_time_ms']:.1f}ms")
        logger.info(f"99th percentile: {summary['p99_response_time_ms']:.1f}ms")
        logger.info(f"Queries under 50ms: {summary['queries_under_50ms']}/{summary['total_queries']} "
                   f"({summary['queries_under_50ms']/summary['total_queries']*100:.1f}%)")
        logger.info(f"Cache hit rate: {summary['cache_hit_rate']:.1f}%")
        
        if report.get("voice_tutor_ready"):
            logger.info("✅ VOICE TUTOR READY: Performance meets <50ms requirements")
        else:
            logger.info("⚠️  OPTIMIZATION NEEDED: Performance below voice tutor standards")
    
    # Show recommendations
    if "recommendations" in report:
        logger.info("\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(report["recommendations"], 1):
            logger.info(f"{i}. {rec}")
    
    logger.info(f"\nFull results saved to: {args.output}")
    logger.info("Benchmark completed! 🎉")

if __name__ == "__main__":
    main()
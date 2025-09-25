#!/usr/bin/env python3
"""
Real-time Performance Monitor
============================

Lightweight monitoring tool for tracking ColBERT retrieval performance
in production educational voice tutor applications.

Features:
- Real-time query performance tracking
- Response time alerts for slow queries  
- System resource monitoring
- Performance trend analysis
- Educational query pattern analysis
"""

import time
import logging
import threading
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import statistics
import psutil

logger = logging.getLogger(__name__)

@dataclass
class QueryMetric:
    """Single query performance metric."""
    timestamp: str
    query: str
    query_type: str
    response_time_ms: float
    results_count: int
    cache_hit: bool
    memory_mb: float
    cpu_percent: float

class PerformanceMonitor:
    """
    Real-time performance monitoring for educational voice tutor.
    
    Tracks query performance and system metrics with configurable
    alerting for response time violations.
    """
    
    def __init__(self,
                 alert_threshold_ms: float = 50.0,
                 history_window_minutes: int = 30,
                 trend_window_minutes: int = 5):
        """
        Initialize performance monitor.
        
        Args:
            alert_threshold_ms: Alert if queries exceed this threshold
            history_window_minutes: How long to keep detailed metrics
            trend_window_minutes: Window for trend analysis
        """
        self.alert_threshold_ms = alert_threshold_ms
        self.history_window = timedelta(minutes=history_window_minutes)
        self.trend_window = timedelta(minutes=trend_window_minutes)
        
        # Metrics storage
        self.query_history = deque()
        self.alert_callbacks: List[Callable] = []
        
        # Performance tracking
        self.total_queries = 0
        self.slow_queries = 0
        self.cache_hits = 0
        
        # Query pattern analysis
        self.query_type_stats = defaultdict(list)
        self.hourly_query_counts = defaultdict(int)
        
        # System monitoring
        self.process = psutil.Process()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024
        
        # Threading
        self._monitoring_active = False
        self._monitor_thread = None
        
        logger.info(f"Performance monitor initialized (alert threshold: {alert_threshold_ms}ms)")
    
    def add_alert_callback(self, callback: Callable[[Dict], None]):
        """Add callback function for performance alerts."""
        self.alert_callbacks.append(callback)
    
    def record_query(self,
                    query: str,
                    query_type: str,
                    response_time_ms: float,
                    results_count: int,
                    cache_hit: bool = False):
        """
        Record a query execution for monitoring.
        
        Args:
            query: The query text
            query_type: Type classification (factual, conceptual, etc.)
            response_time_ms: Query response time in milliseconds
            results_count: Number of results returned
            cache_hit: Whether this was a cache hit
        """
        now = datetime.now()
        
        # Get system metrics
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        cpu_percent = self.process.cpu_percent()
        
        # Create metric record
        metric = QueryMetric(
            timestamp=now.isoformat(),
            query=query,
            query_type=query_type,
            response_time_ms=response_time_ms,
            results_count=results_count,
            cache_hit=cache_hit,
            memory_mb=memory_mb,
            cpu_percent=cpu_percent
        )
        
        # Store metric
        self.query_history.append(metric)
        
        # Update counters
        self.total_queries += 1
        if response_time_ms > self.alert_threshold_ms:
            self.slow_queries += 1
        if cache_hit:
            self.cache_hits += 1
            
        # Track query patterns
        self.query_type_stats[query_type].append(response_time_ms)
        hour_key = now.strftime("%Y-%m-%d %H:00")
        self.hourly_query_counts[hour_key] += 1
        
        # Clean old metrics
        self._cleanup_old_metrics()
        
        # Check for alerts
        self._check_alerts(metric)
        
        logger.debug(f"Recorded query: {response_time_ms:.1f}ms, type: {query_type}")
    
    def _cleanup_old_metrics(self):
        """Remove metrics older than history window."""
        cutoff = datetime.now() - self.history_window
        cutoff_iso = cutoff.isoformat()
        
        while (self.query_history and 
               self.query_history[0].timestamp < cutoff_iso):
            self.query_history.popleft()
    
    def _check_alerts(self, metric: QueryMetric):
        """Check if metric triggers any alerts."""
        alerts = []
        
        # Slow query alert
        if metric.response_time_ms > self.alert_threshold_ms:
            alerts.append({
                "type": "slow_query",
                "severity": "warning" if metric.response_time_ms < 100 else "critical",
                "message": f"Slow query: {metric.response_time_ms:.1f}ms",
                "query": metric.query,
                "response_time": metric.response_time_ms,
                "threshold": self.alert_threshold_ms
            })
        
        # High memory usage alert
        memory_threshold = self.start_memory * 2  # Alert if 2x initial memory
        if metric.memory_mb > memory_threshold:
            alerts.append({
                "type": "high_memory",
                "severity": "warning",
                "message": f"High memory usage: {metric.memory_mb:.1f}MB",
                "memory_mb": metric.memory_mb,
                "threshold": memory_threshold
            })
        
        # Send alerts
        for alert in alerts:
            alert["timestamp"] = metric.timestamp
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
    
    def get_current_stats(self) -> Dict:
        """Get current performance statistics."""
        if not self.query_history:
            return {"error": "No query data available"}
        
        recent_queries = list(self.query_history)
        response_times = [q.response_time_ms for q in recent_queries]
        
        # Calculate percentiles if we have enough data
        if len(response_times) >= 10:
            p95 = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
            p99 = statistics.quantiles(response_times, n=100)[98] if len(response_times) >= 100 else max(response_times)
        else:
            p95 = max(response_times) if response_times else 0
            p99 = p95
        
        # Voice readiness assessment
        under_50ms = sum(1 for rt in response_times if rt < 50)
        voice_ready_percent = (under_50ms / len(response_times)) * 100 if response_times else 0
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_queries": self.total_queries,
            "queries_in_window": len(recent_queries),
            "avg_response_time_ms": statistics.mean(response_times) if response_times else 0,
            "p95_response_time_ms": p95,
            "p99_response_time_ms": p99,
            "slow_query_rate": (self.slow_queries / self.total_queries) * 100 if self.total_queries else 0,
            "cache_hit_rate": (self.cache_hits / self.total_queries) * 100 if self.total_queries else 0,
            "voice_ready_percent": voice_ready_percent,
            "memory_mb": recent_queries[-1].memory_mb if recent_queries else 0,
            "cpu_percent": recent_queries[-1].cpu_percent if recent_queries else 0
        }
    
    def get_trend_analysis(self) -> Dict:
        """Analyze performance trends over trend window."""
        cutoff = datetime.now() - self.trend_window
        cutoff_iso = cutoff.isoformat()
        
        trend_queries = [q for q in self.query_history if q.timestamp >= cutoff_iso]
        
        if len(trend_queries) < 2:
            return {"message": "Insufficient data for trend analysis"}
        
        # Split into two halves for comparison
        mid_point = len(trend_queries) // 2
        early_half = trend_queries[:mid_point]
        recent_half = trend_queries[mid_point:]
        
        early_avg = statistics.mean([q.response_time_ms for q in early_half])
        recent_avg = statistics.mean([q.response_time_ms for q in recent_half])
        
        trend_direction = "improving" if recent_avg < early_avg else "degrading" if recent_avg > early_avg else "stable"
        percent_change = ((recent_avg - early_avg) / early_avg) * 100 if early_avg > 0 else 0
        
        return {
            "trend_direction": trend_direction,
            "percent_change": percent_change,
            "early_avg_ms": early_avg,
            "recent_avg_ms": recent_avg,
            "trend_window_minutes": self.trend_window.total_seconds() / 60,
            "queries_analyzed": len(trend_queries)
        }
    
    def get_query_pattern_analysis(self) -> Dict:
        """Analyze query type performance patterns."""
        patterns = {}
        
        for query_type, response_times in self.query_type_stats.items():
            if response_times:
                patterns[query_type] = {
                    "count": len(response_times),
                    "avg_response_ms": statistics.mean(response_times),
                    "min_response_ms": min(response_times),
                    "max_response_ms": max(response_times),
                    "slow_queries": sum(1 for rt in response_times if rt > self.alert_threshold_ms)
                }
        
        return patterns
    
    def start_monitoring(self, interval_seconds: int = 30):
        """Start background monitoring thread."""
        if self._monitoring_active:
            logger.warning("Monitor already active")
            return
            
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info(f"Background monitoring started (interval: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop background monitoring thread."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Background monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Background monitoring loop."""
        while self._monitoring_active:
            try:
                stats = self.get_current_stats()
                trend = self.get_trend_analysis()
                
                # Log periodic summary
                if "error" not in stats:
                    logger.info(
                        f"Monitor: {stats['queries_in_window']} queries, "
                        f"avg: {stats['avg_response_time_ms']:.1f}ms, "
                        f"p95: {stats['p95_response_time_ms']:.1f}ms, "
                        f"voice-ready: {stats['voice_ready_percent']:.1f}%, "
                        f"trend: {trend.get('trend_direction', 'unknown')}"
                    )
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(interval_seconds)
    
    def export_metrics(self, output_file: str):
        """Export current metrics to JSON file."""
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "current_stats": self.get_current_stats(),
            "trend_analysis": self.get_trend_analysis(),
            "query_patterns": self.get_query_pattern_analysis(),
            "recent_metrics": [asdict(q) for q in list(self.query_history)[-100:]]  # Last 100
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Metrics exported to: {output_file}")

# Example alert callbacks
def console_alert_callback(alert: Dict):
    """Simple console alert callback."""
    severity_emoji = {"warning": "⚠️", "critical": "🚨"}
    emoji = severity_emoji.get(alert["severity"], "ℹ️")
    print(f"{emoji} ALERT: {alert['message']} [{alert['type']}]")

def log_alert_callback(alert: Dict):
    """Log file alert callback."""
    level = logging.WARNING if alert["severity"] == "warning" else logging.ERROR
    logger.log(level, f"ALERT: {alert['message']} [{alert['type']}] - {alert}")

# Example usage
if __name__ == "__main__":
    import random
    
    # Create monitor with console alerts
    monitor = PerformanceMonitor(alert_threshold_ms=50)
    monitor.add_alert_callback(console_alert_callback)
    monitor.add_alert_callback(log_alert_callback)
    
    # Start background monitoring
    monitor.start_monitoring(interval_seconds=10)
    
    # Simulate some queries
    query_types = ["factual", "conceptual", "procedural", "comparative", "definitional"]
    sample_queries = [
        "What is photosynthesis?",
        "How does memory work?", 
        "Steps to solve equations",
        "Compare DNA and RNA",
        "Define cognitive load"
    ]
    
    print("Simulating queries... (Press Ctrl+C to stop)")
    try:
        for i in range(100):
            query = random.choice(sample_queries)
            query_type = random.choice(query_types)
            
            # Simulate various response times
            if i % 10 == 0:  # 10% slow queries
                response_time = random.uniform(60, 120)
            else:
                response_time = random.uniform(15, 45)
            
            cache_hit = random.random() < 0.2  # 20% cache hit rate
            results_count = random.randint(3, 10)
            
            monitor.record_query(query, query_type, response_time, results_count, cache_hit)
            
            time.sleep(0.5)  # 500ms between queries
            
    except KeyboardInterrupt:
        print("\nStopping simulation...")
    
    # Show final stats
    print("\nFinal Statistics:")
    print(json.dumps(monitor.get_current_stats(), indent=2))
    
    # Export metrics
    monitor.export_metrics("./logs/monitor_export.json")
    
    # Stop monitoring
    monitor.stop_monitoring()
    print("Monitoring stopped.")
"""
Metrics collection for the NewsCrawler system.
This module provides functionality for collecting and reporting metrics.
"""

import time
import logging
import threading
import functools
import os
import json
from typing import Dict, Any, Optional, List, Callable, Union, TypeVar
from datetime import datetime, timedelta
import statistics

logger = logging.getLogger(__name__)

# Type variable for function return type
T = TypeVar('T')

# Global metrics registry
_metrics_registry: Dict[str, 'Metric'] = {}
_metrics_lock = threading.RLock()


class Metric:
    """
    Base class for metrics.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize the metric.
        
        Args:
            name: Metric name
            description: Metric description
        """
        self.name = name
        self.description = description
        self.created_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the metric to a dictionary.
        
        Returns:
            Dictionary representation of the metric
        """
        return {
            "name": self.name,
            "description": self.description,
            "type": self.__class__.__name__,
            "created_at": self.created_at
        }


class Counter(Metric):
    """
    Counter metric that tracks a cumulative value.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize the counter.
        
        Args:
            name: Counter name
            description: Counter description
        """
        super().__init__(name, description)
        self.value = 0
        self.lock = threading.RLock()
    
    def increment(self, value: int = 1) -> int:
        """
        Increment the counter.
        
        Args:
            value: Value to increment by
            
        Returns:
            New counter value
        """
        with self.lock:
            self.value += value
            return self.value
    
    def get(self) -> int:
        """
        Get the current counter value.
        
        Returns:
            Current counter value
        """
        with self.lock:
            return self.value
    
    def reset(self) -> None:
        """Reset the counter to zero."""
        with self.lock:
            self.value = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the counter to a dictionary.
        
        Returns:
            Dictionary representation of the counter
        """
        result = super().to_dict()
        result["value"] = self.get()
        return result


class Gauge(Metric):
    """
    Gauge metric that tracks a value that can go up and down.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize the gauge.
        
        Args:
            name: Gauge name
            description: Gauge description
        """
        super().__init__(name, description)
        self.value = 0
        self.lock = threading.RLock()
    
    def set(self, value: float) -> float:
        """
        Set the gauge value.
        
        Args:
            value: New value
            
        Returns:
            New gauge value
        """
        with self.lock:
            self.value = value
            return self.value
    
    def increment(self, value: float = 1) -> float:
        """
        Increment the gauge value.
        
        Args:
            value: Value to increment by
            
        Returns:
            New gauge value
        """
        with self.lock:
            self.value += value
            return self.value
    
    def decrement(self, value: float = 1) -> float:
        """
        Decrement the gauge value.
        
        Args:
            value: Value to decrement by
            
        Returns:
            New gauge value
        """
        with self.lock:
            self.value -= value
            return self.value
    
    def get(self) -> float:
        """
        Get the current gauge value.
        
        Returns:
            Current gauge value
        """
        with self.lock:
            return self.value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the gauge to a dictionary.
        
        Returns:
            Dictionary representation of the gauge
        """
        result = super().to_dict()
        result["value"] = self.get()
        return result


class Histogram(Metric):
    """
    Histogram metric that tracks the distribution of values.
    """
    
    def __init__(self, name: str, description: str = "", buckets: Optional[List[float]] = None):
        """
        Initialize the histogram.
        
        Args:
            name: Histogram name
            description: Histogram description
            buckets: Bucket boundaries (None for default)
        """
        super().__init__(name, description)
        self.values: List[float] = []
        self.buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
        self.lock = threading.RLock()
    
    def observe(self, value: float) -> None:
        """
        Observe a value.
        
        Args:
            value: Value to observe
        """
        with self.lock:
            self.values.append(value)
    
    def get_count(self) -> int:
        """
        Get the number of observations.
        
        Returns:
            Number of observations
        """
        with self.lock:
            return len(self.values)
    
    def get_sum(self) -> float:
        """
        Get the sum of all observations.
        
        Returns:
            Sum of all observations
        """
        with self.lock:
            return sum(self.values)
    
    def get_mean(self) -> float:
        """
        Get the mean of all observations.
        
        Returns:
            Mean of all observations
        """
        with self.lock:
            if not self.values:
                return 0
            return sum(self.values) / len(self.values)
    
    def get_percentile(self, percentile: float) -> float:
        """
        Get a percentile value.
        
        Args:
            percentile: Percentile (0-100)
            
        Returns:
            Percentile value
        """
        with self.lock:
            if not self.values:
                return 0
            
            sorted_values = sorted(self.values)
            k = (len(sorted_values) - 1) * (percentile / 100)
            f = int(k)
            c = k - f
            
            if f + 1 < len(sorted_values):
                return sorted_values[f] * (1 - c) + sorted_values[f + 1] * c
            else:
                return sorted_values[f]
    
    def get_bucket_counts(self) -> Dict[float, int]:
        """
        Get the number of observations in each bucket.
        
        Returns:
            Dictionary mapping bucket upper bounds to counts
        """
        with self.lock:
            counts = {b: 0 for b in self.buckets}
            counts[float('inf')] = 0
            
            for value in self.values:
                for bucket in self.buckets:
                    if value <= bucket:
                        counts[bucket] += 1
                        break
                else:
                    counts[float('inf')] += 1
            
            return counts
    
    def reset(self) -> None:
        """Reset the histogram."""
        with self.lock:
            self.values = []
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the histogram to a dictionary.
        
        Returns:
            Dictionary representation of the histogram
        """
        with self.lock:
            result = super().to_dict()
            
            if self.values:
                result.update({
                    "count": len(self.values),
                    "sum": sum(self.values),
                    "mean": sum(self.values) / len(self.values) if self.values else 0,
                    "min": min(self.values) if self.values else 0,
                    "max": max(self.values) if self.values else 0,
                    "stddev": statistics.stdev(self.values) if len(self.values) > 1 else 0,
                    "percentiles": {
                        "50": self.get_percentile(50),
                        "90": self.get_percentile(90),
                        "95": self.get_percentile(95),
                        "99": self.get_percentile(99)
                    },
                    "buckets": self.get_bucket_counts()
                })
            else:
                result.update({
                    "count": 0,
                    "sum": 0,
                    "mean": 0,
                    "min": 0,
                    "max": 0,
                    "stddev": 0,
                    "percentiles": {
                        "50": 0,
                        "90": 0,
                        "95": 0,
                        "99": 0
                    },
                    "buckets": {b: 0 for b in self.buckets + [float('inf')]}
                })
            
            return result


class Timer:
    """
    Timer for measuring execution time.
    """
    
    def __init__(self, histogram: Histogram):
        """
        Initialize the timer.
        
        Args:
            histogram: Histogram to record measurements in
        """
        self.histogram = histogram
        self.start_time = None
    
    def __enter__(self) -> 'Timer':
        """
        Start the timer.
        
        Returns:
            Self
        """
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Stop the timer and record the duration.
        
        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
        """
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.histogram.observe(duration)


def get_counter(name: str, description: str = "") -> Counter:
    """
    Get or create a counter.
    
    Args:
        name: Counter name
        description: Counter description
        
    Returns:
        Counter instance
    """
    with _metrics_lock:
        if name not in _metrics_registry:
            _metrics_registry[name] = Counter(name, description)
        
        metric = _metrics_registry[name]
        if not isinstance(metric, Counter):
            raise ValueError(f"Metric {name} exists but is not a Counter")
        
        return metric


def get_gauge(name: str, description: str = "") -> Gauge:
    """
    Get or create a gauge.
    
    Args:
        name: Gauge name
        description: Gauge description
        
    Returns:
        Gauge instance
    """
    with _metrics_lock:
        if name not in _metrics_registry:
            _metrics_registry[name] = Gauge(name, description)
        
        metric = _metrics_registry[name]
        if not isinstance(metric, Gauge):
            raise ValueError(f"Metric {name} exists but is not a Gauge")
        
        return metric


def get_histogram(name: str, description: str = "", buckets: Optional[List[float]] = None) -> Histogram:
    """
    Get or create a histogram.
    
    Args:
        name: Histogram name
        description: Histogram description
        buckets: Bucket boundaries (None for default)
        
    Returns:
        Histogram instance
    """
    with _metrics_lock:
        if name not in _metrics_registry:
            _metrics_registry[name] = Histogram(name, description, buckets)
        
        metric = _metrics_registry[name]
        if not isinstance(metric, Histogram):
            raise ValueError(f"Metric {name} exists but is not a Histogram")
        
        return metric


def get_timer(name: str, description: str = "") -> Timer:
    """
    Get a timer that records to a histogram.
    
    Args:
        name: Timer name
        description: Timer description
        
    Returns:
        Timer instance
    """
    histogram = get_histogram(name, description)
    return Timer(histogram)


def timed(name: str, description: str = "") -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for timing function execution.
    
    Args:
        name: Timer name
        description: Timer description
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            with get_timer(name, description):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def increment_counter(name: str, value: int = 1, description: str = "") -> int:
    """
    Increment a counter.
    
    Args:
        name: Counter name
        value: Value to increment by
        description: Counter description
        
    Returns:
        New counter value
    """
    counter = get_counter(name, description)
    return counter.increment(value)


def set_gauge(name: str, value: float, description: str = "") -> float:
    """
    Set a gauge value.
    
    Args:
        name: Gauge name
        value: New value
        description: Gauge description
        
    Returns:
        New gauge value
    """
    gauge = get_gauge(name, description)
    return gauge.set(value)


def observe_value(name: str, value: float, description: str = "") -> None:
    """
    Observe a value in a histogram.
    
    Args:
        name: Histogram name
        value: Value to observe
        description: Histogram description
    """
    histogram = get_histogram(name, description)
    histogram.observe(value)


def get_all_metrics() -> Dict[str, Dict[str, Any]]:
    """
    Get all metrics.
    
    Returns:
        Dictionary mapping metric names to metric data
    """
    with _metrics_lock:
        return {name: metric.to_dict() for name, metric in _metrics_registry.items()}


def export_metrics_json(file_path: str) -> None:
    """
    Export metrics to a JSON file.
    
    Args:
        file_path: Path to JSON file
    """
    metrics = get_all_metrics()
    
    # Create directory if it doesn't exist
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def reset_all_metrics() -> None:
    """Reset all metrics."""
    with _metrics_lock:
        for metric in _metrics_registry.values():
            if hasattr(metric, 'reset'):
                metric.reset() 
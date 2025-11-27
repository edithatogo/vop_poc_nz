"""
Performance Profiling Module

Utilities for monitoring performance bottlenecks in the analysis pipeline.
Includes timing decorators, memory profiling, and profiling reports.
"""

import functools
import os
import time
from contextlib import contextmanager
from typing import Callable

import psutil


class PerformanceProfiler:
    """Track performance metrics across analysis pipeline."""

    def __init__(self):
        self.timings = {}
        self.memory_usage = {}
        self.call_counts = {}

    def record_timing(self, func_name: str, duration: float):
        """Record execution time for a function."""
        if func_name not in self.timings:
            self.timings[func_name] = []
            self.call_counts[func_name] = 0
        self.timings[func_name].append(duration)
        self.call_counts[func_name] += 1

    def record_memory(self, func_name: str, memory_mb: float):
        """Record memory usage for a function."""
        if func_name not in self.memory_usage:
            self.memory_usage[func_name] = []
        self.memory_usage[func_name].append(memory_mb)

    def get_report(self) -> str:
        """Generate profiling report."""
        report = ["=" * 80]
        report.append("PERFORMANCE PROFILING REPORT")
        report.append("=" * 80)
        report.append("")

        # Sort by total time
        sorted_funcs = sorted(
            self.timings.items(), key=lambda x: sum(x[1]), reverse=True
        )

        report.append(
            f"{'Function':<40} {'Calls':>8} {'Total (s)':>12} {'Avg (s)':>12}"
        )
        report.append("-" * 80)

        for func_name, durations in sorted_funcs:
            total_time = sum(durations)
            avg_time = total_time / len(durations)
            calls = self.call_counts[func_name]
            report.append(
                f"{func_name:<40} {calls:>8} {total_time:>12.3f} {avg_time:>12.3f}"
            )

        report.append("")
        report.append("Memory Usage:")
        report.append(f"{'Function':<40} {'Peak (MB)':>15} {'Avg (MB)':>15}")
        report.append("-" * 80)

        for func_name, memory in self.memory_usage.items():
            peak = max(memory)
            avg = sum(memory) / len(memory)
            report.append(f"{func_name:<40} {peak:>15.2f} {avg:>15.2f}")

        report.append("=" * 80)
        return "\n".join(report)

    def save_report(self, filepath: str):
        """Save profiling report to file."""
        os.makedirs(
            os.path.dirname(filepath) if os.path.dirname(filepath) else ".",
            exist_ok=True,
        )
        with open(filepath, "w") as f:
            f.write(self.get_report())
        print(f"Profiling report saved to: {filepath}")


# Global profiler instance
_global_profiler = PerformanceProfiler()


def profile_function(func: Callable) -> Callable:
    """
    Decorator to profile function execution time and memory.

    Usage:
        @profile_function
        def my_function():
            ...
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = f"{func.__module__}.{func.__name__}"

        # Get process for memory tracking
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Time execution
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time

        # Memory after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_used = mem_after - mem_before

        # Record
        _global_profiler.record_timing(func_name, duration)
        _global_profiler.record_memory(func_name, mem_after)

        if duration > 1.0:  # Log slow functions
            print(f"⏱️  {func_name}: {duration:.2f}s (Δmem: {mem_used:+.1f} MB)")

        return result

    return wrapper


@contextmanager
def profile_section(section_name: str):
    """
    Context manager for profiling code sections.

    Usage:
        with profile_section("DSA Analysis"):
            perform_dsa(...)
    """
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024

    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        mem_after = process.memory_info().rss / 1024 / 1024

        _global_profiler.record_timing(section_name, duration)
        _global_profiler.record_memory(section_name, mem_after)

        print(f"✓ {section_name}: {duration:.2f}s (mem: {mem_after:.1f} MB)")


def get_profiler() -> PerformanceProfiler:
    """Get the global profiler instance."""
    return _global_profiler


def reset_profiler():
    """Reset profiling data."""
    global _global_profiler
    _global_profiler = PerformanceProfiler()


def print_profiling_report():
    """Print profiling report to console."""
    print(_global_profiler.get_report())


def save_profiling_report(filepath: str = "output/profiling_report.txt"):
    """Save profiling report to file."""
    _global_profiler.save_report(filepath)

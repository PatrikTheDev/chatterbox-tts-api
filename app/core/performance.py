"""
Performance monitoring and instrumentation for TTS operations
"""

import time
import torch
import psutil
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
from app.core.memory import get_memory_info


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    operation: str
    start_time: float
    end_time: float = 0.0
    cpu_usage_start: float = 0.0
    cpu_usage_end: float = 0.0
    memory_start: Dict[str, float] = field(default_factory=dict)
    memory_end: Dict[str, float] = field(default_factory=dict)
    gpu_utilization: Optional[float] = None
    gpu_memory_usage: Optional[float] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """Get operation duration in seconds"""
        return self.end_time - self.start_time
    
    @property
    def iterations_per_second(self) -> float:
        """Calculate iterations per second (it/s) for TTS operations"""
        if self.duration > 0 and 'text_length' in self.additional_data:
            # Approximate characters processed per second
            return self.additional_data['text_length'] / self.duration
        return 0.0
    
    @property
    def memory_delta_mb(self) -> float:
        """Calculate memory change in MB"""
        if self.memory_start and self.memory_end:
            start_mem = self.memory_start.get('cpu_memory_mb', 0)
            end_mem = self.memory_end.get('cpu_memory_mb', 0)
            return end_mem - start_mem
        return 0.0
    
    @property
    def gpu_memory_delta_mb(self) -> float:
        """Calculate GPU memory change in MB"""
        if self.memory_start and self.memory_end:
            start_gpu = self.memory_start.get('gpu_memory_allocated_mb', 0)
            end_gpu = self.memory_end.get('gpu_memory_allocated_mb', 0)
            return end_gpu - start_gpu
        return 0.0


class PerformanceMonitor:
    """Performance monitoring singleton"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.metrics: List[PerformanceMetrics] = []
            self.current_metrics: Dict[str, PerformanceMetrics] = {}
            self.enabled = True
            self._initialized = True
    
    def start_operation(self, operation: str, **additional_data) -> str:
        """Start monitoring an operation"""
        if not self.enabled:
            return operation
            
        metric = PerformanceMetrics(
            operation=operation,
            start_time=time.time(),
            cpu_usage_start=psutil.cpu_percent(),
            memory_start=get_memory_info(),
            additional_data=additional_data
        )
        
        # Add GPU utilization if available
        if torch.cuda.is_available():
            try:
                import nvidia_ml_py3 as nvml
                nvml.nvmlInit()
                handle = nvml.nvmlDeviceGetHandleByIndex(0)  # First GPU
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                metric.gpu_utilization = util.gpu
                
                # Memory info
                mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                metric.gpu_memory_usage = (mem_info.used / mem_info.total) * 100
            except ImportError:
                # nvidia-ml-py3 not available, use torch fallback
                metric.gpu_utilization = None
            except Exception as e:
                print(f"⚠️ GPU monitoring error: {e}")
        
        operation_id = f"{operation}_{int(metric.start_time * 1000)}"
        self.current_metrics[operation_id] = metric
        return operation_id
    
    def end_operation(self, operation_id: str):
        """End monitoring an operation"""
        if not self.enabled or operation_id not in self.current_metrics:
            return
            
        metric = self.current_metrics.pop(operation_id)
        metric.end_time = time.time()
        metric.cpu_usage_end = psutil.cpu_percent()
        metric.memory_end = get_memory_info()
        
        self.metrics.append(metric)
        
        # Print real-time performance info
        self._print_metric_summary(metric)
    
    def _print_metric_summary(self, metric: PerformanceMetrics):
        """Print a summary of the performance metric"""
        print(f"🔍 {metric.operation}: {metric.duration:.3f}s")
        
        if metric.iterations_per_second > 0:
            print(f"   ⚡ Speed: {metric.iterations_per_second:.1f} chars/s")
        
        if metric.memory_delta_mb != 0:
            print(f"   💾 Memory Δ: {metric.memory_delta_mb:+.1f}MB")
        
        if metric.gpu_memory_delta_mb != 0:
            print(f"   🎮 GPU Memory Δ: {metric.gpu_memory_delta_mb:+.1f}MB")
        
        if metric.gpu_utilization is not None:
            print(f"   🎮 GPU Util: {metric.gpu_utilization:.1f}%")
        
        # Add bottleneck detection
        if metric.duration > 2.0:  # Operations taking > 2 seconds
            print(f"   ⚠️ SLOW OPERATION detected!")
    
    @contextmanager
    def monitor_operation(self, operation: str, **additional_data):
        """Context manager for monitoring operations"""
        operation_id = self.start_operation(operation, **additional_data)
        try:
            yield operation_id
        finally:
            self.end_operation(operation_id)
    
    def get_operation_stats(self, operation_pattern: str = None) -> Dict[str, Any]:
        """Get statistics for operations matching pattern"""
        filtered_metrics = self.metrics
        if operation_pattern:
            filtered_metrics = [m for m in self.metrics if operation_pattern in m.operation]
        
        if not filtered_metrics:
            return {}
        
        durations = [m.duration for m in filtered_metrics]
        speeds = [m.iterations_per_second for m in filtered_metrics if m.iterations_per_second > 0]
        
        stats = {
            'count': len(filtered_metrics),
            'total_time': sum(durations),
            'avg_time': sum(durations) / len(durations),
            'min_time': min(durations),
            'max_time': max(durations),
        }
        
        if speeds:
            stats.update({
                'avg_speed': sum(speeds) / len(speeds),
                'min_speed': min(speeds),
                'max_speed': max(speeds),
            })
        
        return stats
    
    def reset(self):
        """Clear all metrics"""
        self.metrics.clear()
        self.current_metrics.clear()
    
    def disable(self):
        """Disable performance monitoring"""
        self.enabled = False
    
    def enable(self):
        """Enable performance monitoring"""
        self.enabled = True
    
    def export_metrics(self) -> List[Dict[str, Any]]:
        """Export metrics as JSON-serializable format"""
        return [
            {
                'operation': m.operation,
                'duration': m.duration,
                'start_time': m.start_time,
                'end_time': m.end_time,
                'iterations_per_second': m.iterations_per_second,
                'memory_delta_mb': m.memory_delta_mb,
                'gpu_memory_delta_mb': m.gpu_memory_delta_mb,
                'gpu_utilization': m.gpu_utilization,
                'additional_data': m.additional_data,
            }
            for m in self.metrics
        ]


# Global performance monitor instance
perf_monitor = PerformanceMonitor()


@contextmanager
def monitor_tts_operation(operation: str, text_length: int = 0, chunk_index: int = None, **kwargs):
    """Convenience context manager for TTS operations"""
    additional_data = {'text_length': text_length, **kwargs}
    if chunk_index is not None:
        additional_data['chunk_index'] = chunk_index
    
    with perf_monitor.monitor_operation(operation, **additional_data) as op_id:
        yield op_id


def print_performance_summary():
    """Print a comprehensive performance summary"""
    print("\n" + "="*60)
    print("🔍 PERFORMANCE ANALYSIS SUMMARY")
    print("="*60)
    
    # Overall stats
    overall_stats = perf_monitor.get_operation_stats()
    if overall_stats:
        print(f"📊 Total Operations: {overall_stats['count']}")
        print(f"⏱️ Total Time: {overall_stats['total_time']:.2f}s")
        print(f"📈 Avg Operation Time: {overall_stats['avg_time']:.3f}s")
        print(f"⚡ Min/Max Time: {overall_stats['min_time']:.3f}s / {overall_stats['max_time']:.3f}s")
        
        if 'avg_speed' in overall_stats:
            print(f"🚀 Avg Speed: {overall_stats['avg_speed']:.1f} chars/s")
    
    # TTS Generation stats
    tts_stats = perf_monitor.get_operation_stats("TTS_GENERATE")
    if tts_stats:
        print(f"\n🎤 TTS Generation Specific:")
        print(f"   Operations: {tts_stats['count']}")
        print(f"   Avg Time per chunk: {tts_stats['avg_time']:.3f}s")
        if 'avg_speed' in tts_stats:
            print(f"   Avg Speed: {tts_stats['avg_speed']:.1f} chars/s")
            
            # Performance assessment
            if tts_stats['avg_speed'] < 50:
                print(f"   ⚠️ LOW PERFORMANCE: < 50 chars/s")
            elif tts_stats['avg_speed'] < 100:
                print(f"   ⚡ MODERATE PERFORMANCE: 50-100 chars/s")
            else:
                print(f"   🚀 HIGH PERFORMANCE: > 100 chars/s")
    
    # Model loading stats
    model_stats = perf_monitor.get_operation_stats("MODEL_LOAD")
    if model_stats:
        print(f"\n🤖 Model Loading:")
        print(f"   Time: {model_stats['avg_time']:.2f}s")
    
    print("="*60)
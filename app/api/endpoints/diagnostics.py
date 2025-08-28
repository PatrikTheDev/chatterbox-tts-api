"""
Performance diagnostics endpoint
"""

import torch
import psutil
import platform
from typing import Dict, Any, Optional
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.core.performance import perf_monitor, print_performance_summary
from app.core.memory import get_memory_info
from app.core.tts_model import get_device, is_ready
from app.core.profiler import tts_profiler
from app.config import Config


router = APIRouter()


@router.get("/diagnostics/performance", response_model=Dict[str, Any])
async def get_performance_diagnostics():
    """Get comprehensive performance diagnostics"""
    
    # System information
    system_info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": psutil.cpu_count(),
        "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
        "total_ram_gb": psutil.virtual_memory().total / (1024**3),
    }
    
    # GPU information
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "cuda_available": True,
            "cuda_version": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(),
            "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
        }
        
        # Try to get detailed GPU info
        try:
            import nvidia_ml_py3 as nvml
            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            
            # GPU utilization
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_info["gpu_utilization_percent"] = util.gpu
            gpu_info["memory_utilization_percent"] = util.memory
            
            # Temperature
            temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
            gpu_info["temperature_celsius"] = temp
            
            # Clock speeds
            graphics_clock = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_GRAPHICS)
            memory_clock = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM)
            gpu_info["graphics_clock_mhz"] = graphics_clock
            gpu_info["memory_clock_mhz"] = memory_clock
            
        except ImportError:
            gpu_info["detailed_metrics"] = "nvidia-ml-py3 not available"
        except Exception as e:
            gpu_info["detailed_metrics_error"] = str(e)
    else:
        gpu_info["cuda_available"] = False
    
    # Current memory status
    current_memory = get_memory_info()
    
    # TTS model status
    model_status = {
        "model_ready": is_ready(),
        "device": get_device(),
        "configured_device": Config.DEVICE_OVERRIDE,
    }
    
    # Performance statistics
    perf_stats = {
        "total_operations": len(perf_monitor.metrics),
        "overall_stats": perf_monitor.get_operation_stats(),
        "tts_generation_stats": perf_monitor.get_operation_stats("TTS_GENERATE"),
        "model_initialization_stats": perf_monitor.get_operation_stats("MODEL_INIT"),
    }
    
    # Configuration analysis
    config_analysis = {
        "max_chunk_length": Config.MAX_CHUNK_LENGTH,
        "memory_cleanup_interval": Config.MEMORY_CLEANUP_INTERVAL,
        "cuda_cache_clear_interval": Config.CUDA_CACHE_CLEAR_INTERVAL,
        "memory_monitoring_enabled": Config.ENABLE_MEMORY_MONITORING,
        "tts_parameters": {
            "exaggeration": Config.EXAGGERATION,
            "cfg_weight": Config.CFG_WEIGHT,
            "temperature": Config.TEMPERATURE,
        }
    }
    
    return {
        "timestamp": psutil.boot_time(),
        "system_info": system_info,
        "gpu_info": gpu_info,
        "current_memory": current_memory,
        "model_status": model_status,
        "performance_stats": perf_stats,
        "config_analysis": config_analysis,
    }


@router.get("/diagnostics/bottleneck-analysis")
async def analyze_bottlenecks():
    """Analyze potential performance bottlenecks"""
    
    diagnostics = await get_performance_diagnostics()
    analysis = {
        "bottlenecks_detected": [],
        "recommendations": [],
        "severity": "low"
    }
    
    # Analyze TTS performance
    tts_stats = diagnostics["performance_stats"]["tts_generation_stats"]
    if tts_stats and "avg_speed" in tts_stats:
        avg_speed = tts_stats["avg_speed"]
        
        if avg_speed < 40:
            analysis["bottlenecks_detected"].append({
                "type": "LOW_TTS_PERFORMANCE",
                "description": f"TTS speed is {avg_speed:.1f} chars/s (expected >100 chars/s)",
                "severity": "critical"
            })
            analysis["severity"] = "critical"
            
            # GPU-specific analysis
            gpu_info = diagnostics["gpu_info"]
            if gpu_info.get("cuda_available"):
                if gpu_info.get("gpu_utilization_percent", 0) < 50:
                    analysis["recommendations"].append({
                        "type": "GPU_UNDERUTILIZATION",
                        "description": "GPU utilization is low. Check for CPU bottlenecks or memory transfer issues.",
                        "action": "Monitor CPU usage during TTS generation"
                    })
                
                if gpu_info.get("temperature_celsius", 0) > 80:
                    analysis["recommendations"].append({
                        "type": "GPU_THERMAL_THROTTLING",
                        "description": "GPU temperature is high, may cause thermal throttling",
                        "action": "Check GPU cooling and reduce workload if necessary"
                    })
        
        elif avg_speed < 80:
            analysis["bottlenecks_detected"].append({
                "type": "MODERATE_TTS_PERFORMANCE",
                "description": f"TTS speed is {avg_speed:.1f} chars/s (good >100 chars/s)",
                "severity": "warning"
            })
            analysis["severity"] = "warning"
    
    # Memory analysis
    memory_info = diagnostics["current_memory"]
    if memory_info.get("cpu_memory_percent", 0) > 90:
        analysis["bottlenecks_detected"].append({
            "type": "HIGH_MEMORY_USAGE",
            "description": f"CPU memory usage is {memory_info['cpu_memory_percent']:.1f}%",
            "severity": "warning"
        })
    
    # System analysis
    system_info = diagnostics["system_info"]
    cpu_count = system_info.get("cpu_count", 1)
    if cpu_count < 8:
        analysis["recommendations"].append({
            "type": "CPU_CORES",
            "description": f"System has {cpu_count} CPU cores. More cores may improve performance.",
            "action": "Consider upgrading to a CPU with more cores"
        })
    
    # Configuration recommendations
    config = diagnostics["config_analysis"]
    if config["max_chunk_length"] > 300:
        analysis["recommendations"].append({
            "type": "CHUNK_SIZE_OPTIMIZATION",
            "description": f"Chunk length is {config['max_chunk_length']}. Smaller chunks may improve streaming performance.",
            "action": "Try reducing MAX_CHUNK_LENGTH to 200-250"
        })
    
    return analysis


@router.post("/diagnostics/benchmark")
async def run_performance_benchmark():
    """Run a controlled performance benchmark"""
    
    benchmark_results = []
    test_texts = [
        "Short test.",
        "This is a medium length test sentence for benchmarking TTS performance.",
        "This is a much longer test sentence that will help us understand how the TTS system performs with larger chunks of text and whether there are any performance degradations with increased text length.",
    ]
    
    from app.api.endpoints.speech import generate_speech_internal
    from app.config import Config
    import time
    
    for i, test_text in enumerate(test_texts):
        start_time = time.time()
        
        try:
            # Reset performance monitor for clean benchmark
            if i == 0:
                perf_monitor.reset()
            
            # Generate speech
            await generate_speech_internal(
                text=test_text,
                voice_sample_path=Config.VOICE_SAMPLE_PATH,
            )
            
            duration = time.time() - start_time
            chars_per_second = len(test_text) / duration if duration > 0 else 0
            
            benchmark_results.append({
                "test_id": i + 1,
                "text_length": len(test_text),
                "duration_seconds": duration,
                "chars_per_second": chars_per_second,
                "text_preview": test_text[:50] + "..." if len(test_text) > 50 else test_text
            })
            
        except Exception as e:
            benchmark_results.append({
                "test_id": i + 1,
                "text_length": len(test_text),
                "error": str(e),
                "text_preview": test_text[:50] + "..." if len(test_text) > 50 else test_text
            })
    
    # Get performance statistics
    perf_stats = perf_monitor.get_operation_stats("TTS_GENERATE")
    
    return {
        "benchmark_results": benchmark_results,
        "performance_summary": perf_stats,
        "average_performance": sum(r.get("chars_per_second", 0) for r in benchmark_results) / len([r for r in benchmark_results if "chars_per_second" in r]),
        "recommendations": _generate_benchmark_recommendations(benchmark_results)
    }


def _generate_benchmark_recommendations(results):
    """Generate recommendations based on benchmark results"""
    recommendations = []
    
    # Calculate average performance
    valid_results = [r for r in results if "chars_per_second" in r and r["chars_per_second"] > 0]
    if not valid_results:
        return ["No valid benchmark results to analyze"]
    
    avg_performance = sum(r["chars_per_second"] for r in valid_results) / len(valid_results)
    
    if avg_performance < 40:
        recommendations.extend([
            "CRITICAL: Performance is severely degraded (<40 chars/s)",
            "Check GPU utilization and memory usage",
            "Verify model is loaded correctly on GPU",
            "Consider reducing model complexity or switching to a faster device"
        ])
    elif avg_performance < 80:
        recommendations.extend([
            "WARNING: Performance is below optimal (<80 chars/s)",
            "Monitor GPU temperature and utilization",
            "Consider optimizing chunk size",
            "Check for background processes consuming GPU resources"
        ])
    else:
        recommendations.append("Performance is within acceptable range")
    
    # Check for performance variance
    if len(valid_results) > 1:
        performances = [r["chars_per_second"] for r in valid_results]
        max_perf, min_perf = max(performances), min(performances)
        if max_perf / min_perf > 2.0:  # More than 2x difference
            recommendations.append("High performance variance detected - investigate inconsistent performance")
    
    return recommendations


@router.post("/diagnostics/reset-metrics")
async def reset_performance_metrics():
    """Reset all performance metrics"""
    initial_count = len(perf_monitor.metrics)
    perf_monitor.reset()
    
    return {
        "message": f"Reset {initial_count} performance metrics",
        "metrics_cleared": initial_count
    }


@router.get("/diagnostics/export-metrics")
async def export_performance_metrics():
    """Export all performance metrics as JSON"""
    return {
        "metrics": perf_monitor.export_metrics(),
        "summary": perf_monitor.get_operation_stats(),
        "export_timestamp": psutil.boot_time()
    }


@router.post("/diagnostics/profiler/start")
async def start_profiling_session(
    session_name: Optional[str] = None,
    trace_memory: bool = True,
    profile_gpu: Optional[bool] = None,
    warmup_steps: int = 2,
    active_steps: int = 8
):
    """Start a PyTorch profiling session for detailed TTS analysis"""
    try:
        session_id = tts_profiler.start_profiling_session(
            session_name=session_name,
            trace_memory=trace_memory,
            profile_gpu=profile_gpu,
            warmup_steps=warmup_steps,
            active_steps=active_steps
        )
        
        return {
            "status": "profiling_started",
            "session_id": session_id,
            "message": f"PyTorch profiling session '{session_id}' started. Generate TTS audio to collect profiling data.",
            "instructions": [
                f"1. Generate {warmup_steps + active_steps} TTS requests to complete profiling",
                "2. First {} requests are warmup (not profiled)".format(warmup_steps),
                "3. Next {} requests will be actively profiled".format(active_steps),
                "4. Profiler will automatically stop and generate reports"
            ]
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to start profiling: {str(e)}"
        }


@router.post("/diagnostics/profiler/stop")
async def stop_profiling_session():
    """Stop the current PyTorch profiling session"""
    try:
        session_id = tts_profiler.stop_profiling_session()
        
        if session_id:
            return {
                "status": "profiling_stopped",
                "session_id": session_id,
                "message": "Profiling session stopped and reports generated"
            }
        else:
            return {
                "status": "no_active_session",
                "message": "No active profiling session to stop"
            }
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to stop profiling: {str(e)}"
        }


@router.get("/diagnostics/profiler/status")
async def get_profiling_status():
    """Get current profiling session status"""
    return {
        "is_profiling": tts_profiler.is_profiling(),
        "current_session": tts_profiler.current_session,
        "available_profiles": tts_profiler.get_available_profiles()
    }


@router.get("/diagnostics/profiler/profiles")
async def list_available_profiles():
    """List all available profiling reports"""
    profiles = tts_profiler.get_available_profiles()
    
    return {
        "profiles": profiles,
        "count": len(profiles),
        "profile_directory": str(tts_profiler.profile_dir)
    }


@router.post("/diagnostics/profiler/benchmark-with-profiling")
async def run_profiled_benchmark():
    """Run benchmark with PyTorch profiling enabled"""
    
    # Start profiling session
    session_id = tts_profiler.start_profiling_session(
        session_name="benchmark_profile",
        trace_memory=True,
        warmup_steps=1,
        active_steps=3
    )
    
    benchmark_results = []
    test_texts = [
        "Short benchmark test.",
        "Medium length benchmark test for TTS performance analysis.",
        "Longer benchmark test sentence to evaluate TTS performance with extended text input for comprehensive profiling analysis."
    ]
    
    from app.api.endpoints.speech import generate_speech_internal
    from app.config import Config
    import time
    
    try:
        for i, test_text in enumerate(test_texts):
            start_time = time.time()
            
            # Generate speech (this will be profiled)
            await generate_speech_internal(
                text=test_text,
                voice_sample_path=Config.VOICE_SAMPLE_PATH,
            )
            
            duration = time.time() - start_time
            chars_per_second = len(test_text) / duration if duration > 0 else 0
            
            benchmark_results.append({
                "test_id": i + 1,
                "text_length": len(test_text),
                "duration_seconds": duration,
                "chars_per_second": chars_per_second,
                "profiled": True
            })
            
            # Small delay between tests
            time.sleep(0.5)
        
        # Stop profiling
        final_session_id = tts_profiler.stop_profiling_session()
        
        return {
            "benchmark_results": benchmark_results,
            "profiling_session": final_session_id,
            "message": f"Benchmark completed with PyTorch profiling. Check profiles directory for detailed analysis.",
            "average_performance": sum(r.get("chars_per_second", 0) for r in benchmark_results) / len(benchmark_results),
            "profile_location": str(tts_profiler.profile_dir)
        }
        
    except Exception as e:
        # Ensure profiling is stopped even if benchmark fails
        tts_profiler.stop_profiling_session()
        
        return {
            "status": "error",
            "message": f"Profiled benchmark failed: {str(e)}",
            "partial_results": benchmark_results
        }
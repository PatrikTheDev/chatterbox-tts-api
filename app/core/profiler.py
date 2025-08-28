"""
PyTorch native profiling integration for TTS performance analysis
"""

import os
import torch
import time
from contextlib import contextmanager
from typing import Optional, Dict, Any
from pathlib import Path
import json

from app.config import Config


class TTSProfiler:
    """TTS profiling with PyTorch's built-in profiler"""
    
    def __init__(self):
        self.profiler = None
        self.profiling_active = False
        self.profile_dir = Path("./profiles")
        self.profile_dir.mkdir(exist_ok=True)
        self.current_session = None
        
    def start_profiling_session(self, 
                              session_name: str = None,
                              trace_memory: bool = True,
                              profile_gpu: bool = None,
                              export_chrome_trace: bool = True,
                              warmup_steps: int = 2,
                              active_steps: int = 8) -> str:
        """
        Start a PyTorch profiling session
        
        Args:
            session_name: Name for this profiling session
            trace_memory: Track memory allocations
            profile_gpu: Profile GPU operations (auto-detect if None)
            export_chrome_trace: Export Chrome trace for visualization
            warmup_steps: Number of warmup steps before active profiling
            active_steps: Number of steps to actively profile
        
        Returns:
            Session ID for this profiling run
        """
        if self.profiling_active:
            print("⚠️ Profiling session already active. Stopping current session.")
            self.stop_profiling_session()
        
        if session_name is None:
            session_name = f"tts_profile_{int(time.time())}"
        
        if profile_gpu is None:
            profile_gpu = torch.cuda.is_available()
        
        print(f"🔍 Starting PyTorch profiling session: {session_name}")
        print(f"   Memory tracing: {trace_memory}")
        print(f"   GPU profiling: {profile_gpu}")
        print(f"   Warmup/Active steps: {warmup_steps}/{active_steps}")
        
        # Configure profiler activities
        activities = [torch.profiler.ProfilerActivity.CPU]
        if profile_gpu:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        
        # Setup profiler schedule
        schedule = torch.profiler.schedule(
            wait=1,
            warmup=warmup_steps,
            active=active_steps,
            repeat=1
        )
        
        # Create profiler with comprehensive settings
        self.profiler = torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=self._on_trace_ready,
            record_shapes=True,
            profile_memory=trace_memory,
            with_stack=True,
            with_flops=True,  # Count FLOPs for performance analysis
            with_modules=True,  # Track module hierarchy
        )
        
        self.current_session = session_name
        self.profiling_active = True
        
        self.profiler.start()
        return session_name
    
    def _on_trace_ready(self, prof):
        """Callback when profiler trace is ready"""
        if not self.current_session:
            return
            
        timestamp = int(time.time())
        trace_path = self.profile_dir / f"{self.current_session}_{timestamp}"
        trace_path.mkdir(exist_ok=True)
        
        try:
            # Export Chrome trace for visualization
            chrome_trace_path = trace_path / "chrome_trace.json"
            prof.export_chrome_trace(str(chrome_trace_path))
            print(f"📊 Chrome trace exported: {chrome_trace_path}")
            
            # Export stacks for detailed analysis
            stacks_path = trace_path / "stacks.txt"
            with open(stacks_path, 'w') as f:
                f.write(prof.key_averages(group_by_stack_n=5).table(
                    sort_by='self_cuda_time_total', row_limit=50))
            
            # Export detailed table
            table_path = trace_path / "detailed_table.txt"
            with open(table_path, 'w') as f:
                f.write(prof.key_averages().table(
                    sort_by='cuda_time_total' if torch.cuda.is_available() else 'cpu_time_total',
                    row_limit=100))
            
            # Export memory analysis if enabled
            if hasattr(prof, 'profiler') and prof.profiler.config.profile_memory:
                memory_path = trace_path / "memory_analysis.txt"
                with open(memory_path, 'w') as f:
                    f.write("=== Memory Timeline ===\n")
                    f.write(prof.key_averages().table(
                        sort_by='cpu_memory_usage',
                        row_limit=50))
            
            # Generate summary report
            self._generate_summary_report(prof, trace_path)
            
            print(f"📈 Full profiling report available in: {trace_path}")
            
        except Exception as e:
            print(f"❌ Error exporting profiler data: {e}")
    
    def _generate_summary_report(self, prof, output_dir: Path):
        """Generate a human-readable summary report"""
        summary_path = output_dir / "summary_report.json"
        
        try:
            events = prof.key_averages()
            
            # Find TTS-specific operations
            tts_operations = []
            total_time = 0
            gpu_time = 0
            memory_ops = []
            
            for event in events:
                if any(keyword in event.key.lower() for keyword in 
                       ['generate', 'forward', 'conv', 'linear', 'attention', 'transformer']):
                    tts_operations.append({
                        'name': event.key,
                        'cpu_time_ms': event.cpu_time / 1000,  # Convert to ms
                        'cuda_time_ms': event.cuda_time / 1000 if event.cuda_time else 0,
                        'count': event.count,
                        'cpu_memory_mb': event.cpu_memory_usage / (1024*1024) if event.cpu_memory_usage else 0,
                        'cuda_memory_mb': event.cuda_memory_usage / (1024*1024) if event.cuda_memory_usage else 0,
                    })
                
                total_time += event.cpu_time
                if event.cuda_time:
                    gpu_time += event.cuda_time
                
                if event.cpu_memory_usage or event.cuda_memory_usage:
                    memory_ops.append({
                        'name': event.key,
                        'cpu_memory_mb': event.cpu_memory_usage / (1024*1024) if event.cpu_memory_usage else 0,
                        'cuda_memory_mb': event.cuda_memory_usage / (1024*1024) if event.cuda_memory_usage else 0,
                    })
            
            # Generate summary
            summary = {
                'session': self.current_session,
                'timestamp': int(time.time()),
                'total_cpu_time_ms': total_time / 1000,
                'total_gpu_time_ms': gpu_time / 1000,
                'gpu_utilization_ratio': (gpu_time / total_time) if total_time > 0 else 0,
                'top_tts_operations': sorted(tts_operations, 
                                           key=lambda x: x['cuda_time_ms'] + x['cpu_time_ms'], 
                                           reverse=True)[:10],
                'memory_intensive_ops': sorted(memory_ops,
                                             key=lambda x: x['cuda_memory_mb'] + x['cpu_memory_mb'],
                                             reverse=True)[:5],
                'analysis': {
                    'likely_bottlenecks': self._identify_bottlenecks(tts_operations),
                    'optimization_suggestions': self._generate_optimization_suggestions(tts_operations, gpu_time, total_time)
                }
            }
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"📋 Performance summary: {summary_path}")
            
        except Exception as e:
            print(f"❌ Error generating summary report: {e}")
    
    def _identify_bottlenecks(self, tts_operations) -> list:
        """Identify likely performance bottlenecks"""
        bottlenecks = []
        
        if not tts_operations:
            return ["No TTS operations found in profile"]
        
        # Sort by total time
        sorted_ops = sorted(tts_operations, 
                           key=lambda x: x['cuda_time_ms'] + x['cpu_time_ms'], 
                           reverse=True)
        
        # Top operation taking too much time
        if sorted_ops[0]['cuda_time_ms'] + sorted_ops[0]['cpu_time_ms'] > 1000:  # >1s
            bottlenecks.append(f"Slow operation: {sorted_ops[0]['name']} taking {sorted_ops[0]['cuda_time_ms'] + sorted_ops[0]['cpu_time_ms']:.1f}ms")
        
        # Check for CPU-bound operations when GPU is available
        if torch.cuda.is_available():
            cpu_heavy_ops = [op for op in sorted_ops if op['cpu_time_ms'] > op['cuda_time_ms'] * 2]
            if cpu_heavy_ops:
                bottlenecks.append(f"CPU-bound operations detected: {cpu_heavy_ops[0]['name']}")
        
        # Memory-heavy operations
        memory_heavy = [op for op in sorted_ops if op['cuda_memory_mb'] > 1000]  # >1GB
        if memory_heavy:
            bottlenecks.append(f"Memory-intensive operation: {memory_heavy[0]['name']} using {memory_heavy[0]['cuda_memory_mb']:.1f}MB")
        
        return bottlenecks
    
    def _generate_optimization_suggestions(self, tts_operations, gpu_time, total_time) -> list:
        """Generate optimization suggestions based on profile data"""
        suggestions = []
        
        if gpu_time / total_time < 0.7 and torch.cuda.is_available():
            suggestions.append("GPU utilization is low - consider optimizing CPU-GPU data transfers")
        
        if any(op['count'] > 100 for op in tts_operations):
            suggestions.append("High operation count detected - consider batch optimization")
        
        memory_usage = sum(op['cuda_memory_mb'] for op in tts_operations)
        if memory_usage > 8000:  # >8GB
            suggestions.append("High memory usage - consider reducing batch size or model precision")
        
        return suggestions
    
    @contextmanager
    def profile_step(self, step_name: str = "tts_step"):
        """Context manager for profiling individual steps with PyTorch record_function"""
        if not self.profiling_active or not self.profiler:
            yield
            return
        
        # Use PyTorch's record_function for step annotation
        with torch.profiler.record_function(step_name):
            try:
                yield
            finally:
                # Step the profiler
                self.profiler.step()
    
    def stop_profiling_session(self) -> Optional[str]:
        """Stop the current profiling session"""
        if not self.profiling_active or not self.profiler:
            print("⚠️ No active profiling session to stop")
            return None
        
        session_name = self.current_session
        
        try:
            self.profiler.stop()
            print(f"✅ Profiling session '{session_name}' completed")
            
        except Exception as e:
            print(f"❌ Error stopping profiler: {e}")
        
        finally:
            self.profiler = None
            self.profiling_active = False
            self.current_session = None
        
        return session_name
    
    def is_profiling(self) -> bool:
        """Check if profiling is currently active"""
        return self.profiling_active
    
    def get_available_profiles(self) -> list:
        """Get list of available profile reports"""
        profiles = []
        for profile_dir in self.profile_dir.iterdir():
            if profile_dir.is_dir():
                summary_file = profile_dir / "summary_report.json"
                if summary_file.exists():
                    try:
                        with open(summary_file) as f:
                            summary = json.load(f)
                        profiles.append({
                            'name': profile_dir.name,
                            'session': summary.get('session'),
                            'timestamp': summary.get('timestamp'),
                            'path': str(profile_dir)
                        })
                    except:
                        pass  # Skip corrupted profiles
        
        return sorted(profiles, key=lambda x: x['timestamp'], reverse=True)


# Global profiler instance
tts_profiler = TTSProfiler()
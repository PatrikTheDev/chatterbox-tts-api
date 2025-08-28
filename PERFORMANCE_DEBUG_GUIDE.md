# TTS Performance Debugging Guide

This guide helps you identify and resolve the performance bottleneck on your 4090 server (40it/s vs 140it/s on other servers).

## Quick Start

### 1. Deploy Updated Code
```bash
# Install dependencies
pip install nvidia-ml-py3>=7.352.0

# Restart your TTS API
python main.py
```

### 2. Run Immediate Analysis on Slow Server
```bash
# Get comprehensive diagnostics
curl http://your-slow-server:4123/diagnostics/performance > slow_server_diag.json

# Run bottleneck analysis
curl http://your-slow-server:4123/diagnostics/bottleneck-analysis > bottleneck_analysis.json

# Run controlled benchmark
curl -X POST http://your-slow-server:4123/diagnostics/benchmark > benchmark_results.json
```

### 3. Start PyTorch Profiling Session
```bash
# Start detailed profiling
curl -X POST "http://your-slow-server:4123/diagnostics/profiler/start?active_steps=5"

# Generate TTS requests (to collect profile data)
for i in {1..6}; do
  curl -X POST http://your-slow-server:4123/audio/speech \
    -H "Content-Type: application/json" \
    -d "{\"input\": \"Performance test sentence number $i for bottleneck analysis.\", \"voice\": \"alloy\"}" \
    --output /dev/null
done

# Check profiling status
curl http://your-slow-server:4123/diagnostics/profiler/status
```

### 4. Analyze Results
```bash
# Analyze the latest profile
python analyze_profile.py

# List all available profiles
python analyze_profile.py --list

# Compare two profiles (if you have fast server data)
python analyze_profile.py profiles/profile_dir_1 --compare profiles/profile_dir_2
```

## Key Metrics to Compare

### Primary Performance Indicators

1. **TTS Generation Speed**: Target >100 chars/s
   - Look for: `avg_speed` in benchmark results
   - Watch for: Real-time logs showing `⚡ Speed: XX chars/s`

2. **GPU Utilization**: Target >80% during generation
   - Check: `gpu_utilization_percent` in diagnostics
   - PyTorch profiler will show GPU vs CPU time ratios

3. **Operation Timing**: Individual operations should be <1s
   - Monitor: `TTS_GENERATE_CHUNK` duration in logs
   - Profile shows: Detailed breakdown of model operations

### Secondary Factors

1. **GPU Memory Bandwidth**
   - Memory clock speeds in diagnostics
   - Memory utilization patterns in profiler

2. **CPU-GPU Transfer Times**
   - Look for CPU-bound operations in GPU workload
   - Memory copy operations in profiler trace

3. **Thermal Throttling**
   - GPU temperature >83°C indicates throttling
   - Graphics/memory clock reductions

## Expected Output Analysis

### Normal Performance (140it/s server):
```
🔍 PERFORMANCE ANALYSIS SUMMARY
⚡ Avg Speed: 135.2 chars/s
🎤 TTS Generation: 0.45s avg per chunk
🎮 GPU Util: 85.3%
🚀 HIGH PERFORMANCE: > 100 chars/s
```

### Bottlenecked Performance (40it/s server):
```
🔍 PERFORMANCE ANALYSIS SUMMARY  
⚡ Avg Speed: 38.1 chars/s
🎤 TTS Generation: 1.85s avg per chunk  ← SLOW
🎮 GPU Util: 45.2%                      ← LOW
⚠️ LOW PERFORMANCE: < 50 chars/s
⚠️ SLOW OPERATION detected!
```

## Common Bottleneck Patterns

### 1. GPU Underutilization (<50% GPU usage)
**Symptoms**: High CPU time, low GPU time in profiler
**Likely Causes**:
- CPU preprocessing bottleneck
- Inefficient data transfers
- Single-threaded model loading

**Investigation**:
```bash
# Check CPU usage during generation
curl -X POST http://your-server:4123/diagnostics/profiler/benchmark-with-profiling
# Look for high CPU time in profiler results
```

### 2. Memory Transfer Bottleneck
**Symptoms**: Long memory copy operations, PCIe bandwidth limits
**Investigation**: Look for `MemcpyHtoD` and `MemcpyDtoH` operations in profiler

### 3. Thermal Throttling
**Symptoms**: Inconsistent performance, high GPU temperature
**Check**: `temperature_celsius` > 83°C in diagnostics

### 4. CUDA/Driver Issues
**Symptoms**: Suboptimal kernel execution, driver warnings
**Investigation**: Check CUDA version compatibility

## Advanced Profiling

### Chrome Trace Visualization
The profiler generates Chrome trace files for detailed analysis:

1. Generate profile: `curl -X POST http://server:4123/diagnostics/profiler/benchmark-with-profiling`
2. Open `chrome://tracing/` in Chrome browser
3. Load the generated `chrome_trace.json` file
4. Look for:
   - Long-running CUDA kernels
   - Memory transfer gaps
   - CPU-GPU synchronization issues

### Comparative Analysis
Compare slow vs fast server profiles:

```bash
# On fast server
curl -X POST http://fast-server:4123/diagnostics/profiler/benchmark-with-profiling
scp -r profiles/ local_machine:/path/to/fast_profiles/

# On slow server  
curl -X POST http://slow-server:4123/diagnostics/profiler/benchmark-with-profiling
scp -r profiles/ local_machine:/path/to/slow_profiles/

# Compare
python analyze_profile.py slow_profiles/latest --compare fast_profiles/latest
```

## Expected Results & Actions

### If GPU Utilization is Low (<60%)
- **Root Cause**: CPU bottleneck or inefficient data pipeline
- **Action**: Check CPU performance, memory bandwidth, driver versions
- **Profile Focus**: Look for CPU-bound operations in TTS pipeline

### If Memory Usage is High (>8GB fluctuations)
- **Root Cause**: Memory management inefficiency
- **Action**: Optimize batch sizes, check memory leaks
- **Profile Focus**: Memory allocation patterns, garbage collection

### If GPU Clocks are Reduced
- **Root Cause**: Power limiting or thermal throttling
- **Action**: Check power supply, cooling, BIOS settings
- **Diagnostic Focus**: Temperature and clock speed monitoring

## Automated Monitoring

The system now provides real-time performance feedback:

```bash
# Every 5 TTS requests, you'll see:
📊 Request #25 - Final memory: CPU 2.1GB, GPU 5.8GB allocated
🔍 TTS_GENERATE_CHUNK: 1.234s
   ⚡ Speed: 45.2 chars/s
   🎮 GPU Util: 52.1%
   ⚠️ SLOW OPERATION detected!
```

This immediate feedback will help you correlate changes with performance impact during your debugging process.

## Next Steps After Identification

Once you identify the bottleneck:

1. **CPU Bottleneck**: Optimize preprocessing, use faster CPU cores
2. **GPU Memory**: Adjust model precision, batch sizes  
3. **Thermal Issues**: Improve cooling, reduce GPU power limits
4. **Driver Issues**: Update CUDA/drivers, check compatibility
5. **Hardware Issues**: Check PCIe lanes, power delivery, memory speeds

The profiling data will give you precise timing information to quantify any improvements you make.
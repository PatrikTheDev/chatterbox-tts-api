#!/usr/bin/env python3
"""
PyTorch profiling analysis script for TTS bottleneck detection
"""

import json
import argparse
from pathlib import Path
import webbrowser


def analyze_profile_directory(profile_dir: Path):
    """Analyze a PyTorch profiling session directory"""
    
    print(f"🔍 Analyzing profile directory: {profile_dir}")
    
    # Check for summary report
    summary_file = profile_dir / "summary_report.json"
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
        
        print(f"\n📊 SESSION: {summary.get('session', 'Unknown')}")
        print(f"⏱️  Total CPU Time: {summary.get('total_cpu_time_ms', 0):.1f}ms")
        print(f"🎮 Total GPU Time: {summary.get('total_gpu_time_ms', 0):.1f}ms")
        print(f"📈 GPU Utilization: {summary.get('gpu_utilization_ratio', 0)*100:.1f}%")
        
        print(f"\n🔥 TOP TTS OPERATIONS:")
        for i, op in enumerate(summary.get('top_tts_operations', [])[:5], 1):
            total_time = op['cuda_time_ms'] + op['cpu_time_ms']
            print(f"  {i}. {op['name'][:60]}")
            print(f"     Time: {total_time:.1f}ms (CPU: {op['cpu_time_ms']:.1f}ms, GPU: {op['cuda_time_ms']:.1f}ms)")
            print(f"     Count: {op['count']}, Memory: {op['cuda_memory_mb']:.1f}MB")
        
        print(f"\n⚠️  LIKELY BOTTLENECKS:")
        for bottleneck in summary.get('analysis', {}).get('likely_bottlenecks', []):
            print(f"  • {bottleneck}")
        
        print(f"\n💡 OPTIMIZATION SUGGESTIONS:")
        for suggestion in summary.get('analysis', {}).get('optimization_suggestions', []):
            print(f"  • {suggestion}")
    
    # Check for Chrome trace
    chrome_trace = profile_dir / "chrome_trace.json"
    if chrome_trace.exists():
        print(f"\n📊 Chrome trace available: {chrome_trace}")
        print(f"   Open in Chrome: chrome://tracing/ and load {chrome_trace}")
        
        # Offer to open automatically
        try:
            import webbrowser
            response = input("\n🌐 Open Chrome trace in browser? (y/n): ")
            if response.lower() == 'y':
                # Create HTML wrapper for easier opening
                html_file = profile_dir / "trace_viewer.html"
                with open(html_file, 'w') as f:
                    f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>TTS Profile Trace - {profile_dir.name}</title>
</head>
<body>
    <h1>PyTorch TTS Profile Trace</h1>
    <p>Profile session: {profile_dir.name}</p>
    <p><strong>Instructions:</strong></p>
    <ol>
        <li>Go to <a href="chrome://tracing/" target="_blank">chrome://tracing/</a></li>
        <li>Click "Load" button</li>
        <li>Select file: <code>{chrome_trace}</code></li>
    </ol>
    <p>Or copy this path: <code>{chrome_trace}</code></p>
</body>
</html>
                    """)
                webbrowser.open(f"file://{html_file.absolute()}")
        except:
            pass
    
    # Check for detailed tables
    detailed_table = profile_dir / "detailed_table.txt"
    if detailed_table.exists():
        print(f"\n📋 Detailed timing table: {detailed_table}")
        
        # Show top 10 lines of detailed table
        with open(detailed_table) as f:
            lines = f.readlines()
            print("   Top operations:")
            for line in lines[3:13]:  # Skip header lines
                if line.strip():
                    print(f"     {line.strip()}")
    
    print("\n" + "="*80)


def compare_profiles(profile1_dir: Path, profile2_dir: Path):
    """Compare two profile sessions"""
    
    print(f"\n🆚 COMPARING PROFILES")
    print(f"Profile 1: {profile1_dir.name}")
    print(f"Profile 2: {profile2_dir.name}")
    print("="*80)
    
    # Load summaries
    summary1_file = profile1_dir / "summary_report.json"
    summary2_file = profile2_dir / "summary_report.json"
    
    if not (summary1_file.exists() and summary2_file.exists()):
        print("❌ Both profiles must have summary reports for comparison")
        return
    
    with open(summary1_file) as f:
        summary1 = json.load(f)
    with open(summary2_file) as f:
        summary2 = json.load(f)
    
    # Compare key metrics
    cpu1 = summary1.get('total_cpu_time_ms', 0)
    cpu2 = summary2.get('total_cpu_time_ms', 0)
    gpu1 = summary1.get('total_gpu_time_ms', 0)
    gpu2 = summary2.get('total_gpu_time_ms', 0)
    util1 = summary1.get('gpu_utilization_ratio', 0) * 100
    util2 = summary2.get('gpu_utilization_ratio', 0) * 100
    
    print(f"⏱️  CPU Time: {cpu1:.1f}ms vs {cpu2:.1f}ms ({(cpu2/cpu1-1)*100:+.1f}%)")
    print(f"🎮 GPU Time: {gpu1:.1f}ms vs {gpu2:.1f}ms ({(gpu2/gpu1-1)*100:+.1f}%)")
    print(f"📈 GPU Util: {util1:.1f}% vs {util2:.1f}% ({util2-util1:+.1f}pp)")
    
    # Find performance differences
    if cpu2 > cpu1 * 1.2:  # 20% slower
        print("⚠️  Profile 2 shows significantly slower CPU performance")
    elif cpu1 > cpu2 * 1.2:
        print("✅ Profile 2 shows significantly faster CPU performance")
    
    if gpu2 > gpu1 * 1.2:
        print("⚠️  Profile 2 shows significantly slower GPU performance")
    elif gpu1 > gpu2 * 1.2:
        print("✅ Profile 2 shows significantly faster GPU performance")
    
    # Compare top operations
    ops1 = {op['name']: op for op in summary1.get('top_tts_operations', [])}
    ops2 = {op['name']: op for op in summary2.get('top_tts_operations', [])}
    
    common_ops = set(ops1.keys()) & set(ops2.keys())
    if common_ops:
        print(f"\n🔧 OPERATION COMPARISON:")
        for op_name in list(common_ops)[:5]:
            time1 = ops1[op_name]['cuda_time_ms'] + ops1[op_name]['cpu_time_ms']
            time2 = ops2[op_name]['cuda_time_ms'] + ops2[op_name]['cpu_time_ms']
            if time1 > 0:
                diff = (time2 / time1 - 1) * 100
                print(f"  {op_name[:50]}: {time1:.1f}ms vs {time2:.1f}ms ({diff:+.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Analyze PyTorch TTS profiling results")
    parser.add_argument("profile_dir", nargs="?", default="./profiles", 
                       help="Profile directory to analyze")
    parser.add_argument("--compare", 
                       help="Compare with another profile directory")
    parser.add_argument("--list", action="store_true",
                       help="List available profiles")
    
    args = parser.parse_args()
    
    profile_base = Path(args.profile_dir)
    
    if args.list:
        print(f"📁 Available profiles in {profile_base}:")
        if profile_base.exists():
            for i, subdir in enumerate(sorted(profile_base.iterdir()), 1):
                if subdir.is_dir():
                    summary_file = subdir / "summary_report.json"
                    if summary_file.exists():
                        try:
                            with open(summary_file) as f:
                                summary = json.load(f)
                            session = summary.get('session', subdir.name)
                            timestamp = summary.get('timestamp', 0)
                            print(f"  {i}. {session} ({subdir.name})")
                        except:
                            print(f"  {i}. {subdir.name} (no summary)")
        return
    
    if not profile_base.exists():
        print(f"❌ Profile directory not found: {profile_base}")
        return
    
    # Find latest profile if directory contains multiple profiles
    if profile_base.is_dir() and any(subdir.is_dir() for subdir in profile_base.iterdir()):
        # Find most recent profile directory
        profile_dirs = [d for d in profile_base.iterdir() if d.is_dir() and (d / "summary_report.json").exists()]
        if profile_dirs:
            latest_profile = max(profile_dirs, key=lambda d: d.stat().st_mtime)
            analyze_profile_directory(latest_profile)
            
            if args.compare:
                compare_dir = Path(args.compare)
                if compare_dir.exists():
                    compare_profiles(latest_profile, compare_dir)
        else:
            print(f"❌ No valid profile directories found in {profile_base}")
    else:
        # Analyze single profile directory
        analyze_profile_directory(profile_base)


if __name__ == "__main__":
    main()
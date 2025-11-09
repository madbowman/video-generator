"""
CPU Performance Optimization Utilities for Chatterbox TTS
"""

import torch
import psutil
import os
from pathlib import Path


def optimize_cpu_performance():
    """Apply CPU-specific optimizations for better performance"""
    
    # Get CPU info
    cpu_count = psutil.cpu_count(logical=False)  # Physical cores
    logical_count = psutil.cpu_count(logical=True)  # Logical cores
    
    print(f"🖥️  CPU Optimization")
    print(f"   Physical cores: {cpu_count}")
    print(f"   Logical cores: {logical_count}")
    
    # Set optimal thread count (use ALL physical cores for maximum speed)
    optimal_threads = cpu_count  # Use ALL physical cores
    torch.set_num_threads(optimal_threads)
    
    # Set environment variables for maximum CPU performance
    os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
    os.environ['MKL_NUM_THREADS'] = str(optimal_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(optimal_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(optimal_threads)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(optimal_threads)
    
    print(f"   🚀 Optimized for ALL {optimal_threads} cores ({logical_count} threads)")
    
    # Memory optimization
    available_memory = psutil.virtual_memory().available / (1024**3)  # GB
    print(f"   Available RAM: {available_memory:.1f} GB")
    
    if available_memory < 8:
        print("   ⚠️  Low RAM detected - enabling aggressive memory management")
        return True  # Enable aggressive memory management
    
    return False


def get_cpu_recommendations():
    """Get performance recommendations based on CPU"""
    
    cpu_freq = psutil.cpu_freq()
    cpu_percent = psutil.cpu_percent(interval=1)
    
    recommendations = []
    
    if cpu_freq and cpu_freq.current < 2000:  # Less than 2GHz
        recommendations.append("- Use smaller batch sizes")
        recommendations.append("- Enable frequent memory cleanup")
        recommendations.append("- Consider lower quality settings")
    
    if cpu_percent > 80:
        recommendations.append("- Close other applications")
        recommendations.append("- Reduce concurrent processes")
    
    return recommendations


def monitor_cpu_usage(duration=5):
    """Monitor CPU usage during processing"""
    
    print(f"\n📊 CPU Monitoring ({duration}s):")
    
    # Initial reading
    cpu_before = psutil.cpu_percent()
    memory_before = psutil.virtual_memory().percent
    
    import time
    time.sleep(duration)
    
    # Final reading
    cpu_after = psutil.cpu_percent()
    memory_after = psutil.virtual_memory().percent
    
    print(f"   CPU Usage: {cpu_after:.1f}% (was {cpu_before:.1f}%)")
    print(f"   Memory Usage: {memory_after:.1f}% (was {memory_before:.1f}%)")
    
    return {
        'cpu': cpu_after,
        'memory': memory_after,
        'cpu_change': cpu_after - cpu_before,
        'memory_change': memory_after - memory_before
    }


if __name__ == "__main__":
    print("🚀 CPU Performance Optimization Test")
    print("=" * 50)
    
    # Test optimization
    aggressive_mode = optimize_cpu_performance()
    
    # Get recommendations
    recommendations = get_cpu_recommendations()
    if recommendations:
        print(f"\n💡 Performance Recommendations:")
        for rec in recommendations:
            print(f"   {rec}")
    
    # Monitor for 3 seconds
    stats = monitor_cpu_usage(3)
    
    print(f"\n✅ Optimization complete!")
    print(f"   Mode: {'Aggressive' if aggressive_mode else 'Standard'}")
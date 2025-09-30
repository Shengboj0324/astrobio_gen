#!/usr/bin/env python3
"""
📊 RUNPOD MONITORING DASHBOARD
Real-time monitoring for training progress
"""

import time
import psutil
import torch
import json
from datetime import datetime
import subprocess

class RunPodMonitor:
    def __init__(self):
        self.start_time = datetime.now()
    
    def get_gpu_stats(self):
        """Get GPU statistics"""
        if not torch.cuda.is_available():
            return {}
        
        stats = {}
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            stats[f'gpu_{i}'] = {
                'name': props.name,
                'memory_total': props.total_memory,
                'memory_allocated': torch.cuda.memory_allocated(i),
                'memory_reserved': torch.cuda.memory_reserved(i),
                'utilization': self.get_gpu_utilization(i)
            }
        return stats
    
    def get_gpu_utilization(self, device_id):
        """Get GPU utilization percentage"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits', f'--id={device_id}'], 
                                  capture_output=True, text=True)
            return int(result.stdout.strip())
        except:
            return 0
    
    def get_system_stats(self):
        """Get system statistics"""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'uptime': str(datetime.now() - self.start_time)
        }
    
    def monitor_loop(self):
        """Main monitoring loop"""
        print("📊 Starting RunPod monitoring...")
        
        while True:
            timestamp = datetime.now().isoformat()
            
            # Collect stats
            gpu_stats = self.get_gpu_stats()
            system_stats = self.get_system_stats()
            
            # Create monitoring report
            report = {
                'timestamp': timestamp,
                'gpu_stats': gpu_stats,
                'system_stats': system_stats
            }
            
            # Save to file
            with open('/workspace/monitoring_log.json', 'a') as f:
                f.write(json.dumps(report) + '\n')
            
            # Print summary
            print(f"\n📊 {timestamp}")
            print(f"🖥️  CPU: {system_stats['cpu_percent']:.1f}%")
            print(f"💾 RAM: {system_stats['memory_percent']:.1f}%")
            
            for gpu_id, stats in gpu_stats.items():
                memory_used = stats['memory_allocated'] / stats['memory_total'] * 100
                print(f"🔥 {stats['name']}: {stats['utilization']}% GPU, {memory_used:.1f}% VRAM")
            
            time.sleep(30)  # Update every 30 seconds

if __name__ == "__main__":
    monitor = RunPodMonitor()
    monitor.monitor_loop()

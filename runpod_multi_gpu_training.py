#!/usr/bin/env python3
"""
🔥 MULTI-GPU TRAINING SCRIPT FOR RUNPOD
Optimized for 2x RTX A5000 GPUs (48GB total VRAM)
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import argparse
from datetime import datetime

def setup_distributed(rank, world_size):
    """Setup distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()

def train_worker(rank, world_size, args):
    """Training worker for each GPU"""
    print(f"🚀 Starting training worker on GPU {rank}")
    
    # Setup distributed training
    setup_distributed(rank, world_size)
    
    # Create model (replace with your actual model)
    model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512)
    ).cuda(rank)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training loop
    model.train()
    for step in range(args.max_steps):
        # Generate synthetic batch
        batch_size = args.batch_size // world_size  # Split batch across GPUs
        x = torch.randn(batch_size, 1024, device=rank)
        target = torch.randn(batch_size, 512, device=rank)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if rank == 0 and step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
    
    # Cleanup
    cleanup_distributed()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--world_size', type=int, default=2)
    args = parser.parse_args()
    
    print(f"🔥 Starting multi-GPU training with {args.world_size} GPUs")
    
    # Spawn training processes
    mp.spawn(
        train_worker,
        args=(args.world_size, args),
        nprocs=args.world_size,
        join=True
    )
    
    print("✅ Multi-GPU training complete!")

if __name__ == "__main__":
    main()

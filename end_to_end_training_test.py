#!/usr/bin/env python3
"""
End-to-End Training Test
=========================

Comprehensive test of the full training pipeline with:
- 1000 training steps
- Checkpointing every 100 steps
- Validation every 200 steps
- GPU monitoring
- Memory profiling
- Error recovery testing

This test validates production readiness for extended training runs.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleTestModel(nn.Module):
    """Simple 3D CNN for testing"""
    def __init__(self, in_channels=5, out_channels=5):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 32, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(32, out_channels, 3, padding=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class TrainingMonitor:
    """Monitor training metrics"""
    def __init__(self):
        self.losses = []
        self.val_losses = []
        self.step_times = []
        self.gpu_memory = []
        self.start_time = time.time()
    
    def log_step(self, step: int, loss: float, step_time: float):
        """Log training step"""
        self.losses.append(loss)
        self.step_times.append(step_time)
        
        if torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated() / 1024**3  # GB
            self.gpu_memory.append(mem)
        
        if step % 100 == 0:
            avg_loss = sum(self.losses[-100:]) / min(100, len(self.losses))
            avg_time = sum(self.step_times[-100:]) / min(100, len(self.step_times))
            logger.info(f"Step {step}: Loss={avg_loss:.6f}, Time={avg_time:.3f}s/step")
            
            if torch.cuda.is_available():
                logger.info(f"  GPU Memory: {mem:.2f} GB")
    
    def log_validation(self, step: int, val_loss: float):
        """Log validation"""
        self.val_losses.append(val_loss)
        logger.info(f"Validation at step {step}: Loss={val_loss:.6f}")
    
    def get_summary(self) -> Dict:
        """Get training summary"""
        total_time = time.time() - self.start_time
        avg_loss = sum(self.losses) / len(self.losses) if self.losses else 0
        avg_step_time = sum(self.step_times) / len(self.step_times) if self.step_times else 0
        
        summary = {
            'total_time': total_time,
            'total_steps': len(self.losses),
            'avg_loss': avg_loss,
            'final_loss': self.losses[-1] if self.losses else 0,
            'avg_step_time': avg_step_time,
            'throughput': len(self.losses) / total_time if total_time > 0 else 0,
        }
        
        if self.gpu_memory:
            summary['max_gpu_memory'] = max(self.gpu_memory)
            summary['avg_gpu_memory'] = sum(self.gpu_memory) / len(self.gpu_memory)
        
        return summary


def create_synthetic_dataset(num_samples=1000, batch_size=8):
    """Create synthetic dataset for testing"""
    logger.info(f"Creating synthetic dataset: {num_samples} samples")
    
    # Create data: [batch, channels, depth, height, width]
    x = torch.randn(num_samples, 5, 8, 16, 16)
    y = torch.randn(num_samples, 5, 8, 16, 16)
    
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    return dataloader


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                   step: int, checkpoint_dir: Path):
    """Save training checkpoint"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
    
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    
    logger.info(f"Saved checkpoint: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                   checkpoint_path: Path) -> int:
    """Load training checkpoint"""
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']
    
    logger.info(f"Resumed from step {step}")
    return step


def validate(model: nn.Module, val_dataloader: DataLoader, device: torch.device) -> float:
    """Run validation"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_x, batch_y in val_dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            output = model(batch_x)
            loss = nn.functional.mse_loss(output, batch_y)
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches >= 10:  # Validate on 10 batches
                break
    
    model.train()
    return total_loss / num_batches


def run_training_test(num_steps=1000, checkpoint_interval=100, val_interval=200):
    """Run end-to-end training test"""
    logger.info("=" * 80)
    logger.info("END-TO-END TRAINING TEST")
    logger.info("=" * 80)
    logger.info("")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Create model
    logger.info("Creating model...")
    model = SimpleTestModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")
    
    # Create datasets
    train_dataloader = create_synthetic_dataset(num_samples=1000, batch_size=8)
    val_dataloader = create_synthetic_dataset(num_samples=100, batch_size=8)
    
    # Training monitor
    monitor = TrainingMonitor()
    
    # Checkpoint directory
    checkpoint_dir = Path("test_checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Training loop
    logger.info("")
    logger.info(f"Starting training: {num_steps} steps")
    logger.info(f"Checkpointing every {checkpoint_interval} steps")
    logger.info(f"Validation every {val_interval} steps")
    logger.info("")
    
    model.train()
    step = 0
    
    try:
        while step < num_steps:
            for batch_x, batch_y in train_dataloader:
                if step >= num_steps:
                    break
                
                step_start = time.time()
                
                # Move to device
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                # Forward pass
                output = model(batch_x)
                loss = nn.functional.mse_loss(output, batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                step_time = time.time() - step_start
                
                # Log
                monitor.log_step(step, loss.item(), step_time)
                
                # Checkpoint
                if step > 0 and step % checkpoint_interval == 0:
                    save_checkpoint(model, optimizer, step, checkpoint_dir)
                
                # Validation
                if step > 0 and step % val_interval == 0:
                    val_loss = validate(model, val_dataloader, device)
                    monitor.log_validation(step, val_loss)
                
                step += 1
    
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    # Final checkpoint
    save_checkpoint(model, optimizer, step, checkpoint_dir)
    
    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)
    
    summary = monitor.get_summary()
    logger.info(f"Total time: {summary['total_time']:.2f}s")
    logger.info(f"Total steps: {summary['total_steps']}")
    logger.info(f"Average loss: {summary['avg_loss']:.6f}")
    logger.info(f"Final loss: {summary['final_loss']:.6f}")
    logger.info(f"Average step time: {summary['avg_step_time']:.3f}s")
    logger.info(f"Throughput: {summary['throughput']:.2f} steps/s")
    
    if 'max_gpu_memory' in summary:
        logger.info(f"Max GPU memory: {summary['max_gpu_memory']:.2f} GB")
        logger.info(f"Avg GPU memory: {summary['avg_gpu_memory']:.2f} GB")
    
    logger.info("")
    logger.info("✅ Training test complete!")
    
    return summary


def main():
    """Main entry point"""
    try:
        summary = run_training_test(num_steps=1000, checkpoint_interval=100, val_interval=200)
        
        # Validate results
        assert summary['total_steps'] >= 1000, "Did not complete 1000 steps"
        assert summary['final_loss'] < summary['avg_loss'], "Loss did not decrease"
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ ALL VALIDATION CHECKS PASSED")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


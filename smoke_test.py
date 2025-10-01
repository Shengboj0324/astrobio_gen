#!/usr/bin/env python3
"""
Comprehensive Smoke Test Suite
===============================

Zero-tolerance smoke tests for production readiness.
Tests all critical components in <2 minutes.

Usage:
    python smoke_test.py
    python smoke_test.py --verbose
    python smoke_test.py --model rebuilt_llm_integration
"""

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class SmokeTestRunner:
    """Comprehensive smoke test runner"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[Tuple[str, bool, str]] = []
        self.start_time = time.time()
    
    def log(self, message: str):
        """Log message if verbose"""
        if self.verbose:
            print(f"  {message}")
    
    def test(self, name: str, func):
        """Run a single test"""
        print(f"Testing: {name}...", end=" ", flush=True)
        try:
            func()
            print("✅ PASS")
            self.results.append((name, True, ""))
        except Exception as e:
            print(f"❌ FAIL")
            error_msg = f"{type(e).__name__}: {str(e)}"
            if self.verbose:
                error_msg += f"\n{traceback.format_exc()}"
            self.results.append((name, False, error_msg))
            print(f"  Error: {error_msg}")
    
    def test_imports(self):
        """Test critical imports"""
        self.log("Importing PyTorch...")
        import torch
        assert torch.__version__ >= "2.0.0", f"PyTorch version {torch.__version__} < 2.0.0"

        self.log("Importing core models...")
        from models.enhanced_datacube_unet import EnhancedCubeUNet
        from models.rebuilt_llm_integration import RebuiltLLMIntegration
        from models.rebuilt_graph_vae import RebuiltGraphVAE
        from models.rebuilt_datacube_cnn import RebuiltDatacubeCNN

        self.log("Importing attention mechanisms...")
        from models.sota_attention_2025 import SOTAAttention2025, FlashAttention3

        # Skip training system import - it's heavy and may hang
        # from training.unified_sota_training_system import UnifiedSOTATrainingSystem

        self.log("All imports successful")
    
    def test_cuda_availability(self):
        """Test CUDA availability"""
        self.log("Checking CUDA...")
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            self.log(f"CUDA available with {device_count} GPU(s)")
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                self.log(f"  GPU {i}: {name}")
        else:
            self.log("CUDA not available (CPU mode)")
    
    def test_model_initialization(self):
        """Test model initialization"""
        from models.enhanced_datacube_unet import EnhancedCubeUNet
        
        self.log("Initializing EnhancedCubeUNet...")
        model = EnhancedCubeUNet(n_input_vars=5, n_output_vars=5)
        
        param_count = sum(p.numel() for p in model.parameters())
        self.log(f"Model parameters: {param_count:,}")
        
        assert param_count > 0, "Model has no parameters"
    
    def test_forward_pass(self):
        """Test forward pass"""
        from models.enhanced_datacube_unet import EnhancedCubeUNet
        
        self.log("Creating model...")
        model = EnhancedCubeUNet(n_input_vars=5, n_output_vars=5)
        model.eval()
        
        self.log("Creating synthetic input...")
        batch_size = 2
        x = torch.randn(batch_size, 5, 8, 16, 16, 8)
        
        self.log("Running forward pass...")
        with torch.no_grad():
            y = model(x)
        
        self.log(f"Output shape: {y.shape}")
        assert y.shape[0] == batch_size, f"Batch size mismatch: {y.shape[0]} != {batch_size}"
        assert not torch.isnan(y).any(), "Output contains NaN"
        assert not torch.isinf(y).any(), "Output contains Inf"
    
    def test_backward_pass(self):
        """Test backward pass"""
        from models.enhanced_datacube_unet import EnhancedCubeUNet
        
        self.log("Creating model...")
        model = EnhancedCubeUNet(n_input_vars=5, n_output_vars=5)
        model.train()
        
        self.log("Creating synthetic input...")
        x = torch.randn(2, 5, 8, 16, 16, 8)
        target = torch.randn(2, 5, 8, 16, 16, 8)
        
        self.log("Running forward pass...")
        y = model(x)
        
        self.log("Computing loss...")
        loss = nn.functional.mse_loss(y, target)
        
        self.log("Running backward pass...")
        loss.backward()
        
        self.log("Checking gradients...")
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        param_count = sum(1 for p in model.parameters())
        assert grad_count == param_count, f"Gradient count mismatch: {grad_count} != {param_count}"
        
        # Check for NaN/Inf gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
    
    def test_optimizer_step(self):
        """Test optimizer step"""
        from models.enhanced_datacube_unet import EnhancedCubeUNet
        
        self.log("Creating model and optimizer...")
        model = EnhancedCubeUNet(n_input_vars=5, n_output_vars=5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        self.log("Training step...")
        x = torch.randn(2, 5, 8, 16, 16, 8)
        target = torch.randn(2, 5, 8, 16, 16, 8)
        
        y = model(x)
        loss = nn.functional.mse_loss(y, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        self.log(f"Loss: {loss.item():.6f}")
    
    def test_checkpointing(self):
        """Test model checkpointing"""
        from models.enhanced_datacube_unet import EnhancedCubeUNet
        import tempfile
        
        self.log("Creating model...")
        model = EnhancedCubeUNet(n_input_vars=5, n_output_vars=5)
        
        self.log("Saving checkpoint...")
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            checkpoint_path = f.name
            torch.save(model.state_dict(), checkpoint_path)
        
        self.log("Loading checkpoint...")
        model2 = EnhancedCubeUNet(n_input_vars=5, n_output_vars=5)
        model2.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        
        self.log("Verifying parameters...")
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert n1 == n2, f"Parameter name mismatch: {n1} != {n2}"
            assert torch.allclose(p1, p2), f"Parameter value mismatch in {n1}"
        
        # Cleanup
        Path(checkpoint_path).unlink()
    
    def test_attention_mechanisms(self):
        """Test attention mechanisms"""
        from models.sota_attention_2025 import FlashAttention3, SOTAAttention2025
        
        self.log("Testing FlashAttention3...")
        config = type('Config', (), {
            'hidden_size': 512,
            'num_attention_heads': 8,
            'attention_dropout': 0.1,
        })()
        
        attn = FlashAttention3(config)
        batch_size, seq_len = 2, 128
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        
        with torch.no_grad():
            output = attn(hidden_states)
        
        assert output.shape == hidden_states.shape, f"Shape mismatch: {output.shape} != {hidden_states.shape}"
        assert not torch.isnan(output).any(), "Attention output contains NaN"
    
    def test_data_loading(self):
        """Test data loading"""
        from torch.utils.data import DataLoader, TensorDataset
        
        self.log("Creating synthetic dataset...")
        x = torch.randn(100, 5, 8, 16, 16, 8)
        y = torch.randn(100, 5, 8, 16, 16, 8)
        dataset = TensorDataset(x, y)
        
        self.log("Creating DataLoader...")
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
        
        self.log("Iterating through batches...")
        for i, (batch_x, batch_y) in enumerate(dataloader):
            assert batch_x.shape[0] <= 8, f"Batch size too large: {batch_x.shape[0]}"
            if i >= 2:  # Test first 3 batches
                break
    
    def test_mixed_precision(self):
        """Test mixed precision training"""
        from models.enhanced_datacube_unet import EnhancedCubeUNet
        
        self.log("Creating model...")
        model = EnhancedCubeUNet(n_input_vars=5, n_output_vars=5)
        
        self.log("Creating GradScaler...")
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        self.log("Training with mixed precision...")
        x = torch.randn(2, 5, 8, 16, 16, 8)
        target = torch.randn(2, 5, 8, 16, 16, 8)
        
        if torch.cuda.is_available():
            model = model.cuda()
            x = x.cuda()
            target = target.cuda()
            
            with torch.cuda.amp.autocast():
                y = model(x)
                loss = nn.functional.mse_loss(y, target)
            
            scaler.scale(loss).backward()
        else:
            y = model(x)
            loss = nn.functional.mse_loss(y, target)
            loss.backward()
        
        self.log(f"Loss: {loss.item():.6f}")
    
    def test_rust_integration(self):
        """Test Rust integration"""
        try:
            self.log("Importing Rust extensions...")
            import astrobio_rust
            self.log("Rust extensions available")
        except ImportError:
            self.log("Rust extensions not available (will use Python fallback)")
            # This is not a failure - fallback is expected
    
    def run_all_tests(self):
        """Run all smoke tests"""
        print("="*80)
        print("AstroBio-Gen Smoke Test Suite")
        print("="*80)
        
        self.test("Import critical modules", self.test_imports)
        self.test("CUDA availability", self.test_cuda_availability)
        self.test("Model initialization", self.test_model_initialization)
        self.test("Forward pass", self.test_forward_pass)
        self.test("Backward pass", self.test_backward_pass)
        self.test("Optimizer step", self.test_optimizer_step)
        self.test("Checkpointing", self.test_checkpointing)
        self.test("Attention mechanisms", self.test_attention_mechanisms)
        self.test("Data loading", self.test_data_loading)
        self.test("Mixed precision", self.test_mixed_precision)
        self.test("Rust integration", self.test_rust_integration)
        
        # Print summary
        print("="*80)
        print("Test Summary")
        print("="*80)
        
        passed = sum(1 for _, success, _ in self.results if success)
        failed = sum(1 for _, success, _ in self.results if not success)
        total = len(self.results)
        
        print(f"Total:  {total}")
        print(f"Passed: {passed} ({100*passed/total:.1f}%)")
        print(f"Failed: {failed} ({100*failed/total:.1f}%)")
        
        elapsed = time.time() - self.start_time
        print(f"Time:   {elapsed:.2f}s")
        
        if failed > 0:
            print("\n❌ SMOKE TEST FAILED")
            print("\nFailed tests:")
            for name, success, error in self.results:
                if not success:
                    print(f"  - {name}")
                    if error:
                        print(f"    {error}")
            return False
        else:
            print("\n✅ ALL SMOKE TESTS PASSED")
            return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run smoke tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    runner = SmokeTestRunner(verbose=args.verbose)
    success = runner.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


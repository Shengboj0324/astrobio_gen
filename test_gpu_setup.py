#!/usr/bin/env python3
"""
GPU Setup Verification for Astrobiology Project
Test script to verify CUDA, PyTorch, and model training work correctly.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from datetime import datetime
import pandas as pd
import numpy as np
import sys
from pathlib import Path

def test_basic_gpu():
    """Test basic GPU functionality"""
    print("="*60)
    print("BASIC GPU TEST")
    print("="*60)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    else:
        print("❌ No CUDA GPU available!")
        return False
    
    # Test tensor operations
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.mm(x, y)
        print(f"✅ GPU tensor operations work! Result shape: {z.shape}")
        return True
    except Exception as e:
        print(f"❌ GPU tensor operations failed: {e}")
        return False

def test_pytorch_lightning():
    """Test PyTorch Lightning with GPU"""
    print("\n" + "="*60)
    print("PYTORCH LIGHTNING GPU TEST")
    print("="*60)
    
    try:
        # Simple model for testing
        class TestModel(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(10, 1)
                
            def forward(self, x):
                return self.layer(x)
            
            def training_step(self, batch, batch_idx):
                x, y = batch
                y_hat = self(x)
                loss = nn.functional.mse_loss(y_hat, y)
                return loss
            
            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters())
        
        model = TestModel()
        
        # Test GPU availability in Lightning
        if torch.cuda.is_available():
            trainer = pl.Trainer(
                accelerator='gpu',
                devices=1,
                max_epochs=1,
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False
            )
            print("✅ PyTorch Lightning GPU trainer created successfully!")
            return True
        else:
            print("❌ No GPU available for PyTorch Lightning")
            return False
            
    except Exception as e:
        print(f"❌ PyTorch Lightning test failed: {e}")
        return False

def test_project_imports():
    """Test that project modules can be imported"""
    print("\n" + "="*60)
    print("PROJECT IMPORTS TEST")
    print("="*60)
    
    try:
        # Test core project imports
        sys.path.append(str(Path.cwd()))
        
        from models.graph_vae import GVAE
        print("✅ Graph VAE import successful")
        
        from models.fusion_transformer import FusionModel
        print("✅ Fusion Transformer import successful")
        
        # Test if train.py can be imported
        import train
        print("✅ Training script import successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Project imports failed: {e}")
        print("This is normal if you haven't transferred the project files yet.")
        return False

def test_data_processing():
    """Test basic data processing capabilities"""
    print("\n" + "="*60)
    print("DATA PROCESSING TEST")
    print("="*60)
    
    try:
        # Test pandas and numpy
        df = pd.DataFrame({
            'pathway_id': ['map00010', 'map00020'],
            'name': ['Glycolysis', 'TCA Cycle'],
            'reactions': [10, 8]
        })
        print(f"✅ Pandas DataFrame created: {df.shape}")
        
        # Test scientific computing
        import networkx as nx
        G = nx.DiGraph()
        G.add_edges_from([('A', 'B'), ('B', 'C')])
        print(f"✅ NetworkX graph created with {G.number_of_nodes()} nodes")
        
        # Test astronomy libraries
        import astropy
        print(f"✅ Astropy version: {astropy.__version__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data processing test failed: {e}")
        return False

def generate_system_report():
    """Generate comprehensive system report"""
    print("\n" + "="*60)
    print("SYSTEM REPORT")
    print("="*60)
    
    import platform
    import psutil
    
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"CPU: {platform.processor()}")
    print(f"RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  VRAM: {props.total_memory / 1e9:.1f} GB")
            print(f"  Compute: {props.major}.{props.minor}")

def main():
    """Run all tests"""
    print("🚀 Astrobiology Project GPU Setup Verification")
    print(f"Started at: {datetime.now()}")
    
    tests = [
        ("Basic GPU", test_basic_gpu),
        ("PyTorch Lightning", test_pytorch_lightning),
        ("Project Imports", test_project_imports),
        ("Data Processing", test_data_processing)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your setup is ready for training.")
    else:
        print(f"\n⚠️  {total-passed} tests failed. Please check the errors above.")
    
    # Generate system report
    generate_system_report()
    
    print(f"\nCompleted at: {datetime.now()}")

if __name__ == "__main__":
    main() 
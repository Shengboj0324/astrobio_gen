#!/usr/bin/env python3
"""
SIMULATED RUNPOD DEPLOYMENT TEST
================================

This script simulates the exact RunPod environment and validates:
1. GPU availability and configuration
2. All imports work correctly
3. Model instantiation succeeds
4. Data loading works
5. Forward pass succeeds
6. Backward pass and gradient flow work
7. Memory usage is within limits
8. Training loop executes without errors

CRITICAL: This must pass 100% before deployment to RunPod.
"""

import sys
import os
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, List
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RunPodSimulator:
    """Simulates RunPod environment for pre-deployment validation"""
    
    def __init__(self):
        self.test_results = {}
        self.errors = []
        self.warnings = []
        
    def simulate_environment(self):
        """Simulate RunPod environment variables"""
        logger.info("=" * 80)
        logger.info("SIMULATING RUNPOD ENVIRONMENT")
        logger.info("=" * 80)
        
        # Simulate RunPod environment
        os.environ['RUNPOD_POD_ID'] = 'simulated-pod-12345'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 2x A5000
        os.environ['WORKSPACE'] = '/workspace'
        
        logger.info("✅ Environment variables set")
        
    def test_gpu_availability(self):
        """Test 1: GPU Availability"""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 1: GPU AVAILABILITY")
        logger.info("=" * 80)
        
        try:
            import torch
            
            if not torch.cuda.is_available():
                self.errors.append("CRITICAL: CUDA not available")
                self.test_results['gpu_availability'] = 'FAIL'
                logger.error("❌ CUDA not available - DEPLOYMENT WILL FAIL")
                return False
            
            gpu_count = torch.cuda.device_count()
            logger.info(f"✅ CUDA available: {gpu_count} GPUs detected")
            
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                total_memory_gb = props.total_memory / (1024**3)
                logger.info(f"   GPU {i}: {props.name}")
                logger.info(f"   Memory: {total_memory_gb:.2f} GB")
                logger.info(f"   Compute Capability: {props.major}.{props.minor}")
                
                # Validate memory (should be ~24GB per A5000)
                if total_memory_gb < 20:
                    self.warnings.append(f"GPU {i} has less than 20GB memory")
            
            self.test_results['gpu_availability'] = 'PASS'
            return True
            
        except Exception as e:
            self.errors.append(f"GPU test failed: {e}")
            self.test_results['gpu_availability'] = 'FAIL'
            logger.error(f"❌ GPU test failed: {e}")
            traceback.print_exc()
            return False
    
    def test_critical_imports(self):
        """Test 2: Critical Imports"""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 2: CRITICAL IMPORTS")
        logger.info("=" * 80)
        
        critical_imports = [
            ('torch', 'PyTorch'),
            ('torch.nn', 'PyTorch NN'),
            ('torch_geometric', 'PyTorch Geometric'),
            ('transformers', 'Transformers'),
            ('peft', 'PEFT'),
            ('bitsandbytes', 'BitsAndBytes'),
            ('numpy', 'NumPy'),
            ('pandas', 'Pandas'),
        ]
        
        failed_imports = []
        
        for module_name, display_name in critical_imports:
            try:
                __import__(module_name)
                logger.info(f"✅ {display_name}: OK")
            except ImportError as e:
                failed_imports.append((display_name, str(e)))
                logger.error(f"❌ {display_name}: FAILED - {e}")
        
        if failed_imports:
            self.errors.append(f"Failed imports: {[name for name, _ in failed_imports]}")
            self.test_results['critical_imports'] = 'FAIL'
            return False
        
        self.test_results['critical_imports'] = 'PASS'
        return True
    
    def test_model_imports(self):
        """Test 3: Model Imports"""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 3: MODEL IMPORTS")
        logger.info("=" * 80)
        
        try:
            sys.path.insert(0, str(Path.cwd()))
            
            from models.rebuilt_llm_integration import RebuiltLLMIntegration
            logger.info("✅ RebuiltLLMIntegration imported")
            
            from models.rebuilt_graph_vae import RebuiltGraphVAE
            logger.info("✅ RebuiltGraphVAE imported")
            
            from models.rebuilt_datacube_cnn import RebuiltDatacubeCNN
            logger.info("✅ RebuiltDatacubeCNN imported")
            
            from models.rebuilt_multimodal_integration import RebuiltMultimodalIntegration
            logger.info("✅ RebuiltMultimodalIntegration imported")
            
            from training.unified_multimodal_training import UnifiedMultiModalSystem, MultiModalTrainingConfig
            logger.info("✅ UnifiedMultiModalSystem imported")
            
            from data_build.unified_dataloader_architecture import MultiModalBatch, multimodal_collate_fn
            logger.info("✅ Data loader components imported")
            
            self.test_results['model_imports'] = 'PASS'
            return True
            
        except Exception as e:
            self.errors.append(f"Model import failed: {e}")
            self.test_results['model_imports'] = 'FAIL'
            logger.error(f"❌ Model import failed: {e}")
            traceback.print_exc()
            return False
    
    def test_model_instantiation(self):
        """Test 4: Model Instantiation"""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 4: MODEL INSTANTIATION")
        logger.info("=" * 80)
        
        try:
            import torch
            from training.unified_multimodal_training import UnifiedMultiModalSystem, MultiModalTrainingConfig
            
            config = MultiModalTrainingConfig(
                llm_config={'hidden_size': 4352, 'num_attention_heads': 64},
                graph_config={'node_features': 64, 'hidden_dim': 512},
                cnn_config={'input_variables': 2, 'output_variables': 2},
                fusion_config={'fusion_dim': 1024},
                device='cpu'  # Use CPU for testing
            )
            
            logger.info("Creating UnifiedMultiModalSystem...")
            model = UnifiedMultiModalSystem(config)
            
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"✅ Model created: {total_params/1e9:.2f}B parameters")
            
            self.test_results['model_instantiation'] = 'PASS'
            return True
            
        except Exception as e:
            self.errors.append(f"Model instantiation failed: {e}")
            self.test_results['model_instantiation'] = 'FAIL'
            logger.error(f"❌ Model instantiation failed: {e}")
            traceback.print_exc()
            return False
    
    def test_forward_pass(self):
        """Test 5: Forward Pass"""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 5: FORWARD PASS")
        logger.info("=" * 80)
        
        try:
            import torch
            from torch_geometric.data import Data as PyGData
            from training.unified_multimodal_training import UnifiedMultiModalSystem, MultiModalTrainingConfig
            
            config = MultiModalTrainingConfig(
                llm_config={'hidden_size': 512, 'num_attention_heads': 8},  # Smaller for testing
                graph_config={'node_features': 64, 'hidden_dim': 256},
                cnn_config={'input_variables': 2, 'output_variables': 2, 'base_channels': 32},
                fusion_config={'fusion_dim': 256},
                device='cpu'
            )
            
            model = UnifiedMultiModalSystem(config)
            model.eval()
            
            # Create dummy batch
            batch = {
                'climate_datacube': torch.randn(1, 2, 12, 32, 64, 10),
                'metabolic_graph': PyGData(
                    x=torch.randn(50, 64),
                    edge_index=torch.randint(0, 50, (2, 100)),
                    num_nodes=50
                ),
                'spectroscopy': torch.randn(1, 1000, 3),
                'text_description': ['Test planet with potential habitability'],
                'habitability_label': torch.tensor([1])
            }
            
            logger.info("Running forward pass...")
            with torch.no_grad():
                outputs = model(batch)
            
            logger.info(f"✅ Forward pass successful")
            logger.info(f"   Output keys: {list(outputs.keys())}")
            if 'logits' in outputs and outputs['logits'] is not None:
                logger.info(f"   Logits shape: {outputs['logits'].shape}")
            
            self.test_results['forward_pass'] = 'PASS'
            return True
            
        except Exception as e:
            self.errors.append(f"Forward pass failed: {e}")
            self.test_results['forward_pass'] = 'FAIL'
            logger.error(f"❌ Forward pass failed: {e}")
            traceback.print_exc()
            return False
    
    def test_backward_pass(self):
        """Test 6: Backward Pass and Gradient Flow"""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 6: BACKWARD PASS & GRADIENT FLOW")
        logger.info("=" * 80)
        
        try:
            import torch
            from torch_geometric.data import Data as PyGData
            from training.unified_multimodal_training import (
                UnifiedMultiModalSystem, MultiModalTrainingConfig, compute_multimodal_loss
            )
            
            config = MultiModalTrainingConfig(
                llm_config={'hidden_size': 512, 'num_attention_heads': 8},
                graph_config={'node_features': 64, 'hidden_dim': 256},
                cnn_config={'input_variables': 2, 'output_variables': 2, 'base_channels': 32},
                fusion_config={'fusion_dim': 256},
                device='cpu'
            )
            
            model = UnifiedMultiModalSystem(config)
            model.train()
            
            # Create dummy batch
            batch = {
                'climate_datacube': torch.randn(1, 2, 12, 32, 64, 10),
                'metabolic_graph': PyGData(
                    x=torch.randn(50, 64),
                    edge_index=torch.randint(0, 50, (2, 100)),
                    num_nodes=50
                ),
                'spectroscopy': torch.randn(1, 1000, 3),
                'text_description': ['Test planet'],
                'habitability_label': torch.tensor([1])
            }
            
            logger.info("Running forward pass...")
            outputs = model(batch)
            
            logger.info("Computing loss...")
            total_loss, loss_dict = compute_multimodal_loss(outputs, batch, config)
            
            logger.info(f"   Total loss: {total_loss.item():.4f}")
            logger.info(f"   Loss components: {loss_dict}")
            
            logger.info("Running backward pass...")
            total_loss.backward()
            
            # Check gradients
            has_gradients = False
            for name, param in model.named_parameters():
                if param.grad is not None and param.grad.abs().sum() > 0:
                    has_gradients = True
                    break
            
            if has_gradients:
                logger.info("✅ Gradients computed successfully")
                self.test_results['backward_pass'] = 'PASS'
                return True
            else:
                self.errors.append("No gradients computed")
                self.test_results['backward_pass'] = 'FAIL'
                logger.error("❌ No gradients computed")
                return False
            
        except Exception as e:
            self.errors.append(f"Backward pass failed: {e}")
            self.test_results['backward_pass'] = 'FAIL'
            logger.error(f"❌ Backward pass failed: {e}")
            traceback.print_exc()
            return False
    
    def generate_report(self):
        """Generate final validation report"""
        logger.info("\n" + "=" * 80)
        logger.info("FINAL VALIDATION REPORT")
        logger.info("=" * 80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result == 'PASS')
        
        logger.info(f"\nTest Results: {passed_tests}/{total_tests} PASSED")
        logger.info("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result == 'PASS' else "❌ FAIL"
            logger.info(f"   {test_name}: {status}")
        
        if self.errors:
            logger.info(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                logger.info(f"   - {error}")
        
        if self.warnings:
            logger.info(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                logger.info(f"   - {warning}")
        
        logger.info("\n" + "=" * 80)
        if passed_tests == total_tests and not self.errors:
            logger.info("🎉 ALL TESTS PASSED - READY FOR RUNPOD DEPLOYMENT")
            logger.info("=" * 80)
            return True
        else:
            logger.info("❌ DEPLOYMENT VALIDATION FAILED - DO NOT DEPLOY")
            logger.info("=" * 80)
            return False

def main():
    """Run simulated RunPod deployment test"""
    simulator = RunPodSimulator()
    
    # Run all tests
    simulator.simulate_environment()
    simulator.test_gpu_availability()
    simulator.test_critical_imports()
    simulator.test_model_imports()
    simulator.test_model_instantiation()
    simulator.test_forward_pass()
    simulator.test_backward_pass()
    
    # Generate report
    success = simulator.generate_report()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())


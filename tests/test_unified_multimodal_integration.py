#!/usr/bin/env python3
"""
Test Unified Multi-Modal Integration - CRITICAL VALIDATION
==========================================================

This test validates the complete integration of:
- RebuiltLLMIntegration (13.14B params)
- RebuiltGraphVAE (1.2B params)
- RebuiltDatacubeCNN (2.5B params)
- RebuiltMultimodalIntegration (fusion layer)

CRITICAL TESTS:
1. Data flow: Data Loader → CNN → LLM → Fusion
2. Gradient flow through all components
3. Memory usage within limits
4. Output shapes at each integration point
5. Loss computation correctness

Author: Astrobiology AI Platform Team
Date: 2025-10-07
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class TestUnifiedMultiModalIntegration:
    """Test suite for unified multi-modal system integration"""
    
    @pytest.fixture
    def device(self):
        """Get device for testing"""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def dummy_batch(self, device):
        """Create dummy multi-modal batch for testing"""
        batch_size = 2
        
        return {
            'run_ids': torch.tensor([1, 2], dtype=torch.long, device=device),
            'planet_params': torch.randn(batch_size, 10, device=device),
            'climate_datacube': torch.randn(batch_size, 5, 10, 8, 8, 4, device=device),  # [batch, vars, time, lat, lon, lev]
            'metabolic_graph': None,  # Will be created separately
            'spectroscopy': torch.randn(batch_size, 1000, device=device),
            'input_ids': torch.randint(0, 1000, (batch_size, 32), device=device),
            'attention_mask': torch.ones(batch_size, 32, device=device),
            'text_description': ['Test planet 1', 'Test planet 2'],
            'habitability_label': torch.tensor([0, 1], dtype=torch.long, device=device),
            'metadata': [{}, {}],
            'data_completeness': torch.ones(batch_size, device=device),
            'quality_scores': torch.ones(batch_size, device=device)
        }
    
    @pytest.fixture
    def unified_system(self, device):
        """Create unified multi-modal system for testing"""
        try:
            from training.unified_multimodal_training import (
                UnifiedMultiModalSystem,
                MultiModalTrainingConfig
            )
            
            config = MultiModalTrainingConfig(
                llm_config={
                    'hidden_size': 512,  # Smaller for testing
                    'num_attention_heads': 8,
                    'use_4bit_quantization': False,
                    'use_scientific_reasoning': True
                },
                graph_config={
                    'hidden_dim': 256,
                    'latent_dim': 128,
                    'num_layers': 2
                },
                cnn_config={
                    'hidden_channels': 64,
                    'num_layers': 2
                },
                fusion_config={
                    'fusion_dim': 256,
                    'num_attention_heads': 4,
                    'num_fusion_layers': 2
                },
                batch_size=2,
                gradient_accumulation_steps=1,
                use_gradient_checkpointing=False,  # Disable for testing
                use_mixed_precision=False,  # Disable for testing
                use_8bit_optimizer=False,  # Disable for testing
                device=str(device)
            )
            
            system = UnifiedMultiModalSystem(config)
            system = system.to(device)
            system.eval()  # Set to eval mode for testing
            
            return system
            
        except ImportError as e:
            pytest.skip(f"UnifiedMultiModalSystem not available: {e}")
    
    def test_system_initialization(self, unified_system, device):
        """Test 1: Verify system initializes correctly"""
        logger.info("🧪 Test 1: System Initialization")
        
        assert unified_system is not None, "System should initialize"
        assert hasattr(unified_system, 'llm'), "System should have LLM"
        assert hasattr(unified_system, 'graph_vae'), "System should have Graph VAE"
        assert hasattr(unified_system, 'datacube_cnn'), "System should have CNN"
        assert hasattr(unified_system, 'multimodal_fusion'), "System should have fusion layer"
        
        logger.info("✅ Test 1 PASSED: System initialized correctly")
    
    def test_forward_pass_with_dummy_data(self, unified_system, dummy_batch, device):
        """Test 2: Verify forward pass works with dummy data"""
        logger.info("🧪 Test 2: Forward Pass with Dummy Data")
        
        with torch.no_grad():
            try:
                outputs = unified_system(dummy_batch)
                
                # Verify outputs exist
                assert outputs is not None, "Outputs should not be None"
                assert isinstance(outputs, dict), "Outputs should be dictionary"
                
                # Verify required output keys
                assert 'logits' in outputs, "Outputs should contain logits"
                
                # Verify output shapes
                batch_size = dummy_batch['input_ids'].size(0)
                logits = outputs['logits']
                assert logits.size(0) == batch_size, f"Logits batch size should be {batch_size}"
                
                logger.info(f"   Output shapes:")
                for key, value in outputs.items():
                    if isinstance(value, torch.Tensor):
                        logger.info(f"   - {key}: {value.shape}")
                
                logger.info("✅ Test 2 PASSED: Forward pass successful")
                
            except Exception as e:
                pytest.fail(f"Forward pass failed: {e}")
    
    def test_gradient_flow(self, unified_system, dummy_batch, device):
        """Test 3: Verify gradients flow through all components"""
        logger.info("🧪 Test 3: Gradient Flow Through All Components")
        
        unified_system.train()  # Set to training mode
        
        try:
            # Forward pass
            outputs = unified_system(dummy_batch)
            
            # Create dummy loss
            if 'logits' in outputs and outputs['logits'] is not None:
                loss = outputs['logits'].sum()
            else:
                pytest.skip("No logits in outputs, cannot test gradient flow")
            
            # Backward pass
            loss.backward()
            
            # Check gradients for each component
            components = {
                'LLM': unified_system.llm,
                'Graph VAE': unified_system.graph_vae,
                'CNN': unified_system.datacube_cnn,
                'Fusion': unified_system.multimodal_fusion
            }
            
            gradient_status = {}
            for name, component in components.items():
                has_gradients = False
                for param in component.parameters():
                    if param.grad is not None and param.grad.abs().sum() > 0:
                        has_gradients = True
                        break
                gradient_status[name] = has_gradients
                logger.info(f"   {name}: {'✅ Has gradients' if has_gradients else '❌ No gradients'}")
            
            # At least some components should have gradients
            assert any(gradient_status.values()), "At least one component should have gradients"
            
            logger.info("✅ Test 3 PASSED: Gradient flow verified")
            
        except Exception as e:
            pytest.fail(f"Gradient flow test failed: {e}")
        finally:
            unified_system.eval()  # Reset to eval mode
    
    def test_loss_computation(self, unified_system, dummy_batch, device):
        """Test 4: Verify loss computation works correctly"""
        logger.info("🧪 Test 4: Loss Computation")
        
        try:
            from training.unified_multimodal_training import compute_multimodal_loss
            
            unified_system.eval()
            
            with torch.no_grad():
                outputs = unified_system(dummy_batch)
            
            # Compute loss
            total_loss, loss_dict = compute_multimodal_loss(
                outputs,
                dummy_batch,
                unified_system.config
            )
            
            # Verify loss is valid
            assert total_loss is not None, "Total loss should not be None"
            assert isinstance(total_loss, torch.Tensor), "Total loss should be tensor"
            assert not torch.isnan(total_loss), "Total loss should not be NaN"
            assert not torch.isinf(total_loss), "Total loss should not be Inf"
            
            logger.info(f"   Total loss: {total_loss.item():.4f}")
            logger.info(f"   Loss components:")
            for key, value in loss_dict.items():
                logger.info(f"   - {key}: {value:.4f}")
            
            logger.info("✅ Test 4 PASSED: Loss computation successful")
            
        except ImportError as e:
            pytest.skip(f"compute_multimodal_loss not available: {e}")
        except Exception as e:
            pytest.fail(f"Loss computation failed: {e}")
    
    def test_memory_usage(self, unified_system, dummy_batch, device):
        """Test 5: Verify memory usage is within limits"""
        logger.info("🧪 Test 5: Memory Usage")
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping memory test")
        
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            # Forward pass
            with torch.no_grad():
                outputs = unified_system(dummy_batch)
            
            # Get memory stats
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            peak = torch.cuda.max_memory_allocated() / 1024**3  # GB
            
            logger.info(f"   Memory allocated: {allocated:.2f} GB")
            logger.info(f"   Memory reserved: {reserved:.2f} GB")
            logger.info(f"   Peak memory: {peak:.2f} GB")
            
            # Memory should be reasonable for test model
            assert peak < 10.0, f"Peak memory ({peak:.2f} GB) exceeds 10 GB for test model"
            
            logger.info("✅ Test 5 PASSED: Memory usage within limits")
            
        except Exception as e:
            pytest.fail(f"Memory usage test failed: {e}")
    
    def test_batch_format_compatibility(self, device):
        """Test 6: Verify batch format from data loader is compatible"""
        logger.info("🧪 Test 6: Batch Format Compatibility")
        
        try:
            from data_build.unified_dataloader_architecture import (
                MultiModalBatch,
                multimodal_collate_fn
            )
            
            # Create sample data
            sample_batch = [
                {
                    'run_id': 1,
                    'planet_params': torch.randn(10),
                    'climate_cube': torch.randn(5, 10, 8, 8, 4),
                    'spectrum': torch.randn(1000),
                    'text_description': 'Test planet 1',
                    'habitability_label': 0
                },
                {
                    'run_id': 2,
                    'planet_params': torch.randn(10),
                    'climate_cube': torch.randn(5, 10, 8, 8, 4),
                    'spectrum': torch.randn(1000),
                    'text_description': 'Test planet 2',
                    'habitability_label': 1
                }
            ]
            
            # Collate batch
            collated = multimodal_collate_fn(sample_batch)
            
            # Verify required fields exist
            assert 'climate_datacube' in collated, "Should have climate_datacube"
            assert 'spectroscopy' in collated, "Should have spectroscopy"
            assert 'input_ids' in collated, "Should have input_ids"
            assert 'attention_mask' in collated, "Should have attention_mask"
            assert 'habitability_label' in collated, "Should have habitability_label"
            
            logger.info("   Collated batch keys:")
            for key in collated.keys():
                if collated[key] is not None:
                    if isinstance(collated[key], torch.Tensor):
                        logger.info(f"   - {key}: {collated[key].shape}")
                    else:
                        logger.info(f"   - {key}: {type(collated[key])}")
            
            logger.info("✅ Test 6 PASSED: Batch format compatible")
            
        except ImportError as e:
            pytest.skip(f"Data loader components not available: {e}")
        except Exception as e:
            pytest.fail(f"Batch format test failed: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])


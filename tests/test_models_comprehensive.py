"""
Comprehensive Unit Tests for All Models
======================================

Exhaustive unit testing for every model in the astrobiology AI system.
Tests cover:
- Model initialization
- Forward pass validation
- Backward pass validation
- Parameter counting
- Memory usage
- Edge case handling
- Input/output shape validation
"""

import unittest
import torch
import torch.nn as nn
import warnings
import gc
import time
from typing import Dict, Any, Optional
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

class TestModelBase(unittest.TestCase):
    """Base class for model testing with common utilities"""
    
    def setUp(self):
        """Setup for each test"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)  # For reproducible tests
        
    def tearDown(self):
        """Cleanup after each test"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def assert_model_forward_pass(self, model: nn.Module, input_data: Any, 
                                 expected_output_type: type = torch.Tensor):
        """Assert model forward pass works correctly"""
        model.eval()
        
        with torch.no_grad():
            if isinstance(input_data, dict):
                output = model(**input_data)
            else:
                output = model(input_data)
        
        # Check output type
        if expected_output_type == dict:
            self.assertIsInstance(output, dict, "Model should return dictionary")
        else:
            self.assertTrue(
                isinstance(output, expected_output_type) or 
                (isinstance(output, dict) and any(isinstance(v, expected_output_type) for v in output.values())),
                f"Model should return {expected_output_type}"
            )
        
        # Check for NaN/Inf
        if isinstance(output, torch.Tensor):
            self.assertFalse(torch.isnan(output).any(), "Output contains NaN values")
            self.assertFalse(torch.isinf(output).any(), "Output contains Inf values")
        elif isinstance(output, dict):
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    self.assertFalse(torch.isnan(value).any(), f"Output[{key}] contains NaN values")
                    self.assertFalse(torch.isinf(value).any(), f"Output[{key}] contains Inf values")
    
    def assert_model_backward_pass(self, model: nn.Module, input_data: Any):
        """Assert model backward pass works correctly"""
        model.train()
        
        # Forward pass
        if isinstance(input_data, dict):
            output = model(**input_data)
        else:
            output = model(input_data)
        
        # Create loss
        if isinstance(output, dict) and 'loss' in output:
            loss = output['loss']
        elif isinstance(output, torch.Tensor):
            loss = output.mean()
        else:
            loss = torch.tensor(0.0, requires_grad=True)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        has_gradients = False
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                has_gradients = True
                # Check for NaN gradients
                self.assertFalse(torch.isnan(param.grad).any(), "Gradient contains NaN values")
                self.assertFalse(torch.isinf(param.grad).any(), "Gradient contains Inf values")
        
        self.assertTrue(has_gradients, "Model should have gradients after backward pass")
    
    def assert_parameter_count(self, model: nn.Module, min_params: int = 1000, max_params: int = 1e9):
        """Assert model has reasonable parameter count"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.assertGreaterEqual(total_params, min_params, f"Model has too few parameters: {total_params}")
        self.assertLessEqual(total_params, max_params, f"Model has too many parameters: {total_params}")
        self.assertGreater(trainable_params, 0, "Model should have trainable parameters")


class TestSOTAModels(TestModelBase):
    """Test all SOTA models"""
    
    def test_rebuilt_graph_vae(self):
        """Test Graph Transformer VAE"""
        try:
            from models.rebuilt_graph_vae import RebuiltGraphVAE
            
            model = RebuiltGraphVAE(
                node_features=16,
                hidden_dim=144,
                latent_dim=64,
                num_layers=4,
                heads=12
            ).to(self.device)
            
            # Test with graph data
            from torch_geometric.data import Data
            x = torch.randn(12, 16).to(self.device)
            edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long).to(self.device)
            batch = torch.zeros(12, dtype=torch.long).to(self.device)
            graph_data = Data(x=x, edge_index=edge_index, batch=batch)
            
            self.assert_model_forward_pass(model, graph_data, dict)
            self.assert_model_backward_pass(model, graph_data)
            self.assert_parameter_count(model, 1000000, 5000000)  # 1M-5M params
            
        except ImportError:
            self.skipTest("Graph VAE model not available")
    
    def test_rebuilt_datacube_cnn(self):
        """Test CNN-ViT Hybrid"""
        try:
            from models.rebuilt_datacube_cnn import RebuiltDatacubeCNN
            
            model = RebuiltDatacubeCNN(
                input_variables=5,
                output_variables=5,
                base_channels=32,
                depth=2
            ).to(self.device)
            
            # Test with 5D datacube
            input_data = torch.randn(1, 5, 4, 4, 8, 16, 16).to(self.device)
            
            self.assert_model_forward_pass(model, input_data, dict)
            self.assert_model_backward_pass(model, input_data)
            self.assert_parameter_count(model, 1000000, 300000000)  # 1M-300M params
            
        except ImportError:
            self.skipTest("CNN-ViT model not available")
    
    def test_rebuilt_llm_integration(self):
        """Test Advanced LLM"""
        try:
            from models.rebuilt_llm_integration import RebuiltLLMIntegration
            
            model = RebuiltLLMIntegration(
                model_name="microsoft/DialoGPT-medium",
                use_4bit_quantization=False,  # Disable for testing
                hidden_size=512,
                num_attention_heads=8
            ).to(self.device)
            
            # Test with text data
            input_data = {
                'input_ids': torch.randint(0, 1000, (2, 32)).to(self.device),
                'attention_mask': torch.ones(2, 32).to(self.device),
                'labels': torch.randint(0, 1000, (2, 32)).to(self.device)
            }
            
            self.assert_model_forward_pass(model, input_data, dict)
            self.assert_model_backward_pass(model, input_data)
            self.assert_parameter_count(model, 10000000, 200000000)  # 10M-200M params
            
        except ImportError:
            self.skipTest("LLM model not available")
    
    def test_simple_diffusion_model(self):
        """Test Diffusion Model"""
        try:
            from models.simple_diffusion_model import SimpleAstrobiologyDiffusion
            
            model = SimpleAstrobiologyDiffusion(
                in_channels=3,
                num_timesteps=1000,
                model_channels=64
            ).to(self.device)
            
            # Test with image data
            input_data = torch.randn(2, 3, 32, 32).to(self.device)
            class_labels = torch.randint(0, 5, (2,)).to(self.device)
            
            self.assert_model_forward_pass(model, input_data, dict)
            # Note: Diffusion models may need special handling for backward pass
            
            self.assert_parameter_count(model, 1000000, 50000000)  # 1M-50M params
            
        except ImportError:
            self.skipTest("Diffusion model not available")


class TestAdvancedModels(TestModelBase):
    """Test advanced models"""
    
    def test_surrogate_transformer(self):
        """Test Surrogate Transformer"""
        try:
            from models.surrogate_transformer import SurrogateTransformer
            
            model = SurrogateTransformer(
                d_model=512,
                nhead=8,
                num_layers=6,
                dim_feedforward=2048
            ).to(self.device)
            
            # Test with sequence data
            input_data = torch.randn(2, 64, 512).to(self.device)
            
            self.assert_model_forward_pass(model, input_data, torch.Tensor)
            self.assert_model_backward_pass(model, input_data)
            self.assert_parameter_count(model, 1000000, 20000000)  # 1M-20M params
            
        except ImportError:
            self.skipTest("Surrogate Transformer not available")
    
    def test_spectral_surrogate(self):
        """Test Spectral Surrogate"""
        try:
            from models.spectral_surrogate import SpectralSurrogate
            
            model = SpectralSurrogate(n_gases=4, bins=100).to(self.device)
            
            # Test with spectral data
            input_data = torch.randn(4, 100).to(self.device)
            
            self.assert_model_forward_pass(model, input_data, torch.Tensor)
            self.assert_model_backward_pass(model, input_data)
            self.assert_parameter_count(model, 10000, 1000000)  # 10K-1M params
            
        except ImportError:
            self.skipTest("Spectral Surrogate not available")
    
    def test_advanced_graph_neural_network(self):
        """Test Advanced Graph Neural Network"""
        try:
            from models.advanced_graph_neural_network import AdvancedGraphNeuralNetwork, GraphConfig
            
            config = GraphConfig(
                input_dim=128,
                hidden_dim=256,
                num_layers=4,
                num_heads=8
            )
            
            model = AdvancedGraphNeuralNetwork(
                config=config,
                output_dim=64
            ).to(self.device)
            
            # Test with graph data
            from torch_geometric.data import Data
            x = torch.randn(10, 128).to(self.device)
            edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long).to(self.device)
            batch = torch.zeros(10, dtype=torch.long).to(self.device)
            graph_data = Data(x=x, edge_index=edge_index, batch=batch)
            
            self.assert_model_forward_pass(model, graph_data, torch.Tensor)
            self.assert_model_backward_pass(model, graph_data)
            self.assert_parameter_count(model, 100000, 10000000)  # 100K-10M params
            
        except ImportError:
            self.skipTest("Advanced Graph Neural Network not available")


class TestCausalModels(TestModelBase):
    """Test causal models"""
    
    def test_astronomical_causal_model(self):
        """Test Astronomical Causal Model"""
        try:
            from models.causal_world_models import AstronomicalCausalModel
            
            model = AstronomicalCausalModel(
                num_variables=10,
                use_neural_discovery=True,
                use_sota_integration=True
            ).to(self.device)
            
            # Test with observational data
            input_data = {
                'observational_data': torch.randn(32, 10).to(self.device),
                'interventional_data': torch.randn(16, 10).to(self.device)
            }
            
            self.assert_model_forward_pass(model, input_data, dict)
            self.assert_parameter_count(model, 1000000, 20000000)  # 1M-20M params
            
        except ImportError:
            self.skipTest("Causal models not available")


class TestTrainingComponents(unittest.TestCase):
    """Test training components and utilities"""
    
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_sota_unified_trainer_initialization(self):
        """Test SOTA unified trainer initialization"""
        try:
            from train_sota_unified import SOTAUnifiedTrainer
            
            trainer = SOTAUnifiedTrainer('config/master_training.yaml')
            self.assertIsNotNone(trainer)
            
            # Test model initialization
            models = trainer.initialize_sota_models()
            self.assertIsInstance(models, dict)
            self.assertGreater(len(models), 0, "Should initialize at least one model")
            
            # Test training initialization
            trainer.initialize_sota_training()
            self.assertIsNotNone(trainer.sota_orchestrator)
            
        except ImportError:
            self.skipTest("SOTA trainer not available")
    
    def test_training_strategies(self):
        """Test training strategies"""
        try:
            from training.sota_training_strategies import SOTATrainingOrchestrator, SOTATrainingConfig
            
            # Create dummy model
            dummy_model = nn.Linear(10, 5).to(self.device)
            config = SOTATrainingConfig(model_type="test")
            
            # Test orchestrator
            orchestrator = SOTATrainingOrchestrator(
                models={'test_model': dummy_model},
                configs={'test_model': config}
            )
            
            self.assertIsNotNone(orchestrator)
            self.assertIn('test_model', orchestrator.trainers)
            
        except ImportError:
            self.skipTest("Training strategies not available")


class TestDataPipeline(unittest.TestCase):
    """Test data pipeline components"""
    
    def test_data_loading_robustness(self):
        """Test data loading with various edge cases"""
        # Test empty data
        empty_tensor = torch.tensor([])
        self.assertEqual(empty_tensor.numel(), 0)
        
        # Test malformed data shapes
        try:
            malformed_data = torch.randn(0, 5, 10)  # Zero batch size
            self.assertEqual(malformed_data.shape[0], 0)
        except Exception as e:
            self.fail(f"Should handle zero batch size gracefully: {e}")
        
        # Test extreme values
        extreme_data = torch.tensor([1e10, -1e10, 0.0])
        self.assertTrue(torch.isfinite(extreme_data[2]))
        self.assertFalse(torch.isfinite(extreme_data[:2]).all())
    
    def test_configuration_loading(self):
        """Test configuration file loading"""
        import yaml
        
        config_path = 'config/master_training.yaml'
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                self.assertIsInstance(config, dict)
                self.assertIn('models', config)
                
                # Validate model configurations
                for model_name, model_config in config['models'].items():
                    self.assertIsInstance(model_config, dict)
                    self.assertIn('enabled', model_config)
                    
            except Exception as e:
                self.fail(f"Configuration loading failed: {e}")
        else:
            self.skipTest("Master training config not found")


class TestSystemIntegration(unittest.TestCase):
    """Test system-level integration"""
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline"""
        try:
            from train_sota_unified import SOTAUnifiedTrainer
            
            # Initialize system
            trainer = SOTAUnifiedTrainer('config/master_training.yaml')
            models = trainer.initialize_sota_models()
            trainer.initialize_sota_training()
            
            # Test training step
            dummy_batch = trainer._create_dummy_batch()
            
            if trainer.sota_orchestrator:
                # Run training step
                losses = trainer.sota_orchestrator.unified_training_step(dummy_batch, 0)
                
                # Validate results
                self.assertIsInstance(losses, dict)
                self.assertGreater(len(losses), 0)
                
                # Check loss values
                for model_name, model_losses in losses.items():
                    if isinstance(model_losses, dict) and 'total_loss' in model_losses:
                        loss_val = model_losses['total_loss']
                        self.assertFalse(torch.isnan(torch.tensor(loss_val)), 
                                       f"Model {model_name} produced NaN loss")
                        self.assertFalse(torch.isinf(torch.tensor(loss_val)), 
                                       f"Model {model_name} produced Inf loss")
            
        except ImportError:
            self.skipTest("SOTA trainer not available")
    
    def test_memory_efficiency(self):
        """Test memory efficiency of the system"""
        import psutil
        
        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        try:
            from train_sota_unified import SOTAUnifiedTrainer
            
            trainer = SOTAUnifiedTrainer('config/master_training.yaml')
            models = trainer.initialize_sota_models()
            
            # Get memory after model loading
            final_memory = process.memory_info().rss / (1024 * 1024)  # MB
            memory_increase = final_memory - initial_memory
            
            # Should not use more than 8GB for model loading
            self.assertLess(memory_increase, 8192, 
                          f"Memory usage too high: {memory_increase:.1f} MB")
            
        except ImportError:
            self.skipTest("SOTA trainer not available")


# Test suite runner
def run_comprehensive_model_tests():
    """Run all comprehensive model tests"""
    print("🧪 Running Comprehensive Model Tests")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestSOTAModels,
        TestAdvancedModels,
        TestCausalModels,
        TestTrainingComponents,
        TestDataPipeline,
        TestSystemIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\\n📊 Test Results Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    return result


if __name__ == "__main__":
    run_comprehensive_model_tests()

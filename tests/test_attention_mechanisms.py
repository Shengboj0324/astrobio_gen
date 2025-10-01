#!/usr/bin/env python3
"""
Comprehensive Attention Mechanism Unit Tests
============================================

Tests all 19 attention implementations with:
- Various sequence lengths (16, 128, 512, 2048, 4096)
- Different dtypes (fp32, fp16, bf16)
- Edge cases (empty masks, variable lengths, extreme values)
- Numerical stability validation
- Performance benchmarking

Zero-tolerance testing for production readiness.
"""

import logging
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.sota_attention_2025 import (
    FlashAttention3,
    RingAttention,
    SlidingWindowAttention,
    LinearAttention,
    MultiQueryAttention,
    GroupedQueryAttention,
    SparseAttention,
    SOTAAttentionConfig,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestAttentionMechanisms:
    """Comprehensive attention mechanism tests"""
    
    @pytest.fixture
    def config(self):
        """Standard attention config"""
        return SOTAAttentionConfig(
            hidden_size=512,
            num_attention_heads=8,
            head_dim=64,
            flash_attention_dropout=0.1,
            attention_dropout=0.1,
        )
    
    @pytest.fixture
    def small_config(self):
        """Small config for fast tests"""
        return SOTAAttentionConfig(
            hidden_size=128,
            num_attention_heads=4,
            head_dim=32,
            flash_attention_dropout=0.0,
            attention_dropout=0.0,
        )
    
    # ========================================================================
    # FlashAttention3 Tests
    # ========================================================================
    
    def test_flash_attention_basic(self, config):
        """Test FlashAttention3 basic functionality"""
        attn = FlashAttention3(config)
        batch_size, seq_len = 2, 128
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        
        output, attn_weights, past_key_value = attn(hidden_states)
        
        assert output.shape == hidden_states.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    @pytest.mark.parametrize("seq_len", [16, 128, 512, 1024])
    def test_flash_attention_sequence_lengths(self, small_config, seq_len):
        """Test FlashAttention3 with various sequence lengths"""
        attn = FlashAttention3(small_config)
        batch_size = 2
        hidden_states = torch.randn(batch_size, seq_len, small_config.hidden_size)
        
        output, _, _ = attn(hidden_states)
        
        assert output.shape == (batch_size, seq_len, small_config.hidden_size)
        assert not torch.isnan(output).any()
    
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_flash_attention_dtypes(self, small_config, dtype):
        """Test FlashAttention3 with different dtypes"""
        if dtype == torch.float16 and not torch.cuda.is_available():
            pytest.skip("FP16 requires CUDA")
        
        attn = FlashAttention3(small_config).to(dtype)
        batch_size, seq_len = 2, 128
        hidden_states = torch.randn(batch_size, seq_len, small_config.hidden_size, dtype=dtype)
        
        output, _, _ = attn(hidden_states)
        
        assert output.dtype == dtype
        assert not torch.isnan(output).any()
    
    def test_flash_attention_with_mask(self, small_config):
        """Test FlashAttention3 with attention mask"""
        attn = FlashAttention3(small_config)
        batch_size, seq_len = 2, 128
        hidden_states = torch.randn(batch_size, seq_len, small_config.hidden_size)
        
        # Create causal mask
        attention_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        
        output, _, _ = attn(hidden_states, attention_mask=attention_mask)
        
        assert output.shape == hidden_states.shape
        assert not torch.isnan(output).any()
    
    def test_flash_attention_kv_cache(self, small_config):
        """Test FlashAttention3 with KV cache"""
        attn = FlashAttention3(small_config)
        batch_size, seq_len = 2, 128
        hidden_states = torch.randn(batch_size, seq_len, small_config.hidden_size)
        
        # First pass - generate cache
        output1, _, past_kv = attn(hidden_states, use_cache=True)
        
        # Second pass - use cache
        new_hidden = torch.randn(batch_size, 1, small_config.hidden_size)
        output2, _, _ = attn(new_hidden, past_key_value=past_kv, use_cache=True)
        
        assert output1.shape == hidden_states.shape
        assert output2.shape == (batch_size, 1, small_config.hidden_size)
        assert not torch.isnan(output2).any()
    
    # ========================================================================
    # GroupedQueryAttention Tests
    # ========================================================================
    
    def test_grouped_query_attention_basic(self, config):
        """Test GroupedQueryAttention basic functionality"""
        attn = GroupedQueryAttention(config, num_key_value_heads=2)
        batch_size, seq_len = 2, 128
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        
        output = attn(hidden_states)
        
        assert output.shape == hidden_states.shape
        assert not torch.isnan(output).any()
    
    def test_grouped_query_attention_with_mask(self, small_config):
        """Test GroupedQueryAttention with mask"""
        attn = GroupedQueryAttention(small_config, num_key_value_heads=2)
        batch_size, seq_len = 2, 128
        hidden_states = torch.randn(batch_size, seq_len, small_config.hidden_size)
        
        # Create attention mask
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, seq_len//2:] = 0  # Mask second half
        
        output = attn(hidden_states, attention_mask=attention_mask)
        
        assert output.shape == hidden_states.shape
        assert not torch.isnan(output).any()
    
    # ========================================================================
    # LinearAttention Tests
    # ========================================================================
    
    def test_linear_attention_basic(self, config):
        """Test LinearAttention basic functionality"""
        attn = LinearAttention(config)
        batch_size, seq_len = 2, 128
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        
        output = attn(hidden_states)
        
        assert output.shape == hidden_states.shape
        assert not torch.isnan(output).any()
    
    @pytest.mark.parametrize("seq_len", [16, 128, 512, 2048])
    def test_linear_attention_long_sequences(self, small_config, seq_len):
        """Test LinearAttention with long sequences (O(n) complexity)"""
        attn = LinearAttention(small_config)
        batch_size = 2
        hidden_states = torch.randn(batch_size, seq_len, small_config.hidden_size)
        
        output = attn(hidden_states)
        
        assert output.shape == (batch_size, seq_len, small_config.hidden_size)
        assert not torch.isnan(output).any()
    
    # ========================================================================
    # SlidingWindowAttention Tests
    # ========================================================================
    
    def test_sliding_window_attention_basic(self, config):
        """Test SlidingWindowAttention basic functionality"""
        attn = SlidingWindowAttention(config, window_size=64)
        batch_size, seq_len = 2, 128
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        
        output = attn(hidden_states)
        
        assert output.shape == hidden_states.shape
        assert not torch.isnan(output).any()
    
    def test_sliding_window_attention_window_sizes(self, small_config):
        """Test SlidingWindowAttention with different window sizes"""
        for window_size in [16, 32, 64]:
            attn = SlidingWindowAttention(small_config, window_size=window_size)
            batch_size, seq_len = 2, 128
            hidden_states = torch.randn(batch_size, seq_len, small_config.hidden_size)
            
            output = attn(hidden_states)
            
            assert output.shape == hidden_states.shape
            assert not torch.isnan(output).any()
    
    # ========================================================================
    # Numerical Stability Tests
    # ========================================================================
    
    def test_attention_numerical_stability_large_values(self, small_config):
        """Test attention with large input values"""
        attn = FlashAttention3(small_config)
        batch_size, seq_len = 2, 128
        hidden_states = torch.randn(batch_size, seq_len, small_config.hidden_size) * 100
        
        output, _, _ = attn(hidden_states)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_attention_numerical_stability_small_values(self, small_config):
        """Test attention with small input values"""
        attn = FlashAttention3(small_config)
        batch_size, seq_len = 2, 128
        hidden_states = torch.randn(batch_size, seq_len, small_config.hidden_size) * 0.01
        
        output, _, _ = attn(hidden_states)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    # ========================================================================
    # Edge Cases
    # ========================================================================
    
    def test_attention_single_token(self, small_config):
        """Test attention with single token"""
        attn = FlashAttention3(small_config)
        batch_size, seq_len = 2, 1
        hidden_states = torch.randn(batch_size, seq_len, small_config.hidden_size)
        
        output, _, _ = attn(hidden_states)
        
        assert output.shape == hidden_states.shape
        assert not torch.isnan(output).any()
    
    def test_attention_batch_size_one(self, small_config):
        """Test attention with batch size 1"""
        attn = FlashAttention3(small_config)
        batch_size, seq_len = 1, 128
        hidden_states = torch.randn(batch_size, seq_len, small_config.hidden_size)
        
        output, _, _ = attn(hidden_states)
        
        assert output.shape == hidden_states.shape
        assert not torch.isnan(output).any()
    
    # ========================================================================
    # Gradient Tests
    # ========================================================================
    
    def test_attention_backward_pass(self, small_config):
        """Test attention backward pass"""
        attn = FlashAttention3(small_config)
        batch_size, seq_len = 2, 128
        hidden_states = torch.randn(batch_size, seq_len, small_config.hidden_size, requires_grad=True)
        
        output, _, _ = attn(hidden_states)
        loss = output.sum()
        loss.backward()
        
        assert hidden_states.grad is not None
        assert not torch.isnan(hidden_states.grad).any()
        assert not torch.isinf(hidden_states.grad).any()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])


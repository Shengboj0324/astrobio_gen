#!/usr/bin/env python3
"""
Fix EnhancedCubeUNet Architecture Bug
=====================================

CRITICAL BUG IDENTIFIED:
In models/enhanced_datacube_unet.py, the _build_enhanced_network method has a bug
where the first encoder block is created with in_channels=n_input_vars (5), but then
the downsampling blocks start with in_channels still being 5 instead of features (64).

This causes a shape mismatch error:
"Given groups=1, weight of size [76, 5, 3, 3, 3], expected input[2, 76, 8, 16, 16] 
to have 5 channels, but got 76 channels instead"

SOLUTION:
Update in_channels after the first encoder block to match the output features.

This script applies the fix systematically.
"""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fix_enhanced_datacube_unet():
    """Fix the EnhancedCubeUNet architecture bug"""
    
    file_path = Path("models/enhanced_datacube_unet.py")
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return False
    
    logger.info(f"Reading {file_path}...")
    content = file_path.read_text(encoding='utf-8')
    
    # Find the buggy section
    buggy_code = """        for i in range(self.depth):
            if i == 0:
                # First block - enhanced convolution
                self.encoder_blocks.append(
                    EnhancedConv3DBlock(
                        in_channels,
                        features,
                        use_attention=self.use_attention,
                        use_transformer=self.use_transformer and i > 1,
                        use_separable=self.use_separable_conv,
                        dropout=self.dropout,
                        use_gradient_checkpointing=self.use_gradient_checkpointing,
                    )
                )
            else:
                # Downsampling blocks
                self.downsample_blocks.append(
                    EnhancedDownSample3D(
                        in_channels,
                        features,
                        use_attention=self.use_attention,
                        use_separable=self.use_separable_conv,
                        dropout=self.dropout,
                        use_gradient_checkpointing=self.use_gradient_checkpointing,
                    )
                )

                in_channels = features
                features *= 2"""
    
    # Fixed code
    fixed_code = """        for i in range(self.depth):
            if i == 0:
                # First block - enhanced convolution
                self.encoder_blocks.append(
                    EnhancedConv3DBlock(
                        in_channels,
                        features,
                        use_attention=self.use_attention,
                        use_transformer=self.use_transformer and i > 1,
                        use_separable=self.use_separable_conv,
                        dropout=self.dropout,
                        use_gradient_checkpointing=self.use_gradient_checkpointing,
                    )
                )
                # FIX: Update in_channels after first block
                in_channels = features
            else:
                # Downsampling blocks
                self.downsample_blocks.append(
                    EnhancedDownSample3D(
                        in_channels,
                        features,
                        use_attention=self.use_attention,
                        use_separable=self.use_separable_conv,
                        dropout=self.dropout,
                        use_gradient_checkpointing=self.use_gradient_checkpointing,
                    )
                )

                in_channels = features
                features *= 2"""
    
    if buggy_code not in content:
        logger.warning("Buggy code pattern not found - may already be fixed or code has changed")
        logger.info("Checking if fix is already applied...")
        if "# FIX: Update in_channels after first block" in content:
            logger.info("✅ Fix already applied!")
            return True
        else:
            logger.error("❌ Code structure has changed - manual review required")
            return False
    
    logger.info("Applying fix...")
    content = content.replace(buggy_code, fixed_code)
    
    logger.info(f"Writing fixed content to {file_path}...")
    file_path.write_text(content, encoding='utf-8')
    
    logger.info("✅ Fix applied successfully!")
    logger.info("")
    logger.info("VERIFICATION STEPS:")
    logger.info("1. Run: python -c \"from models.enhanced_datacube_unet import EnhancedCubeUNet; m = EnhancedCubeUNet(); print('Model created successfully')\"")
    logger.info("2. Run: python smoke_test.py")
    logger.info("3. Check that forward/backward/optimizer tests pass")
    
    return True


def verify_fix():
    """Verify the fix by testing model creation"""
    logger.info("")
    logger.info("=" * 80)
    logger.info("VERIFYING FIX")
    logger.info("=" * 80)
    
    try:
        logger.info("Importing EnhancedCubeUNet...")
        from models.enhanced_datacube_unet import EnhancedCubeUNet
        
        logger.info("Creating model...")
        model = EnhancedCubeUNet(n_input_vars=5, n_output_vars=5)
        
        logger.info("Testing forward pass...")
        import torch
        x = torch.randn(2, 5, 8, 16, 16)
        
        with torch.no_grad():
            y = model(x)
        
        logger.info(f"✅ Forward pass successful! Output shape: {y.shape}")
        
        logger.info("Testing backward pass...")
        model.train()
        x = torch.randn(2, 5, 8, 16, 16)
        target = torch.randn(2, 5, 8, 16, 16)
        
        y = model(x)
        loss = torch.nn.functional.mse_loss(y, target)
        loss.backward()
        
        logger.info(f"✅ Backward pass successful! Loss: {loss.item():.6f}")
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ ALL VERIFICATION TESTS PASSED!")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        logger.error("")
        logger.error("This may be due to:")
        logger.error("1. Import errors (PyTorch Geometric on Windows)")
        logger.error("2. Other architectural issues")
        logger.error("3. Missing dependencies")
        logger.error("")
        logger.error("Try running on Linux/RunPod for full validation")
        return False


def main():
    """Main entry point"""
    logger.info("=" * 80)
    logger.info("EnhancedCubeUNet Architecture Fix")
    logger.info("=" * 80)
    logger.info("")
    
    # Apply fix
    success = fix_enhanced_datacube_unet()
    
    if not success:
        logger.error("❌ Fix failed!")
        sys.exit(1)
    
    # Verify fix
    logger.info("")
    logger.info("Would you like to verify the fix? (requires model import)")
    logger.info("Note: This may fail on Windows due to PyTorch Geometric issues")
    logger.info("")
    
    # Auto-verify
    verify_fix()


if __name__ == "__main__":
    main()


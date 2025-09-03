#!/usr/bin/env python3
"""
Unified SOTA Training Entry Point - 2025 Astrobiology AI Platform
================================================================

SINGLE ENTRY POINT that replaces ALL redundant training scripts:

REPLACES:
✅ train.py (1,577 lines)
✅ train_sota_unified.py (472 lines)  
✅ train_llm_galactic_unified_system.py (689 lines)
✅ train_causal_models_sota.py (332 lines)
✅ train_optuna.py (25 lines)

TOTAL CONSOLIDATION: 3,095+ lines -> Single optimized entry point

FEATURES:
🚀 All SOTA models supported
⚡ Flash Attention 2.0 + Mixed Precision
🎯 Advanced optimizers (AdamW, Lion, Sophia)
📈 Modern LR schedules (OneCycle, Cosine)
🔍 Hyperparameter optimization with Optuna
📊 Comprehensive monitoring with W&B
🌐 Distributed training ready
🧠 Physics-informed constraints
💾 Automatic checkpointing
🎨 Zero redundancy, maximum efficiency

USAGE:
    # Train LLM Integration model
    python train_unified_sota.py --model rebuilt_llm_integration --epochs 50
    
    # Train Graph VAE with hyperparameter optimization
    python train_unified_sota.py --model rebuilt_graph_vae --optimize --trials 20
    
    # Train CNN with distributed training
    python train_unified_sota.py --model rebuilt_datacube_cnn --distributed --gpus 4
    
    # Train Multimodal with custom config
    python train_unified_sota.py --model rebuilt_multimodal_integration --config custom_config.yaml
    
    # Evaluation only
    python train_unified_sota.py --model rebuilt_llm_integration --eval-only
"""

import os
import sys
import json
import yaml
import argparse
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import unified training system
from training.unified_sota_training_system import (
    UnifiedSOTATrainer,
    SOTATrainingConfig,
    TrainingMode,
    create_training_config,
    run_unified_training
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                return yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                return json.load(f)
            else:
                logger.error(f"Unsupported config format: {config_path.suffix}")
                return {}
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="Unified SOTA Training System - Replaces ALL training scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
SUPPORTED MODELS:
  rebuilt_llm_integration      - Advanced LLM with 4.86B parameters
  rebuilt_graph_vae           - Graph VAE with 42M+ parameters  
  rebuilt_datacube_cnn        - Enhanced CNN with physics constraints
  rebuilt_multimodal_integration - Multi-modal fusion system

TRAINING MODES:
  full_pipeline               - Complete training pipeline (default)
  single_model               - Train single model only
  hyperopt                   - Hyperparameter optimization
  eval_only                  - Evaluation only
  distributed                - Distributed training

EXAMPLES:
  # Basic training
  python train_unified_sota.py --model rebuilt_llm_integration
  
  # With hyperparameter optimization
  python train_unified_sota.py --model rebuilt_graph_vae --optimize --trials 50
  
  # Distributed training
  python train_unified_sota.py --model rebuilt_datacube_cnn --distributed --gpus 4 --nodes 2
  
  # Custom configuration
  python train_unified_sota.py --model rebuilt_multimodal_integration --config config/custom.yaml
        """
    )
    
    # Model selection
    parser.add_argument(
        '--model', 
        type=str, 
        default='rebuilt_llm_integration',
        choices=[
            'rebuilt_llm_integration',
            'rebuilt_graph_vae', 
            'rebuilt_datacube_cnn',
            'rebuilt_multimodal_integration'
        ],
        help='Model to train'
    )
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=None, help='Weight decay')
    
    # Optimization
    parser.add_argument('--optimizer', type=str, default='adamw', 
                       choices=['adamw', 'lion', 'sophia'], help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='onecycle',
                       choices=['onecycle', 'cosine', 'cosine_restarts'], help='LR scheduler')
    
    # SOTA features
    parser.add_argument('--no-flash-attention', action='store_true', 
                       help='Disable Flash Attention')
    parser.add_argument('--no-mixed-precision', action='store_true',
                       help='Disable mixed precision training')
    parser.add_argument('--no-gradient-checkpointing', action='store_true',
                       help='Disable gradient checkpointing')
    parser.add_argument('--no-compile', action='store_true',
                       help='Disable torch.compile optimization')
    
    # Training modes
    parser.add_argument('--optimize', action='store_true',
                       help='Run hyperparameter optimization')
    parser.add_argument('--trials', type=int, default=50,
                       help='Number of optimization trials')
    parser.add_argument('--eval-only', action='store_true',
                       help='Evaluation only mode')
    parser.add_argument('--distributed', action='store_true',
                       help='Enable distributed training')
    
    # Hardware
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--nodes', type=int, default=1, help='Number of nodes')
    
    # Configuration
    parser.add_argument('--config', type=str, default=None,
                       help='Configuration file path')
    parser.add_argument('--output-dir', type=str, default='outputs/sota_training',
                       help='Output directory')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name')
    
    # Monitoring
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    parser.add_argument('--log-every', type=int, default=50,
                       help='Log every N steps')
    
    # Physics constraints
    parser.add_argument('--physics-constraints', action='store_true',
                       help='Enable physics constraints')
    parser.add_argument('--physics-weight', type=float, default=0.1,
                       help='Physics constraint weight')
    
    # Checkpointing
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--save-every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    return parser


def main():
    """Main entry point - replaces all training scripts"""
    
    # Print banner
    print("🚀 UNIFIED SOTA TRAINING SYSTEM - 2025 ASTROBIOLOGY AI PLATFORM")
    print("=" * 70)
    print("📦 CONSOLIDATES: train.py + train_sota_unified.py + train_llm_galactic_unified_system.py")
    print("                + train_causal_models_sota.py + train_optuna.py")
    print("⚡ FEATURES: Flash Attention + Mixed Precision + Advanced Optimizers")
    print("🎯 ZERO REDUNDANCY - MAXIMUM SOTA PERFORMANCE")
    print("=" * 70)
    
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Load config file if provided
    config_overrides = {}
    if args.config:
        config_overrides.update(load_config_file(args.config))
    
    # Override with command line arguments
    if args.batch_size is not None:
        config_overrides['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config_overrides['learning_rate'] = args.learning_rate
    if args.weight_decay is not None:
        config_overrides['weight_decay'] = args.weight_decay
    
    config_overrides.update({
        'max_epochs': args.epochs,
        'optimizer_name': args.optimizer,
        'scheduler_name': args.scheduler,
        'use_flash_attention': not args.no_flash_attention,
        'use_mixed_precision': not args.no_mixed_precision,
        'use_gradient_checkpointing': not args.no_gradient_checkpointing,
        'use_compile': not args.no_compile,
        'use_distributed': args.distributed,
        'num_gpus': args.gpus,
        'num_nodes': args.nodes,
        'output_dir': args.output_dir,
        'experiment_name': args.experiment_name or f"sota_{args.model}",
        'use_wandb': not args.no_wandb,
        'log_every_n_steps': args.log_every,
        'use_physics_constraints': args.physics_constraints,
        'physics_weight': args.physics_weight,
        'save_every_n_epochs': args.save_every
    })
    
    # Determine training mode
    if args.optimize:
        mode = TrainingMode.HYPERPARAMETER_OPTIMIZATION
    elif args.eval_only:
        mode = TrainingMode.EVALUATION_ONLY
    elif args.distributed:
        mode = TrainingMode.DISTRIBUTED
    else:
        mode = TrainingMode.FULL_PIPELINE
    
    # Print configuration
    logger.info(f"🎯 Model: {args.model}")
    logger.info(f"📊 Mode: {mode.value}")
    logger.info(f"⚡ SOTA Features: Flash Attention={not args.no_flash_attention}, "
               f"Mixed Precision={not args.no_mixed_precision}")
    logger.info(f"🔧 Optimizer: {args.optimizer}, Scheduler: {args.scheduler}")
    
    # Run training
    try:
        results = asyncio.run(run_unified_training(
            model_name=args.model,
            config_overrides=config_overrides,
            mode=mode
        ))
        
        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / f"training_results_{args.model}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"📊 Results saved to: {results_file}")
        
        if results.get('status') == 'completed':
            logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("🚀 Model ready for production deployment!")
        else:
            logger.info("✅ Training process completed")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

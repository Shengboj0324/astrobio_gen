#!/bin/bash
#
# Training Entry Point Script
# ===========================
#
# Production-grade training script with comprehensive error handling,
# logging, and configuration management.
#
# Usage:
#   ./train.sh [options]
#
# Environment Variables:
#   MODEL_NAME          - Model to train (default: rebuilt_llm_integration)
#   BATCH_SIZE          - Batch size (default: 32)
#   LEARNING_RATE       - Learning rate (default: 1e-4)
#   MAX_EPOCHS          - Maximum epochs (default: 100)
#   NUM_GPUS            - Number of GPUs (default: auto-detect)
#   CHECKPOINT_DIR      - Checkpoint directory (default: ./checkpoints)
#   LOG_DIR             - Log directory (default: ./logs)
#   WANDB_PROJECT       - WandB project name (default: astrobio-gen)
#   WANDB_ENTITY        - WandB entity (default: none)
#   RESUME_CHECKPOINT   - Path to checkpoint to resume from (default: none)
#

set -euo pipefail  # Exit on error, undefined variable, or pipe failure

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Error handler
error_handler() {
    log_error "Training failed at line $1"
    log_error "Check logs at: $LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
    exit 1
}

trap 'error_handler $LINENO' ERR

# Configuration
MODEL_NAME=${MODEL_NAME:-"rebuilt_llm_integration"}
BATCH_SIZE=${BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
MAX_EPOCHS=${MAX_EPOCHS:-100}
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi --list-gpus | wc -l)}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"./checkpoints"}
LOG_DIR=${LOG_DIR:-"./logs"}
WANDB_PROJECT=${WANDB_PROJECT:-"astrobio-gen"}
WANDB_ENTITY=${WANDB_ENTITY:-""}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-""}

# Create directories
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"

# Log file
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

# Print configuration
log_info "==================================================================="
log_info "AstroBio-Gen Training"
log_info "==================================================================="
log_info "Model:           $MODEL_NAME"
log_info "Batch Size:      $BATCH_SIZE"
log_info "Learning Rate:   $LEARNING_RATE"
log_info "Max Epochs:      $MAX_EPOCHS"
log_info "GPUs:            $NUM_GPUS"
log_info "Checkpoint Dir:  $CHECKPOINT_DIR"
log_info "Log Dir:         $LOG_DIR"
log_info "WandB Project:   $WANDB_PROJECT"
if [ -n "$RESUME_CHECKPOINT" ]; then
    log_info "Resume From:     $RESUME_CHECKPOINT"
fi
log_info "==================================================================="

# Check CUDA availability
log_info "Checking CUDA availability..."
if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"; then
    log_error "CUDA is not available. Please check your GPU setup."
    exit 1
fi
log_success "CUDA is available"

# Check GPU count
DETECTED_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
log_info "Detected $DETECTED_GPUS GPU(s)"

if [ "$NUM_GPUS" -gt "$DETECTED_GPUS" ]; then
    log_warning "Requested $NUM_GPUS GPUs but only $DETECTED_GPUS available"
    NUM_GPUS=$DETECTED_GPUS
fi

# Build training command
TRAIN_CMD="python train_unified_sota.py"
TRAIN_CMD="$TRAIN_CMD --model $MODEL_NAME"
TRAIN_CMD="$TRAIN_CMD --batch_size $BATCH_SIZE"
TRAIN_CMD="$TRAIN_CMD --learning_rate $LEARNING_RATE"
TRAIN_CMD="$TRAIN_CMD --max_epochs $MAX_EPOCHS"
TRAIN_CMD="$TRAIN_CMD --checkpoint_dir $CHECKPOINT_DIR"
TRAIN_CMD="$TRAIN_CMD --log_dir $LOG_DIR"

# Multi-GPU configuration
if [ "$NUM_GPUS" -gt 1 ]; then
    log_info "Configuring multi-GPU training with $NUM_GPUS GPUs"
    TRAIN_CMD="$TRAIN_CMD --num_gpus $NUM_GPUS"
    TRAIN_CMD="$TRAIN_CMD --distributed_backend nccl"
fi

# WandB configuration
if [ -n "$WANDB_ENTITY" ]; then
    TRAIN_CMD="$TRAIN_CMD --wandb_project $WANDB_PROJECT"
    TRAIN_CMD="$TRAIN_CMD --wandb_entity $WANDB_ENTITY"
fi

# Resume from checkpoint
if [ -n "$RESUME_CHECKPOINT" ]; then
    if [ -f "$RESUME_CHECKPOINT" ]; then
        log_info "Resuming from checkpoint: $RESUME_CHECKPOINT"
        TRAIN_CMD="$TRAIN_CMD --resume_from_checkpoint $RESUME_CHECKPOINT"
    else
        log_error "Checkpoint not found: $RESUME_CHECKPOINT"
        exit 1
    fi
fi

# Add optimizations
TRAIN_CMD="$TRAIN_CMD --use_flash_attention"
TRAIN_CMD="$TRAIN_CMD --use_mixed_precision"
TRAIN_CMD="$TRAIN_CMD --use_gradient_checkpointing"
TRAIN_CMD="$TRAIN_CMD --gradient_clip_val 1.0"
TRAIN_CMD="$TRAIN_CMD --checkpoint_every 1000"
TRAIN_CMD="$TRAIN_CMD --log_every 100"

# Print command
log_info "Training command:"
log_info "$TRAIN_CMD"
log_info "==================================================================="

# Start training
log_info "Starting training..."
log_info "Logs will be saved to: $LOG_FILE"

# Run training with output to both console and log file
$TRAIN_CMD 2>&1 | tee "$LOG_FILE"

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    log_success "Training completed successfully!"
    log_success "Checkpoints saved to: $CHECKPOINT_DIR"
    log_success "Logs saved to: $LOG_FILE"
else
    log_error "Training failed!"
    log_error "Check logs at: $LOG_FILE"
    exit 1
fi

# Print final checkpoint location
FINAL_CHECKPOINT="$CHECKPOINT_DIR/last.ckpt"
if [ -f "$FINAL_CHECKPOINT" ]; then
    log_success "Final checkpoint: $FINAL_CHECKPOINT"
    
    # Print checkpoint info
    log_info "Checkpoint info:"
    python -c "
import torch
ckpt = torch.load('$FINAL_CHECKPOINT', map_location='cpu')
print(f\"  Epoch: {ckpt.get('epoch', 'N/A')}\")
print(f\"  Global Step: {ckpt.get('global_step', 'N/A')}\")
if 'state_dict' in ckpt:
    params = sum(p.numel() for p in ckpt['state_dict'].values())
    print(f\"  Parameters: {params:,}\")
" 2>/dev/null || log_warning "Could not read checkpoint info"
fi

log_info "==================================================================="
log_success "Training script completed!"
log_info "==================================================================="


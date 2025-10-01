#!/bin/bash
#
# Evaluation Entry Point Script
# ==============================
#
# Production-grade evaluation script for model validation and testing.
#
# Usage:
#   ./eval.sh [options]
#
# Environment Variables:
#   MODEL_NAME          - Model to evaluate (default: rebuilt_llm_integration)
#   CHECKPOINT_PATH     - Path to checkpoint (required)
#   EVAL_DATASET        - Dataset to evaluate on (default: validation)
#   BATCH_SIZE          - Batch size (default: 64)
#   NUM_GPUS            - Number of GPUs (default: 1)
#   OUTPUT_DIR          - Output directory (default: ./results)
#   METRICS             - Metrics to compute (default: all)
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

error_handler() {
    log_error "Evaluation failed at line $1"
    exit 1
}

trap 'error_handler $LINENO' ERR

# Configuration
MODEL_NAME=${MODEL_NAME:-"rebuilt_llm_integration"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-""}
EVAL_DATASET=${EVAL_DATASET:-"validation"}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_GPUS=${NUM_GPUS:-1}
OUTPUT_DIR=${OUTPUT_DIR:-"./results"}
METRICS=${METRICS:-"all"}

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print configuration
log_info "==================================================================="
log_info "AstroBio-Gen Evaluation"
log_info "==================================================================="
log_info "Model:           $MODEL_NAME"
log_info "Checkpoint:      $CHECKPOINT_PATH"
log_info "Dataset:         $EVAL_DATASET"
log_info "Batch Size:      $BATCH_SIZE"
log_info "GPUs:            $NUM_GPUS"
log_info "Output Dir:      $OUTPUT_DIR"
log_info "Metrics:         $METRICS"
log_info "==================================================================="

# Check checkpoint
if [ -z "$CHECKPOINT_PATH" ]; then
    log_error "CHECKPOINT_PATH is required"
    log_info "Usage: CHECKPOINT_PATH=path/to/checkpoint.ckpt ./eval.sh"
    exit 1
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
    log_error "Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

log_success "Checkpoint found: $CHECKPOINT_PATH"

# Check CUDA
log_info "Checking CUDA availability..."
if ! python -c "import torch; assert torch.cuda.is_available()"; then
    log_warning "CUDA not available, using CPU"
fi

# Build evaluation command
EVAL_CMD="python -c \"
import torch
import json
from pathlib import Path
from models.${MODEL_NAME} import ${MODEL_NAME^}
from validation.benchmark_suite import run_comprehensive_evaluation

# Load model
print('Loading model from checkpoint...')
model = ${MODEL_NAME^}.load_from_checkpoint('$CHECKPOINT_PATH')
model.eval()

# Run evaluation
print('Running evaluation...')
results = run_comprehensive_evaluation(
    model=model,
    dataset='$EVAL_DATASET',
    batch_size=$BATCH_SIZE,
    metrics='$METRICS'.split(',') if '$METRICS' != 'all' else None
)

# Save results
output_file = Path('$OUTPUT_DIR') / 'eval_results_$(date +%Y%m%d_%H%M%S).json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f'Results saved to: {output_file}')
print('\\nEvaluation Summary:')
for key, value in results.get('summary', {}).items():
    print(f'  {key}: {value}')
\""

log_info "Starting evaluation..."
eval "$EVAL_CMD"

log_success "Evaluation completed successfully!"
log_info "==================================================================="


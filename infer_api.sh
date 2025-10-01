#!/bin/bash
#
# Inference API Entry Point Script
# =================================
#
# Production-grade inference API server with FastAPI.
#
# Usage:
#   ./infer_api.sh [options]
#
# Environment Variables:
#   MODEL_NAME          - Model to serve (default: rebuilt_llm_integration)
#   CHECKPOINT_PATH     - Path to checkpoint (required)
#   HOST                - Host to bind to (default: 0.0.0.0)
#   PORT                - Port to bind to (default: 8000)
#   NUM_WORKERS         - Number of workers (default: 1)
#   BATCH_SIZE          - Batch size for inference (default: 8)
#   MAX_LENGTH          - Maximum sequence length (default: 2048)
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
    log_error "API server failed at line $1"
    exit 1
}

trap 'error_handler $LINENO' ERR

# Configuration
MODEL_NAME=${MODEL_NAME:-"rebuilt_llm_integration"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-""}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8000}
NUM_WORKERS=${NUM_WORKERS:-1}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_LENGTH=${MAX_LENGTH:-2048}

# Print configuration
log_info "==================================================================="
log_info "AstroBio-Gen Inference API"
log_info "==================================================================="
log_info "Model:           $MODEL_NAME"
log_info "Checkpoint:      $CHECKPOINT_PATH"
log_info "Host:            $HOST"
log_info "Port:            $PORT"
log_info "Workers:         $NUM_WORKERS"
log_info "Batch Size:      $BATCH_SIZE"
log_info "Max Length:      $MAX_LENGTH"
log_info "==================================================================="

# Check checkpoint
if [ -z "$CHECKPOINT_PATH" ]; then
    log_error "CHECKPOINT_PATH is required"
    log_info "Usage: CHECKPOINT_PATH=path/to/checkpoint.ckpt ./infer_api.sh"
    exit 1
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
    log_error "Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

log_success "Checkpoint found: $CHECKPOINT_PATH"

# Check FastAPI
log_info "Checking FastAPI installation..."
if ! python -c "import fastapi, uvicorn" 2>/dev/null; then
    log_error "FastAPI or uvicorn not installed"
    log_info "Install with: pip install fastapi uvicorn"
    exit 1
fi
log_success "FastAPI is installed"

# Start API server
log_info "Starting inference API server..."
log_info "API will be available at: http://$HOST:$PORT"
log_info "Documentation at: http://$HOST:$PORT/docs"
log_info "==================================================================="

# Export environment variables for the API
export MODEL_NAME
export CHECKPOINT_PATH
export BATCH_SIZE
export MAX_LENGTH

# Start uvicorn
uvicorn api.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$NUM_WORKERS" \
    --log-level info \
    --access-log \
    --use-colors

log_info "==================================================================="
log_info "API server stopped"
log_info "==================================================================="


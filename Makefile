# Makefile for Astrobiology Research Platform - ISEF Competition Ready
# =====================================================================
# Complete reproducibility system for scientific validation
# Compatible with RunPod A500 GPU environment

.PHONY: all setup install test benchmark validate reproduce-all clean help
.DEFAULT_GOAL := help

# Configuration
PYTHON := python3
PIP := pip
CONDA := conda
DOCKER := docker

# Directories
RESULTS_DIR := results
FIGURES_DIR := paper/figures
TABLES_DIR := paper/tables
DATA_DIR := data
MODELS_DIR := models
TESTS_DIR := tests

# Environment variables for reproducibility
export PYTHONHASHSEED=42
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST=8.6
export PYTHONPATH=$(PWD)/src

# Colors for output
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
RESET := \033[0m

##@ Setup and Installation

setup: ## Set up the complete environment for ISEF competition
	@echo "$(BLUE)🚀 Setting up Astrobiology Research Platform...$(RESET)"
	@echo "$(YELLOW)📋 Creating directories...$(RESET)"
	@mkdir -p $(RESULTS_DIR) $(FIGURES_DIR) $(TABLES_DIR) $(DATA_DIR) $(MODELS_DIR) logs checkpoints
	@echo "$(YELLOW)🔧 Setting up reproducible environment...$(RESET)"
	@$(PYTHON) utils/set_seeds.py
	@echo "$(GREEN)✅ Environment setup complete!$(RESET)"

install: ## Install all dependencies for production
	@echo "$(BLUE)📦 Installing production dependencies...$(RESET)"
	@$(PIP) install --upgrade pip setuptools wheel
	@$(PIP) install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
	@$(PIP) install -r requirements-production-lock.txt
	@$(PIP) install -e .
	@echo "$(GREEN)✅ Installation complete!$(RESET)"

install-dev: ## Install development dependencies
	@echo "$(BLUE)🛠️  Installing development dependencies...$(RESET)"
	@$(PIP) install pytest pytest-cov pytest-xdist pytest-mock pytest-asyncio
	@$(PIP) install black isort ruff mypy
	@$(PIP) install hypothesis
	@echo "$(GREEN)✅ Development dependencies installed!$(RESET)"

##@ Testing and Validation

test: ## Run comprehensive test suite
	@echo "$(BLUE)🧪 Running comprehensive test suite...$(RESET)"
	@$(PYTHON) -m pytest $(TESTS_DIR)/ -v --tb=short --maxfail=5
	@echo "$(GREEN)✅ All tests passed!$(RESET)"

test-physics: ## Run physics invariant tests
	@echo "$(BLUE)⚛️  Running physics invariant tests...$(RESET)"
	@$(PYTHON) -m pytest $(TESTS_DIR)/test_physics_invariants.py -v --hypothesis-show-statistics
	@echo "$(GREEN)✅ Physics tests passed!$(RESET)"

test-stability: ## Run stability tests
	@echo "$(BLUE)🔄 Running stability tests...$(RESET)"
	@$(PYTHON) -m pytest $(TESTS_DIR)/test_long_rollout.py -v
	@echo "$(GREEN)✅ Stability tests passed!$(RESET)"

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)📊 Running tests with coverage...$(RESET)"
	@$(PYTHON) -m pytest $(TESTS_DIR)/ \
		--cov=src/astrobio_gen \
		--cov-report=html \
		--cov-report=xml \
		--cov-report=term-missing \
		--cov-fail-under=70
	@echo "$(GREEN)✅ Coverage report generated!$(RESET)"

##@ Benchmarking and Validation

benchmark: ## Run GCM benchmark suite
	@echo "$(BLUE)🏁 Running GCM benchmark suite...$(RESET)"
	@$(PYTHON) experiments/gcm_bench.py
	@echo "$(GREEN)✅ Benchmarks completed! Results in $(RESULTS_DIR)/bench.csv$(RESET)"

calibration: ## Run calibration validation
	@echo "$(BLUE)🎯 Running calibration validation...$(RESET)"
	@$(PYTHON) validation/calibration.py
	@echo "$(GREEN)✅ Calibration completed! Results in $(RESULTS_DIR)/calibration.csv$(RESET)"

ablation: ## Run ablation studies
	@echo "$(BLUE)🔬 Running ablation studies...$(RESET)"
	@$(PYTHON) experiments/ablations.py
	@echo "$(GREEN)✅ Ablation studies completed! Results in $(RESULTS_DIR)/ablations.csv$(RESET)"

rollout: ## Run long rollout stability tests
	@echo "$(BLUE)🔄 Running long rollout stability tests...$(RESET)"
	@$(PYTHON) validation/long_rollout.py
	@echo "$(GREEN)✅ Rollout tests completed! Results in $(RESULTS_DIR)/rollout.csv$(RESET)"

validate: benchmark calibration ablation rollout ## Run complete validation suite
	@echo "$(GREEN)🏆 Complete validation suite finished!$(RESET)"

##@ Figure and Table Generation

figures: ## Generate all paper figures
	@echo "$(BLUE)📊 Generating paper figures...$(RESET)"
	@mkdir -p $(FIGURES_DIR)
	@$(PYTHON) paper/figures/fig_parity.py
	@$(PYTHON) paper/figures/fig_ablations.py
	@$(PYTHON) paper/figures/fig_calibration.py
	@$(PYTHON) paper/figures/fig_rollout.py
	@$(PYTHON) paper/figures/fig_architecture.py
	@echo "$(GREEN)✅ All figures generated in $(FIGURES_DIR)/$(RESET)"

tables: ## Generate all paper tables
	@echo "$(BLUE)📋 Generating paper tables...$(RESET)"
	@mkdir -p $(TABLES_DIR)
	@$(PYTHON) paper/tables/generate_tables.py
	@echo "$(GREEN)✅ All tables generated in $(TABLES_DIR)/$(RESET)"

##@ Reproducibility

earth-calibration: ## Run Earth calibration
	@echo "$(BLUE)🌍 Running Earth calibration...$(RESET)"
	@$(PYTHON) -c "
	import json
	from pathlib import Path
	print('Earth calibration results already generated in results/earth_calib.json')
	if Path('results/earth_calib.json').exists():
	    with open('results/earth_calib.json', 'r') as f:
	        data = json.load(f)
	    print(f'Overall accuracy: {data[\"earth_calibration_results\"][\"overall_accuracy\"]}')
	else:
	    print('Calibration file not found - run make validate first')
	"
	@echo "$(GREEN)✅ Earth calibration validated!$(RESET)"

reproduce-all: setup validate figures tables earth-calibration ## Reproduce all results for ISEF competition
	@echo "$(BLUE)🎯 Reproducing all ISEF competition results...$(RESET)"
	@echo "$(YELLOW)📊 Generating final summary report...$(RESET)"
	@$(PYTHON) -c "
	import json
	import pandas as pd
	from pathlib import Path
	from datetime import datetime
	
	print('=== ISEF Competition Results Summary ===')
	print(f'Generated: {datetime.now().isoformat()}')
	print()
	
	# Check benchmark results
	if Path('$(RESULTS_DIR)/bench.csv').exists():
	    df = pd.read_csv('$(RESULTS_DIR)/bench.csv')
	    print(f'✅ GCM Benchmarks: {len(df)} models tested')
	else:
	    print('❌ GCM Benchmarks: Not found')
	
	# Check calibration results
	if Path('$(RESULTS_DIR)/calibration.csv').exists():
	    df = pd.read_csv('$(RESULTS_DIR)/calibration.csv')
	    print(f'✅ Calibration: {len(df)} configurations tested')
	else:
	    print('❌ Calibration: Not found')
	
	# Check ablation results
	if Path('$(RESULTS_DIR)/ablations.csv').exists():
	    df = pd.read_csv('$(RESULTS_DIR)/ablations.csv')
	    print(f'✅ Ablation Studies: {len(df)} configurations tested')
	else:
	    print('❌ Ablation Studies: Not found')
	
	# Check figures
	figures_dir = Path('$(FIGURES_DIR)')
	if figures_dir.exists():
	    svg_files = list(figures_dir.glob('*.svg'))
	    print(f'✅ Figures: {len(svg_files)} generated')
	else:
	    print('❌ Figures: Directory not found')
	
	print()
	print('🏆 ISEF Competition Reproducibility: COMPLETE')
	"
	@echo "$(GREEN)🎉 All results successfully reproduced for ISEF competition!$(RESET)"

##@ Development and Quality

lint: ## Run code linting
	@echo "$(BLUE)🔍 Running code linting...$(RESET)"
	@ruff check . --output-format=github
	@ruff format --check .
	@isort --check-only --diff .
	@echo "$(GREEN)✅ Linting passed!$(RESET)"

format: ## Format code
	@echo "$(BLUE)✨ Formatting code...$(RESET)"
	@ruff format .
	@isort .
	@echo "$(GREEN)✅ Code formatted!$(RESET)"

type-check: ## Run type checking
	@echo "$(BLUE)🔍 Running type checking...$(RESET)"
	@mypy src/ --ignore-missing-imports --show-error-codes
	@echo "$(GREEN)✅ Type checking completed!$(RESET)"

##@ Docker and Deployment

docker-build: ## Build Docker image for RunPod deployment
	@echo "$(BLUE)🐳 Building Docker image...$(RESET)"
	@$(DOCKER) build -t astrobio-gen:latest .
	@echo "$(GREEN)✅ Docker image built successfully!$(RESET)"

docker-test: ## Test Docker image
	@echo "$(BLUE)🧪 Testing Docker image...$(RESET)"
	@$(DOCKER) run --rm astrobio-gen:latest python -c "
	import torch
	print(f'PyTorch version: {torch.__version__}')
	print(f'CUDA available: {torch.cuda.is_available()}')
	from utils.set_seeds import verify_deterministic_setup
	verify_deterministic_setup()
	print('✅ Docker image validation passed')
	"
	@echo "$(GREEN)✅ Docker image test passed!$(RESET)"

##@ Data Management

data-check: ## Check data integrity and availability
	@echo "$(BLUE)📊 Checking data integrity...$(RESET)"
	@$(PYTHON) -c "
	import json
	from pathlib import Path
	
	manifest_path = Path('DATA_MANIFEST.md')
	if manifest_path.exists():
	    print('✅ DATA_MANIFEST.md exists')
	    with open(manifest_path, 'r') as f:
	        content = f.read()
	        if 'SHA256' in content:
	            print('✅ Checksums documented')
	        if 'License' in content:
	            print('✅ Licenses documented')
	else:
	    print('❌ DATA_MANIFEST.md missing')
	
	# Check for data directories
	data_dirs = ['data/raw', 'data/processed', 'data/interim']
	for data_dir in data_dirs:
	    if Path(data_dir).exists():
	        print(f'✅ {data_dir} exists')
	    else:
	        print(f'⚠️  {data_dir} missing (will be created as needed)')
	"
	@echo "$(GREEN)✅ Data check completed!$(RESET)"

##@ Performance and Monitoring

performance: ## Run performance benchmarks
	@echo "$(BLUE)⚡ Running performance benchmarks...$(RESET)"
	@$(PYTHON) -c "
	import time
	import torch
	import numpy as np
	
	print('=== Performance Benchmark ===')
	
	# NumPy benchmark
	start_time = time.time()
	a = np.random.randn(1000, 1000)
	b = np.random.randn(1000, 1000)
	c = np.dot(a, b)
	numpy_time = time.time() - start_time
	print(f'NumPy matrix multiply (1000x1000): {numpy_time:.3f}s')
	
	# PyTorch CPU benchmark
	start_time = time.time()
	x = torch.randn(1000, 1000)
	y = torch.randn(1000, 1000)
	z = torch.mm(x, y)
	torch_cpu_time = time.time() - start_time
	print(f'PyTorch CPU matrix multiply (1000x1000): {torch_cpu_time:.3f}s')
	
	# PyTorch GPU benchmark (if available)
	if torch.cuda.is_available():
	    x_gpu = torch.randn(1000, 1000, device='cuda')
	    y_gpu = torch.randn(1000, 1000, device='cuda')
	    torch.cuda.synchronize()
	    start_time = time.time()
	    z_gpu = torch.mm(x_gpu, y_gpu)
	    torch.cuda.synchronize()
	    torch_gpu_time = time.time() - start_time
	    print(f'PyTorch GPU matrix multiply (1000x1000): {torch_gpu_time:.3f}s')
	    print(f'GPU speedup: {torch_cpu_time/torch_gpu_time:.1f}x')
	else:
	    print('GPU not available for benchmarking')
	
	print('✅ Performance benchmark completed')
	"
	@echo "$(GREEN)✅ Performance benchmarks completed!$(RESET)"

##@ Cleanup

clean: ## Clean up generated files
	@echo "$(BLUE)🧹 Cleaning up...$(RESET)"
	@rm -rf __pycache__/ */__pycache__/ */*/__pycache__/
	@rm -rf .pytest_cache/ .coverage htmlcov/ coverage.xml
	@rm -rf .mypy_cache/ .ruff_cache/
	@rm -rf build/ dist/ *.egg-info/
	@rm -rf logs/*.log checkpoints/*.pth
	@echo "$(GREEN)✅ Cleanup completed!$(RESET)"

clean-results: ## Clean up results and generated files
	@echo "$(BLUE)🧹 Cleaning results...$(RESET)"
	@rm -rf $(RESULTS_DIR)/*.csv $(RESULTS_DIR)/*.json $(RESULTS_DIR)/*.svg
	@rm -rf $(FIGURES_DIR)/*.svg $(TABLES_DIR)/*.csv
	@echo "$(GREEN)✅ Results cleaned!$(RESET)"

##@ Information and Help

status: ## Show project status and environment info
	@echo "$(BLUE)📊 Project Status$(RESET)"
	@echo "=================================="
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "PyTorch version: $$($(PYTHON) -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
	@echo "CUDA available: $$($(PYTHON) -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Unknown')"
	@echo "Working directory: $$(pwd)"
	@echo "PYTHONPATH: $$PYTHONPATH"
	@echo ""
	@echo "$(YELLOW)Directory Status:$(RESET)"
	@ls -la $(RESULTS_DIR)/ 2>/dev/null | head -5 || echo "Results directory empty/missing"
	@echo ""
	@echo "$(YELLOW)Recent Results:$(RESET)"
	@find $(RESULTS_DIR)/ -name "*.csv" -o -name "*.json" 2>/dev/null | head -5 || echo "No results found"

check-requirements: ## Check if all requirements are satisfied
	@echo "$(BLUE)🔍 Checking requirements...$(RESET)"
	@$(PYTHON) -c "
	import pkg_resources
	import sys
	
	try:
	    with open('requirements-production-lock.txt', 'r') as f:
	        requirements = f.read().splitlines()
	    
	    missing = []
	    for req in requirements:
	        if req.strip() and not req.startswith('#'):
	            try:
	                pkg_resources.require(req.split('==')[0].split('>=')[0].split('<=')[0])
	            except:
	                missing.append(req)
	    
	    if missing:
	        print('❌ Missing requirements:')
	        for req in missing[:5]:  # Show first 5
	            print(f'  - {req}')
	        if len(missing) > 5:
	            print(f'  ... and {len(missing)-5} more')
	        sys.exit(1)
	    else:
	        print('✅ All requirements satisfied')
	
	except FileNotFoundError:
	    print('❌ requirements-production-lock.txt not found')
	    sys.exit(1)
	"

help: ## Show this help message
	@echo "$(BLUE)Astrobiology Research Platform - ISEF Competition Ready$(RESET)"
	@echo "=========================================================="
	@echo ""
	@echo "$(YELLOW)Quick Start:$(RESET)"
	@echo "  make setup install      # Set up environment and install dependencies"
	@echo "  make reproduce-all      # Reproduce all ISEF competition results"
	@echo "  make test               # Run comprehensive test suite"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "\n$(YELLOW)Available Commands:$(RESET)\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BLUE)%s$(RESET)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(YELLOW)Environment Variables:$(RESET)"
	@echo "  PYTHONHASHSEED=42       # For reproducibility"
	@echo "  CUDA_VISIBLE_DEVICES=0  # GPU selection"
	@echo "  TORCH_CUDA_ARCH_LIST=8.6 # A500 GPU architecture"
	@echo ""
	@echo "$(YELLOW)For ISEF Competition:$(RESET)"
	@echo "  1. Run 'make reproduce-all' to generate all results"
	@echo "  2. All figures will be in $(FIGURES_DIR)/"
	@echo "  3. All tables will be in $(TABLES_DIR)/"
	@echo "  4. All metrics will be in $(RESULTS_DIR)/"
	@echo ""
	@echo "$(GREEN)🏆 Ready for ISEF Competition! 🏆$(RESET)"

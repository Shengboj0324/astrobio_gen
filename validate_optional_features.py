#!/usr/bin/env python3
"""
Validate Optional Feature Guards
=================================

Checks that all optional features (wandb, optuna, bitsandbytes, flash_attn)
are properly guarded with availability checks.
"""

import ast
import sys
from pathlib import Path

# Color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
RESET = '\033[0m'

class OptionalFeatureValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []
        
    def validate_file(self, file_path: Path) -> bool:
        """Validate a single file for proper optional feature guards"""
        print(f"\n{CYAN}Checking: {file_path}{RESET}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse AST
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            self.errors.append(f"{file_path}: Syntax error at line {e.lineno}")
            return False
        
        # Check for unguarded usage
        lines = content.split('\n')
        
        # Find all wandb.* calls
        wandb_lines = [i+1 for i, line in enumerate(lines) if 'wandb.' in line and 'import wandb' not in line]
        optuna_lines = [i+1 for i, line in enumerate(lines) if 'optuna.' in line and 'import optuna' not in line]
        bnb_lines = [i+1 for i, line in enumerate(lines) if 'bnb.' in line and 'import bitsandbytes' not in line]
        flash_lines = [i+1 for i, line in enumerate(lines) if 'flash_attn' in line and 'import flash_attn' not in line and 'from flash_attn' not in line]
        
        # Check if these are properly guarded
        for line_num in wandb_lines:
            # Look backwards for if WANDB_AVAILABLE
            found_guard = False
            for i in range(max(0, line_num - 20), line_num):
                if 'WANDB_AVAILABLE' in lines[i] and 'if' in lines[i]:
                    found_guard = True
                    break
            
            if not found_guard:
                # Check if it's in a try/except block
                found_try = False
                for i in range(max(0, line_num - 10), line_num):
                    if 'try:' in lines[i]:
                        found_try = True
                        break
                
                if not found_try:
                    self.warnings.append(f"{file_path}:{line_num}: wandb usage may not be guarded")
        
        for line_num in optuna_lines:
            found_guard = False
            for i in range(max(0, line_num - 20), line_num):
                if 'OPTUNA_AVAILABLE' in lines[i] and 'if' in lines[i]:
                    found_guard = True
                    break
            
            if not found_guard:
                self.warnings.append(f"{file_path}:{line_num}: optuna usage may not be guarded")
        
        for line_num in bnb_lines:
            found_guard = False
            for i in range(max(0, line_num - 20), line_num):
                if 'BITSANDBYTES_AVAILABLE' in lines[i] and 'if' in lines[i]:
                    found_guard = True
                    break
            
            if not found_guard:
                self.warnings.append(f"{file_path}:{line_num}: bitsandbytes usage may not be guarded")
        
        # Check availability flags are defined
        has_wandb_flag = 'WANDB_AVAILABLE' in content
        has_optuna_flag = 'OPTUNA_AVAILABLE' in content
        has_bnb_flag = 'BITSANDBYTES_AVAILABLE' in content
        has_flash_flag = 'FLASH_ATTENTION_AVAILABLE' in content
        
        uses_wandb = len(wandb_lines) > 0
        uses_optuna = len(optuna_lines) > 0
        uses_bnb = len(bnb_lines) > 0
        uses_flash = len(flash_lines) > 0
        
        if uses_wandb and not has_wandb_flag:
            self.errors.append(f"{file_path}: Uses wandb but WANDB_AVAILABLE not defined")
        
        if uses_optuna and not has_optuna_flag:
            self.errors.append(f"{file_path}: Uses optuna but OPTUNA_AVAILABLE not defined")
        
        if uses_bnb and not has_bnb_flag:
            self.errors.append(f"{file_path}: Uses bitsandbytes but BITSANDBYTES_AVAILABLE not defined")
        
        if uses_flash and not has_flash_flag:
            self.errors.append(f"{file_path}: Uses flash_attn but FLASH_ATTENTION_AVAILABLE not defined")
        
        # Print summary for this file
        if uses_wandb:
            print(f"  wandb: {len(wandb_lines)} usages, flag defined: {has_wandb_flag}")
        if uses_optuna:
            print(f"  optuna: {len(optuna_lines)} usages, flag defined: {has_optuna_flag}")
        if uses_bnb:
            print(f"  bitsandbytes: {len(bnb_lines)} usages, flag defined: {has_bnb_flag}")
        if uses_flash:
            print(f"  flash_attn: {len(flash_lines)} usages, flag defined: {has_flash_flag}")
        
        return True
    
    def run(self):
        """Run validation on critical files"""
        print(f"\n{MAGENTA}{'='*80}{RESET}")
        print(f"{MAGENTA}OPTIONAL FEATURE GUARD VALIDATION{RESET}")
        print(f"{MAGENTA}{'='*80}{RESET}")
        
        critical_files = [
            Path('training/unified_sota_training_system.py'),
            Path('training/unified_multimodal_training.py'),
            Path('train_unified_sota.py'),
        ]
        
        for file_path in critical_files:
            if file_path.exists():
                self.validate_file(file_path)
            else:
                self.warnings.append(f"File not found: {file_path}")
        
        # Print summary
        print(f"\n{MAGENTA}{'='*80}{RESET}")
        print(f"{MAGENTA}VALIDATION SUMMARY{RESET}")
        print(f"{MAGENTA}{'='*80}{RESET}\n")
        
        if self.errors:
            print(f"{RED}✗ ERRORS: {len(self.errors)}{RESET}")
            for error in self.errors:
                print(f"  {error}")
        else:
            print(f"{GREEN}✓ No critical errors found{RESET}")
        
        if self.warnings:
            print(f"\n{YELLOW}⚠ WARNINGS: {len(self.warnings)}{RESET}")
            for warning in self.warnings[:10]:  # Show first 10
                print(f"  {warning}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more")
        
        print(f"\n{MAGENTA}{'='*80}{RESET}")
        if len(self.errors) == 0:
            print(f"{GREEN}✓ OPTIONAL FEATURES PROPERLY GUARDED{RESET}")
        else:
            print(f"{RED}✗ CRITICAL ERRORS MUST BE FIXED{RESET}")
        print(f"{MAGENTA}{'='*80}{RESET}\n")
        
        return len(self.errors) == 0

if __name__ == "__main__":
    validator = OptionalFeatureValidator()
    success = validator.run()
    sys.exit(0 if success else 1)


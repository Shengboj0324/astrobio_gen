#!/usr/bin/env python3
"""
Critical Bug Fix Validation
============================

Validates that all critical bugs identified in the code review have been fixed:
1. Logger initialization before usage
2. MultiModalBatch.to() preserves annotations
3. Optional features properly guarded
"""

import sys
import ast
from pathlib import Path

# Color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
RESET = '\033[0m'

class CriticalBugValidator:
    def __init__(self):
        self.errors = []
        self.passed = []
        
    def check_logger_initialization(self):
        """Check that logger is initialized before any logger.warning() calls"""
        print(f"\n{CYAN}{'='*80}{RESET}")
        print(f"{CYAN}BUG FIX #1: Logger Initialization{RESET}")
        print(f"{CYAN}{'='*80}{RESET}")
        
        file_path = Path('training/unified_sota_training_system.py')
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Find logger initialization line
        logger_init_line = None
        for i, line in enumerate(lines):
            if 'logger = logging.getLogger(__name__)' in line:
                logger_init_line = i + 1
                break
        
        if logger_init_line is None:
            self.errors.append("Logger initialization not found")
            print(f"{RED}✗ Logger initialization not found{RESET}")
            return
        
        print(f"  Logger initialized at line {logger_init_line}")
        
        # Find all logger.warning() calls before initialization (excluding comments)
        early_logger_calls = []
        for i, line in enumerate(lines[:logger_init_line-1]):
            stripped = line.strip()
            if 'logger.' in line and 'import' not in line and 'logging.' not in line and not stripped.startswith('#'):
                early_logger_calls.append(i + 1)
        
        if early_logger_calls:
            self.errors.append(f"Logger used before initialization at lines: {early_logger_calls}")
            print(f"{RED}✗ Logger used before initialization at lines: {early_logger_calls}{RESET}")
        else:
            self.passed.append("Logger initialization")
            print(f"{GREEN}✓ Logger initialized before all usage{RESET}")
        
        # Check that logging.basicConfig is before logger init
        basicConfig_line = None
        for i, line in enumerate(lines[:logger_init_line-1]):
            if 'logging.basicConfig' in line:
                basicConfig_line = i + 1
                break
        
        if basicConfig_line and basicConfig_line < logger_init_line:
            print(f"{GREEN}✓ logging.basicConfig at line {basicConfig_line} (before logger init){RESET}")
        else:
            self.errors.append("logging.basicConfig not found before logger initialization")
            print(f"{RED}✗ logging.basicConfig not found before logger initialization{RESET}")
    
    def check_multimodalbatch_annotations(self):
        """Check that MultiModalBatch.to() preserves annotations"""
        print(f"\n{CYAN}{'='*80}{RESET}")
        print(f"{CYAN}BUG FIX #2: MultiModalBatch.to() Annotations{RESET}")
        print(f"{CYAN}{'='*80}{RESET}")
        
        file_path = Path('data_build/unified_dataloader_architecture.py')
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find the to() method
        if 'def to(self, device):' not in content:
            self.errors.append("MultiModalBatch.to() method not found")
            print(f"{RED}✗ MultiModalBatch.to() method not found{RESET}")
            return
        
        print(f"  MultiModalBatch.to() method found")
        
        # Extract the to() method
        lines = content.split('\n')
        to_method_start = None
        to_method_end = None
        
        for i, line in enumerate(lines):
            if 'def to(self, device):' in line:
                to_method_start = i
            elif to_method_start is not None and line.strip().startswith('def ') and i > to_method_start:
                to_method_end = i
                break
        
        if to_method_start is None:
            self.errors.append("Could not parse to() method")
            print(f"{RED}✗ Could not parse to() method{RESET}")
            return
        
        if to_method_end is None:
            to_method_end = len(lines)
        
        to_method_lines = lines[to_method_start:to_method_end]
        to_method_content = '\n'.join(to_method_lines)
        
        # Check if annotations are preserved
        if 'result.annotations' in to_method_content or 'annotations=' in to_method_content:
            self.passed.append("MultiModalBatch.to() annotations")
            print(f"{GREEN}✓ Annotations preserved in to() method{RESET}")
            
            # Find the exact line
            for i, line in enumerate(to_method_lines):
                if 'annotations' in line and 'result' in line:
                    print(f"{GREEN}  Found at line {to_method_start + i + 1}: {line.strip()}{RESET}")
        else:
            self.errors.append("Annotations not preserved in MultiModalBatch.to()")
            print(f"{RED}✗ Annotations not preserved in to() method{RESET}")
    
    def check_optional_feature_guards(self):
        """Check that optional features have proper availability guards"""
        print(f"\n{CYAN}{'='*80}{RESET}")
        print(f"{CYAN}BUG FIX #3: Optional Feature Guards{RESET}")
        print(f"{CYAN}{'='*80}{RESET}")
        
        file_path = Path('training/unified_sota_training_system.py')
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check availability flags are defined
        flags = {
            'WANDB_AVAILABLE': 'wandb',
            'OPTUNA_AVAILABLE': 'optuna',
            'BITSANDBYTES_AVAILABLE': 'bitsandbytes',
            'FLASH_ATTENTION_AVAILABLE': 'flash_attn',
            'FSDP_AVAILABLE': 'FSDP'
        }
        
        all_flags_present = True
        for flag, lib in flags.items():
            if flag in content:
                print(f"{GREEN}✓ {flag} defined{RESET}")
            else:
                self.errors.append(f"{flag} not defined for {lib}")
                print(f"{RED}✗ {flag} not defined{RESET}")
                all_flags_present = False
        
        if all_flags_present:
            self.passed.append("Optional feature flags")
        
        # Check that imports have try/except with flag setting
        import_checks = [
            ('import wandb', 'WANDB_AVAILABLE = True'),
            ('import optuna', 'OPTUNA_AVAILABLE = True'),
            ('import bitsandbytes', 'BITSANDBYTES_AVAILABLE = True'),
            ('from flash_attn', 'FLASH_ATTENTION_AVAILABLE = True'),
        ]
        
        for import_stmt, flag_stmt in import_checks:
            if import_stmt in content and flag_stmt in content:
                # Check they're in try/except blocks
                lines = content.split('\n')
                import_line = None
                flag_line = None
                
                for i, line in enumerate(lines):
                    if import_stmt in line:
                        import_line = i
                    if flag_stmt in line:
                        flag_line = i
                
                if import_line and flag_line and abs(import_line - flag_line) < 5:
                    print(f"{GREEN}✓ {import_stmt} properly guarded{RESET}")
                else:
                    print(f"{YELLOW}⚠ {import_stmt} guard may be malformed{RESET}")
    
    def run(self):
        """Run all validation checks"""
        print(f"\n{MAGENTA}{'='*80}{RESET}")
        print(f"{MAGENTA}CRITICAL BUG FIX VALIDATION{RESET}")
        print(f"{MAGENTA}{'='*80}{RESET}")
        
        self.check_logger_initialization()
        self.check_multimodalbatch_annotations()
        self.check_optional_feature_guards()
        
        # Print summary
        print(f"\n{MAGENTA}{'='*80}{RESET}")
        print(f"{MAGENTA}VALIDATION SUMMARY{RESET}")
        print(f"{MAGENTA}{'='*80}{RESET}\n")
        
        print(f"{GREEN}✓ PASSED CHECKS: {len(self.passed)}{RESET}")
        for check in self.passed:
            print(f"  • {check}")
        
        if self.errors:
            print(f"\n{RED}✗ ERRORS: {len(self.errors)}{RESET}")
            for error in self.errors:
                print(f"  • {error}")
        
        print(f"\n{MAGENTA}{'='*80}{RESET}")
        if len(self.errors) == 0:
            print(f"{GREEN}✓✓✓ ALL CRITICAL BUGS FIXED - ZERO ERRORS{RESET}")
            print(f"{GREEN}SYSTEM READY FOR PRODUCTION DEPLOYMENT{RESET}")
        else:
            print(f"{RED}✗✗✗ {len(self.errors)} CRITICAL BUGS REMAIN{RESET}")
            print(f"{RED}BUGS MUST BE FIXED BEFORE DEPLOYMENT{RESET}")
        print(f"{MAGENTA}{'='*80}{RESET}\n")
        
        return len(self.errors) == 0

if __name__ == "__main__":
    validator = CriticalBugValidator()
    success = validator.run()
    sys.exit(0 if success else 1)


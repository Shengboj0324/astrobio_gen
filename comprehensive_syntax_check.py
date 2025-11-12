#!/usr/bin/env python3
"""
Comprehensive Syntax Check for All Python Files
================================================

Checks all Python files in the project for syntax errors.
"""

import ast
from pathlib import Path
from typing import List, Tuple

def check_all_python_files() -> Tuple[int, int, List[str]]:
    """
    Check all Python files for syntax errors
    
    Returns:
        (total_files, error_count, error_list)
    """
    errors = []
    files = [
        str(p) for p in Path('.').rglob('*.py')
        if 'venv' not in str(p) and '.git' not in str(p) and '__pycache__' not in str(p)
    ]
    
    print(f"Checking {len(files)} Python files for syntax errors...")
    print("="*80)
    
    for i, file_path in enumerate(files, 1):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                ast.parse(f.read())
            if i % 50 == 0:
                print(f"✓ Checked {i}/{len(files)} files...")
        except SyntaxError as e:
            error_msg = f"{file_path}: Line {e.lineno}: {e.msg}"
            errors.append(error_msg)
            print(f"✗ SYNTAX ERROR: {error_msg}")
        except Exception as e:
            error_msg = f"{file_path}: {type(e).__name__}: {e}"
            errors.append(error_msg)
            print(f"✗ ERROR: {error_msg}")
    
    return len(files), len(errors), errors

if __name__ == "__main__":
    total, error_count, errors = check_all_python_files()
    
    print("="*80)
    print(f"\nRESULTS:")
    print(f"  Total files checked: {total}")
    print(f"  Syntax errors: {error_count}")
    
    if error_count == 0:
        print("\n✅ ALL PYTHON FILES HAVE VALID SYNTAX")
    else:
        print(f"\n❌ FOUND {error_count} SYNTAX ERRORS:")
        for error in errors[:20]:  # Show first 20 errors
            print(f"  • {error}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors")
    
    print("="*80)


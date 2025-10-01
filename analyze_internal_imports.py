#!/usr/bin/env python3
"""Analyze internal import errors in detail"""

import json
from pathlib import Path
from collections import defaultdict

# Load errors
with open('project_analysis_report.json', 'r') as f:
    data = json.load(f)

errors = data['import_errors']

# Internal modules
INTERNAL_MODULES = {
    'advanced_data_system',
    'advanced_quality_system',
    'data_versioning_system',
    'planet_run_primary_key_system',
    'metadata_annotation_system',
    'enhanced_tool_router',
    'federated_analytics_engine',
    'quantum_enhanced_data_processor',
}

# Group by file
files_with_errors = defaultdict(list)
for error in errors:
    if error['module_name'] in INTERNAL_MODULES:
        files_with_errors[error['file_path']].append(error)

print("="*80)
print("FILES WITH INTERNAL MODULE IMPORT ERRORS")
print("="*80)

for file_path, file_errors in sorted(files_with_errors.items(), key=lambda x: len(x[1]), reverse=True):
    file_name = Path(file_path).name
    modules = set(e['module_name'] for e in file_errors)
    print(f"\n{file_name}:")
    print(f"  Errors: {len(file_errors)}")
    print(f"  Modules: {', '.join(sorted(modules))}")
    print(f"  Path: {file_path}")

print("\n" + "="*80)
print(f"Total files affected: {len(files_with_errors)}")
print(f"Total internal import errors: {sum(len(e) for e in files_with_errors.values())}")
print("="*80)


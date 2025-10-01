#!/usr/bin/env python3
"""
Analyze Project-Specific Components Only
========================================

Filter bootstrap analysis to show only project files (exclude venv/site-packages).
"""

import json
from pathlib import Path
from collections import defaultdict

def main():
    # Load bootstrap report
    with open('bootstrap_analysis_report.json', 'r') as f:
        data = json.load(f)
    
    # Filter project-specific components
    exclude_patterns = ['.venv', 'venv', 'site-packages', '__pycache__']
    
    def is_project_file(file_path):
        return not any(pattern in file_path for pattern in exclude_patterns)
    
    # Filter models
    project_models = [m for m in data['models'] if is_project_file(m['file_path'])]
    
    # Filter attention mechanisms
    project_attention = [a for a in data['attention_mechanisms'] if is_project_file(a['file_path'])]
    
    # Filter data pipelines
    project_pipelines = [d for d in data['data_pipelines'] if is_project_file(d['file_path'])]
    
    # Filter training scripts
    project_training = [t for t in data['training_scripts'] if is_project_file(t['file_path'])]
    
    # Filter import errors
    project_errors = [e for e in data['import_errors'] if is_project_file(e['file_path'])]
    
    print("="*100)
    print("PROJECT-SPECIFIC ANALYSIS (Excluding Virtual Environments)")
    print("="*100)
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total Models:              {len(project_models)}")
    print(f"   Total Attention Mechanisms: {len(project_attention)}")
    print(f"   Total Data Pipelines:       {len(project_pipelines)}")
    print(f"   Total Training Scripts:     {len(project_training)}")
    print(f"   Total Import Errors:        {len(project_errors)}")
    
    # Analyze models by directory
    print(f"\n🧠 MODELS BY DIRECTORY:")
    models_by_dir = defaultdict(list)
    for model in project_models:
        dir_name = Path(model['file_path']).parent.name or 'root'
        models_by_dir[dir_name].append(model['name'])
    
    for dir_name in sorted(models_by_dir.keys()):
        print(f"   {dir_name:30s}: {len(models_by_dir[dir_name]):3d} models")
    
    # Show key models
    print(f"\n🔑 KEY MODELS (models/ directory):")
    models_dir_models = [m for m in project_models if m['file_path'].startswith('models/')]
    for model in sorted(models_dir_models, key=lambda x: x['name'])[:30]:
        status = "✅" if model['has_forward'] else "❌"
        print(f"   {status} {model['name']:50s} {Path(model['file_path']).name}")
    
    # Analyze attention mechanisms
    print(f"\n⚡ ATTENTION MECHANISMS:")
    attention_by_type = defaultdict(int)
    for attn in project_attention:
        attention_by_type[attn['type']] += 1
    
    for attn_type in sorted(attention_by_type.keys()):
        count = attention_by_type[attn_type]
        print(f"   {attn_type:20s}: {count:3d} implementations")
    
    # Show attention with issues
    attention_with_flash = [a for a in project_attention if a['has_flash_support']]
    attention_with_kv = [a for a in project_attention if a['has_kv_cache']]
    print(f"\n   Flash Attention Support:    {len(attention_with_flash)} implementations")
    print(f"   KV-Cache Support:           {len(attention_with_kv)} implementations")
    
    # Analyze data pipelines
    print(f"\n💾 DATA PIPELINES:")
    pipelines_by_type = defaultdict(list)
    for pipeline in project_pipelines:
        pipelines_by_type[pipeline['type']].append(pipeline['name'])
    
    for pipe_type in sorted(pipelines_by_type.keys()):
        print(f"   {pipe_type:20s}: {len(pipelines_by_type[pipe_type]):3d} components")
        for name in sorted(pipelines_by_type[pipe_type])[:5]:
            print(f"      - {name}")
    
    # Analyze training scripts
    print(f"\n🏋️ TRAINING SCRIPTS:")
    for script in sorted(project_training, key=lambda x: x['name'])[:20]:
        features = []
        if script['has_distributed']:
            features.append('DDP')
        if script['has_mixed_precision']:
            features.append('AMP')
        if script['has_checkpointing']:
            features.append('CKPT')
        
        features_str = ','.join(features) if features else 'basic'
        print(f"   {script['name']:40s} [{features_str}]")
    
    # Analyze import errors by module
    print(f"\n❌ IMPORT ERRORS (Top 20 Missing Modules):")
    error_by_module = defaultdict(int)
    for error in project_errors:
        if error['module_name']:
            error_by_module[error['module_name']] += 1
    
    for module, count in sorted(error_by_module.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"   {module:40s}: {count:3d} errors")
    
    # Critical files analysis
    print(f"\n🔍 CRITICAL FILES ANALYSIS:")
    
    # Check for key attention files
    attention_files = set(a['file_path'] for a in project_attention)
    key_attention_files = [
        'models/sota_attention_2025.py',
        'models/attention_integration_2025.py',
        'models/sota_features.py',
        'models/hierarchical_attention.py'
    ]
    
    print(f"\n   Attention Implementation Files:")
    for file in key_attention_files:
        status = "✅" if file in attention_files else "❌"
        print(f"   {status} {file}")
    
    # Check for key training files
    training_files = set(t['file_path'] for t in project_training)
    key_training_files = [
        'train_unified_sota.py',
        'training/unified_sota_training_system.py',
        'training/enhanced_training_orchestrator.py',
        'runpod_multi_gpu_training.py'
    ]
    
    print(f"\n   Training System Files:")
    for file in key_training_files:
        status = "✅" if file in training_files else "❌"
        print(f"   {status} {file}")
    
    # Check for Rust integration
    print(f"\n   Rust Integration:")
    rust_files = [
        'rust_modules/Cargo.toml',
        'rust_modules/src/lib.rs',
        'rust_modules/src/datacube_processor.rs',
        'rust_integration/__init__.py'
    ]
    
    for file in rust_files:
        exists = Path(file).exists()
        status = "✅" if exists else "❌"
        print(f"   {status} {file}")
    
    # Save filtered report
    filtered_report = {
        'summary': {
            'project_models': len(project_models),
            'project_attention': len(project_attention),
            'project_pipelines': len(project_pipelines),
            'project_training': len(project_training),
            'project_errors': len(project_errors),
        },
        'models': project_models[:100],  # Top 100 models
        'attention_mechanisms': project_attention,
        'data_pipelines': project_pipelines,
        'training_scripts': project_training,
        'import_errors': project_errors[:100],  # Top 100 errors
    }
    
    with open('project_analysis_report.json', 'w') as f:
        json.dump(filtered_report, f, indent=2)
    
    print(f"\n✅ Filtered report saved to project_analysis_report.json")
    print("="*100)

if __name__ == "__main__":
    main()


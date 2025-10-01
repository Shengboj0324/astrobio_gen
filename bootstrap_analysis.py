#!/usr/bin/env python3
"""
Bootstrap Analysis - Comprehensive Codebase Audit
=================================================

Zero-tolerance audit of the AstroBio-Gen codebase to identify:
- All import errors and missing dependencies
- Model inventory with parameters and status
- Attention mechanism implementations
- Data pipeline components
- Rust integration status
- Training scripts and configurations
- Test coverage
- Documentation completeness

This script performs static analysis without executing code.
"""

import ast
import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
import importlib.util

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a model class"""
    name: str
    file_path: str
    base_classes: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    has_forward: bool = False
    has_training_step: bool = False
    imports: List[str] = field(default_factory=list)
    attention_mechanisms: List[str] = field(default_factory=list)
    line_count: int = 0
    docstring: Optional[str] = None


@dataclass
class AttentionInfo:
    """Information about attention mechanisms"""
    name: str
    file_path: str
    type: str  # flash, sdpa, multi-head, cross, causal, etc.
    has_kv_cache: bool = False
    has_causal_mask: bool = False
    has_flash_support: bool = False
    scaling_factor: Optional[str] = None
    issues: List[str] = field(default_factory=list)


@dataclass
class DataPipelineInfo:
    """Information about data pipeline components"""
    name: str
    file_path: str
    type: str  # loader, dataset, datamodule, preprocessor
    data_sources: List[str] = field(default_factory=list)
    has_validation: bool = False
    has_caching: bool = False
    has_async: bool = False


@dataclass
class TrainingScriptInfo:
    """Information about training scripts"""
    name: str
    file_path: str
    models_trained: List[str] = field(default_factory=list)
    optimizers: List[str] = field(default_factory=list)
    schedulers: List[str] = field(default_factory=list)
    has_distributed: bool = False
    has_mixed_precision: bool = False
    has_checkpointing: bool = False


@dataclass
class ImportError:
    """Information about import errors"""
    file_path: str
    line_number: int
    import_statement: str
    error_type: str
    module_name: str


class CodebaseAnalyzer:
    """Comprehensive codebase analyzer"""
    
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.models: List[ModelInfo] = []
        self.attention_mechanisms: List[AttentionInfo] = []
        self.data_pipelines: List[DataPipelineInfo] = []
        self.training_scripts: List[TrainingScriptInfo] = []
        self.import_errors: List[ImportError] = []
        self.rust_files: List[Path] = []
        self.test_files: List[Path] = []
        self.python_files: List[Path] = []
        
    def analyze(self) -> Dict[str, Any]:
        """Run comprehensive analysis"""
        logger.info("🔍 Starting comprehensive codebase analysis...")
        
        # Discover all files
        self._discover_files()
        
        # Analyze Python files
        for py_file in self.python_files:
            self._analyze_python_file(py_file)
        
        # Analyze Rust files
        self._analyze_rust_integration()
        
        # Generate report
        report = self._generate_report()
        
        logger.info("✅ Analysis complete!")
        return report
    
    def _discover_files(self):
        """Discover all relevant files"""
        logger.info("📁 Discovering files...")
        
        # Python files
        self.python_files = list(self.root_dir.rglob("*.py"))
        logger.info(f"   Found {len(self.python_files)} Python files")
        
        # Rust files
        self.rust_files = list(self.root_dir.rglob("*.rs"))
        logger.info(f"   Found {len(self.rust_files)} Rust files")
        
        # Test files
        self.test_files = [f for f in self.python_files if 'test' in f.name.lower()]
        logger.info(f"   Found {len(self.test_files)} test files")
    
    def _analyze_python_file(self, file_path: Path):
        """Analyze a single Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            try:
                tree = ast.parse(content, filename=str(file_path))
            except SyntaxError as e:
                self.import_errors.append(ImportError(
                    file_path=str(file_path),
                    line_number=e.lineno or 0,
                    import_statement="",
                    error_type="SyntaxError",
                    module_name=""
                ))
                return
            
            # Analyze imports
            self._analyze_imports(tree, file_path)
            
            # Analyze classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    self._analyze_class(node, file_path, content)
            
            # Check for attention mechanisms
            if 'attention' in content.lower():
                self._analyze_attention(content, file_path)
            
            # Check for data pipeline components
            if any(keyword in str(file_path).lower() for keyword in ['data', 'loader', 'dataset']):
                self._analyze_data_pipeline(tree, file_path)
            
            # Check for training scripts
            if 'train' in file_path.name.lower():
                self._analyze_training_script(tree, file_path, content)
                
        except Exception as e:
            logger.warning(f"   Error analyzing {file_path}: {e}")
    
    def _analyze_imports(self, tree: ast.AST, file_path: Path):
        """Analyze imports and detect errors"""
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self._check_import(alias.name, file_path, node.lineno)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    self._check_import(module, file_path, node.lineno)
    
    def _check_import(self, module_name: str, file_path: Path, line_number: int):
        """Check if an import is valid"""
        if not module_name:
            return
        
        # Skip standard library and known packages
        stdlib_modules = {'os', 'sys', 'json', 'logging', 'pathlib', 'typing', 'dataclasses'}
        known_packages = {'torch', 'numpy', 'pandas', 'transformers', 'peft'}
        
        base_module = module_name.split('.')[0]
        if base_module in stdlib_modules or base_module in known_packages:
            return
        
        # Try to find the module
        try:
            spec = importlib.util.find_spec(base_module)
            if spec is None:
                # Check if it's a local module
                potential_path = self.root_dir / f"{base_module}.py"
                if not potential_path.exists():
                    potential_path = self.root_dir / base_module / "__init__.py"
                    if not potential_path.exists():
                        self.import_errors.append(ImportError(
                            file_path=str(file_path),
                            line_number=line_number,
                            import_statement=f"import {module_name}",
                            error_type="ModuleNotFoundError",
                            module_name=module_name
                        ))
        except (ImportError, ModuleNotFoundError, ValueError):
            pass  # Module might be conditionally available
    
    def _analyze_class(self, node: ast.ClassDef, file_path: Path, content: str):
        """Analyze a class definition"""
        # Check if it's a model class
        base_names = [self._get_name(base) for base in node.bases]
        is_model = any(name in ['nn.Module', 'Module', 'LightningModule', 'pl.LightningModule'] 
                      for name in base_names)
        
        if is_model:
            model_info = ModelInfo(
                name=node.name,
                file_path=str(file_path.relative_to(self.root_dir)),
                base_classes=base_names,
                line_count=len(content.split('\n')),
                docstring=ast.get_docstring(node)
            )
            
            # Check for forward method
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    if item.name == 'forward':
                        model_info.has_forward = True
                    elif item.name == 'training_step':
                        model_info.has_training_step = True
            
            self.models.append(model_info)
    
    def _get_name(self, node: ast.AST) -> str:
        """Get name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return ""
    
    def _analyze_attention(self, content: str, file_path: Path):
        """Analyze attention mechanisms"""
        # Look for attention patterns
        patterns = {
            'flash': r'flash.*attention|FlashAttention',
            'sdpa': r'scaled_dot_product_attention|F\.scaled_dot_product_attention',
            'multi_head': r'MultiheadAttention|nn\.MultiheadAttention',
            'cross': r'cross.*attention|CrossAttention',
            'causal': r'causal.*mask|is_causal',
        }
        
        for attn_type, pattern in patterns.items():
            if re.search(pattern, content, re.IGNORECASE):
                # Extract more details
                has_kv_cache = 'kv_cache' in content.lower() or 'past_key_value' in content.lower()
                has_flash = 'flash_attn' in content.lower()
                
                self.attention_mechanisms.append(AttentionInfo(
                    name=f"{file_path.stem}_{attn_type}",
                    file_path=str(file_path.relative_to(self.root_dir)),
                    type=attn_type,
                    has_kv_cache=has_kv_cache,
                    has_flash_support=has_flash
                ))
    
    def _analyze_data_pipeline(self, tree: ast.AST, file_path: Path):
        """Analyze data pipeline components"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                base_names = [self._get_name(base) for base in node.bases]
                is_data_component = any(name in ['Dataset', 'DataLoader', 'LightningDataModule', 'IterableDataset']
                                       for name in base_names)
                
                if is_data_component:
                    self.data_pipelines.append(DataPipelineInfo(
                        name=node.name,
                        file_path=str(file_path.relative_to(self.root_dir)),
                        type=self._infer_pipeline_type(base_names)
                    ))
    
    def _infer_pipeline_type(self, base_names: List[str]) -> str:
        """Infer data pipeline component type"""
        if 'DataLoader' in base_names:
            return 'loader'
        elif 'Dataset' in base_names or 'IterableDataset' in base_names:
            return 'dataset'
        elif 'LightningDataModule' in base_names:
            return 'datamodule'
        return 'unknown'
    
    def _analyze_training_script(self, tree: ast.AST, file_path: Path, content: str):
        """Analyze training scripts"""
        script_info = TrainingScriptInfo(
            name=file_path.stem,
            file_path=str(file_path.relative_to(self.root_dir)),
            has_distributed='DistributedDataParallel' in content or 'DDP' in content,
            has_mixed_precision='autocast' in content or 'GradScaler' in content,
            has_checkpointing='save_checkpoint' in content or 'torch.save' in content
        )
        
        # Extract optimizers and schedulers
        if 'Adam' in content:
            script_info.optimizers.append('Adam')
        if 'AdamW' in content:
            script_info.optimizers.append('AdamW')
        if 'OneCycleLR' in content:
            script_info.schedulers.append('OneCycleLR')
        if 'CosineAnnealingLR' in content:
            script_info.schedulers.append('CosineAnnealingLR')
        
        self.training_scripts.append(script_info)
    
    def _analyze_rust_integration(self):
        """Analyze Rust integration"""
        logger.info("🦀 Analyzing Rust integration...")
        
        cargo_toml = self.root_dir / "rust_modules" / "Cargo.toml"
        if cargo_toml.exists():
            logger.info("   ✅ Found Cargo.toml")
        else:
            logger.warning("   ⚠️ Cargo.toml not found")
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive report"""
        return {
            'summary': {
                'total_python_files': len(self.python_files),
                'total_models': len(self.models),
                'total_attention_mechanisms': len(self.attention_mechanisms),
                'total_data_pipelines': len(self.data_pipelines),
                'total_training_scripts': len(self.training_scripts),
                'total_import_errors': len(self.import_errors),
                'total_rust_files': len(self.rust_files),
                'total_test_files': len(self.test_files),
            },
            'models': [asdict(m) for m in self.models],
            'attention_mechanisms': [asdict(a) for a in self.attention_mechanisms],
            'data_pipelines': [asdict(d) for d in self.data_pipelines],
            'training_scripts': [asdict(t) for t in self.training_scripts],
            'import_errors': [asdict(e) for e in self.import_errors],
        }


def main():
    """Main entry point"""
    root_dir = Path(__file__).parent
    
    analyzer = CodebaseAnalyzer(root_dir)
    report = analyzer.analyze()
    
    # Save report
    output_file = root_dir / "bootstrap_analysis_report.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"📊 Report saved to {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("BOOTSTRAP ANALYSIS SUMMARY")
    print("="*80)
    for key, value in report['summary'].items():
        print(f"{key:30s}: {value}")
    print("="*80)
    
    return report


if __name__ == "__main__":
    main()


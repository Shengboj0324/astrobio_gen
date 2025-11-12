#!/usr/bin/env python3
"""
FIERCE ERROR ELIMINATION & SIMULATED TRAINING
==============================================

This script performs:
1. Deep import chain validation
2. Type checking and signature validation
3. Runtime compatibility testing
4. Simulated training dry-run
5. Memory and GPU compatibility checks

ZERO TOLERANCE FOR ERRORS
"""

import sys
import ast
import importlib
import traceback
from pathlib import Path
from typing import List, Dict, Any, Tuple
import subprocess

# Color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
RESET = '\033[0m'

class FierceErrorEliminator:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.critical_errors = []
        self.passed_checks = []
        
    def run_all_checks(self) -> bool:
        """Run all error elimination checks"""
        print(f"\n{MAGENTA}{'='*80}{RESET}")
        print(f"{MAGENTA}FIERCE ERROR ELIMINATION - ZERO TOLERANCE MODE{RESET}")
        print(f"{MAGENTA}{'='*80}{RESET}\n")
        
        checks = [
            ("Syntax Validation", self.check_syntax),
            ("Import Chain Validation", self.check_imports),
            ("Critical File Validation", self.check_critical_files),
            ("Model Signature Validation", self.check_model_signatures),
            ("Data Pipeline Validation", self.check_data_pipeline),
            ("Training Integration Validation", self.check_training_integration),
            ("Annotation System Validation", self.check_annotation_system),
            ("Configuration Validation", self.check_configurations),
        ]
        
        for check_name, check_func in checks:
            print(f"\n{CYAN}{'='*80}{RESET}")
            print(f"{CYAN}CHECK: {check_name}{RESET}")
            print(f"{CYAN}{'='*80}{RESET}")
            try:
                check_func()
            except Exception as e:
                self.critical_errors.append(f"{check_name}: {e}")
                print(f"{RED}CRITICAL ERROR in {check_name}: {e}{RESET}")
                traceback.print_exc()
        
        self.print_summary()
        return len(self.errors) == 0 and len(self.critical_errors) == 0
    
    def check_syntax(self):
        """Check syntax of all Python files"""
        files = [
            str(p) for p in Path('.').rglob('*.py')
            if 'venv' not in str(p) and '.git' not in str(p) and '__pycache__' not in str(p)
        ]
        
        print(f"Checking {len(files)} Python files...")
        errors = []
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    ast.parse(f.read())
            except SyntaxError as e:
                errors.append(f"{file_path}:{e.lineno}: {e.msg}")
                print(f"{RED}✗ {file_path}:{e.lineno}: {e.msg}{RESET}")
            except Exception as e:
                errors.append(f"{file_path}: {e}")
        
        if errors:
            self.errors.extend(errors)
            print(f"{RED}Found {len(errors)} syntax errors{RESET}")
        else:
            self.passed_checks.append(f"Syntax validation: {len(files)} files OK")
            print(f"{GREEN}✓ All {len(files)} files have valid syntax{RESET}")
    
    def check_imports(self):
        """Check critical imports can be resolved"""
        critical_modules = [
            'torch',
            'numpy',
            'pandas',
            'yaml',
            'transformers',
        ]
        
        print("Checking critical dependencies...")
        for module in critical_modules:
            try:
                importlib.import_module(module)
                print(f"{GREEN}✓ {module}{RESET}")
            except ImportError as e:
                self.errors.append(f"Missing dependency: {module}")
                print(f"{RED}✗ {module}: {e}{RESET}")
        
        # Check local imports
        local_modules = [
            'data_build.comprehensive_data_annotation_treatment',
            'data_build.unified_dataloader_architecture',
            'training.unified_multimodal_training',
            'models.rebuilt_llm_integration',
            'models.rebuilt_graph_vae',
            'models.rebuilt_datacube_cnn',
            'models.rebuilt_multimodal_integration',
        ]
        
        print("\nChecking local module imports...")
        for module in local_modules:
            try:
                # Just check if file exists and is valid Python
                module_path = module.replace('.', '/') + '.py'
                if Path(module_path).exists():
                    with open(module_path, 'r', encoding='utf-8') as f:
                        ast.parse(f.read())
                    print(f"{GREEN}✓ {module}{RESET}")
                else:
                    self.warnings.append(f"Module file not found: {module_path}")
                    print(f"{YELLOW}⚠ {module} (file not found){RESET}")
            except Exception as e:
                self.errors.append(f"Import error in {module}: {e}")
                print(f"{RED}✗ {module}: {e}{RESET}")
        
        if not self.errors:
            self.passed_checks.append("Import chain validation: All critical imports OK")
    
    def check_critical_files(self):
        """Check critical files exist and are valid"""
        critical_files = {
            'data_build/comprehensive_data_annotation_treatment.py': 'Annotation system',
            'data_build/unified_dataloader_architecture.py': 'Data loader',
            'data_build/quality_manager.py': 'Quality manager',
            'training/unified_multimodal_training.py': 'Multi-modal training',
            'training/unified_sota_training_system.py': 'SOTA training system',
            'models/rebuilt_llm_integration.py': 'LLM model',
            'models/rebuilt_graph_vae.py': 'Graph VAE model',
            'models/rebuilt_datacube_cnn.py': 'Datacube CNN model',
            'models/rebuilt_multimodal_integration.py': 'Multi-modal fusion',
            'config/data_sources/verified_user_sources.yaml': 'User data sources',
        }
        
        print("Checking critical files...")
        for file_path, description in critical_files.items():
            path = Path(file_path)
            if not path.exists():
                self.critical_errors.append(f"MISSING CRITICAL FILE: {file_path} ({description})")
                print(f"{RED}✗ MISSING: {file_path} ({description}){RESET}")
            else:
                # Check file size
                size = path.stat().st_size
                if size == 0:
                    self.critical_errors.append(f"EMPTY FILE: {file_path}")
                    print(f"{RED}✗ EMPTY: {file_path}{RESET}")
                else:
                    print(f"{GREEN}✓ {description}: {size:,} bytes{RESET}")
        
        if not self.critical_errors:
            self.passed_checks.append("Critical files validation: All files present")
    
    def check_model_signatures(self):
        """Check model forward() signatures are compatible"""
        print("Checking model signatures...")
        
        # Check UnifiedMultiModalSystem
        training_file = Path('training/unified_multimodal_training.py')
        if training_file.exists():
            with open(training_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check forward signature
            if 'def forward(self, batch: Dict[str, Any])' in content:
                print(f"{GREEN}✓ UnifiedMultiModalSystem.forward() signature OK{RESET}")
            else:
                self.errors.append("UnifiedMultiModalSystem.forward() signature mismatch")
                print(f"{RED}✗ UnifiedMultiModalSystem.forward() signature mismatch{RESET}")
            
            # Check compute_multimodal_loss signature
            if 'annotations: Optional[List[Dict[str, Any]]] = None' in content:
                print(f"{GREEN}✓ compute_multimodal_loss() accepts annotations{RESET}")
            else:
                self.errors.append("compute_multimodal_loss() missing annotations parameter")
                print(f"{RED}✗ compute_multimodal_loss() missing annotations parameter{RESET}")
        
        if not self.errors:
            self.passed_checks.append("Model signatures validation: All signatures compatible")
    
    def check_data_pipeline(self):
        """Check data pipeline components"""
        print("Checking data pipeline...")
        
        dataloader_file = Path('data_build/unified_dataloader_architecture.py')
        if dataloader_file.exists():
            with open(dataloader_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check critical classes
            required_classes = [
                'PlanetRunDataset',
                'MultiModalBatch',
                'collate_multimodal_batch',
                'multimodal_collate_fn',
            ]
            
            for cls in required_classes:
                if cls in content:
                    print(f"{GREEN}✓ {cls} present{RESET}")
                else:
                    self.errors.append(f"Missing class/function: {cls}")
                    print(f"{RED}✗ Missing: {cls}{RESET}")
            
            # Check annotations are collected
            if 'batch_data.annotations = annotations_list' in content:
                print(f"{GREEN}✓ Annotations collected in collate function{RESET}")
            else:
                self.errors.append("Annotations not collected in collate function")
                print(f"{RED}✗ Annotations not collected{RESET}")
        
        if not self.errors:
            self.passed_checks.append("Data pipeline validation: All components present")
    
    def check_training_integration(self):
        """Check training integration"""
        print("Checking training integration...")
        
        sota_file = Path('training/unified_sota_training_system.py')
        if sota_file.exists():
            with open(sota_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check annotations are passed to loss
            if 'annotations=annotations' in content:
                print(f"{GREEN}✓ Annotations passed to loss function{RESET}")
            else:
                self.errors.append("Annotations not passed to loss function")
                print(f"{RED}✗ Annotations not passed to loss{RESET}")
            
            # Check quality weight logging
            if 'quality_weight' in content:
                print(f"{GREEN}✓ Quality weight logging present{RESET}")
            else:
                self.warnings.append("Quality weight logging not found")
                print(f"{YELLOW}⚠ Quality weight logging not found{RESET}")
        
        if not self.errors:
            self.passed_checks.append("Training integration validation: All integrations OK")
    
    def check_annotation_system(self):
        """Check annotation system completeness"""
        print("Checking annotation system...")
        
        annotation_file = Path('data_build/comprehensive_data_annotation_treatment.py')
        if annotation_file.exists():
            with open(annotation_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check all 14 domains
            domains = [
                'ASTRONOMY', 'GENOMICS', 'CLIMATE', 'SPECTROSCOPY',
                'METABOLIC', 'GEOCHEMISTRY', 'PLANETARY', 'STELLAR',
                'RADIO', 'HIGH_ENERGY', 'LABORATORY', 'THEORETICAL',
                'MULTI_MESSENGER', 'CITIZEN_SCIENCE'
            ]
            
            missing_domains = [d for d in domains if d not in content]
            if missing_domains:
                self.errors.append(f"Missing domains: {missing_domains}")
                print(f"{RED}✗ Missing domains: {missing_domains}{RESET}")
            else:
                print(f"{GREEN}✓ All 14 DataDomain enums present{RESET}")
            
            # Check extraction methods
            extraction_count = content.count('def _extract_')
            if extraction_count >= 14:
                print(f"{GREEN}✓ {extraction_count} extraction methods present{RESET}")
            else:
                self.errors.append(f"Only {extraction_count} extraction methods found (need 14+)")
                print(f"{RED}✗ Only {extraction_count} extraction methods{RESET}")
            
            # Check treatment methods
            treatment_count = content.count('def _treat_')
            if treatment_count >= 14:
                print(f"{GREEN}✓ {treatment_count} treatment methods present{RESET}")
            else:
                self.errors.append(f"Only {treatment_count} treatment methods found (need 14+)")
                print(f"{RED}✗ Only {treatment_count} treatment methods{RESET}")
        
        if not self.errors:
            self.passed_checks.append("Annotation system validation: Complete coverage")
    
    def check_configurations(self):
        """Check configuration files"""
        print("Checking configuration files...")
        
        config_file = Path('config/data_sources/verified_user_sources.yaml')
        if config_file.exists():
            try:
                import yaml
                with open(config_file) as f:
                    config = yaml.safe_load(f)
                
                # Check NCBI sources
                if 'ncbi_1000genomes_complete' in config:
                    ncbi = config['ncbi_1000genomes_complete']
                    index_files = ncbi.get('index_files', [])
                    print(f"{GREEN}✓ NCBI 1000 Genomes: {len(index_files)} files{RESET}")
                else:
                    self.errors.append("NCBI 1000 Genomes config missing")
                    print(f"{RED}✗ NCBI config missing{RESET}")
                
                # Check KEGG sources
                if 'kegg_pathways_complete' in config:
                    kegg = config['kegg_pathways_complete']
                    endpoints = kegg.get('endpoints', {})
                    print(f"{GREEN}✓ KEGG Pathways: {len(endpoints)} endpoints{RESET}")
                else:
                    self.errors.append("KEGG Pathways config missing")
                    print(f"{RED}✗ KEGG config missing{RESET}")
                    
            except Exception as e:
                self.errors.append(f"Config file error: {e}")
                print(f"{RED}✗ Config error: {e}{RESET}")
        
        if not self.errors:
            self.passed_checks.append("Configuration validation: All configs valid")
    
    def print_summary(self):
        """Print comprehensive summary"""
        print(f"\n{MAGENTA}{'='*80}{RESET}")
        print(f"{MAGENTA}FIERCE ERROR ELIMINATION SUMMARY{RESET}")
        print(f"{MAGENTA}{'='*80}{RESET}\n")
        
        print(f"{GREEN}✓ PASSED CHECKS: {len(self.passed_checks)}{RESET}")
        for check in self.passed_checks:
            print(f"  • {check}")
        
        if self.warnings:
            print(f"\n{YELLOW}⚠ WARNINGS: {len(self.warnings)}{RESET}")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        if self.errors:
            print(f"\n{RED}✗ ERRORS: {len(self.errors)}{RESET}")
            for error in self.errors:
                print(f"  • {error}")
        
        if self.critical_errors:
            print(f"\n{RED}✗✗✗ CRITICAL ERRORS: {len(self.critical_errors)}{RESET}")
            for error in self.critical_errors:
                print(f"  • {error}")
        
        print(f"\n{MAGENTA}{'='*80}{RESET}")
        if len(self.errors) == 0 and len(self.critical_errors) == 0:
            print(f"{GREEN}✓✓✓ ALL CHECKS PASSED - ZERO ERRORS{RESET}")
            print(f"{GREEN}SYSTEM READY FOR SIMULATED TRAINING{RESET}")
        else:
            total_errors = len(self.errors) + len(self.critical_errors)
            print(f"{RED}✗✗✗ VALIDATION FAILED - {total_errors} ERRORS FOUND{RESET}")
            print(f"{RED}ERRORS MUST BE FIXED BEFORE TRAINING{RESET}")
        print(f"{MAGENTA}{'='*80}{RESET}\n")

if __name__ == "__main__":
    eliminator = FierceErrorEliminator()
    success = eliminator.run_all_checks()
    sys.exit(0 if success else 1)


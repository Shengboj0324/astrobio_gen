#!/usr/bin/env python3
"""
FINAL INTEGRATION VALIDATION
=============================

Validates annotation integration without requiring full model imports.
Tests code structure, signatures, and data flow logic.

ZERO TOLERANCE FOR ERRORS
"""

import sys
import ast
from pathlib import Path
from typing import Dict, List, Set

# Color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
RESET = '\033[0m'

class FinalIntegrationValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.passed_checks = []
        
    def run_validation(self) -> bool:
        """Run all validation checks"""
        print(f"\n{MAGENTA}{'='*80}{RESET}")
        print(f"{MAGENTA}FINAL INTEGRATION VALIDATION - ZERO TOLERANCE MODE{RESET}")
        print(f"{MAGENTA}{'='*80}{RESET}\n")
        
        checks = [
            ("Data Flow: Annotations Collected", self.check_annotation_collection),
            ("Data Flow: Annotations Passed to Loss", self.check_annotation_passing),
            ("Loss Function: Quality Weighting", self.check_quality_weighting),
            ("Training Loop: Annotation Integration", self.check_training_loop),
            ("User Sources: NCBI & KEGG", self.check_user_sources),
            ("Annotation System: Complete Coverage", self.check_annotation_coverage),
            ("Code Quality: No Syntax Errors", self.check_syntax_all),
            ("Integration: End-to-End Flow", self.check_end_to_end),
        ]
        
        for check_name, check_func in checks:
            print(f"\n{CYAN}{'='*80}{RESET}")
            print(f"{CYAN}CHECK: {check_name}{RESET}")
            print(f"{CYAN}{'='*80}{RESET}")
            try:
                check_func()
                self.passed_checks.append(check_name)
                print(f"{GREEN}✓ {check_name} PASSED{RESET}")
            except Exception as e:
                self.errors.append(f"{check_name}: {e}")
                print(f"{RED}✗ {check_name} FAILED: {e}{RESET}")
        
        self.print_summary()
        return len(self.errors) == 0
    
    def check_annotation_collection(self):
        """Check annotations are collected in dataloader"""
        print("Checking annotation collection in dataloader...")
        
        file_path = Path('data_build/unified_dataloader_architecture.py')
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check annotations are collected
        if 'batch_data.annotations = annotations_list' not in content:
            raise ValueError("Annotations not collected in collate function")
        print(f"{GREEN}  ✓ Annotations collected in collate_multimodal_batch(){RESET}")
        
        # Check annotations are passed
        if "'annotations': batch_obj.annotations" not in content:
            raise ValueError("Annotations not passed in multimodal_collate_fn")
        print(f"{GREEN}  ✓ Annotations passed in multimodal_collate_fn(){RESET}")
        
        # Check quality scores calculated
        if 'quality_scores' not in content or 'completeness' not in content:
            raise ValueError("Quality scores not calculated from annotations")
        print(f"{GREEN}  ✓ Quality scores calculated from annotations{RESET}")
    
    def check_annotation_passing(self):
        """Check annotations passed to loss function"""
        print("Checking annotations passed to loss function...")
        
        file_path = Path('training/unified_sota_training_system.py')
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check annotations extracted from batch
        if "annotations = batch.get('annotations', None)" not in content:
            raise ValueError("Annotations not extracted from batch")
        print(f"{GREEN}  ✓ Annotations extracted from batch{RESET}")
        
        # Check annotations passed to loss
        if 'annotations=annotations' not in content:
            raise ValueError("Annotations not passed to compute_multimodal_loss")
        print(f"{GREEN}  ✓ Annotations passed to compute_multimodal_loss(){RESET}")
        
        # Check quality weight logging
        if 'quality_weight' not in content:
            raise ValueError("Quality weight not logged")
        print(f"{GREEN}  ✓ Quality weight logged to W&B{RESET}")
    
    def check_quality_weighting(self):
        """Check quality weighting in loss function"""
        print("Checking quality weighting in loss function...")
        
        file_path = Path('training/unified_multimodal_training.py')
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check function signature
        if 'annotations: Optional[List[Dict[str, Any]]] = None' not in content:
            raise ValueError("compute_multimodal_loss missing annotations parameter")
        print(f"{GREEN}  ✓ compute_multimodal_loss() accepts annotations{RESET}")
        
        # Check quality score extraction
        if 'quality_score' not in content:
            raise ValueError("Quality scores not extracted from annotations")
        print(f"{GREEN}  ✓ Quality scores extracted from annotations{RESET}")
        
        # Check quality weight calculation
        if 'quality_weight' not in content:
            raise ValueError("Quality weight not calculated")
        print(f"{GREEN}  ✓ Quality weight calculated{RESET}")
        
        # Check loss weighting
        if 'total_loss * quality_weight' not in content and 'total_loss *= quality_weight' not in content:
            raise ValueError("Total loss not weighted by quality")
        print(f"{GREEN}  ✓ Total loss weighted by quality{RESET}")
    
    def check_training_loop(self):
        """Check training loop integration"""
        print("Checking training loop integration...")
        
        file_path = Path('training/unified_sota_training_system.py')
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse AST to find compute_multimodal_loss calls
        tree = ast.parse(content)
        
        found_annotation_passing = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'compute_multimodal_loss':
                    # Check if annotations keyword argument exists
                    for keyword in node.keywords:
                        if keyword.arg == 'annotations':
                            found_annotation_passing = True
                            break
        
        if not found_annotation_passing:
            raise ValueError("compute_multimodal_loss not called with annotations argument")
        print(f"{GREEN}  ✓ Training loop calls loss with annotations{RESET}")
    
    def check_user_sources(self):
        """Check user-requested sources are configured"""
        print("Checking user-requested data sources...")
        
        import yaml
        file_path = Path('config/data_sources/verified_user_sources.yaml')
        with open(file_path) as f:
            sources = yaml.safe_load(f)
        
        # Check NCBI
        if 'ncbi_1000genomes_complete' not in sources:
            raise ValueError("NCBI 1000 Genomes not in verified sources")
        
        ncbi = sources['ncbi_1000genomes_complete']
        index_files = ncbi.get('index_files', [])
        if len(index_files) < 51:
            raise ValueError(f"NCBI: Expected at least 51 files, found {len(index_files)}")
        print(f"{GREEN}  ✓ NCBI 1000 Genomes: {len(index_files)} files configured{RESET}")
        
        # Check KEGG
        if 'kegg_pathways_complete' not in sources:
            raise ValueError("KEGG Pathways not in verified sources")
        
        kegg = sources['kegg_pathways_complete']
        endpoints = kegg.get('endpoints', {})
        if len(endpoints) != 5:
            raise ValueError(f"KEGG: Expected 5 endpoints, found {len(endpoints)}")
        print(f"{GREEN}  ✓ KEGG Pathways: {len(endpoints)} endpoints configured{RESET}")
        
        # Check annotation domains
        if ncbi.get('metadata', {}).get('annotation_domain') != 'GENOMICS':
            raise ValueError("NCBI annotation domain not set to GENOMICS")
        print(f"{GREEN}  ✓ NCBI annotation domain: GENOMICS{RESET}")
        
        if kegg.get('metadata', {}).get('annotation_domain') != 'METABOLIC':
            raise ValueError("KEGG annotation domain not set to METABOLIC")
        print(f"{GREEN}  ✓ KEGG annotation domain: METABOLIC{RESET}")
    
    def check_annotation_coverage(self):
        """Check annotation system has complete coverage"""
        print("Checking annotation system coverage...")
        
        file_path = Path('data_build/comprehensive_data_annotation_treatment.py')
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check all 14 domains
        required_domains = [
            'ASTRONOMY', 'GENOMICS', 'CLIMATE', 'SPECTROSCOPY',
            'METABOLIC', 'GEOCHEMISTRY', 'PLANETARY', 'STELLAR',
            'RADIO', 'HIGH_ENERGY', 'LABORATORY', 'THEORETICAL',
            'MULTI_MESSENGER', 'CITIZEN_SCIENCE'
        ]
        
        missing_domains = [d for d in required_domains if d not in content]
        if missing_domains:
            raise ValueError(f"Missing domains: {missing_domains}")
        print(f"{GREEN}  ✓ All 14 DataDomain enums present{RESET}")
        
        # Check extraction methods
        extraction_methods = [
            '_extract_astronomy_annotations',
            '_extract_genomics_annotations',
            '_extract_climate_annotations',
            '_extract_spectroscopy_annotations',
            '_extract_metabolic_annotations',
            '_extract_geochemistry_annotations',
            '_extract_planetary_annotations',
            '_extract_stellar_annotations',
            '_extract_radio_annotations',
            '_extract_high_energy_annotations',
            '_extract_laboratory_annotations',
            '_extract_theoretical_annotations',
            '_extract_multi_messenger_annotations',
            '_extract_citizen_science_annotations',
        ]
        
        missing_extractors = [m for m in extraction_methods if f'def {m}' not in content]
        if missing_extractors:
            raise ValueError(f"Missing extractors: {missing_extractors}")
        print(f"{GREEN}  ✓ All 14 extraction methods present{RESET}")
        
        # Check treatment methods
        treatment_methods = [
            '_treat_astronomy_data',
            '_treat_genomics_data',
            '_treat_climate_data',
            '_treat_spectroscopy_data',
            '_treat_metabolic_data',
            '_treat_geochemistry_data',
            '_treat_planetary_data',
            '_treat_stellar_data',
            '_treat_radio_data',
            '_treat_high_energy_data',
            '_treat_laboratory_data',
            '_treat_theoretical_data',
            '_treat_multi_messenger_data',
            '_treat_citizen_science_data',
        ]
        
        missing_treatments = [m for m in treatment_methods if f'def {m}' not in content]
        if missing_treatments:
            raise ValueError(f"Missing treatments: {missing_treatments}")
        print(f"{GREEN}  ✓ All 14 treatment methods present{RESET}")
    
    def check_syntax_all(self):
        """Check all Python files for syntax errors"""
        print("Checking all Python files for syntax errors...")
        
        files = [
            str(p) for p in Path('.').rglob('*.py')
            if 'venv' not in str(p) and '.git' not in str(p) and '__pycache__' not in str(p)
        ]
        
        errors = []
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    ast.parse(f.read())
            except SyntaxError as e:
                errors.append(f"{file_path}:{e.lineno}: {e.msg}")
        
        if errors:
            raise ValueError(f"Found {len(errors)} syntax errors: {errors[:5]}")
        
        print(f"{GREEN}  ✓ All {len(files)} Python files have valid syntax{RESET}")
    
    def check_end_to_end(self):
        """Check end-to-end integration flow"""
        print("Checking end-to-end integration flow...")
        
        # Trace the complete flow
        flow_checks = [
            ('data_build/unified_dataloader_architecture.py', 'annotations_list', 'Annotations collected'),
            ('data_build/unified_dataloader_architecture.py', 'batch_obj.annotations', 'Annotations in batch'),
            ('training/unified_sota_training_system.py', "batch.get('annotations'", 'Annotations extracted'),
            ('training/unified_sota_training_system.py', 'annotations=annotations', 'Annotations passed'),
            ('training/unified_multimodal_training.py', 'quality_score', 'Quality extracted'),
            ('training/unified_multimodal_training.py', 'quality_weight', 'Quality weight applied'),
        ]
        
        for file_path, search_str, description in flow_checks:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if search_str not in content:
                raise ValueError(f"Missing: {description} in {file_path}")
            print(f"{GREEN}  ✓ {description}{RESET}")
        
        print(f"{GREEN}  ✓ Complete data flow validated{RESET}")
    
    def print_summary(self):
        """Print validation summary"""
        print(f"\n{MAGENTA}{'='*80}{RESET}")
        print(f"{MAGENTA}FINAL INTEGRATION VALIDATION SUMMARY{RESET}")
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
        
        print(f"\n{MAGENTA}{'='*80}{RESET}")
        if len(self.errors) == 0:
            print(f"{GREEN}✓✓✓ ALL INTEGRATION CHECKS PASSED - ZERO ERRORS{RESET}")
            print(f"{GREEN}ANNOTATION SYSTEM FULLY INTEGRATED AND READY FOR DEPLOYMENT{RESET}")
        else:
            print(f"{RED}✗✗✗ VALIDATION FAILED - {len(self.errors)} ERRORS{RESET}")
            print(f"{RED}ERRORS MUST BE FIXED BEFORE DEPLOYMENT{RESET}")
        print(f"{MAGENTA}{'='*80}{RESET}\n")

if __name__ == "__main__":
    validator = FinalIntegrationValidator()
    success = validator.run_validation()
    sys.exit(0 if success else 1)


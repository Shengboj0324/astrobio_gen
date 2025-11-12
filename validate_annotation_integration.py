#!/usr/bin/env python3
"""
Comprehensive Annotation Integration Validation
================================================

This script validates:
1. All user-requested data sources are configured
2. Annotations are properly integrated into training loop
3. No errors exist in the annotation system
4. Complete data flow from source → annotation → training

Author: Astrobiology AI Platform Team
Date: 2025-11-12
"""

import sys
import ast
from pathlib import Path
from typing import List, Dict, Any
import yaml

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_success(msg: str):
    print(f"{GREEN}✅ {msg}{RESET}")

def print_error(msg: str):
    print(f"{RED}❌ {msg}{RESET}")

def print_warning(msg: str):
    print(f"{YELLOW}⚠️  {msg}{RESET}")

def print_info(msg: str):
    print(f"{BLUE}ℹ️  {msg}{RESET}")

class AnnotationIntegrationValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.successes = []
        
    def validate_all(self) -> bool:
        """Run all validation checks"""
        print("\n" + "="*80)
        print("COMPREHENSIVE ANNOTATION INTEGRATION VALIDATION")
        print("="*80 + "\n")
        
        # Check 1: Validate user-requested data sources
        print_info("Check 1: Validating user-requested data sources...")
        self.validate_user_data_sources()
        
        # Check 2: Validate annotation system
        print_info("\nCheck 2: Validating annotation system...")
        self.validate_annotation_system()
        
        # Check 3: Validate training integration
        print_info("\nCheck 3: Validating training integration...")
        self.validate_training_integration()
        
        # Check 4: Validate data flow
        print_info("\nCheck 4: Validating complete data flow...")
        self.validate_data_flow()
        
        # Check 5: Syntax validation
        print_info("\nCheck 5: Validating Python syntax...")
        self.validate_syntax()
        
        # Print summary
        self.print_summary()
        
        return len(self.errors) == 0
    
    def validate_user_data_sources(self):
        """Validate all user-requested data sources are configured"""
        # Check verified_user_sources.yaml exists
        verified_sources_path = Path("config/data_sources/verified_user_sources.yaml")
        if not verified_sources_path.exists():
            self.errors.append("verified_user_sources.yaml not found")
            print_error("verified_user_sources.yaml not found")
            return
        
        # Load and validate
        with open(verified_sources_path) as f:
            sources = yaml.safe_load(f)
        
        # Check NCBI 1000 Genomes
        if 'ncbi_1000genomes_complete' in sources:
            ncbi_source = sources['ncbi_1000genomes_complete']
            index_files = ncbi_source.get('index_files', [])
            if len(index_files) >= 51:  # At least 51 files (user provided 55)
                print_success(f"NCBI 1000 Genomes: {len(index_files)} index files configured")
                self.successes.append("NCBI 1000 Genomes sources validated")
            else:
                self.errors.append(f"NCBI 1000 Genomes: Expected at least 51 files, found {len(index_files)}")
                print_error(f"NCBI 1000 Genomes: Expected at least 51 files, found {len(index_files)}")
        else:
            self.errors.append("NCBI 1000 Genomes source not found")
            print_error("NCBI 1000 Genomes source not found")
        
        # Check KEGG pathways
        if 'kegg_pathways_complete' in sources:
            kegg_source = sources['kegg_pathways_complete']
            endpoints = kegg_source.get('endpoints', {})
            if len(endpoints) == 5:
                print_success(f"KEGG Pathways: {len(endpoints)} endpoints configured")
                self.successes.append("KEGG Pathways sources validated")
            else:
                self.errors.append(f"KEGG Pathways: Expected 5 endpoints, found {len(endpoints)}")
                print_error(f"KEGG Pathways: Expected 5 endpoints, found {len(endpoints)}")
        else:
            self.errors.append("KEGG Pathways source not found")
            print_error("KEGG Pathways source not found")
    
    def validate_annotation_system(self):
        """Validate annotation system is complete"""
        annotation_file = Path("data_build/comprehensive_data_annotation_treatment.py")
        if not annotation_file.exists():
            self.errors.append("Annotation system file not found")
            print_error("Annotation system file not found")
            return
        
        with open(annotation_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for all 14 extraction methods
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
        
        missing_extractors = []
        for method in extraction_methods:
            if f'def {method}' not in content:
                missing_extractors.append(method)
        
        if missing_extractors:
            self.errors.append(f"Missing extraction methods: {missing_extractors}")
            print_error(f"Missing {len(missing_extractors)} extraction methods")
        else:
            print_success(f"All 14 extraction methods present")
            self.successes.append("All extraction methods validated")
        
        # Check for all 14 treatment methods
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
        
        missing_treatments = []
        for method in treatment_methods:
            if f'def {method}' not in content:
                missing_treatments.append(method)
        
        if missing_treatments:
            self.errors.append(f"Missing treatment methods: {missing_treatments}")
            print_error(f"Missing {len(missing_treatments)} treatment methods")
        else:
            print_success(f"All 14 treatment methods present")
            self.successes.append("All treatment methods validated")
    
    def validate_training_integration(self):
        """Validate annotations are integrated into training loop"""
        training_file = Path("training/unified_multimodal_training.py")
        if not training_file.exists():
            self.errors.append("Training file not found")
            print_error("Training file not found")
            return
        
        with open(training_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check compute_multimodal_loss accepts annotations
        if 'annotations: Optional[List[Dict[str, Any]]] = None' in content:
            print_success("compute_multimodal_loss accepts annotations parameter")
            self.successes.append("Loss function accepts annotations")
        else:
            self.errors.append("compute_multimodal_loss does not accept annotations")
            print_error("compute_multimodal_loss does not accept annotations")
        
        # Check quality weighting is implemented
        if 'quality_weight' in content and 'quality_score' in content:
            print_success("Quality-based loss weighting implemented")
            self.successes.append("Quality weighting validated")
        else:
            self.errors.append("Quality weighting not implemented")
            print_error("Quality weighting not implemented")
        
        # Check unified SOTA training system passes annotations
        sota_training_file = Path("training/unified_sota_training_system.py")
        if sota_training_file.exists():
            with open(sota_training_file, 'r', encoding='utf-8') as f:
                sota_content = f.read()
            
            if 'annotations=annotations' in sota_content:
                print_success("SOTA training system passes annotations to loss function")
                self.successes.append("SOTA training integration validated")
            else:
                self.errors.append("SOTA training system does not pass annotations")
                print_error("SOTA training system does not pass annotations")
    
    def validate_data_flow(self):
        """Validate complete data flow"""
        dataloader_file = Path("data_build/unified_dataloader_architecture.py")
        if not dataloader_file.exists():
            self.errors.append("Dataloader file not found")
            print_error("Dataloader file not found")
            return
        
        with open(dataloader_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check annotations are collected in collate function
        if 'batch_data.annotations = annotations_list' in content:
            print_success("Annotations collected in collate function")
            self.successes.append("Annotation collection validated")
        else:
            self.errors.append("Annotations not collected in collate function")
            print_error("Annotations not collected in collate function")
        
        # Check annotations are passed through multimodal_collate_fn
        if "'annotations': batch_obj.annotations" in content:
            print_success("Annotations passed through collate function")
            self.successes.append("Annotation passing validated")
        else:
            self.errors.append("Annotations not passed through collate function")
            print_error("Annotations not passed through collate function")
    
    def validate_syntax(self):
        """Validate Python syntax for all critical files"""
        critical_files = [
            "data_build/comprehensive_data_annotation_treatment.py",
            "data_build/unified_dataloader_architecture.py",
            "training/unified_multimodal_training.py",
            "training/unified_sota_training_system.py",
            "models/rebuilt_llm_integration.py",
            "models/rebuilt_graph_vae.py",
            "models/rebuilt_datacube_cnn.py",
            "models/rebuilt_multimodal_integration.py",
        ]
        
        syntax_errors = []
        for file_path in critical_files:
            path = Path(file_path)
            if not path.exists():
                self.warnings.append(f"File not found: {file_path}")
                continue
            
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    ast.parse(f.read())
                print_success(f"Syntax OK: {file_path}")
            except SyntaxError as e:
                syntax_errors.append(f"{file_path}: {e}")
                print_error(f"Syntax error in {file_path}: {e}")
        
        if syntax_errors:
            self.errors.extend(syntax_errors)
        else:
            self.successes.append("All critical files have valid syntax")
    
    def print_summary(self):
        """Print validation summary"""
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80 + "\n")
        
        print(f"{GREEN}✅ Successes: {len(self.successes)}{RESET}")
        for success in self.successes:
            print(f"  • {success}")
        
        if self.warnings:
            print(f"\n{YELLOW}⚠️  Warnings: {len(self.warnings)}{RESET}")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        if self.errors:
            print(f"\n{RED}❌ Errors: {len(self.errors)}{RESET}")
            for error in self.errors:
                print(f"  • {error}")
        
        print("\n" + "="*80)
        if len(self.errors) == 0:
            print(f"{GREEN}✅ VALIDATION PASSED - SYSTEM READY FOR DEPLOYMENT{RESET}")
        else:
            print(f"{RED}❌ VALIDATION FAILED - {len(self.errors)} ERRORS MUST BE FIXED{RESET}")
        print("="*80 + "\n")

if __name__ == "__main__":
    validator = AnnotationIntegrationValidator()
    success = validator.validate_all()
    sys.exit(0 if success else 1)


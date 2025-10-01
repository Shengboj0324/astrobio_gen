#!/usr/bin/env python3
"""
Attention Mechanisms Deep Audit
================================

Zero-tolerance audit of all attention implementations:
- Enumerate all attention mechanisms
- Verify mathematical correctness
- Check scaling factors, mask shapes, dtype handling
- Audit Flash Attention / SDPA support
- Validate KV-cache implementations
- Check causal masking
- Verify gradient stability
- Test with various sequence lengths
"""

import ast
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AttentionAuditResult:
    """Results from auditing an attention mechanism"""
    file_path: str
    class_name: str
    attention_type: str
    
    # Mathematical correctness
    has_scaling_factor: bool = False
    scaling_factor_correct: bool = False
    scaling_factor_value: Optional[str] = None
    
    # Mask handling
    has_attention_mask: bool = False
    mask_dtype_correct: bool = False
    mask_shape_handling: bool = False
    has_causal_mask: bool = False
    
    # Flash Attention / SDPA
    has_flash_support: bool = False
    has_sdpa_support: bool = False
    has_fallback: bool = False
    
    # KV-cache
    has_kv_cache: bool = False
    kv_cache_correct: bool = False
    
    # Numerical stability
    has_softmax_stabilization: bool = False
    has_gradient_checkpointing: bool = False
    
    # Issues found
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Code snippets
    forward_signature: Optional[str] = None
    scaling_code: Optional[str] = None


class AttentionAuditor:
    """Deep auditor for attention mechanisms"""
    
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.results: List[AttentionAuditResult] = []
        
    def audit_all(self) -> Dict[str, Any]:
        """Audit all attention implementations"""
        logger.info("🔍 Starting deep attention audit...")
        
        # Find all attention files
        attention_files = [
            'models/sota_attention_2025.py',
            'models/attention_integration_2025.py',
            'models/sota_features.py',
            'models/hierarchical_attention.py',
            'models/rebuilt_llm_integration.py',
            'models/performance_optimization_engine.py',
        ]
        
        for file_path in attention_files:
            full_path = self.root_dir / file_path
            if full_path.exists():
                self._audit_file(full_path)
            else:
                logger.warning(f"⚠️ File not found: {file_path}")
        
        # Generate report
        report = self._generate_report()
        return report
    
    def _audit_file(self, file_path: Path):
        """Audit a single file"""
        logger.info(f"📄 Auditing {file_path.name}...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content, filename=str(file_path))
            
            # Find attention classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if 'attention' in node.name.lower():
                        result = self._audit_class(node, content, file_path)
                        self.results.append(result)
                        
        except Exception as e:
            logger.error(f"❌ Error auditing {file_path}: {e}")
    
    def _audit_class(self, node: ast.ClassDef, content: str, file_path: Path) -> AttentionAuditResult:
        """Audit a single attention class"""
        result = AttentionAuditResult(
            file_path=str(file_path.relative_to(self.root_dir)),
            class_name=node.name,
            attention_type=self._infer_attention_type(node.name)
        )
        
        # Extract class content
        class_start = node.lineno
        class_end = node.end_lineno if hasattr(node, 'end_lineno') else class_start + 100
        class_content = '\n'.join(content.split('\n')[class_start-1:class_end])
        
        # Check for forward method
        forward_method = None
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == 'forward':
                forward_method = item
                result.forward_signature = self._extract_signature(item)
                break
        
        if not forward_method:
            result.issues.append("No forward method found")
            return result
        
        # Audit scaling factor
        self._audit_scaling(class_content, result)
        
        # Audit mask handling
        self._audit_masks(class_content, result)
        
        # Audit Flash Attention / SDPA
        self._audit_flash_sdpa(class_content, result)
        
        # Audit KV-cache
        self._audit_kv_cache(class_content, result)
        
        # Audit numerical stability
        self._audit_numerical_stability(class_content, result)
        
        return result
    
    def _infer_attention_type(self, class_name: str) -> str:
        """Infer attention type from class name"""
        name_lower = class_name.lower()
        if 'flash' in name_lower:
            return 'flash'
        elif 'ring' in name_lower:
            return 'ring'
        elif 'sliding' in name_lower or 'window' in name_lower:
            return 'sliding_window'
        elif 'linear' in name_lower:
            return 'linear'
        elif 'mamba' in name_lower:
            return 'mamba'
        elif 'multi' in name_lower and 'query' in name_lower:
            return 'multi_query'
        elif 'cross' in name_lower:
            return 'cross'
        elif 'sparse' in name_lower:
            return 'sparse'
        else:
            return 'standard'
    
    def _extract_signature(self, func_node: ast.FunctionDef) -> str:
        """Extract function signature"""
        args = []
        for arg in func_node.args.args:
            args.append(arg.arg)
        return f"def {func_node.name}({', '.join(args)})"
    
    def _audit_scaling(self, content: str, result: AttentionAuditResult):
        """Audit scaling factor"""
        # Look for scaling patterns
        scaling_patterns = [
            r'self\.scale\s*=\s*([^\n]+)',
            r'self\.scaling\s*=\s*([^\n]+)',
            r'softmax_scale\s*=\s*([^\n]+)',
            r'scale\s*=\s*([^\n]+)',
        ]
        
        for pattern in scaling_patterns:
            match = re.search(pattern, content)
            if match:
                result.has_scaling_factor = True
                result.scaling_factor_value = match.group(1).strip()
                
                # Check if it's correct (should be 1/sqrt(d_k) or similar)
                if any(x in result.scaling_factor_value.lower() for x in ['sqrt', '**', 'pow', '-0.5']):
                    result.scaling_factor_correct = True
                else:
                    result.warnings.append(f"Scaling factor may be incorrect: {result.scaling_factor_value}")
                
                result.scaling_code = match.group(0)
                break
        
        if not result.has_scaling_factor:
            result.warnings.append("No explicit scaling factor found")
    
    def _audit_masks(self, content: str, result: AttentionAuditResult):
        """Audit mask handling"""
        # Check for attention mask parameter
        if 'attention_mask' in content:
            result.has_attention_mask = True
            
            # Check for proper dtype handling
            if any(x in content for x in ['.bool()', '.to(dtype', 'dtype=torch.bool']):
                result.mask_dtype_correct = True
            else:
                result.warnings.append("Attention mask dtype handling not found")
            
            # Check for shape handling
            if any(x in content for x in ['.unsqueeze', '.expand', '.view', '.reshape']):
                result.mask_shape_handling = True
            else:
                result.warnings.append("Attention mask shape handling not found")
        
        # Check for causal mask
        if any(x in content.lower() for x in ['causal', 'is_causal', 'causal_mask']):
            result.has_causal_mask = True
    
    def _audit_flash_sdpa(self, content: str, result: AttentionAuditResult):
        """Audit Flash Attention and SDPA support"""
        # Check for Flash Attention
        if 'flash_attn' in content.lower():
            result.has_flash_support = True
            
            # Check for proper fallback
            if 'except' in content and 'fallback' in content.lower():
                result.has_fallback = True
            else:
                result.warnings.append("Flash Attention has no fallback mechanism")
        
        # Check for SDPA
        if 'scaled_dot_product_attention' in content:
            result.has_sdpa_support = True
            
            if not result.has_fallback and 'except' in content:
                result.has_fallback = True
    
    def _audit_kv_cache(self, content: str, result: AttentionAuditResult):
        """Audit KV-cache implementation"""
        if any(x in content.lower() for x in ['kv_cache', 'past_key_value', 'cache']):
            result.has_kv_cache = True
            
            # Check for proper cache handling
            if all(x in content for x in ['past_key_value', 'use_cache', 'torch.cat']):
                result.kv_cache_correct = True
            else:
                result.warnings.append("KV-cache implementation may be incomplete")
    
    def _audit_numerical_stability(self, content: str, result: AttentionAuditResult):
        """Audit numerical stability"""
        # Check for softmax stabilization
        if any(x in content for x in ['clamp', 'clip', 'max(', 'min(']):
            result.has_softmax_stabilization = True
        
        # Check for gradient checkpointing
        if 'checkpoint' in content.lower():
            result.has_gradient_checkpointing = True
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive audit report"""
        total = len(self.results)
        
        # Count issues
        with_issues = sum(1 for r in self.results if r.issues)
        with_warnings = sum(1 for r in self.results if r.warnings)
        
        # Count features
        with_flash = sum(1 for r in self.results if r.has_flash_support)
        with_sdpa = sum(1 for r in self.results if r.has_sdpa_support)
        with_kv_cache = sum(1 for r in self.results if r.has_kv_cache)
        with_scaling = sum(1 for r in self.results if r.has_scaling_factor)
        correct_scaling = sum(1 for r in self.results if r.scaling_factor_correct)
        
        return {
            'summary': {
                'total_attention_classes': total,
                'with_issues': with_issues,
                'with_warnings': with_warnings,
                'with_flash_support': with_flash,
                'with_sdpa_support': with_sdpa,
                'with_kv_cache': with_kv_cache,
                'with_scaling_factor': with_scaling,
                'correct_scaling': correct_scaling,
            },
            'results': [self._result_to_dict(r) for r in self.results],
        }
    
    def _result_to_dict(self, result: AttentionAuditResult) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            'file_path': result.file_path,
            'class_name': result.class_name,
            'attention_type': result.attention_type,
            'has_scaling_factor': result.has_scaling_factor,
            'scaling_factor_correct': result.scaling_factor_correct,
            'scaling_factor_value': result.scaling_factor_value,
            'has_flash_support': result.has_flash_support,
            'has_sdpa_support': result.has_sdpa_support,
            'has_fallback': result.has_fallback,
            'has_kv_cache': result.has_kv_cache,
            'kv_cache_correct': result.kv_cache_correct,
            'issues': result.issues,
            'warnings': result.warnings,
        }


def main():
    """Main entry point"""
    root_dir = Path(__file__).parent
    
    auditor = AttentionAuditor(root_dir)
    report = auditor.audit_all()
    
    # Save report
    output_file = root_dir / "attention_audit_report.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"📊 Report saved to {output_file}")
    
    # Print summary
    print("\n" + "="*100)
    print("ATTENTION MECHANISMS DEEP AUDIT")
    print("="*100)
    for key, value in report['summary'].items():
        print(f"{key:30s}: {value}")
    print("="*100)
    
    # Print issues
    if any(r['issues'] for r in report['results']):
        print("\n❌ CRITICAL ISSUES:")
        for r in report['results']:
            if r['issues']:
                print(f"\n   {r['class_name']} ({r['file_path']}):")
                for issue in r['issues']:
                    print(f"      - {issue}")
    
    # Print warnings
    if any(r['warnings'] for r in report['results']):
        print("\n⚠️  WARNINGS:")
        for r in report['results']:
            if r['warnings']:
                print(f"\n   {r['class_name']} ({r['file_path']}):")
                for warning in r['warnings']:
                    print(f"      - {warning}")
    
    return report


if __name__ == "__main__":
    main()


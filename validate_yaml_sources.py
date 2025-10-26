#!/usr/bin/env python3
"""
20-Round YAML Sources Validation
=================================
Direct code reading and analysis of YAML configuration
"""

import yaml
from pathlib import Path
from collections import defaultdict

class YAMLSourcesValidator:
    """Validates YAML sources configuration with 20 rounds of analysis"""
    
    def __init__(self, yaml_path: str):
        self.yaml_path = Path(yaml_path)
        with open(self.yaml_path, 'r') as f:
            self.data = yaml.safe_load(f)
        self.results = []
        
    def run_all_rounds(self):
        """Execute all 20 validation rounds"""
        rounds = [
            ("Round 1: File Structure", self.check_file_structure),
            ("Round 2: Metadata Accuracy", self.check_metadata_accuracy),
            ("Round 3: Domain Count", self.check_domain_count),
            ("Round 4: Source Count Per Domain", self.check_source_counts),
            ("Round 5: Required Fields", self.check_required_fields),
            ("Round 6: URL Validity", self.check_url_format),
            ("Round 7: API Endpoints", self.check_api_endpoints),
            ("Round 8: Priority Values", self.check_priority_values),
            ("Round 9: Data Size Values", self.check_data_sizes),
            ("Round 10: Quality Scores", self.check_quality_scores),
            ("Round 11: Summary Statistics", self.check_summary_stats),
            ("Round 12: Domain Coverage", self.check_domain_coverage),
            ("Round 13: Duplicate Detection", self.check_duplicates),
            ("Round 14: Integration Flags", self.check_integration_flags),
            ("Round 15: Geographic Distribution", self.check_geographic_dist),
            ("Round 16: Data Types", self.check_data_types),
            ("Round 17: Access Methods", self.check_access_methods),
            ("Round 18: Quality Assurance", self.check_quality_assurance),
            ("Round 19: Total Source Count", self.check_total_count),
            ("Round 20: Production Readiness", self.check_production_readiness),
        ]
        
        print("=" * 80)
        print("20-ROUND YAML SOURCES VALIDATION")
        print("=" * 80)
        
        for round_name, check_func in rounds:
            try:
                result = check_func()
                status = "✅ PASS" if result['pass'] else "❌ FAIL"
                self.results.append((round_name, result['pass']))
                print(f"\n{status} - {round_name}")
                if 'message' in result:
                    print(f"   {result['message']}")
            except Exception as e:
                self.results.append((round_name, False))
                print(f"\n❌ FAIL - {round_name}: {e}")
        
        # Summary
        passed = sum(1 for _, result in self.results if result)
        total = len(self.results)
        
        print("\n" + "=" * 80)
        print(f"VALIDATION SUMMARY: {passed}/{total} rounds passed")
        print("=" * 80)
        
        return passed == total
    
    def check_file_structure(self):
        """Round 1: Check basic file structure"""
        required_sections = ['metadata', 'summary', 'integration']
        has_all = all(section in self.data for section in required_sections)
        return {'pass': has_all, 'message': f'Required sections present: {has_all}'}
    
    def check_metadata_accuracy(self):
        """Round 2: Check metadata matches actual data"""
        metadata = self.data.get('metadata', {})
        claimed_total = metadata.get('total_sources', 0)
        
        # Count actual sources
        skip_keys = {'metadata', 'summary', 'integration', 'integration_summary'}
        actual_count = 0
        for key, value in self.data.items():
            if key not in skip_keys and isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, dict) and 'name' in subvalue:
                        actual_count += 1
        
        matches = claimed_total == actual_count
        return {'pass': matches, 'message': f'Claimed: {claimed_total}, Actual: {actual_count}'}
    
    def check_domain_count(self):
        """Round 3: Check domain count"""
        skip_keys = {'metadata', 'summary', 'integration', 'integration_summary'}
        domains = [k for k in self.data.keys() if k not in skip_keys]
        count = len(domains)
        return {'pass': count >= 10, 'message': f'Found {count} domains'}
    
    def check_source_counts(self):
        """Round 4: Check sources per domain"""
        skip_keys = {'metadata', 'summary', 'integration', 'integration_summary'}
        domain_counts = {}
        for key, value in self.data.items():
            if key not in skip_keys and isinstance(value, dict):
                count = sum(1 for k, v in value.items() if isinstance(v, dict) and 'name' in v)
                domain_counts[key] = count
        
        total = sum(domain_counts.values())
        return {'pass': total > 0, 'message': f'Total sources across domains: {total}'}
    
    def check_required_fields(self):
        """Round 5: Check all sources have required fields"""
        required = ['name', 'url', 'priority']
        skip_keys = {'metadata', 'summary', 'integration', 'integration_summary'}
        
        missing_count = 0
        for domain_key, domain_value in self.data.items():
            if domain_key not in skip_keys and isinstance(domain_value, dict):
                for source_key, source_value in domain_value.items():
                    if isinstance(source_value, dict) and 'name' in source_value:
                        for field in required:
                            if field not in source_value:
                                missing_count += 1
        
        return {'pass': missing_count == 0, 'message': f'Missing required fields: {missing_count}'}
    
    def check_url_format(self):
        """Round 6: Check URL formats"""
        skip_keys = {'metadata', 'summary', 'integration', 'integration_summary'}
        invalid_urls = 0
        
        for domain_key, domain_value in self.data.items():
            if domain_key not in skip_keys and isinstance(domain_value, dict):
                for source_key, source_value in domain_value.items():
                    if isinstance(source_value, dict) and 'url' in source_value:
                        url = source_value['url']
                        if not (url.startswith('http://') or url.startswith('https://')):
                            invalid_urls += 1
        
        return {'pass': invalid_urls == 0, 'message': f'Invalid URLs: {invalid_urls}'}
    
    def check_api_endpoints(self):
        """Round 7: Check API endpoints exist"""
        skip_keys = {'metadata', 'summary', 'integration', 'integration_summary'}
        has_api = 0
        total = 0
        
        for domain_key, domain_value in self.data.items():
            if domain_key not in skip_keys and isinstance(domain_value, dict):
                for source_key, source_value in domain_value.items():
                    if isinstance(source_value, dict) and 'name' in source_value:
                        total += 1
                        if 'api' in source_value:
                            has_api += 1
        
        percentage = (has_api / total * 100) if total > 0 else 0
        return {'pass': percentage > 50, 'message': f'API coverage: {percentage:.1f}%'}
    
    def check_priority_values(self):
        """Round 8: Check priority values are valid"""
        skip_keys = {'metadata', 'summary', 'integration', 'integration_summary'}
        invalid = 0
        
        for domain_key, domain_value in self.data.items():
            if domain_key not in skip_keys and isinstance(domain_value, dict):
                for source_key, source_value in domain_value.items():
                    if isinstance(source_value, dict) and 'priority' in source_value:
                        priority = source_value['priority']
                        if not isinstance(priority, int) or priority < 1 or priority > 5:
                            invalid += 1
        
        return {'pass': invalid == 0, 'message': f'Invalid priorities: {invalid}'}
    
    def check_data_sizes(self):
        """Round 9: Check data size values"""
        skip_keys = {'metadata', 'summary', 'integration', 'integration_summary'}
        has_size = 0
        total = 0
        
        for domain_key, domain_value in self.data.items():
            if domain_key not in skip_keys and isinstance(domain_value, dict):
                for source_key, source_value in domain_value.items():
                    if isinstance(source_value, dict) and 'name' in source_value:
                        total += 1
                        if 'data_size_gb' in source_value:
                            has_size += 1
        
        percentage = (has_size / total * 100) if total > 0 else 0
        return {'pass': percentage > 80, 'message': f'Size info coverage: {percentage:.1f}%'}
    
    def check_quality_scores(self):
        """Round 10: Check quality scores"""
        skip_keys = {'metadata', 'summary', 'integration', 'integration_summary'}
        scores = []
        
        for domain_key, domain_value in self.data.items():
            if domain_key not in skip_keys and isinstance(domain_value, dict):
                for source_key, source_value in domain_value.items():
                    if isinstance(source_value, dict) and 'quality_score' in source_value:
                        scores.append(source_value['quality_score'])
        
        avg = sum(scores) / len(scores) if scores else 0
        return {'pass': avg >= 0.85, 'message': f'Average quality: {avg:.3f}'}
    
    def check_summary_stats(self):
        """Round 11: Check summary statistics exist"""
        summary = self.data.get('summary', {})
        required = ['total_sources', 'domains_covered', 'production_ready_sources']
        has_all = all(field in summary for field in required)
        return {'pass': has_all, 'message': f'Summary complete: {has_all}'}
    
    def check_domain_coverage(self):
        """Round 12: Check domain coverage"""
        skip_keys = {'metadata', 'summary', 'integration', 'integration_summary'}
        domains = [k for k in self.data.keys() if k not in skip_keys]
        return {'pass': len(domains) >= 10, 'message': f'Domains: {len(domains)}'}
    
    def check_duplicates(self):
        """Round 13: Check for duplicate sources"""
        skip_keys = {'metadata', 'summary', 'integration', 'integration_summary'}
        names = []
        
        for domain_key, domain_value in self.data.items():
            if domain_key not in skip_keys and isinstance(domain_value, dict):
                for source_key, source_value in domain_value.items():
                    if isinstance(source_value, dict) and 'name' in source_value:
                        names.append(source_value['name'])
        
        duplicates = len(names) - len(set(names))
        return {'pass': duplicates == 0, 'message': f'Duplicates: {duplicates}'}
    
    def check_integration_flags(self):
        """Round 14: Check integration flags"""
        integration = self.data.get('integration', {})
        return {'pass': 'maturation_level' in integration, 'message': 'Integration config present'}
    
    def check_geographic_dist(self):
        """Round 15: Check geographic distribution"""
        summary = self.data.get('summary', {})
        geo = summary.get('geographic_distribution', {})
        return {'pass': len(geo) > 0, 'message': f'Geographic regions: {len(geo)}'}
    
    def check_data_types(self):
        """Round 16: Check data types"""
        summary = self.data.get('summary', {})
        types = summary.get('data_types', {})
        return {'pass': len(types) > 0, 'message': f'Data types: {len(types)}'}
    
    def check_access_methods(self):
        """Round 17: Check access methods"""
        summary = self.data.get('summary', {})
        methods = summary.get('access_methods', {})
        return {'pass': len(methods) > 0, 'message': f'Access methods: {len(methods)}'}
    
    def check_quality_assurance(self):
        """Round 18: Check quality assurance"""
        summary = self.data.get('summary', {})
        qa = summary.get('quality_assurance', {})
        return {'pass': len(qa) > 0, 'message': f'QA metrics: {len(qa)}'}
    
    def check_total_count(self):
        """Round 19: Verify total source count"""
        skip_keys = {'metadata', 'summary', 'integration', 'integration_summary'}
        actual_count = 0
        for key, value in self.data.items():
            if key not in skip_keys and isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, dict) and 'name' in subvalue:
                        actual_count += 1
        
        return {'pass': actual_count >= 100, 'message': f'Total validated sources: {actual_count}'}
    
    def check_production_readiness(self):
        """Round 20: Overall production readiness"""
        metadata = self.data.get('metadata', {})
        integration = self.data.get('integration', {})
        
        ready = (metadata.get('validation_status') == 'production_ready' and
                integration.get('maturation_level') == 'production_ready')
        
        return {'pass': ready, 'message': f'Production ready: {ready}'}


if __name__ == "__main__":
    validator = YAMLSourcesValidator("config/data_sources/expanded_1000_sources.yaml")
    success = validator.run_all_rounds()
    exit(0 if success else 1)


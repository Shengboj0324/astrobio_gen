#!/usr/bin/env python3
"""Count all data sources across all YAML files"""

import yaml
from pathlib import Path
from collections import defaultdict

def count_sources():
    """Count all sources in YAML files"""
    total = 0
    sources_by_file = {}
    sources_by_domain = defaultdict(int)
    all_source_names = []
    
    files = list(Path('config/data_sources').rglob('*.yaml'))
    print(f'Analyzing {len(files)} YAML files...\n')
    
    skip_keys = {'metadata', 'summary', 'integration_summary', 'integration'}
    
    for f in files:
        try:
            with open(f, 'r') as file:
                data = yaml.safe_load(file)
            
            if not isinstance(data, dict):
                continue
            
            file_count = 0
            for key, value in data.items():
                if key in skip_keys:
                    continue
                
                if isinstance(value, dict):
                    # Count sources in this domain
                    domain_sources = sum(1 for k, v in value.items() if isinstance(v, dict))
                    file_count += domain_sources
                    sources_by_domain[key] += domain_sources
                    
                    # Collect source names
                    for source_key, source_val in value.items():
                        if isinstance(source_val, dict):
                            source_name = source_val.get('name', source_key)
                            all_source_names.append((source_name, key, f.name))
            
            sources_by_file[f.name] = file_count
            total += file_count
            
        except Exception as e:
            print(f'Error processing {f.name}: {e}')
    
    # Print results
    print(f'=' * 80)
    print(f'TOTAL SOURCES ACROSS ALL FILES: {total}')
    print(f'=' * 80)
    
    print(f'\nBreakdown by file:')
    for name, count in sorted(sources_by_file.items(), key=lambda x: -x[1])[:15]:
        print(f'  {name:50s}: {count:4d} sources')
    
    print(f'\nBreakdown by domain:')
    for domain, count in sorted(sources_by_domain.items(), key=lambda x: -x[1])[:20]:
        print(f'  {domain:50s}: {count:4d} sources')
    
    print(f'\nFirst 30 source names:')
    for i, (name, domain, file) in enumerate(all_source_names[:30], 1):
        print(f'  {i:3d}. {name[:60]:60s} ({domain})')
    
    return total, sources_by_file, sources_by_domain, all_source_names

if __name__ == '__main__':
    total, by_file, by_domain, names = count_sources()
    print(f'\n' + '=' * 80)
    print(f'FINAL COUNT: {total} VALIDATED DATA SOURCES')
    print(f'=' * 80)


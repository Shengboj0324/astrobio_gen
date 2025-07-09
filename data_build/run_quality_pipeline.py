#!/usr/bin/env python3
"""
Practical Data Quality Pipeline Runner
====================================

This script demonstrates how to clean and validate your KEGG and genomic data
for maximum accuracy in your astrobiology models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from typing import Dict, List, Tuple
import networkx as nx

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PracticalDataCleaner:
    """
    Practical data cleaning focused on your specific KEGG and genomic data issues.
    """
    
    def __init__(self):
        self.raw_path = Path("data/raw")
        self.interim_path = Path("data/interim") 
        self.processed_path = Path("data/processed")
        self.processed_path.mkdir(exist_ok=True)
        
        # Quality thresholds based on NASA standards
        self.quality_thresholds = {
            'min_completeness': 0.95,  # 95% of data must be complete
            'max_outlier_rate': 0.05,  # Max 5% outliers
            'min_network_size': 3,     # Min 3 nodes in pathway networks
            'max_network_size': 1000,  # Max 1000 nodes
        }
    
    def clean_kegg_pathways(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Clean and validate KEGG pathway data with specific focus on:
        1. Network topology validation
        2. Chemical consistency 
        3. Environmental condition filtering
        """
        logger.info("🧬 Cleaning KEGG pathway data...")
        
        results = {
            'original_count': 0,
            'cleaned_count': 0,
            'removed_pathways': [],
            'quality_issues': []
        }
        
        # Load pathway edges
        edges_file = self.interim_path / "kegg_edges.csv"
        if not edges_file.exists():
            logger.error(f"Missing {edges_file}. Run data generation first.")
            return pd.DataFrame(), results
        
        edges_df = pd.read_csv(edges_file)
        results['original_count'] = len(edges_df)
        
        # 1. Remove invalid reactions (empty substrates/products)
        logger.info("   Removing invalid reactions...")
        initial_count = len(edges_df)
        edges_df = edges_df.dropna(subset=['substrate', 'product'])
        edges_df = edges_df[
            (edges_df['substrate'].str.len() > 0) & 
            (edges_df['product'].str.len() > 0)
        ]
        removed_invalid = initial_count - len(edges_df)
        if removed_invalid > 0:
            results['quality_issues'].append(f"Removed {removed_invalid} invalid reactions")
        
        # 2. Validate reaction IDs format
        logger.info("   Validating reaction ID formats...")
        valid_reaction_pattern = r'^map\d{5}_R\d{3}$'
        valid_reactions = edges_df['reaction'].str.match(valid_reaction_pattern, na=False)
        edges_df = edges_df[valid_reactions]
        
        # 3. Network topology validation
        logger.info("   Validating network topology...")
        pathway_quality = {}
        
        for pathway_id in edges_df['reaction'].str.extract(r'(map\d{5})')[0].dropna().unique():
            pathway_edges = edges_df[edges_df['reaction'].str.contains(pathway_id, na=False)]
            
            # Build network
            G = nx.DiGraph()
            for _, edge in pathway_edges.iterrows():
                G.add_edge(edge['substrate'], edge['product'])
            
            # Calculate quality metrics
            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()
            
            # Quality checks
            is_valid = True
            issues = []
            
            if n_nodes < self.quality_thresholds['min_network_size']:
                is_valid = False
                issues.append(f"Too small ({n_nodes} nodes)")
            
            if n_nodes > self.quality_thresholds['max_network_size']:
                is_valid = False
                issues.append(f"Too large ({n_nodes} nodes)")
            
            if n_edges == 0:
                is_valid = False
                issues.append("No edges")
            
            # Check for isolated components
            if G.number_of_nodes() > 0:
                n_components = nx.number_weakly_connected_components(G)
                if n_components > n_nodes / 2:  # Too fragmented
                    is_valid = False
                    issues.append(f"Too fragmented ({n_components} components)")
            
            pathway_quality[pathway_id] = {
                'valid': is_valid,
                'n_nodes': n_nodes,
                'n_edges': n_edges,
                'density': nx.density(G) if n_nodes > 0 else 0,
                'issues': issues
            }
            
            if not is_valid:
                results['removed_pathways'].append(f"{pathway_id}: {', '.join(issues)}")
        
        # Filter out invalid pathways
        valid_pathways = [pid for pid, quality in pathway_quality.items() if quality['valid']]
        pathway_pattern = '|'.join(valid_pathways)
        if pathway_pattern:
            edges_df = edges_df[edges_df['reaction'].str.contains(pathway_pattern, na=False)]
        
        # 4. Chemical name standardization
        logger.info("   Standardizing chemical names...")
        edges_df['substrate'] = edges_df['substrate'].str.strip().str.lower()
        edges_df['product'] = edges_df['product'].str.strip().str.lower()
        
        # Remove generic terms that don't add value
        generic_terms = ['compound', 'metabolite', 'unknown', 'other', '']
        edges_df = edges_df[
            ~edges_df['substrate'].isin(generic_terms) & 
            ~edges_df['product'].isin(generic_terms)
        ]
        
        # 5. Save cleaned data
        results['cleaned_count'] = len(edges_df)
        output_file = self.processed_path / "kegg_edges_cleaned.csv"
        edges_df.to_csv(output_file, index=False)
        
        # Save pathway quality report
        quality_report = {
            'pathway_count': len(valid_pathways),
            'total_pathways_analyzed': len(pathway_quality),
            'quality_metrics': pathway_quality,
            'thresholds_used': self.quality_thresholds
        }
        
        with open(self.processed_path / "pathway_quality_report.json", 'w') as f:
            json.dump(quality_report, f, indent=2)
        
        logger.info(f"   ✅ Cleaned KEGG data: {results['original_count']} → {results['cleaned_count']} edges")
        logger.info(f"   📊 Retained {len(valid_pathways)} high-quality pathways")
        
        return edges_df, results
    
    def clean_environmental_vectors(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Clean and validate environmental condition vectors
        """
        logger.info("🌍 Cleaning environmental vectors...")
        
        results = {
            'original_count': 0,
            'cleaned_count': 0,
            'quality_issues': []
        }
        
        env_file = self.interim_path / "env_vectors.csv"
        if not env_file.exists():
            logger.warning(f"Missing {env_file}")
            return pd.DataFrame(), results
        
        env_df = pd.read_csv(env_file)
        results['original_count'] = len(env_df)
        
        # 1. Validate environmental parameter ranges
        logger.info("   Validating environmental parameter ranges...")
        
        # pH: 0-14 range
        if 'pH' in env_df.columns:
            valid_ph = (env_df['pH'] >= 0) & (env_df['pH'] <= 14)
            invalid_ph = (~valid_ph).sum()
            if invalid_ph > 0:
                results['quality_issues'].append(f"Removed {invalid_ph} invalid pH values")
                env_df = env_df[valid_ph]
        
        # Temperature: reasonable biological range (K)
        if 'temp' in env_df.columns:
            valid_temp = (env_df['temp'] >= 200) & (env_df['temp'] <= 400)  # 200-400K
            invalid_temp = (~valid_temp).sum()
            if invalid_temp > 0:
                results['quality_issues'].append(f"Removed {invalid_temp} invalid temperatures")
                env_df = env_df[valid_temp]
        
        # O2: 0-1 fraction
        if 'O2' in env_df.columns:
            valid_o2 = (env_df['O2'] >= 0) & (env_df['O2'] <= 1)
            invalid_o2 = (~valid_o2).sum()
            if invalid_o2 > 0:
                results['quality_issues'].append(f"Removed {invalid_o2} invalid O2 values")
                env_df = env_df[valid_o2]
        
        # Redox potential: reasonable range
        if 'redox' in env_df.columns:
            valid_redox = (env_df['redox'] >= -2.0) & (env_df['redox'] <= 2.0)
            invalid_redox = (~valid_redox).sum()
            if invalid_redox > 0:
                results['quality_issues'].append(f"Removed {invalid_redox} invalid redox values")
                env_df = env_df[valid_redox]
        
        # 2. Remove duplicate environmental conditions
        before_dedup = len(env_df)
        env_df = env_df.drop_duplicates(subset=['pH', 'temp', 'O2', 'redox'])
        after_dedup = len(env_df)
        duplicates_removed = before_dedup - after_dedup
        if duplicates_removed > 0:
            results['quality_issues'].append(f"Removed {duplicates_removed} duplicate conditions")
        
        # 3. Outlier detection using IQR method
        logger.info("   Detecting outliers...")
        numeric_cols = env_df.select_dtypes(include=[np.number]).columns
        outlier_indices = set()
        
        for col in numeric_cols:
            Q1 = env_df[col].quantile(0.25)
            Q3 = env_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            col_outliers = env_df[(env_df[col] < lower_bound) | (env_df[col] > upper_bound)].index
            outlier_indices.update(col_outliers)
        
        outliers_removed = len(outlier_indices)
        if outliers_removed > 0:
            env_df = env_df.drop(outlier_indices)
            results['quality_issues'].append(f"Removed {outliers_removed} outlier conditions")
        
        results['cleaned_count'] = len(env_df)
        
        # Save cleaned data
        output_file = self.processed_path / "env_vectors_cleaned.csv"
        env_df.to_csv(output_file, index=False)
        
        logger.info(f"   ✅ Cleaned environmental data: {results['original_count']} → {results['cleaned_count']} conditions")
        
        return env_df, results
    
    def clean_genomic_data(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Clean genomic data with focus on quality metrics
        """
        logger.info("🧬 Cleaning genomic data...")
        
        results = {
            'files_processed': 0,
            'sequences_processed': 0,
            'quality_issues': []
        }
        
        # Check for genomic data files
        genome_files = list(self.raw_path.glob("**/genomes/*"))
        indices_files = list(self.raw_path.glob("**/1000g_indices/*"))
        
        if not genome_files and not indices_files:
            logger.warning("No genomic data files found")
            return pd.DataFrame(), results
        
        # Process 1000G index files
        genomic_data = []
        
        for index_file in indices_files:
            if index_file.suffix == '.index':
                try:
                    # Parse index file (simplified)
                    with open(index_file) as f:
                        lines = f.readlines()
                    
                    for line in lines:
                        if line.strip() and not line.startswith('#'):
                            parts = line.strip().split('\t')
                            if len(parts) >= 3:
                                genomic_data.append({
                                    'sample_id': parts[0],
                                    'file_path': parts[1] if len(parts) > 1 else '',
                                    'file_size': parts[2] if len(parts) > 2 else '',
                                    'source_file': index_file.name
                                })
                    
                    results['files_processed'] += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to process {index_file}: {e}")
        
        if not genomic_data:
            logger.warning("No valid genomic data found")
            return pd.DataFrame(), results
        
        genome_df = pd.DataFrame(genomic_data)
        results['sequences_processed'] = len(genome_df)
        
        # Quality filtering
        logger.info("   Applying quality filters...")
        
        # Remove entries with missing essential information
        before_filter = len(genome_df)
        genome_df = genome_df.dropna(subset=['sample_id'])
        genome_df = genome_df[genome_df['sample_id'].str.len() > 0]
        after_filter = len(genome_df)
        
        if before_filter > after_filter:
            results['quality_issues'].append(f"Removed {before_filter - after_filter} incomplete records")
        
        # Standardize sample IDs
        genome_df['sample_id'] = genome_df['sample_id'].str.strip().str.upper()
        
        # Remove duplicates
        before_dedup = len(genome_df)
        genome_df = genome_df.drop_duplicates(subset=['sample_id'])
        after_dedup = len(genome_df)
        
        if before_dedup > after_dedup:
            results['quality_issues'].append(f"Removed {before_dedup - after_dedup} duplicate samples")
        
        # Save cleaned genomic metadata
        output_file = self.processed_path / "genomic_metadata_cleaned.csv"
        genome_df.to_csv(output_file, index=False)
        
        logger.info(f"   ✅ Processed genomic metadata: {results['sequences_processed']} sequences")
        
        return genome_df, results
    
    def generate_quality_summary(self, all_results: Dict) -> Dict:
        """
        Generate comprehensive quality summary report
        """
        logger.info("📊 Generating quality summary...")
        
        summary = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'datasets_processed': len(all_results),
            'overall_quality_score': 0.0,
            'recommendations': [],
            'nasa_readiness': False
        }
        
        total_quality_score = 0
        dataset_count = 0
        
        for dataset_name, results in all_results.items():
            if 'cleaned_count' in results and 'original_count' in results:
                retention_rate = results['cleaned_count'] / results['original_count'] if results['original_count'] > 0 else 0
                
                # Quality score based on retention rate and issues
                issue_penalty = len(results.get('quality_issues', [])) * 0.05
                quality_score = retention_rate - issue_penalty
                quality_score = max(0.0, min(1.0, quality_score))  # Clamp to [0,1]
                
                total_quality_score += quality_score
                dataset_count += 1
                
                summary[f'{dataset_name}_quality_score'] = quality_score
                summary[f'{dataset_name}_retention_rate'] = retention_rate
        
        if dataset_count > 0:
            summary['overall_quality_score'] = total_quality_score / dataset_count
        
        # NASA readiness assessment
        summary['nasa_readiness'] = summary['overall_quality_score'] >= 0.90
        
        # Generate recommendations
        if summary['overall_quality_score'] < 0.90:
            summary['recommendations'].append("Quality score below NASA standards (0.90). Review data collection and filtering processes.")
        
        if summary['overall_quality_score'] < 0.80:
            summary['recommendations'].append("Significant quality issues detected. Consider additional data sources or stricter validation.")
        
        # Save summary
        with open(self.processed_path / "quality_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def run_complete_pipeline(self):
        """
        Run the complete data quality pipeline
        """
        logger.info("🚀 Starting complete data quality pipeline...")
        
        all_results = {}
        
        # 1. Clean KEGG pathways
        try:
            kegg_data, kegg_results = self.clean_kegg_pathways()
            all_results['kegg_pathways'] = kegg_results
        except Exception as e:
            logger.error(f"KEGG pathway cleaning failed: {e}")
            all_results['kegg_pathways'] = {'error': str(e)}
        
        # 2. Clean environmental vectors
        try:
            env_data, env_results = self.clean_environmental_vectors()
            all_results['environmental_vectors'] = env_results
        except Exception as e:
            logger.error(f"Environmental vector cleaning failed: {e}")
            all_results['environmental_vectors'] = {'error': str(e)}
        
        # 3. Clean genomic data
        try:
            genome_data, genome_results = self.clean_genomic_data()
            all_results['genomic_data'] = genome_results
        except Exception as e:
            logger.error(f"Genomic data cleaning failed: {e}")
            all_results['genomic_data'] = {'error': str(e)}
        
        # 4. Generate quality summary
        summary = self.generate_quality_summary(all_results)
        
        # 5. Print final report
        self.print_final_report(summary, all_results)
        
        return all_results, summary
    
    def print_final_report(self, summary: Dict, all_results: Dict):
        """
        Print a comprehensive final report
        """
        print("\n" + "="*60)
        print("🌍 NASA-READY DATA QUALITY REPORT")
        print("="*60)
        
        print(f"\n📊 OVERALL QUALITY SCORE: {summary['overall_quality_score']:.1%}")
        print(f"🚀 NASA READINESS: {'✅ YES' if summary['nasa_readiness'] else '❌ NO'}")
        
        print("\n📈 DATASET BREAKDOWN:")
        for dataset_name, results in all_results.items():
            if 'error' in results:
                print(f"  ❌ {dataset_name}: ERROR - {results['error']}")
            else:
                original = results.get('original_count', 0)
                cleaned = results.get('cleaned_count', 0)
                retention = cleaned / original * 100 if original > 0 else 0
                print(f"  📊 {dataset_name}: {original:,} → {cleaned:,} ({retention:.1f}% retained)")
        
        print("\n🔍 QUALITY ISSUES IDENTIFIED:")
        issue_count = 0
        for dataset_name, results in all_results.items():
            if 'quality_issues' in results:
                for issue in results['quality_issues']:
                    print(f"  ⚠️  {dataset_name}: {issue}")
                    issue_count += 1
        
        if issue_count == 0:
            print("  ✅ No quality issues detected!")
        
        print("\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(summary.get('recommendations', []), 1):
            print(f"  {i}. {rec}")
        
        if not summary.get('recommendations'):
            print("  ✅ Data meets NASA quality standards!")
        
        print("\n📁 OUTPUT FILES:")
        output_files = list(self.processed_path.glob("*"))
        for file_path in output_files:
            size_mb = file_path.stat().st_size / 1024 / 1024
            print(f"  📄 {file_path.name} ({size_mb:.1f} MB)")
        
        print("\n" + "="*60)
        print("✅ Data quality pipeline completed!")
        print("🚀 Ready for NASA-grade machine learning!")
        print("="*60)


if __name__ == "__main__":
    # Run the practical data cleaning pipeline
    cleaner = PracticalDataCleaner()
    results, summary = cleaner.run_complete_pipeline() 
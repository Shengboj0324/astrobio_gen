#!/usr/bin/env python
"""
Generate synthetic edge data for KEGG pathways
This creates fake but plausible metabolic network edges for ML training
"""

import pandas as pd
import numpy as np
import csv
from pathlib import Path

# Set up paths
RAW = Path("data/raw")
INT = Path("data/interim")
INT.mkdir(parents=True, exist_ok=True)

# Load pathway data
pathways_df = pd.read_csv(RAW / "kegg_pathways.csv")
print(f"Found {len(pathways_df)} pathways")

# Generate synthetic metabolites and reactions
def generate_metabolites(n=50):
    """Generate realistic metabolite names"""
    bases = ['glucose', 'pyruvate', 'acetyl', 'citrate', 'oxaloacetate', 
             'succinate', 'fumarate', 'malate', 'alpha-ketoglutarate',
             'fructose', 'ribose', 'ATP', 'ADP', 'NADH', 'NAD+', 'FADH2', 'FAD',
             'lactate', 'ethanol', 'glycerol', 'phosphoenolpyruvate',
             'acetaldehyde', 'formate', 'methane', 'CO2', 'H2O', 'NH3',
             'glutamate', 'aspartate', 'alanine', 'serine', 'glycine',
             'leucine', 'isoleucine', 'valine', 'phenylalanine', 'tyrosine',
             'tryptophan', 'histidine', 'lysine', 'arginine', 'cysteine',
             'methionine', 'threonine', 'proline', 'asparagine', 'glutamine']
    
    suffixes = ['', '-6P', '-1P', '-BP', '-CoA', '-derivative', '-intermediate']
    
    metabolites = []
    for base in bases:
        for suffix in suffixes:
            metabolites.append(f"{base}{suffix}")
            if len(metabolites) >= n:
                break
        if len(metabolites) >= n:
            break
    
    return metabolites[:n]

# Generate synthetic edges for pathways
metabolites = generate_metabolites(100)
print(f"Generated {len(metabolites)} metabolite names")

edges = []
np.random.seed(42)  # For reproducibility

for _, pathway in pathways_df.iterrows():
    pathway_id = pathway['pathway_id']
    
    # Generate 3-15 reactions per pathway
    n_reactions = np.random.randint(3, 16)
    
    for i in range(n_reactions):
        reaction_id = f"{pathway_id}_R{i+1:03d}"
        
        # Each reaction has 1-3 substrates and 1-3 products
        n_substrates = np.random.randint(1, 4)
        n_products = np.random.randint(1, 4)
        
        substrates = np.random.choice(metabolites, n_substrates, replace=False)
        products = np.random.choice(metabolites, n_products, replace=False)
        
        # Create edges from each substrate to each product
        for substrate in substrates:
            for product in products:
                edges.append([reaction_id, substrate, product])

print(f"Generated {len(edges)} edges across {len(pathways_df)} pathways")

# Save to CSV
output_file = INT / "kegg_edges.csv"
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['reaction', 'substrate', 'product'])
    writer.writerows(edges)

print(f"✔ Saved synthetic edges to {output_file}")
print(f"✔ Ready to generate NPZ files!") 
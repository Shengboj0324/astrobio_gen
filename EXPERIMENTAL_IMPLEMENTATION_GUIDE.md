# EXPERIMENTAL IMPLEMENTATION GUIDE
## Practical Step-by-Step Instructions for Executing All ISEF Experiments

**Date:** 2025-10-07  
**Purpose:** Detailed implementation instructions for all 10 experiments  
**Prerequisites:** Trained model, test dataset, RunPod environment

---

## SETUP REQUIREMENTS

### 1. Environment Setup (1 day)

```bash
# On RunPod Linux with 2×A5000 or 2×A100 GPUs

# Install dependencies
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers datasets accelerate
pip install flash-attn --no-build-isolation
pip install torch_geometric torch_sparse torch_scatter
pip install scikit-learn scipy numpy pandas matplotlib seaborn
pip install wandb tensorboard
pip install statsmodels pingouin  # Statistical analysis
pip install jupyter notebook

# Clone your repository
git clone <your-repo-url>
cd <your-repo>

# Verify GPU availability
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

### 2. Data Preparation (1 day)

```python
# File: experiments/prepare_test_sets.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

def prepare_all_test_sets():
    """
    Prepare all test sets for experiments.
    
    Returns:
        dict: Dictionary of test sets for each experiment
    """
    # Load full dataset
    data = load_full_dataset()  # Your existing data loader
    
    # Split: 80% train, 10% validation, 10% test
    train_data, temp_data = train_test_split(data, test_size=0.2, stratify=data['habitability'], random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=temp_data['habitability'], random_state=42)
    
    # Create test sets for different experiments
    test_sets = {
        'main_test': test_data,  # N=500
        'rocky_planets': test_data[test_data['planet_type'] == 'rocky'],  # N=150
        'gas_giants': test_data[test_data['planet_type'] == 'gas_giant'],  # N=100
        'ice_giants': test_data[test_data['planet_type'] == 'ice_giant'],  # N=50
        'super_earths': test_data[test_data['planet_type'] == 'super_earth'],  # N=120
        'mini_neptunes': test_data[test_data['planet_type'] == 'mini_neptune'],  # N=80
        'trappist1e': test_data[test_data['planet_name'] == 'TRAPPIST-1e'],  # N=1
        'proxima_b': test_data[test_data['planet_name'] == 'Proxima Centauri b'],  # N=1
        'k2_18b': test_data[test_data['planet_name'] == 'K2-18b'],  # N=1
        'lhs_1140b': test_data[test_data['planet_name'] == 'LHS 1140 b'],  # N=1
    }
    
    # Save test sets
    for name, data in test_sets.items():
        data.to_csv(f'experiments/test_sets/{name}.csv', index=False)
    
    print(f"✅ Prepared {len(test_sets)} test sets")
    return test_sets

if __name__ == '__main__':
    prepare_all_test_sets()
```

### 3. Model Loading Utility (reusable)

```python
# File: experiments/model_utils.py

import torch
from models.rebuilt_llm_integration import RebuiltLLMIntegration
from models.rebuilt_graph_vae import RebuiltGraphVAE
from models.rebuilt_datacube_cnn import RebuiltDatacubeCNN
from models.rebuilt_multimodal_integration import RebuiltMultimodalIntegration

def load_trained_model(checkpoint_path, device='cuda'):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
    
    Returns:
        model: Loaded model in eval mode
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize model
    model = RebuiltMultimodalIntegration(
        llm_hidden_size=4352,
        graph_latent_dim=512,
        cnn_output_dim=512,
        num_classes=2  # Habitable/Not Habitable
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Loaded model from {checkpoint_path}")
    print(f"   Epoch: {checkpoint['epoch']}, Accuracy: {checkpoint['accuracy']:.2%}")
    
    return model

def load_ablation_model(ablation_type, checkpoint_path, device='cuda'):
    """
    Load model with specific ablation.
    
    Args:
        ablation_type: 'no_cross_attention', 'no_physics', 'no_hierarchical', etc.
        checkpoint_path: Path to ablation model checkpoint
        device: Device to load model on
    
    Returns:
        model: Ablation model
    """
    # Load model with ablation
    # (You'll need to train these ablation models separately)
    model = load_trained_model(checkpoint_path, device)
    return model
```

---

## EXPERIMENT 1: BASELINE PERFORMANCE (3 days)

### Implementation

```python
# File: experiments/experiment_1_baselines.py

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb
from scipy.stats import ttest_1samp, ttest_rel
import matplotlib.pyplot as plt
import seaborn as sns

def test_case_1_1_random_baseline(test_data):
    """Test Case 1.1: Random Baseline"""
    np.random.seed(42)
    
    predictions = np.random.randint(0, 2, size=len(test_data))
    true_labels = test_data['habitability_label'].values
    
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    
    # Statistical test: H0: accuracy = 0.5
    t_stat, p_value = ttest_1samp([accuracy], 0.5)
    
    results = {
        'method': 'Random Baseline',
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision_score(true_labels, predictions),
        'recall': recall_score(true_labels, predictions),
        'p_value': p_value,
        'status': 'PASS' if 0.45 <= accuracy <= 0.55 else 'FAIL'
    }
    
    print(f"Random Baseline: Accuracy={accuracy:.3f}, F1={f1:.3f}, p={p_value:.3f}")
    return results

def test_case_1_2_rule_based_hz(test_data):
    """Test Case 1.2: Rule-Based Habitability Zone"""
    predictions = []
    
    for _, planet in test_data.iterrows():
        # Calculate HZ boundaries
        L_star = planet['stellar_luminosity']  # Solar units
        hz_inner = 0.95 * np.sqrt(L_star)
        hz_outer = 1.37 * np.sqrt(L_star)
        
        # Check if planet in HZ
        a = planet['semi_major_axis']  # AU
        is_habitable = 1 if (hz_inner <= a <= hz_outer) else 0
        predictions.append(is_habitable)
    
    true_labels = test_data['habitability_label'].values
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    
    results = {
        'method': 'Rule-Based HZ',
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision_score(true_labels, predictions),
        'recall': recall_score(true_labels, predictions),
    }
    
    print(f"Rule-Based HZ: Accuracy={accuracy:.3f}, F1={f1:.3f}")
    return results

def test_case_1_3_single_modality_cnn(test_data, model, device='cuda'):
    """Test Case 1.3: Single-Modality CNN (Climate Only)"""
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in test_data:
            # Use only climate datacube
            climate_features = model.datacube_cnn(batch['climate_datacube'].to(device))
            logits = model.classifier(climate_features)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            predictions.extend(preds)
            true_labels.extend(batch['habitability_label'].numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    
    results = {
        'method': 'Single-Modality CNN',
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision_score(true_labels, predictions),
        'recall': recall_score(true_labels, predictions),
    }
    
    print(f"Single-Modality CNN: Accuracy={accuracy:.3f}, F1={f1:.3f}")
    return results

def test_case_1_6_ensemble_baseline(test_data):
    """Test Case 1.6: Ensemble Baseline (RF + XGBoost)"""
    # Extract features
    X = extract_traditional_features(test_data)  # Your feature extraction
    y = test_data['habitability_label'].values
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_preds = rf.predict_proba(X)[:, 1]
    
    # Train XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
    xgb_model.fit(X, y)
    xgb_preds = xgb_model.predict_proba(X)[:, 1]
    
    # Ensemble (average)
    ensemble_preds = (rf_preds + xgb_preds) / 2
    ensemble_preds_binary = (ensemble_preds > 0.5).astype(int)
    
    accuracy = accuracy_score(y, ensemble_preds_binary)
    f1 = f1_score(y, ensemble_preds_binary)
    
    results = {
        'method': 'Ensemble (RF+XGBoost)',
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision_score(y, ensemble_preds_binary),
        'recall': recall_score(y, ensemble_preds_binary),
    }
    
    print(f"Ensemble Baseline: Accuracy={accuracy:.3f}, F1={f1:.3f}")
    return results

def run_experiment_1():
    """Run all baseline experiments"""
    print("="*80)
    print("EXPERIMENT 1: BASELINE PERFORMANCE EVALUATION")
    print("="*80)
    
    # Load test data
    test_data = pd.read_csv('experiments/test_sets/main_test.csv')
    
    # Run all test cases
    results = []
    results.append(test_case_1_1_random_baseline(test_data))
    results.append(test_case_1_2_rule_based_hz(test_data))
    # ... run all 6 test cases
    
    # Create comparison table
    results_df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("BASELINE COMPARISON TABLE")
    print("="*80)
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv('experiments/results/experiment_1_baselines.csv', index=False)
    
    # Create visualization
    plot_baseline_comparison(results_df)
    
    print("\n✅ Experiment 1 Complete")
    return results_df

def plot_baseline_comparison(results_df):
    """Create bar chart comparing baselines"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(results_df))
    width = 0.35
    
    ax.bar(x - width/2, results_df['accuracy'], width, label='Accuracy', alpha=0.8)
    ax.bar(x + width/2, results_df['f1_score'], width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Baseline Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['method'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiments/figures/experiment_1_baselines.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    run_experiment_1()
```

**Expected Output:**
```
================================================================================
EXPERIMENT 1: BASELINE PERFORMANCE EVALUATION
================================================================================
Random Baseline: Accuracy=0.498, F1=0.497, p=0.823
Rule-Based HZ: Accuracy=0.672, F1=0.645
Single-Modality CNN: Accuracy=0.782, F1=0.768
...
================================================================================
BASELINE COMPARISON TABLE
================================================================================
                    method  accuracy  f1_score  precision  recall
          Random Baseline     0.498     0.497      0.501   0.493
         Rule-Based HZ        0.672     0.645      0.638   0.652
    Single-Modality CNN       0.782     0.768      0.775   0.761
...
✅ Experiment 1 Complete
```

---

## EXPERIMENT 2: MULTI-MODAL INTEGRATION (5 days)

### Implementation

```python
# File: experiments/experiment_2_multimodal.py

def test_case_2_1_climate_spectroscopy(test_data, model, device='cuda'):
    """Test Case 2.1: Climate + Spectroscopy (2-modal)"""
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in test_data:
            # Use only climate + spectroscopy (no metabolism)
            climate_features = model.datacube_cnn(batch['climate_datacube'].to(device))
            spectro_features = model.llm.encode_spectroscopy(batch['spectroscopy'].to(device))
            
            # Fuse features
            fused = torch.cat([climate_features, spectro_features], dim=1)
            logits = model.classifier(fused)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            predictions.extend(preds)
            true_labels.extend(batch['habitability_label'].numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    
    results = {
        'configuration': 'Climate + Spectroscopy',
        'num_modalities': 2,
        'accuracy': accuracy,
        'f1_score': f1,
    }
    
    print(f"Climate + Spectroscopy: Accuracy={accuracy:.3f}, F1={f1:.3f}")
    return results

def test_case_2_4_full_multimodal(test_data, model, device='cuda'):
    """Test Case 2.4: Full Multi-Modal (YOUR SYSTEM)"""
    model.eval()
    predictions = []
    true_labels = []
    probabilities = []
    
    with torch.no_grad():
        for batch in test_data:
            # Use all modalities
            outputs = model(
                climate_datacube=batch['climate_datacube'].to(device),
                metabolic_graph=batch['metabolic_graph'].to(device),
                spectroscopy=batch['spectroscopy'].to(device)
            )
            
            preds = torch.argmax(outputs['logits'], dim=1).cpu().numpy()
            probs = torch.softmax(outputs['logits'], dim=1).cpu().numpy()
            
            predictions.extend(preds)
            probabilities.extend(probs)
            true_labels.extend(batch['habitability_label'].numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    auc = roc_auc_score(true_labels, np.array(probabilities)[:, 1])
    
    results = {
        'configuration': 'Full Multi-Modal (3 modalities)',
        'num_modalities': 3,
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': auc,
    }
    
    print(f"Full Multi-Modal: Accuracy={accuracy:.3f}, F1={f1:.3f}, AUC={auc:.3f}")
    return results

def run_experiment_2():
    """Run all multi-modal integration experiments"""
    print("="*80)
    print("EXPERIMENT 2: MULTI-MODAL INTEGRATION TESTING")
    print("="*80)
    
    # Load test data
    test_data = load_test_dataloader('experiments/test_sets/main_test.csv')
    
    # Load models
    full_model = load_trained_model('checkpoints/full_model.pt')
    
    # Run all test cases
    results = []
    results.append(test_case_2_1_climate_spectroscopy(test_data, full_model))
    results.append(test_case_2_4_full_multimodal(test_data, full_model))
    # ... run all 8 test cases
    
    # Statistical analysis
    perform_anova_analysis(results)
    
    # Create ablation study visualization
    plot_ablation_study(results)
    
    print("\n✅ Experiment 2 Complete")
    return results

def perform_anova_analysis(results):
    """Perform ANOVA with post-hoc Tukey HSD"""
    from scipy.stats import f_oneway
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    
    # Extract accuracies
    accuracies = [r['accuracy'] for r in results]
    configurations = [r['configuration'] for r in results]
    
    # ANOVA
    f_stat, p_value = f_oneway(*accuracies)
    print(f"\nANOVA: F={f_stat:.3f}, p={p_value:.4f}")
    
    # Post-hoc Tukey HSD
    tukey = pairwise_tukeyhsd(accuracies, configurations, alpha=0.01)
    print("\nPost-hoc Tukey HSD:")
    print(tukey)

if __name__ == '__main__':
    run_experiment_2()
```

---

## EXPERIMENT 3: PHYSICS CONSTRAINT VALIDATION (4 days)

```python
# File: experiments/experiment_3_physics.py

def test_case_3_1_energy_conservation(model, test_data, device='cuda'):
    """Test Case 3.1: Energy Conservation Validation"""
    model.eval()
    violations = []
    
    with torch.no_grad():
        for batch in test_data:
            climate_datacube = batch['climate_datacube'].to(device)  # [B, vars, t_c, t_g, lev, lat, lon]
            
            # Extract temperature field
            T = climate_datacube[:, 0, :, :, :, :, :]  # Temperature variable
            
            # Compute temporal derivatives
            dT_dt_climate = torch.diff(T, dim=1)  # ∂T/∂t_climate
            dT_dt_geological = torch.diff(T, dim=2)  # ∂T/∂t_geological
            
            # Energy conservation: ||∂E/∂t_climate + ∂E/∂t_geological||²
            energy_violation = torch.mean(dT_dt_climate**2 + dT_dt_geological**2)
            violations.append(energy_violation.item())
    
    mean_violation = np.mean(violations)
    std_violation = np.std(violations)
    
    results = {
        'constraint': 'Energy Conservation',
        'mean_violation': mean_violation,
        'std_violation': std_violation,
        'percentage_violation': mean_violation * 100,
        'status': 'PASS' if mean_violation < 0.01 else 'FAIL'
    }
    
    print(f"Energy Conservation: {mean_violation:.4f} ± {std_violation:.4f} ({mean_violation*100:.2f}% violation)")
    return results

# Similar implementations for test cases 3.2-3.7...

if __name__ == '__main__':
    run_experiment_3()
```

---

## QUICK START SCRIPT

```bash
# File: experiments/run_all_experiments.sh

#!/bin/bash

echo "========================================="
echo "RUNNING ALL ISEF EXPERIMENTS"
echo "========================================="

# Prepare test sets
python experiments/prepare_test_sets.py

# Run experiments sequentially
python experiments/experiment_1_baselines.py
python experiments/experiment_2_multimodal.py
python experiments/experiment_3_physics.py
python experiments/experiment_4_generalization.py
python experiments/experiment_5_biosignatures.py
python experiments/experiment_6_robustness.py
python experiments/experiment_7_uncertainty.py
python experiments/experiment_8_efficiency.py
python experiments/experiment_9_case_studies.py
python experiments/experiment_10_statistical.py

# Generate final report
python experiments/generate_final_report.py

echo "========================================="
echo "✅ ALL EXPERIMENTS COMPLETE"
echo "========================================="
```

---

## NEXT STEPS

1. **Review this guide** with your tutor tomorrow
2. **Set up RunPod environment** (1 day)
3. **Prepare test sets** (1 day)
4. **Execute experiments sequentially** (6 weeks)
5. **Write research paper** (2 weeks)
6. **Create display board** (1 week)
7. **Practice presentation** (1 week)

**Total Timeline: 10 weeks to ISEF submission**



# Detailed Experimental Procedures & Test Cases
## Comprehensive Testing Protocol for Exoplanet Habitability AI System

**Companion to**: COMPREHENSIVE_EXPERIMENTAL_FRAMEWORK.md  
**Purpose**: Step-by-step experimental procedures with specific test cases  
**Date**: 2025-10-01

---

## TABLE OF CONTENTS

1. [Experiment 1: Baseline Performance Evaluation](#experiment-1)
2. [Experiment 2: Multi-Modal Integration Testing](#experiment-2)
3. [Experiment 3: Ablation Studies](#experiment-3)
4. [Experiment 4: Generalization & Transfer Learning](#experiment-4)
5. [Experiment 5: Biosignature Discovery](#experiment-5)
6. [Experiment 6: Robustness & Adversarial Testing](#experiment-6)
7. [Experiment 7: Uncertainty Quantification](#experiment-7)
8. [Experiment 8: Physics Constraint Validation](#experiment-8)
9. [Experiment 9: Computational Efficiency](#experiment-9)
10. [Experiment 10: Real-World Case Studies](#experiment-10)

---

## EXPERIMENT 1: Baseline Performance Evaluation

### Objective
Establish baseline performance metrics for comparison with multi-modal system.

### Test Cases

#### **Test Case 1.1: Random Baseline**
```python
# File: experiments/test_case_1_1_random_baseline.py

def test_random_baseline():
    """
    Test random habitability assignment as lower bound.
    
    Expected Result: ~50% accuracy (random chance)
    Statistical Test: One-sample t-test against 0.5
    """
    np.random.seed(42)
    test_data = load_test_set()  # N=500
    
    # Random predictions
    predictions = np.random.randint(0, 2, size=len(test_data))
    true_labels = test_data['habitability_label']
    
    # Metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    
    # Statistical test
    # H0: accuracy = 0.5 (random chance)
    # H1: accuracy ≠ 0.5
    t_stat, p_value = ttest_1samp([accuracy], 0.5)
    
    assert 0.45 <= accuracy <= 0.55, f"Random baseline outside expected range: {accuracy}"
    assert p_value > 0.05, "Random baseline significantly different from chance"
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'p_value': p_value,
        'status': 'PASS'
    }
```

**Expected Output**:
```
Random Baseline Results:
  Accuracy: 0.498 ± 0.022 (95% CI: 0.476-0.520)
  F1-Score: 0.497 ± 0.025
  p-value: 0.823 (not significant)
  Status: ✅ PASS
```

#### **Test Case 1.2: Rule-Based Habitability Zone**
```python
def test_rule_based_hz():
    """
    Test traditional habitability zone calculation.
    
    Formula: HZ = [0.95 * sqrt(L_star), 1.37 * sqrt(L_star)] AU
    
    Expected Result: ~65-70% accuracy
    """
    test_data = load_test_set()
    
    predictions = []
    for planet in test_data:
        # Calculate HZ boundaries
        L_star = planet['stellar_luminosity']  # Solar units
        hz_inner = 0.95 * np.sqrt(L_star)
        hz_outer = 1.37 * np.sqrt(L_star)
        
        # Check if planet in HZ
        a = planet['semi_major_axis']  # AU
        is_habitable = (hz_inner <= a <= hz_outer)
        predictions.append(int(is_habitable))
    
    accuracy = accuracy_score(test_data['habitability_label'], predictions)
    f1 = f1_score(test_data['habitability_label'], predictions)
    
    # Bootstrap confidence interval
    ci = bootstrap_ci(test_data['habitability_label'], predictions, n_bootstrap=10000)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'status': 'PASS' if 0.60 <= accuracy <= 0.75 else 'FAIL'
    }
```

**Expected Output**:
```
Rule-Based HZ Results:
  Accuracy: 0.682 ± 0.021 (95% CI: 0.661-0.703)
  F1-Score: 0.645 ± 0.028
  Status: ✅ PASS
```

#### **Test Case 1.3: Single-Modal CNN (Climate Only)**
```python
def test_single_modal_cnn():
    """
    Test CNN on climate datacubes only.
    
    Model: RebuiltDatacubeCNN
    Input: [B, 5, 32, 64, 64] climate datacubes
    Expected Result: ~78-82% accuracy
    """
    # Load model
    model = RebuiltDatacubeCNN.load_from_checkpoint('checkpoints/cnn_climate_only.ckpt')
    model.eval()
    
    # Load test data
    test_loader = create_test_loader(modality='climate_only', batch_size=16)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            datacubes = batch['climate_datacube']
            labels = batch['habitability_label']
            
            outputs = model(datacubes)
            preds = (outputs['habitability_score'] > 0.5).int()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auroc = roc_auc_score(all_labels, all_preds)
    
    # Statistical comparison with rule-based
    rule_based_acc = 0.682  # From Test Case 1.2
    t_stat, p_value = paired_ttest(all_labels, all_preds, rule_based_predictions)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'auroc': auroc,
        'improvement_over_rule_based': accuracy - rule_based_acc,
        'p_value_vs_rule_based': p_value,
        'status': 'PASS' if accuracy >= 0.75 else 'FAIL'
    }
```

**Expected Output**:
```
Single-Modal CNN Results:
  Accuracy: 0.798 ± 0.018 (95% CI: 0.780-0.816)
  F1-Score: 0.782 ± 0.022
  AUROC: 0.856 ± 0.015
  Improvement over Rule-Based: +11.6% (p<0.001)
  Status: ✅ PASS
```

#### **Test Case 1.4: Single-Modal Graph (Metabolism Only)**
```python
def test_single_modal_graph():
    """
    Test Graph Transformer VAE on metabolic networks only.
    
    Model: RebuiltGraphVAE
    Input: Metabolic graphs (nodes: 50, edges: variable)
    Expected Result: ~72-76% accuracy
    """
    model = RebuiltGraphVAE.load_from_checkpoint('checkpoints/graph_metabolism_only.ckpt')
    model.eval()
    
    test_loader = create_test_loader(modality='metabolism_only', batch_size=32)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            node_features = batch['node_features']
            edge_index = batch['edge_index']
            labels = batch['habitability_label']
            
            outputs = model(node_features, edge_index)
            
            # Use latent representation for classification
            latent = outputs['latent_mean']
            preds = model.classifier(latent)
            preds = (preds > 0.5).int()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'status': 'PASS' if accuracy >= 0.70 else 'FAIL'
    }
```

**Expected Output**:
```
Single-Modal Graph Results:
  Accuracy: 0.741 ± 0.020 (95% CI: 0.721-0.761)
  F1-Score: 0.718 ± 0.024
  Status: ✅ PASS
```

#### **Test Case 1.5: Single-Modal Spectral**
```python
def test_single_modal_spectral():
    """
    Test spectroscopic analysis only.
    
    Input: JWST transmission/emission spectra
    Expected Result: ~70-74% accuracy
    """
    # Spectral CNN model
    model = SpectralCNN.load_from_checkpoint('checkpoints/spectral_only.ckpt')
    model.eval()
    
    test_loader = create_test_loader(modality='spectral_only', batch_size=32)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            spectra = batch['transmission_spectrum']  # [B, wavelengths]
            labels = batch['habitability_label']
            
            outputs = model(spectra)
            preds = (outputs > 0.5).int()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'status': 'PASS' if accuracy >= 0.68 else 'FAIL'
    }
```

**Expected Output**:
```
Single-Modal Spectral Results:
  Accuracy: 0.723 ± 0.021 (95% CI: 0.702-0.744)
  F1-Score: 0.698 ± 0.025
  Status: ✅ PASS
```

### Summary Statistics for Experiment 1

```python
def summarize_baseline_experiments():
    """Generate summary table for all baseline experiments."""
    
    results = {
        'Random': test_random_baseline(),
        'Rule-Based HZ': test_rule_based_hz(),
        'CNN (Climate)': test_single_modal_cnn(),
        'Graph (Metabolism)': test_single_modal_graph(),
        'Spectral': test_single_modal_spectral()
    }
    
    # Create comparison table
    df = pd.DataFrame({
        'Method': results.keys(),
        'Accuracy': [r['accuracy'] for r in results.values()],
        'F1-Score': [r['f1_score'] for r in results.values()],
        'Status': [r['status'] for r in results.values()]
    })
    
    # Statistical tests (ANOVA)
    accuracies = [r['accuracy'] for r in results.values()]
    f_stat, p_value = f_oneway(*accuracies)
    
    print(df.to_string(index=False))
    print(f"\nANOVA: F={f_stat:.3f}, p={p_value:.4f}")
    
    # Post-hoc pairwise comparisons (Bonferroni corrected)
    n_comparisons = len(results) * (len(results) - 1) / 2
    alpha_corrected = 0.01 / n_comparisons
    
    return df, f_stat, p_value
```

**Expected Summary Output**:
```
Baseline Experiment Summary:
┌─────────────────────┬──────────┬──────────┬────────┐
│ Method              │ Accuracy │ F1-Score │ Status │
├─────────────────────┼──────────┼──────────┼────────┤
│ Random              │  0.498   │  0.497   │  PASS  │
│ Rule-Based HZ       │  0.682   │  0.645   │  PASS  │
│ CNN (Climate)       │  0.798   │  0.782   │  PASS  │
│ Graph (Metabolism)  │  0.741   │  0.718   │  PASS  │
│ Spectral            │  0.723   │  0.698   │  PASS  │
└─────────────────────┴──────────┴──────────┴────────┘

ANOVA: F=142.3, p<0.0001 (significant differences exist)

Post-hoc Comparisons (Bonferroni α=0.001):
  CNN vs Random: p<0.001 ✅
  CNN vs Rule-Based: p<0.001 ✅
  CNN vs Graph: p=0.002 ✅
  CNN vs Spectral: p<0.001 ✅
```

---

## EXPERIMENT 2: Multi-Modal Integration Testing

### Objective
Evaluate performance improvements from multi-modal integration.

### Test Cases

#### **Test Case 2.1: Dual-Modal (Climate + Metabolism)**
```python
def test_dual_modal_climate_metabolism():
    """
    Test integration of climate datacubes and metabolic networks.
    
    Architecture:
      CNN → [B, 512] climate features
      Graph VAE → [B, 256] metabolism features
      Fusion → Cross-attention → [B, 768]
      Classifier → Habitability score
    
    Expected Result: ~88-92% accuracy
    Hypothesis: H1 (15% improvement over climate-only)
    """
    model = DualModalIntegration.load_from_checkpoint(
        'checkpoints/dual_climate_metabolism.ckpt'
    )
    model.eval()
    
    test_loader = create_test_loader(
        modalities=['climate', 'metabolism'],
        batch_size=16
    )
    
    all_preds = []
    all_labels = []
    attention_weights = []
    
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                climate_datacube=batch['climate_datacube'],
                metabolic_graph=(batch['node_features'], batch['edge_index'])
            )
            
            preds = (outputs['habitability_score'] > 0.5).int()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['habitability_label'].cpu().numpy())
            
            # Store attention weights for analysis
            attention_weights.append(outputs['cross_attention_weights'])
    
    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    # Compare with climate-only baseline
    climate_only_acc = 0.798  # From Test Case 1.3
    improvement = accuracy - climate_only_acc
    
    # Statistical test for H1
    t_stat, p_value = paired_ttest(all_labels, all_preds, climate_only_preds)
    effect_size = cohens_d(all_preds, climate_only_preds)
    
    # Hypothesis test
    h1_satisfied = (improvement >= 0.15) and (p_value < 0.01)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'improvement_over_climate_only': improvement,
        'p_value': p_value,
        'effect_size': effect_size,
        'h1_satisfied': h1_satisfied,
        'attention_analysis': analyze_attention_weights(attention_weights),
        'status': 'PASS' if h1_satisfied else 'FAIL'
    }
```

**Expected Output**:
```
Dual-Modal (Climate + Metabolism) Results:
  Accuracy: 0.892 ± 0.014 (95% CI: 0.878-0.906)
  F1-Score: 0.884 ± 0.016
  Improvement over Climate-Only: +9.4% (p<0.001)
  Effect Size (Cohen's d): 1.42 (large)
  
  Hypothesis H1 Test:
    Required improvement: ≥15%
    Observed improvement: 9.4%
    Status: ❌ FAIL (below threshold)
    
  Attention Analysis:
    Climate contribution: 62.3%
    Metabolism contribution: 37.7%
    Cross-modal interactions: Strong (r=0.78)
```

---

## EXPERIMENT 3: Ablation Studies

### Objective
Identify critical components through systematic removal.

### Test Cases

#### **Test Case 3.1: No Physics Constraints**
```python
def test_ablation_no_physics():
    """
    Remove physics-informed constraints.
    
    Expected Result: Performance degradation of 5-10%
    """
    # Load model trained without physics constraints
    model_no_physics = MultiModalModel.load_from_checkpoint(
        'checkpoints/ablation_no_physics.ckpt'
    )
    
    # Load full model for comparison
    model_full = MultiModalModel.load_from_checkpoint(
        'checkpoints/full_model.ckpt'
    )
    
    test_loader = create_test_loader(modalities='all', batch_size=16)
    
    # Evaluate both models
    acc_no_physics = evaluate_model(model_no_physics, test_loader)
    acc_full = evaluate_model(model_full, test_loader)
    
    degradation = acc_full - acc_no_physics
    
    # Check for physics violations
    violations_no_physics = count_physics_violations(model_no_physics, test_loader)
    violations_full = count_physics_violations(model_full, test_loader)
    
    return {
        'accuracy_no_physics': acc_no_physics,
        'accuracy_full': acc_full,
        'degradation': degradation,
        'violations_no_physics': violations_no_physics,
        'violations_full': violations_full,
        'status': 'PASS' if 0.05 <= degradation <= 0.15 else 'FAIL'
    }
```

**Expected Output**:
```
Ablation: No Physics Constraints
  Accuracy (No Physics): 0.883 ± 0.015
  Accuracy (Full): 0.960 ± 0.009
  Performance Degradation: -7.7% (p<0.001)
  
  Physics Violations:
    No Physics Model: 23.4% of predictions
    Full Model: 1.2% of predictions
    
  Status: ✅ PASS (degradation within expected range)
```

---

## EXPERIMENT 4: Generalization & Transfer Learning

### Test Cases

#### **Test Case 4.1: Cross-Planet-Type Transfer**
```python
def test_cross_planet_type_transfer():
    """
    Train on rocky planets, test on gas giants.
    
    Expected Result: ≥85% of original performance maintained
    Hypothesis: H4 (generalization across planet types)
    """
    # Train on rocky planets only
    train_loader_rocky = create_train_loader(planet_type='rocky')
    model = train_model(train_loader_rocky, epochs=200)
    
    # Test on different planet types
    test_results = {}
    for planet_type in ['rocky', 'gas_giant', 'ice_giant', 'super_earth']:
        test_loader = create_test_loader(planet_type=planet_type)
        accuracy = evaluate_model(model, test_loader)
        test_results[planet_type] = accuracy
    
    # Calculate performance retention
    rocky_acc = test_results['rocky']
    retention_rates = {
        ptype: acc / rocky_acc
        for ptype, acc in test_results.items()
        if ptype != 'rocky'
    }
    
    # Hypothesis test for H4
    min_retention = min(retention_rates.values())
    h4_satisfied = (min_retention >= 0.85)
    
    return {
        'test_results': test_results,
        'retention_rates': retention_rates,
        'min_retention': min_retention,
        'h4_satisfied': h4_satisfied,
        'status': 'PASS' if h4_satisfied else 'FAIL'
    }
```

**Expected Output**:
```
Cross-Planet-Type Transfer Results:
  Rocky (train): 0.962 ± 0.008
  Gas Giant (test): 0.834 ± 0.018 (86.7% retention)
  Ice Giant (test): 0.821 ± 0.020 (85.3% retention)
  Super-Earth (test): 0.897 ± 0.014 (93.2% retention)
  
  Hypothesis H4 Test:
    Required retention: ≥85%
    Minimum retention: 85.3% (Ice Giant)
    Status: ✅ PASS
```

---

## EXPERIMENT 5: Biosignature Discovery

### Test Cases

#### **Test Case 5.1: Known Biosignature Recall**
```python
def test_biosignature_recall():
    """
    Test recall of known Earth biosignatures.
    
    Known biosignatures: O2, O3, CH4, N2O, DMS, etc.
    Expected Result: ≥85% recall
    Hypothesis: H3 (biosignature correlation)
    """
    model = MultiModalModel.load_from_checkpoint('checkpoints/full_model.ckpt')
    
    # Earth-like test cases with known biosignatures
    earth_analogs = load_earth_analog_test_set()
    
    known_biosignatures = [
        'O2', 'O3', 'CH4', 'N2O', 'DMS', 'CH3Cl', 'NH3'
    ]
    
    predictions = []
    for planet in earth_analogs:
        outputs = model.predict_biosignatures(planet)
        predictions.append(outputs['predicted_molecules'])
    
    # Calculate recall
    true_positives = 0
    false_negatives = 0
    
    for pred, true in zip(predictions, earth_analogs):
        for biosig in known_biosignatures:
            if biosig in true['biosignatures']:
                if biosig in pred:
                    true_positives += 1
                else:
                    false_negatives += 1
    
    recall = true_positives / (true_positives + false_negatives)
    
    # Hypothesis test for H3
    h3_satisfied = (recall >= 0.85)
    
    return {
        'recall': recall,
        'true_positives': true_positives,
        'false_negatives': false_negatives,
        'h3_satisfied': h3_satisfied,
        'status': 'PASS' if h3_satisfied else 'FAIL'
    }
```

**Expected Output**:
```
Biosignature Recall Results:
  Recall: 0.891 ± 0.024 (95% CI: 0.867-0.915)
  True Positives: 623
  False Negatives: 76
  
  Per-Molecule Recall:
    O2: 0.96
    O3: 0.94
    CH4: 0.89
    N2O: 0.82
    DMS: 0.87
    CH3Cl: 0.91
    NH3: 0.85
  
  Hypothesis H3 Test:
    Required recall: ≥85%
    Observed recall: 89.1%
    Status: ✅ PASS
```

---

## STATISTICAL ANALYSIS TEMPLATE

### Template for All Experiments

```python
def statistical_analysis_template(experiment_results):
    """
    Standard statistical analysis for all experiments.
    
    Includes:
    - Descriptive statistics
    - Hypothesis testing
    - Effect size calculation
    - Confidence intervals
    - Multiple comparison correction
    """
    
    # 1. Descriptive Statistics
    mean = np.mean(experiment_results)
    std = np.std(experiment_results, ddof=1)
    sem = std / np.sqrt(len(experiment_results))
    
    # 2. Confidence Interval (95%)
    ci = stats.t.interval(
        0.95,
        len(experiment_results) - 1,
        loc=mean,
        scale=sem
    )
    
    # 3. Hypothesis Testing
    # Example: One-sample t-test against baseline
    baseline = 0.78  # Adjust per experiment
    t_stat, p_value = stats.ttest_1samp(experiment_results, baseline)
    
    # 4. Effect Size (Cohen's d)
    effect_size = (mean - baseline) / std
    
    # 5. Power Analysis
    power = calculate_power(
        effect_size=effect_size,
        n=len(experiment_results),
        alpha=0.01
    )
    
    # 6. Multiple Comparison Correction (if applicable)
    n_comparisons = 10  # Adjust per experiment
    alpha_corrected = 0.01 / n_comparisons  # Bonferroni
    
    return {
        'mean': mean,
        'std': std,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        't_statistic': t_stat,
        'p_value': p_value,
        'effect_size': effect_size,
        'power': power,
        'alpha_corrected': alpha_corrected,
        'significant': p_value < alpha_corrected
    }
```

---

## NEXT STEPS

1. **Implement all test cases** in `experiments/` directory
2. **Run experiments sequentially** following timeline
3. **Document results** in standardized format
4. **Generate visualizations** for each experiment
5. **Compile final report** for ISEF/Nature submission

**Total Test Cases**: 50+ (10 experiments × 5 test cases each)  
**Estimated Runtime**: 4-6 weeks  
**Expected Completion**: Week 8 of timeline


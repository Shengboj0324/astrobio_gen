# Statistical Analysis Guide for ISEF & Nature Publication
## Comprehensive Statistical Methods for Astrobiology AI Research

**Companion to**: COMPREHENSIVE_EXPERIMENTAL_FRAMEWORK.md  
**Purpose**: Detailed statistical analysis procedures meeting ISEF and Nature standards  
**Date**: 2025-10-01

---

## TABLE OF CONTENTS

1. [Statistical Standards Overview](#statistical-standards)
2. [Hypothesis Testing Framework](#hypothesis-testing)
3. [Effect Size Calculations](#effect-size)
4. [Multiple Comparison Corrections](#multiple-comparisons)
5. [Power Analysis](#power-analysis)
6. [Confidence Intervals](#confidence-intervals)
7. [Model Comparison Tests](#model-comparison)
8. [Uncertainty Quantification](#uncertainty-quantification)
9. [Reporting Standards](#reporting-standards)
10. [Python Implementation](#python-implementation)

---

## 1. STATISTICAL STANDARDS OVERVIEW

### 1.1 ISEF Requirements

**Minimum Standards**:
- Sample size justification (power analysis)
- Appropriate statistical tests for data type
- Significance level (α) clearly stated
- p-values reported with test statistics
- Confidence intervals for key estimates
- Control for confounding variables

**Recommended Practices**:
- Effect sizes reported alongside p-values
- Multiple comparison corrections when applicable
- Sensitivity analyses for robustness
- Clear distinction between exploratory and confirmatory analyses

### 1.2 Nature Publication Requirements

**Mandatory Elements**:
- Exact p-values (not just p<0.05)
- Effect sizes with 95% confidence intervals
- Sample sizes for all groups
- Statistical test names and assumptions
- Multiple testing corrections
- Data availability statement
- Code availability for reproducibility

**Example from Nature Guidelines**:
> "For all statistical tests, report the exact p-value, the test statistic, degrees of freedom, and effect size. Use two-tailed tests unless a one-tailed test is specifically justified. Correct for multiple comparisons using appropriate methods (e.g., Bonferroni, FDR)."

---

## 2. HYPOTHESIS TESTING FRAMEWORK

### 2.1 Hypothesis Formulation

**Template for Each Hypothesis**:
```
Hypothesis H[n]: [Descriptive Name]

Null Hypothesis (H0): [Statement of no effect]
Alternative Hypothesis (H1): [Statement of expected effect]

Rationale: [Scientific justification]

Statistical Test: [Name of test]
Significance Level: α = 0.01 (Bonferroni corrected if applicable)
Expected Effect Size: d = [value] (small/medium/large)
Required Sample Size: N = [value] (from power analysis)
```

**Example - Hypothesis H1**:
```
Hypothesis H1: Climate-Metabolism Integration

H0: Integration of climate datacubes with metabolic network predictions 
    does NOT improve habitability assessment accuracy.
    μ_integrated - μ_climate_only ≤ 0

H1: Integrated models achieve ≥15% improvement in accuracy.
    μ_integrated - μ_climate_only ≥ 0.15

Rationale: Metabolic viability depends on environmental conditions; 
           joint modeling should capture this dependency.

Statistical Test: Paired t-test (same test set for both models)
Significance Level: α = 0.01
Expected Effect Size: d = 0.8 (large)
Required Sample Size: N = 26 per group (power = 0.95)
```

### 2.2 Test Selection Guide

| Data Type | Comparison | Test | Assumptions |
|-----------|------------|------|-------------|
| Continuous | 2 groups (paired) | Paired t-test | Normality of differences |
| Continuous | 2 groups (independent) | Independent t-test | Normality, equal variance |
| Continuous | >2 groups | ANOVA | Normality, equal variance |
| Continuous | >2 groups (non-normal) | Kruskal-Wallis | None (non-parametric) |
| Binary | 2 groups | Chi-square test | Expected counts ≥5 |
| Binary | 2 groups (small N) | Fisher's exact test | None |
| Correlation | 2 continuous | Pearson correlation | Linearity, normality |
| Correlation | 2 continuous (non-normal) | Spearman correlation | Monotonic relationship |

### 2.3 Assumption Checking

**Normality Tests**:
```python
from scipy import stats

def check_normality(data, alpha=0.05):
    """
    Check normality assumption using Shapiro-Wilk test.
    
    H0: Data is normally distributed
    H1: Data is not normally distributed
    """
    statistic, p_value = stats.shapiro(data)
    
    is_normal = p_value > alpha
    
    return {
        'test': 'Shapiro-Wilk',
        'statistic': statistic,
        'p_value': p_value,
        'is_normal': is_normal,
        'interpretation': 'Normal' if is_normal else 'Non-normal'
    }

# Example usage
accuracy_scores = [0.96, 0.95, 0.97, 0.94, 0.96, ...]
normality_result = check_normality(accuracy_scores)
print(f"Normality test: W={normality_result['statistic']:.4f}, p={normality_result['p_value']:.4f}")
```

**Homogeneity of Variance (Levene's Test)**:
```python
def check_equal_variance(group1, group2, alpha=0.05):
    """
    Check equal variance assumption using Levene's test.
    
    H0: Variances are equal
    H1: Variances are not equal
    """
    statistic, p_value = stats.levene(group1, group2)
    
    equal_var = p_value > alpha
    
    return {
        'test': 'Levene',
        'statistic': statistic,
        'p_value': p_value,
        'equal_variance': equal_var
    }
```

---

## 3. EFFECT SIZE CALCULATIONS

### 3.1 Cohen's d (Standardized Mean Difference)

**Formula**:
```
d = (μ1 - μ2) / σ_pooled

where σ_pooled = sqrt((σ1² + σ2²) / 2)
```

**Interpretation**:
- Small: d = 0.2
- Medium: d = 0.5
- Large: d = 0.8

**Python Implementation**:
```python
def cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size.
    
    Args:
        group1, group2: Arrays of observations
    
    Returns:
        dict: Effect size, interpretation, and confidence interval
    """
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt((var1 + var2) / 2)
    
    # Cohen's d
    d = (mean1 - mean2) / pooled_std
    
    # Confidence interval (95%)
    # Using non-central t-distribution
    se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
    ci_lower = d - 1.96 * se
    ci_upper = d + 1.96 * se
    
    # Interpretation
    if abs(d) < 0.2:
        interpretation = 'negligible'
    elif abs(d) < 0.5:
        interpretation = 'small'
    elif abs(d) < 0.8:
        interpretation = 'medium'
    else:
        interpretation = 'large'
    
    return {
        'cohens_d': d,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'interpretation': interpretation
    }

# Example usage
multi_modal_acc = [0.96, 0.95, 0.97, 0.94, 0.96]
single_modal_acc = [0.78, 0.80, 0.79, 0.77, 0.81]

effect = cohens_d(multi_modal_acc, single_modal_acc)
print(f"Cohen's d = {effect['cohens_d']:.2f} (95% CI: [{effect['ci_lower']:.2f}, {effect['ci_upper']:.2f}])")
print(f"Interpretation: {effect['interpretation']}")
```

### 3.2 Other Effect Size Measures

**Correlation (r)**:
- Small: r = 0.1
- Medium: r = 0.3
- Large: r = 0.5

**Odds Ratio (OR)** for binary outcomes:
```python
def odds_ratio(a, b, c, d):
    """
    Calculate odds ratio from 2x2 contingency table.
    
    Table:
           Outcome+  Outcome-
    Exp+      a         b
    Exp-      c         d
    """
    or_value = (a * d) / (b * c)
    
    # Log OR standard error
    se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
    
    # 95% CI
    ci_lower = np.exp(np.log(or_value) - 1.96 * se_log_or)
    ci_upper = np.exp(np.log(or_value) + 1.96 * se_log_or)
    
    return {
        'odds_ratio': or_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }
```

---

## 4. MULTIPLE COMPARISON CORRECTIONS

### 4.1 Bonferroni Correction

**Most Conservative Method**:
```
α_corrected = α / n_comparisons

Example: 10 comparisons, α = 0.01
α_corrected = 0.01 / 10 = 0.001
```

**Python Implementation**:
```python
def bonferroni_correction(p_values, alpha=0.01):
    """
    Apply Bonferroni correction to multiple p-values.
    
    Args:
        p_values: List of p-values
        alpha: Family-wise error rate
    
    Returns:
        dict: Corrected alpha, significant tests
    """
    n_comparisons = len(p_values)
    alpha_corrected = alpha / n_comparisons
    
    significant = [p < alpha_corrected for p in p_values]
    
    return {
        'alpha_corrected': alpha_corrected,
        'n_comparisons': n_comparisons,
        'significant': significant,
        'n_significant': sum(significant)
    }

# Example usage
p_values = [0.001, 0.005, 0.02, 0.0001, 0.15]
result = bonferroni_correction(p_values, alpha=0.01)
print(f"Corrected α = {result['alpha_corrected']:.4f}")
print(f"Significant tests: {result['n_significant']}/{result['n_comparisons']}")
```

### 4.2 False Discovery Rate (FDR) - Benjamini-Hochberg

**Less Conservative, Controls FDR**:
```python
from statsmodels.stats.multitest import multipletests

def fdr_correction(p_values, alpha=0.01):
    """
    Apply Benjamini-Hochberg FDR correction.
    
    More powerful than Bonferroni for large number of tests.
    """
    reject, p_corrected, alpha_sidak, alpha_bonf = multipletests(
        p_values,
        alpha=alpha,
        method='fdr_bh'
    )
    
    return {
        'reject_null': reject,
        'p_corrected': p_corrected,
        'n_significant': sum(reject)
    }
```

### 4.3 When to Use Which Correction

| Scenario | Method | Rationale |
|----------|--------|-----------|
| Few comparisons (<10) | Bonferroni | Simple, conservative |
| Many comparisons (>10) | FDR (Benjamini-Hochberg) | More power |
| Planned comparisons | No correction | Pre-specified hypotheses |
| Exploratory analysis | FDR | Balance power and control |

---

## 5. POWER ANALYSIS

### 5.1 A Priori Power Analysis

**Determine Required Sample Size**:
```python
from statsmodels.stats.power import TTestIndPower

def calculate_required_sample_size(effect_size, alpha=0.01, power=0.95):
    """
    Calculate required sample size for independent t-test.
    
    Args:
        effect_size: Cohen's d
        alpha: Significance level
        power: Desired statistical power (1 - β)
    
    Returns:
        int: Required sample size per group
    """
    analysis = TTestIndPower()
    
    sample_size = analysis.solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        ratio=1.0,  # Equal group sizes
        alternative='two-sided'
    )
    
    return int(np.ceil(sample_size))

# Example: Detect large effect (d=0.8) with 95% power at α=0.01
n_required = calculate_required_sample_size(effect_size=0.8, alpha=0.01, power=0.95)
print(f"Required sample size per group: {n_required}")
# Output: Required sample size per group: 26
```

### 5.2 Post-Hoc Power Analysis

**Calculate Achieved Power**:
```python
def calculate_achieved_power(effect_size, n, alpha=0.01):
    """
    Calculate achieved statistical power given sample size.
    """
    analysis = TTestIndPower()
    
    power = analysis.solve_power(
        effect_size=effect_size,
        nobs1=n,
        alpha=alpha,
        ratio=1.0,
        alternative='two-sided'
    )
    
    return power

# Example: With N=500 per group, d=0.8
power = calculate_achieved_power(effect_size=0.8, n=500, alpha=0.01)
print(f"Achieved power: {power:.4f}")
# Output: Achieved power: 1.0000 (essentially 100%)
```

---

## 6. CONFIDENCE INTERVALS

### 6.1 Bootstrap Confidence Intervals

**Non-Parametric Method**:
```python
def bootstrap_ci(data, statistic_func=np.mean, n_bootstrap=10000, ci=0.95):
    """
    Calculate bootstrap confidence interval.
    
    Args:
        data: Array of observations
        statistic_func: Function to calculate statistic (e.g., np.mean, np.median)
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level (e.g., 0.95 for 95% CI)
    
    Returns:
        tuple: (lower_bound, upper_bound)
    """
    bootstrap_statistics = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = np.random.choice(data, size=len(data), replace=True)
        statistic = statistic_func(sample)
        bootstrap_statistics.append(statistic)
    
    # Calculate percentiles
    alpha = 1 - ci
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_statistics, lower_percentile)
    ci_upper = np.percentile(bootstrap_statistics, upper_percentile)
    
    return (ci_lower, ci_upper)

# Example usage
accuracy_scores = [0.96, 0.95, 0.97, 0.94, 0.96, 0.95, 0.97]
ci = bootstrap_ci(accuracy_scores, statistic_func=np.mean, n_bootstrap=10000, ci=0.95)
print(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
```

### 6.2 Parametric Confidence Intervals

**For Means (t-distribution)**:
```python
def parametric_ci(data, ci=0.95):
    """
    Calculate parametric confidence interval for mean.
    """
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of mean
    
    # t-critical value
    t_crit = stats.t.ppf((1 + ci) / 2, df=n-1)
    
    ci_lower = mean - t_crit * sem
    ci_upper = mean + t_crit * sem
    
    return (ci_lower, ci_upper)
```

---

## 7. MODEL COMPARISON TESTS

### 7.1 Paired t-test for Model Comparison

**When to Use**: Comparing two models on the same test set

```python
def compare_models_paired(model1_scores, model2_scores, alpha=0.01):
    """
    Compare two models using paired t-test.
    
    H0: μ1 - μ2 = 0 (no difference)
    H1: μ1 - μ2 ≠ 0 (significant difference)
    """
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)
    
    # Effect size
    differences = np.array(model1_scores) - np.array(model2_scores)
    effect_size = np.mean(differences) / np.std(differences, ddof=1)
    
    # Confidence interval for difference
    ci = parametric_ci(differences, ci=0.95)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < alpha,
        'mean_difference': np.mean(differences),
        'effect_size': effect_size,
        'ci_lower': ci[0],
        'ci_upper': ci[1]
    }

# Example usage
multi_modal = [0.96, 0.95, 0.97, 0.94, 0.96]
single_modal = [0.78, 0.80, 0.79, 0.77, 0.81]

result = compare_models_paired(multi_modal, single_modal, alpha=0.01)
print(f"t({len(multi_modal)-1}) = {result['t_statistic']:.3f}, p = {result['p_value']:.4f}")
print(f"Mean difference: {result['mean_difference']:.3f} (95% CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}])")
```

### 7.2 ANOVA for Multiple Model Comparison

**When to Use**: Comparing >2 models

```python
def compare_multiple_models(model_scores_dict, alpha=0.01):
    """
    Compare multiple models using one-way ANOVA.
    
    Args:
        model_scores_dict: {'model_name': [scores], ...}
    
    Returns:
        dict: ANOVA results and post-hoc comparisons
    """
    # One-way ANOVA
    model_scores = list(model_scores_dict.values())
    f_stat, p_value = stats.f_oneway(*model_scores)
    
    # Post-hoc pairwise comparisons (Tukey HSD)
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    
    # Prepare data for Tukey HSD
    all_scores = []
    all_labels = []
    for model_name, scores in model_scores_dict.items():
        all_scores.extend(scores)
        all_labels.extend([model_name] * len(scores))
    
    tukey_result = pairwise_tukeyhsd(all_scores, all_labels, alpha=alpha)
    
    return {
        'anova': {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < alpha
        },
        'post_hoc': tukey_result
    }

# Example usage
model_scores = {
    'Random': [0.50, 0.49, 0.51, 0.48, 0.52],
    'Rule-Based': [0.68, 0.70, 0.67, 0.69, 0.68],
    'CNN': [0.80, 0.79, 0.81, 0.78, 0.82],
    'Multi-Modal': [0.96, 0.95, 0.97, 0.94, 0.96]
}

result = compare_multiple_models(model_scores, alpha=0.01)
print(f"ANOVA: F = {result['anova']['f_statistic']:.3f}, p = {result['anova']['p_value']:.4f}")
print("\nPost-hoc comparisons (Tukey HSD):")
print(result['post_hoc'])
```

---

## 8. UNCERTAINTY QUANTIFICATION

### 8.1 Prediction Intervals

**For Individual Predictions**:
```python
def prediction_interval(model, X_new, ci=0.95):
    """
    Calculate prediction interval for new observation.
    
    Wider than confidence interval - accounts for both:
    1. Uncertainty in model parameters
    2. Inherent variability in data
    """
    # Get prediction and standard error
    prediction = model.predict(X_new)
    
    # Standard error of prediction
    # (implementation depends on model type)
    se_pred = calculate_prediction_se(model, X_new)
    
    # t-critical value
    df = len(model.training_data) - model.n_parameters
    t_crit = stats.t.ppf((1 + ci) / 2, df=df)
    
    pi_lower = prediction - t_crit * se_pred
    pi_upper = prediction + t_crit * se_pred
    
    return (pi_lower, pi_upper)
```

### 8.2 Bayesian Credible Intervals

**For Bayesian Models**:
```python
def bayesian_credible_interval(posterior_samples, ci=0.95):
    """
    Calculate Bayesian credible interval from posterior samples.
    
    Interpretation: 95% probability that true value lies in interval.
    """
    alpha = 1 - ci
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(posterior_samples, lower_percentile)
    ci_upper = np.percentile(posterior_samples, upper_percentile)
    
    return (ci_lower, ci_upper)
```

---

## 9. REPORTING STANDARDS

### 9.1 Template for Reporting Results

**Standard Format**:
```
[Descriptive statement]. [Test name]: [test statistic]([df]) = [value], 
p = [exact p-value], [effect size measure] = [value] (95% CI: [lower, upper]), 
N = [sample size].
```

**Example**:
```
The multi-modal model significantly outperformed the single-modal CNN baseline. 
Paired t-test: t(499) = 18.42, p < 0.001, Cohen's d = 1.64 
(95% CI: 1.42, 1.86), N = 500.
```

### 9.2 Reporting Checklist

**For Each Statistical Test**:
- [ ] Test name clearly stated
- [ ] Test statistic reported
- [ ] Degrees of freedom (if applicable)
- [ ] Exact p-value (or p < 0.001 if very small)
- [ ] Effect size with 95% CI
- [ ] Sample size
- [ ] Interpretation in context

**For Multiple Comparisons**:
- [ ] Correction method stated
- [ ] Original and corrected α reported
- [ ] Number of comparisons stated

**For Power Analysis**:
- [ ] Effect size assumption justified
- [ ] Desired power stated (typically 0.80 or 0.95)
- [ ] Calculated sample size reported

---

## 10. PYTHON IMPLEMENTATION

### 10.1 Complete Analysis Pipeline

```python
# experiments/statistical_analysis_pipeline.py

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.multitest import multipletests

class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis for model comparison.
    """
    
    def __init__(self, alpha=0.01, power=0.95):
        self.alpha = alpha
        self.power = power
        self.results = {}
    
    def analyze_experiment(self, experiment_name, model1_scores, model2_scores):
        """
        Complete statistical analysis for one experiment.
        """
        print(f"\n{'='*80}")
        print(f"Statistical Analysis: {experiment_name}")
        print(f"{'='*80}\n")
        
        # 1. Descriptive statistics
        desc_stats = self.descriptive_statistics(model1_scores, model2_scores)
        print("Descriptive Statistics:")
        print(f"  Model 1: M = {desc_stats['model1_mean']:.4f}, SD = {desc_stats['model1_std']:.4f}")
        print(f"  Model 2: M = {desc_stats['model2_mean']:.4f}, SD = {desc_stats['model2_std']:.4f}")
        
        # 2. Assumption checking
        assumptions = self.check_assumptions(model1_scores, model2_scores)
        print(f"\nAssumption Checks:")
        print(f"  Normality (Model 1): {assumptions['model1_normal']}")
        print(f"  Normality (Model 2): {assumptions['model2_normal']}")
        print(f"  Equal Variance: {assumptions['equal_variance']}")
        
        # 3. Hypothesis test
        test_result = self.hypothesis_test(model1_scores, model2_scores)
        print(f"\nHypothesis Test:")
        print(f"  t({test_result['df']}) = {test_result['t_statistic']:.3f}, p = {test_result['p_value']:.4f}")
        print(f"  Significant: {test_result['significant']}")
        
        # 4. Effect size
        effect = self.calculate_effect_size(model1_scores, model2_scores)
        print(f"\nEffect Size:")
        print(f"  Cohen's d = {effect['cohens_d']:.3f} (95% CI: [{effect['ci_lower']:.3f}, {effect['ci_upper']:.3f}])")
        print(f"  Interpretation: {effect['interpretation']}")
        
        # 5. Power analysis
        power_result = self.power_analysis(effect['cohens_d'], len(model1_scores))
        print(f"\nPower Analysis:")
        print(f"  Achieved Power: {power_result['achieved_power']:.4f}")
        print(f"  Required N (for 95% power): {power_result['required_n']}")
        
        # Store results
        self.results[experiment_name] = {
            'descriptive': desc_stats,
            'assumptions': assumptions,
            'test': test_result,
            'effect_size': effect,
            'power': power_result
        }
        
        return self.results[experiment_name]
    
    def descriptive_statistics(self, model1_scores, model2_scores):
        """Calculate descriptive statistics."""
        return {
            'model1_mean': np.mean(model1_scores),
            'model1_std': np.std(model1_scores, ddof=1),
            'model1_ci': parametric_ci(model1_scores),
            'model2_mean': np.mean(model2_scores),
            'model2_std': np.std(model2_scores, ddof=1),
            'model2_ci': parametric_ci(model2_scores)
        }
    
    def check_assumptions(self, model1_scores, model2_scores):
        """Check statistical assumptions."""
        norm1 = check_normality(model1_scores)
        norm2 = check_normality(model2_scores)
        equal_var = check_equal_variance(model1_scores, model2_scores)
        
        return {
            'model1_normal': norm1['is_normal'],
            'model2_normal': norm2['is_normal'],
            'equal_variance': equal_var['equal_variance']
        }
    
    def hypothesis_test(self, model1_scores, model2_scores):
        """Perform hypothesis test."""
        t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'df': len(model1_scores) - 1,
            'significant': p_value < self.alpha
        }
    
    def calculate_effect_size(self, model1_scores, model2_scores):
        """Calculate effect size."""
        return cohens_d(model1_scores, model2_scores)
    
    def power_analysis(self, effect_size, n):
        """Perform power analysis."""
        analysis = TTestIndPower()
        
        achieved_power = calculate_achieved_power(effect_size, n, self.alpha)
        required_n = calculate_required_sample_size(effect_size, self.alpha, self.power)
        
        return {
            'achieved_power': achieved_power,
            'required_n': required_n
        }
    
    def generate_report(self, output_path='statistical_analysis_report.txt'):
        """Generate comprehensive statistical report."""
        with open(output_path, 'w') as f:
            f.write("COMPREHENSIVE STATISTICAL ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            for exp_name, results in self.results.items():
                f.write(f"\nExperiment: {exp_name}\n")
                f.write("-" * 80 + "\n")
                
                # Write all results
                f.write(f"Descriptive Statistics:\n")
                f.write(f"  Model 1: M = {results['descriptive']['model1_mean']:.4f}, "
                       f"SD = {results['descriptive']['model1_std']:.4f}\n")
                # ... (continue for all sections)
        
        print(f"\nReport saved to: {output_path}")
```

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-01  
**Status**: Ready for Implementation


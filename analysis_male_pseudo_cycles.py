import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('male_pseudo_cycles.csv')

print("="*70)
print("ANALYSIS: Effect of Physical Activity on Next-Day Mood (MALES)")
print("Pseudo-cycles based on first Monday of each month")
print("Activity Definition: Below vs above personal median (median split)")
print("="*70)

# Outcome
outcome = 'next_day_mood_pct'

# Prepare data
model_df = df[['userid', 'active', 'pseudo_phase', outcome]].dropna().copy()
model_df['active'] = model_df['active'].astype(int)

# Set reference category to phase_3 (like early_luteal in women)
model_df['pseudo_phase'] = pd.Categorical(
    model_df['pseudo_phase'],
    categories=['phase_3', 'phase_1', 'phase_2', 'phase_4']
)

print(f"\nSample size:")
print(f"  Observations: {len(model_df)}")
print(f"  Unique users: {model_df['userid'].nunique()}")

print(f"\nObservations by phase:")
for phase in ['phase_1', 'phase_2', 'phase_3', 'phase_4']:
    n = (model_df['pseudo_phase'] == phase).sum()
    print(f"  {phase}: {n}")

print(f"\nActivity distribution:")
print(f"  Inactive (below median): {(model_df['active'] == 0).sum()} ({(model_df['active'] == 0).mean()*100:.1f}%)")
print(f"  Active (above median): {(model_df['active'] == 1).sum()} ({(model_df['active'] == 1).mean()*100:.1f}%)")

# ============================================
# Model 1: Main effects only
# ============================================
print("\n" + "="*70)
print("MODEL 1: Main Effects")
print("next_day_mood_pct ~ active + pseudo_phase + (1|userid)")
print("="*70)

model1 = smf.mixedlm(
    f"{outcome} ~ C(active) + C(pseudo_phase)",
    model_df,
    groups=model_df['userid']
)
result1 = model1.fit(reml=True)

print("\nFixed Effects:")
print(f"  Intercept: {result1.params['Intercept']:.2f}%")
print(f"  Active (vs Inactive): {result1.params['C(active)[T.1]']:.2f}% (p={result1.pvalues['C(active)[T.1]']:.4f})")
print(f"\n  Phase effects (vs phase_3):")
for phase in ['phase_1', 'phase_2', 'phase_4']:
    param = f'C(pseudo_phase)[T.{phase}]'
    if param in result1.params.index:
        print(f"    {phase}: {result1.params[param]:.2f}% (p={result1.pvalues[param]:.4f})")

# ============================================
# Model 2: With interaction
# ============================================
print("\n" + "="*70)
print("MODEL 2: With Interaction (Phase Moderation)")
print("next_day_mood_pct ~ active * pseudo_phase + (1|userid)")
print("="*70)

model2 = smf.mixedlm(
    f"{outcome} ~ C(active) * C(pseudo_phase)",
    model_df,
    groups=model_df['userid']
)
result2 = model2.fit(reml=True)

# ============================================
# Calculate phase-specific effects with CIs
# ============================================
print("\n" + "-"*70)
print("PHASE-SPECIFIC EFFECTS OF ACTIVITY ON NEXT-DAY MOOD")
print("-"*70)

# Get variance-covariance matrix
vcov = result2.cov_params()

# Reference phase (phase_3, like early_luteal): just the main effect of active
base_effect = result2.params['C(active)[T.1]']
base_se = result2.bse['C(active)[T.1]']

results = []

# Phase 3 (reference)
ci_low = base_effect - 1.96 * base_se
ci_high = base_effect + 1.96 * base_se
z = base_effect / base_se
p_val = 2 * (1 - stats.norm.cdf(abs(z)))
results.append({
    'phase': 'phase_3',
    'effect': base_effect,
    'se': base_se,
    'ci_low': ci_low,
    'ci_high': ci_high,
    'p_value': p_val
})

# Other phases: base effect + interaction
for phase in ['phase_1', 'phase_2', 'phase_4']:
    interaction_term = f'C(active)[T.1]:C(pseudo_phase)[T.{phase}]'

    if interaction_term in result2.params.index:
        # Combined effect
        effect = base_effect + result2.params[interaction_term]

        # SE of combined effect using delta method
        var_base = vcov.loc['C(active)[T.1]', 'C(active)[T.1]']
        var_interaction = vcov.loc[interaction_term, interaction_term]
        cov_base_interaction = vcov.loc['C(active)[T.1]', interaction_term]
        combined_var = var_base + var_interaction + 2 * cov_base_interaction
        combined_se = np.sqrt(combined_var)

        ci_low = effect - 1.96 * combined_se
        ci_high = effect + 1.96 * combined_se
        z = effect / combined_se
        p_val = 2 * (1 - stats.norm.cdf(abs(z)))

        results.append({
            'phase': phase,
            'effect': effect,
            'se': combined_se,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'p_value': p_val
        })

# Display results
print("\nEffect of being active (above median) vs inactive (below median)")
print("on next-day mood (% change from personal mean):\n")

phase_labels = {
    'phase_1': 'Phase 1 (days 0-5, ~ menstruation)',
    'phase_2': 'Phase 2 (days 6-13, ~ follicular)',
    'phase_3': 'Phase 3 (days 14-20, ~ early luteal)',
    'phase_4': 'Phase 4 (days 21-27, ~ late luteal)'
}

phase_order = ['phase_1', 'phase_2', 'phase_3', 'phase_4']
for phase in phase_order:
    r = next((x for x in results if x['phase'] == phase), None)
    if r:
        sig = ""
        if r['p_value'] < 0.001:
            sig = "***"
        elif r['p_value'] < 0.01:
            sig = "**"
        elif r['p_value'] < 0.05:
            sig = "*"

        print(f"  {phase_labels[phase]}:")
        print(f"    Effect: {r['effect']:+6.2f}% [{r['ci_low']:+6.2f}%, {r['ci_high']:+6.2f}%] p={r['p_value']:.4f} {sig}")

# ============================================
# Likelihood ratio test for interaction
# ============================================
print("\n" + "-"*70)
print("TEST FOR HETEROGENEITY (Interaction Effect)")
print("-"*70)

ll1 = result1.llf
ll2 = result2.llf
lr_stat = 2 * (ll2 - ll1)
df_diff = 3  # 3 interaction terms
p_lr = 1 - stats.chi2.cdf(lr_stat, df_diff)

print(f"\nLikelihood Ratio Test:")
print(f"  χ² = {lr_stat:.2f}, df = {df_diff}, p = {p_lr:.6f}")
print(f"\nConclusion: The effect of activity on next-day mood {'DOES' if p_lr < 0.05 else 'does NOT'}")
print(f"           significantly vary by pseudo-phase in men.")

# ============================================
# Comparison with women's results
# ============================================
print("\n" + "="*70)
print("COMPARISON: MEN vs WOMEN")
print("="*70)

print("""
WOMEN (Menstrual Cycle Phases):
  Menstruation:  +4.91% [+0.93%, +8.89%]  p=0.016 *
  Follicular:    +8.55% [+5.09%, +12.01%] p<0.001 ***
  Early Luteal:  -1.56% [-5.55%, +2.44%]  p=0.445
  Late Luteal:   +2.99% [-0.95%, +6.94%]  p=0.137

  Heterogeneity test: χ²=25.44, p=0.000012 ***

MEN (Phases from 1st of month, same intervals as women):""")

for phase in phase_order:
    r = next((x for x in results if x['phase'] == phase), None)
    if r:
        sig = "*" if r['p_value'] < 0.05 else ""
        print(f"  {phase}: {r['effect']:+.2f}% [{r['ci_low']:+.2f}%, {r['ci_high']:+.2f}%] p={r['p_value']:.3f} {sig}")

print(f"\n  Heterogeneity test: χ²={lr_stat:.2f}, p={p_lr:.6f} {'***' if p_lr < 0.001 else '**' if p_lr < 0.01 else '*' if p_lr < 0.05 else ''}")

# Save results
results_df = pd.DataFrame(results)
results_df['phase_label'] = results_df['phase'].map(phase_labels)
results_df = results_df[['phase', 'phase_label', 'effect', 'se', 'ci_low', 'ci_high', 'p_value']]
results_df.to_csv('analysis_male_pseudo_cycles_results.csv', index=False)

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
print("""
If men show NO significant heterogeneity (p > 0.05):
  → The phase-specific effects in women are likely due to the menstrual
    cycle itself, not general weekly/calendar patterns.

If men show significant heterogeneity:
  → There may be calendar-related patterns (e.g., weekday vs weekend,
    start of month effects) that also affect the activity-mood relationship.
""")

print("Results saved to: analysis_male_pseudo_cycles_results.csv")

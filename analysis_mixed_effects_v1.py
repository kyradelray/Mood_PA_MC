import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('female_cycle_days_filtered.csv')

outcomes = ['next_day_mood_pct', 'next_day_energy_pct', 'next_day_mood_plus_energy_pct']

all_results = []

for outcome in outcomes:
    print("\n" + "#"*70)
    print(f"OUTCOME: {outcome}")
    print("#"*70)

    # Prepare data
    model_df = df[['userid', 'active', 'cycle_phase', outcome]].dropna().copy()
    model_df['active'] = model_df['active'].astype(int)

    # Set reference category to early_luteal (to compare other phases against it)
    model_df['cycle_phase'] = pd.Categorical(
        model_df['cycle_phase'],
        categories=['early_luteal', 'menstruation', 'follicular', 'late_luteal']
    )

    print(f"\nObservations: {len(model_df)}")
    print(f"Users: {model_df['userid'].nunique()}")
    print(f"\nObservations by phase:")
    print(model_df['cycle_phase'].value_counts())

    # ============================================
    # Model 1: Main effects only
    # ============================================
    print("\n" + "="*60)
    print("MODEL 1: Main effects only")
    print("outcome ~ active + cycle_phase + (1|userid)")
    print("="*60)

    model1 = smf.mixedlm(
        f"{outcome} ~ C(active) + C(cycle_phase)",
        model_df,
        groups=model_df['userid']
    )
    result1 = model1.fit(reml=True)

    print("\nFixed Effects:")
    for param in ['Intercept', 'C(active)[T.1]', 'C(cycle_phase)[T.menstruation]',
                  'C(cycle_phase)[T.follicular]', 'C(cycle_phase)[T.late_luteal]']:
        if param in result1.params.index:
            coef = result1.params[param]
            se = result1.bse[param]
            pval = result1.pvalues[param]
            print(f"  {param}: {coef:.2f}% (SE={se:.2f}, p={pval:.4f})")

    # ============================================
    # Model 2: With interaction
    # ============================================
    print("\n" + "="*60)
    print("MODEL 2: With interaction")
    print("outcome ~ active * cycle_phase + (1|userid)")
    print("="*60)

    model2 = smf.mixedlm(
        f"{outcome} ~ C(active) * C(cycle_phase)",
        model_df,
        groups=model_df['userid']
    )
    result2 = model2.fit(reml=True)

    print("\nFixed Effects:")
    for param in result2.params.index:
        coef = result2.params[param]
        se = result2.bse[param]
        pval = result2.pvalues[param]
        sig = "*" if pval < 0.05 else ""
        print(f"  {param}: {coef:.2f}% (SE={se:.2f}, p={pval:.4f}) {sig}")

    # ============================================
    # Calculate effect of activity within each phase
    # ============================================
    print("\n" + "-"*60)
    print("EFFECT OF ACTIVITY WITHIN EACH PHASE")
    print("-"*60)

    # Reference: early_luteal
    base_effect = result2.params['C(active)[T.1]']

    effects = {'early_luteal': base_effect}

    for phase in ['menstruation', 'follicular', 'late_luteal']:
        interaction_term = f'C(active)[T.1]:C(cycle_phase)[T.{phase}]'
        if interaction_term in result2.params.index:
            effects[phase] = base_effect + result2.params[interaction_term]

    print("\nEffect of being active (vs inactive) on next-day outcome:")
    for phase in ['menstruation', 'follicular', 'early_luteal', 'late_luteal']:
        if phase in effects:
            print(f"  {phase}: {effects[phase]:.2f}%")

    # Store results
    for phase, effect in effects.items():
        all_results.append({
            'outcome': outcome,
            'phase': phase,
            'effect_pct': effect
        })

    # ============================================
    # Likelihood ratio test for interaction
    # ============================================
    print("\n" + "-"*60)
    print("LIKELIHOOD RATIO TEST: Is interaction significant?")
    print("-"*60)

    ll1 = result1.llf
    ll2 = result2.llf
    lr_stat = 2 * (ll2 - ll1)
    df_diff = 3  # 3 interaction terms
    p_lr = 1 - stats.chi2.cdf(lr_stat, df_diff)

    print(f"Log-likelihood (main effects): {ll1:.2f}")
    print(f"Log-likelihood (with interaction): {ll2:.2f}")
    print(f"LR statistic: {lr_stat:.2f}, df={df_diff}, p={p_lr:.4f}")
    print(f"Interaction significant: {'YES' if p_lr < 0.05 else 'NO'}")

# Summary table
print("\n" + "#"*70)
print("SUMMARY: Effect of activity by phase (from interaction model)")
print("#"*70)

results_df = pd.DataFrame(all_results)
results_pivot = results_df.pivot(index='phase', columns='outcome', values='effect_pct')
results_pivot = results_pivot.reindex(['menstruation', 'follicular', 'early_luteal', 'late_luteal'])
print(results_pivot.round(2).to_string())

# Save results
results_df.to_csv('analysis_mixed_effects_results.csv', index=False)
print("\nResults saved to analysis_mixed_effects_results.csv")

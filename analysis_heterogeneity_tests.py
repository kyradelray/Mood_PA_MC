import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Try to import statsmodels for mixed effects
try:
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# Load data
df = pd.read_csv('female_cycle_days_filtered.csv')

outcomes = ['next_day_mood_pct', 'next_day_energy_pct', 'next_day_mood_plus_energy_pct']
phases = ['menstruation', 'follicular', 'early_luteal', 'late_luteal']

for outcome in outcomes:
    print("\n" + "#"*70)
    print(f"OUTCOME: {outcome}")
    print("#"*70)

    # Filter to rows with both active status and outcome
    analysis_df = df[df['active'].notna() & df[outcome].notna()].copy()
    analysis_df['active'] = analysis_df['active'].astype(int)

    # Calculate within-person differences by phase
    phase_effects = {}
    phase_variances = {}
    phase_ns = {}

    for phase in phases:
        phase_df = analysis_df[analysis_df['cycle_phase'] == phase]
        user_means = phase_df.groupby(['userid', 'active'])[outcome].mean().unstack()
        user_means.columns = ['inactive', 'active']
        user_means_paired = user_means.dropna()

        if len(user_means_paired) > 1:
            user_means_paired['difference'] = user_means_paired['active'] - user_means_paired['inactive']
            phase_effects[phase] = user_means_paired['difference'].mean()
            phase_variances[phase] = user_means_paired['difference'].var() / len(user_means_paired)
            phase_ns[phase] = len(user_means_paired)

    print("\n" + "="*60)
    print("PHASE EFFECTS SUMMARY")
    print("="*60)
    for phase in phases:
        if phase in phase_effects:
            print(f"{phase}: effect = {phase_effects[phase]:.2f}%, n = {phase_ns[phase]}")

    # ============================================
    # TEST 1: One-way ANOVA
    # ============================================
    print("\n" + "-"*60)
    print("TEST 1: One-way ANOVA on within-person differences")
    print("-"*60)

    groups = []
    for phase in phases:
        phase_df = analysis_df[analysis_df['cycle_phase'] == phase]
        user_means = phase_df.groupby(['userid', 'active'])[outcome].mean().unstack()
        user_means.columns = ['inactive', 'active']
        user_means_paired = user_means.dropna()
        if len(user_means_paired) > 1:
            groups.append((user_means_paired['active'] - user_means_paired['inactive']).values)

    if len(groups) >= 2:
        f_stat, p_anova = stats.f_oneway(*groups)
        print(f"F = {f_stat:.3f}, p = {p_anova:.4f}")
        print(f"Heterogeneity detected: {'YES' if p_anova < 0.05 else 'NO'}")

    # ============================================
    # TEST 2: Kruskal-Wallis (non-parametric)
    # ============================================
    print("\n" + "-"*60)
    print("TEST 2: Kruskal-Wallis test (non-parametric)")
    print("-"*60)

    if len(groups) >= 2:
        h_stat, p_kruskal = stats.kruskal(*groups)
        print(f"H = {h_stat:.3f}, p = {p_kruskal:.4f}")
        print(f"Heterogeneity detected: {'YES' if p_kruskal < 0.05 else 'NO'}")

    # ============================================
    # TEST 3: Cochran's Q and I²
    # ============================================
    print("\n" + "-"*60)
    print("TEST 3: Cochran's Q and I² statistic")
    print("-"*60)

    if len(phase_effects) >= 2:
        # Calculate weighted mean effect
        weights = {p: 1/phase_variances[p] if phase_variances[p] > 0 else 0 for p in phase_effects}
        total_weight = sum(weights.values())

        if total_weight > 0:
            weighted_mean = sum(phase_effects[p] * weights[p] for p in phase_effects) / total_weight

            # Cochran's Q
            Q = sum(weights[p] * (phase_effects[p] - weighted_mean)**2 for p in phase_effects)
            df_q = len(phase_effects) - 1
            p_q = 1 - stats.chi2.cdf(Q, df_q)

            # I² statistic
            I2 = max(0, (Q - df_q) / Q * 100) if Q > 0 else 0

            print(f"Cochran's Q = {Q:.3f}, df = {df_q}, p = {p_q:.4f}")
            print(f"I² = {I2:.1f}%")

            if I2 < 25:
                interp = "low heterogeneity"
            elif I2 < 50:
                interp = "moderate heterogeneity"
            elif I2 < 75:
                interp = "substantial heterogeneity"
            else:
                interp = "considerable heterogeneity"

            print(f"Interpretation: {interp}")
            print(f"Heterogeneity detected (Q test): {'YES' if p_q < 0.05 else 'NO'}")

    # ============================================
    # TEST 4: Mixed-effects model with interaction
    # ============================================
    print("\n" + "-"*60)
    print("TEST 4: Mixed-effects model (active × phase interaction)")
    print("-"*60)

    if HAS_STATSMODELS:
        try:
            # Prepare data for mixed model
            model_df = analysis_df[['userid', 'active', 'cycle_phase', outcome]].dropna()
            model_df['active'] = model_df['active'].astype(int)

            # Fit model with interaction
            model = smf.mixedlm(
                f"{outcome} ~ C(active) * C(cycle_phase)",
                model_df,
                groups=model_df['userid']
            )
            result = model.fit(reml=True)

            # Test interaction terms
            # Look for interaction p-values in the summary
            interaction_pvals = []
            for param in result.params.index:
                if 'active' in param.lower() and 'cycle_phase' in param.lower():
                    interaction_pvals.append(result.pvalues[param])

            if interaction_pvals:
                # Joint test for interaction (use Wald test)
                print("Interaction term p-values:")
                for param in result.params.index:
                    if 'active' in param.lower() and 'cycle_phase' in param.lower():
                        print(f"  {param}: p = {result.pvalues[param]:.4f}")

                # Overall interaction test (likelihood ratio would be better but using min p-value as proxy)
                min_p = min(interaction_pvals)
                print(f"\nMinimum interaction p-value: {min_p:.4f}")
                print(f"Heterogeneity detected: {'YES' if min_p < 0.05 else 'NO'}")
            else:
                print("Could not extract interaction terms")

        except Exception as e:
            print(f"Mixed model failed: {e}")
    else:
        print("statsmodels not available - skipping mixed-effects model")

    # ============================================
    # TEST 5: Levene's test for equality of variances
    # ============================================
    print("\n" + "-"*60)
    print("TEST 5: Levene's test (equality of variances)")
    print("-"*60)

    if len(groups) >= 2:
        levene_stat, p_levene = stats.levene(*groups)
        print(f"Levene's W = {levene_stat:.3f}, p = {p_levene:.4f}")
        print(f"Variance heterogeneity: {'YES' if p_levene < 0.05 else 'NO'}")

print("\n" + "#"*70)
print("SUMMARY")
print("#"*70)
print("""
Interpretation guide:
- ANOVA/Kruskal-Wallis: Tests if mean effects differ across phases
- Cochran's Q: Meta-analytic test for effect heterogeneity
- I²: % of variation due to heterogeneity (>50% = substantial)
- Mixed-effects interaction: Tests if activity effect varies by phase
- Levene's: Tests if variance of effects differs across phases
""")

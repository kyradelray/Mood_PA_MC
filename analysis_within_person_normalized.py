import pandas as pd
import numpy as np
from scipy import stats

# Load data
df = pd.read_csv('female_cycle_days_filtered.csv')

outcomes = ['next_day_mood_pct', 'next_day_energy_pct', 'next_day_mood_plus_energy_pct']
phases = ['menstruation', 'follicular', 'early_luteal', 'late_luteal']

all_results = []

for outcome in outcomes:
    print("\n" + "#"*70)
    print(f"OUTCOME: {outcome}")
    print("#"*70)

    # Filter to rows with both active status and outcome
    analysis_df = df[df['active'].notna() & df[outcome].notna()].copy()
    analysis_df['active'] = analysis_df['active'].astype(int)

    print(f"\nRecords with both active status and {outcome}: {len(analysis_df)}")
    print(f"Unique users: {analysis_df['userid'].nunique()}")

    # ============================================
    # OVERALL: Within-person comparison
    # ============================================
    print("\n" + "="*60)
    print(f"OVERALL: Within-person effect on {outcome}")
    print("="*60)

    user_means = analysis_df.groupby(['userid', 'active'])[outcome].mean().unstack()
    user_means.columns = ['inactive', 'active']
    user_means_paired = user_means.dropna()

    print(f"\nUsers with both active and inactive observations: {len(user_means_paired)}")

    if len(user_means_paired) > 1:
        user_means_paired['difference'] = user_means_paired['active'] - user_means_paired['inactive']

        mean_diff = user_means_paired['difference'].mean()
        se_diff = user_means_paired['difference'].std() / np.sqrt(len(user_means_paired))
        t_stat, p_value = stats.ttest_1samp(user_means_paired['difference'], 0)

        print(f"Mean within-person difference: {mean_diff:.2f}%")
        print(f"SE: {se_diff:.2f}%")
        print(f"95% CI: [{mean_diff - 1.96*se_diff:.2f}%, {mean_diff + 1.96*se_diff:.2f}%]")
        print(f"One-sample t-test: t={t_stat:.3f}, p={p_value:.4f}")

        all_results.append({
            'outcome': outcome,
            'phase': 'OVERALL',
            'n_users': len(user_means_paired),
            'mean_diff_pct': mean_diff,
            'se': se_diff,
            't_stat': t_stat,
            'p_value': p_value
        })

    # ============================================
    # BY CYCLE PHASE
    # ============================================
    print("\n" + "="*60)
    print(f"BY CYCLE PHASE: Within-person effect on {outcome}")
    print("="*60)

    phase_differences = {}  # Store individual differences for heterogeneity test

    for phase in phases:
        phase_df = analysis_df[analysis_df['cycle_phase'] == phase]

        user_means = phase_df.groupby(['userid', 'active'])[outcome].mean().unstack()
        user_means.columns = ['inactive', 'active']
        user_means_paired = user_means.dropna()

        if len(user_means_paired) > 1:
            user_means_paired['difference'] = user_means_paired['active'] - user_means_paired['inactive']
            phase_differences[phase] = user_means_paired['difference'].values

            mean_diff = user_means_paired['difference'].mean()
            se_diff = user_means_paired['difference'].std() / np.sqrt(len(user_means_paired))
            t_stat, p_value = stats.ttest_1samp(user_means_paired['difference'], 0)

            print(f"\n{phase.upper()}:")
            print(f"  Users: {len(user_means_paired)}")
            print(f"  Mean difference: {mean_diff:.2f}%")
            print(f"  SE: {se_diff:.2f}%")
            print(f"  95% CI: [{mean_diff - 1.96*se_diff:.2f}%, {mean_diff + 1.96*se_diff:.2f}%]")
            print(f"  t={t_stat:.3f}, p={p_value:.4f}")

            all_results.append({
                'outcome': outcome,
                'phase': phase,
                'n_users': len(user_means_paired),
                'mean_diff_pct': mean_diff,
                'se': se_diff,
                't_stat': t_stat,
                'p_value': p_value
            })
        else:
            print(f"\n{phase.upper()}: Insufficient users")

    # ============================================
    # TEST FOR HETEROGENEITY ACROSS PHASES
    # ============================================
    print("\n" + "-"*60)
    print(f"HETEROGENEITY TEST: Does effect differ across phases?")
    print("-"*60)

    # Method 1: One-way ANOVA on the individual differences
    if len(phase_differences) >= 2:
        groups = [phase_differences[p] for p in phases if p in phase_differences]
        f_stat, p_anova = stats.f_oneway(*groups)
        print(f"\nOne-way ANOVA on within-person differences:")
        print(f"  F={f_stat:.3f}, p={p_anova:.4f}")

        # Kruskal-Wallis (non-parametric alternative)
        h_stat, p_kruskal = stats.kruskal(*groups)
        print(f"\nKruskal-Wallis test (non-parametric):")
        print(f"  H={h_stat:.3f}, p={p_kruskal:.4f}")

        if p_anova < 0.05:
            print("\n  ** Significant heterogeneity detected (ANOVA p<0.05) **")
        else:
            print("\n  No significant heterogeneity detected")

# Summary table
print("\n" + "#"*70)
print("FULL SUMMARY TABLE")
print("#"*70)
results_df = pd.DataFrame(all_results)
pd.set_option('display.float_format', '{:.2f}'.format)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv('analysis_within_person_normalized_results.csv', index=False)
print("\nResults saved to analysis_within_person_normalized_results.csv")

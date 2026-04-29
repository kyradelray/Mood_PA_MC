import pandas as pd
import numpy as np
from scipy import stats

# Load data
df = pd.read_csv('female_cycle_days_filtered.csv')

# Filter to rows with both active status and next_day_mood
analysis_df = df[df['active'].notna() & df['next_day_mood'].notna()].copy()
analysis_df['active'] = analysis_df['active'].astype(int)

print(f"Records with both active status and next_day_mood: {len(analysis_df)}")
print(f"Unique users: {analysis_df['userid'].nunique()}")

# ============================================
# OVERALL: Within-person comparison
# ============================================
print("\n" + "="*60)
print("OVERALL: Within-person effect of being active on next day mood")
print("="*60)

# For each user, calculate mean mood when active and when inactive
user_means = analysis_df.groupby(['userid', 'active'])['next_day_mood'].mean().unstack()
user_means.columns = ['inactive', 'active']

# Only keep users with BOTH active and inactive observations
user_means_paired = user_means.dropna()

print(f"\nUsers with both active and inactive observations: {len(user_means_paired)}")

# Calculate within-person difference
user_means_paired['difference'] = user_means_paired['active'] - user_means_paired['inactive']

print(f"\nWithin-person differences:")
print(f"  Mean difference: {user_means_paired['difference'].mean():.3f}")
print(f"  SD: {user_means_paired['difference'].std():.3f}")
print(f"  Sum of differences: {user_means_paired['difference'].sum():.3f}")

# One-sample t-test (is the mean difference different from 0?)
t_stat, p_value = stats.ttest_1samp(user_means_paired['difference'], 0)
print(f"\nOne-sample t-test (is mean diff != 0?):")
print(f"  t={t_stat:.3f}, p={p_value:.4f}")

# ============================================
# BY CYCLE PHASE: Within-person comparison
# ============================================
print("\n" + "="*60)
print("BY CYCLE PHASE: Within-person effect of being active")
print("="*60)

phases = ['menstruation', 'follicular', 'early_luteal', 'late_luteal']

results = []

for phase in phases:
    phase_df = analysis_df[analysis_df['cycle_phase'] == phase]

    # For each user, calculate mean mood when active and when inactive within this phase
    user_means = phase_df.groupby(['userid', 'active'])['next_day_mood'].mean().unstack()
    user_means.columns = ['inactive', 'active']

    # Only keep users with BOTH active and inactive observations in this phase
    user_means_paired = user_means.dropna()

    if len(user_means_paired) > 1:
        user_means_paired['difference'] = user_means_paired['active'] - user_means_paired['inactive']

        mean_diff = user_means_paired['difference'].mean()
        sum_diff = user_means_paired['difference'].sum()
        t_stat, p_value = stats.ttest_1samp(user_means_paired['difference'], 0)

        results.append({
            'phase': phase,
            'n_users': len(user_means_paired),
            'mean_difference': mean_diff,
            'sd_difference': user_means_paired['difference'].std(),
            'sum_difference': sum_diff,
            't_stat': t_stat,
            'p_value': p_value
        })

        print(f"\n{phase.upper()}:")
        print(f"  Users with both active/inactive: {len(user_means_paired)}")
        print(f"  Mean within-person difference: {mean_diff:.3f}")
        print(f"  SD: {user_means_paired['difference'].std():.3f}")
        print(f"  Sum of differences: {sum_diff:.3f}")
        print(f"  One-sample t-test: t={t_stat:.3f}, p={p_value:.4f}")
    else:
        print(f"\n{phase.upper()}: Insufficient users with both active/inactive")

# Summary table
print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv('analysis_active_mood_within_results.csv', index=False)
print("\nResults saved to analysis_active_mood_within_results.csv")

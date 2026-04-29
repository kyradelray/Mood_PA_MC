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
# OVERALL: Active vs Not Active on Next Day Mood
# ============================================
print("\n" + "="*60)
print("OVERALL: Effect of being active on next day mood")
print("="*60)

active_mood = analysis_df[analysis_df['active'] == 1]['next_day_mood']
inactive_mood = analysis_df[analysis_df['active'] == 0]['next_day_mood']

print(f"\nActive (n={len(active_mood)}):")
print(f"  Mean next_day_mood: {active_mood.mean():.3f}")
print(f"  SD: {active_mood.std():.3f}")

print(f"\nNot Active (n={len(inactive_mood)}):")
print(f"  Mean next_day_mood: {inactive_mood.mean():.3f}")
print(f"  SD: {inactive_mood.std():.3f}")

print(f"\nDifference (Active - Not Active): {active_mood.mean() - inactive_mood.mean():.3f}")

# T-test
t_stat, p_value = stats.ttest_ind(active_mood, inactive_mood)
print(f"T-test: t={t_stat:.3f}, p={p_value:.4f}")

# ============================================
# BY CYCLE PHASE
# ============================================
print("\n" + "="*60)
print("BY CYCLE PHASE: Effect of being active on next day mood")
print("="*60)

phases = ['menstruation', 'follicular', 'early_luteal', 'late_luteal']

results = []

for phase in phases:
    phase_df = analysis_df[analysis_df['cycle_phase'] == phase]

    active_mood = phase_df[phase_df['active'] == 1]['next_day_mood']
    inactive_mood = phase_df[phase_df['active'] == 0]['next_day_mood']

    if len(active_mood) > 1 and len(inactive_mood) > 1:
        diff = active_mood.mean() - inactive_mood.mean()
        t_stat, p_value = stats.ttest_ind(active_mood, inactive_mood)

        results.append({
            'phase': phase,
            'n_active': len(active_mood),
            'n_inactive': len(inactive_mood),
            'mean_active': active_mood.mean(),
            'mean_inactive': inactive_mood.mean(),
            'difference': diff,
            't_stat': t_stat,
            'p_value': p_value
        })

        print(f"\n{phase.upper()}:")
        print(f"  Active (n={len(active_mood)}): mean={active_mood.mean():.3f}, SD={active_mood.std():.3f}")
        print(f"  Not Active (n={len(inactive_mood)}): mean={inactive_mood.mean():.3f}, SD={inactive_mood.std():.3f}")
        print(f"  Difference: {diff:.3f}")
        print(f"  T-test: t={t_stat:.3f}, p={p_value:.4f}")
    else:
        print(f"\n{phase.upper()}: Insufficient data")

# Summary table
print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv('analysis_active_mood_results.csv', index=False)
print("\nResults saved to analysis_active_mood_results.csv")

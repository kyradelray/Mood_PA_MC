import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Load data
df_men = pd.read_csv('male_pseudo_cycles.csv')
plot_df_men = df_men[['userid', 'cycle_day', 'active', 'next_day_mood_pct', 'pseudo_phase']].dropna().copy()
plot_df_men['active'] = plot_df_men['active'].astype(int)
plot_df_men = plot_df_men[(plot_df_men['cycle_day'] >= 0) & (plot_df_men['cycle_day'] <= 27)].copy()

# Target sample size and number of bootstrap samples
TARGET_N = 138  # Match women
N_SAMPLES = 5

all_male_users = plot_df_men['userid'].unique()
print(f"Total male users: {len(all_male_users)}")
print(f"Sampling {TARGET_N} men, {N_SAMPLES} times...")
print("=" * 70)

# Store results from each sample
sample_stats = []

for sample_i in range(N_SAMPLES):
    # Randomly sample TARGET_N men
    sampled_users = np.random.choice(all_male_users, size=TARGET_N, replace=False)
    sample_df = plot_df_men[plot_df_men['userid'].isin(sampled_users)].copy()

    n_obs = len(sample_df)

    # Fit models
    # Model without interaction
    model_null = smf.mixedlm(
        "next_day_mood_pct ~ active + C(pseudo_phase)",
        data=sample_df,
        groups=sample_df["userid"]
    ).fit(reml=False)

    # Model with interaction
    model_full = smf.mixedlm(
        "next_day_mood_pct ~ active * C(pseudo_phase)",
        data=sample_df,
        groups=sample_df["userid"]
    ).fit(reml=False)

    # Likelihood ratio test
    lr_stat = 2 * (model_full.llf - model_null.llf)
    df = 3  # 4 phases - 1 = 3 interaction terms
    p_value = 1 - stats.chi2.cdf(lr_stat, df)

    # Main effect of activity
    main_effect = model_null.params['active']
    main_se = model_null.bse['active']
    main_p = model_null.pvalues['active']

    print(f"\nSample {sample_i + 1}: {TARGET_N} users, {n_obs} observations")
    print(f"  Main effect of activity: {main_effect:.2f}% (SE={main_se:.2f}, p={main_p:.4f})")
    print(f"  Heterogeneity test: χ² = {lr_stat:.2f}, df = {df}, p = {p_value:.4f}")

    sample_stats.append({
        'sample': sample_i + 1,
        'n_users': TARGET_N,
        'n_obs': n_obs,
        'main_effect': main_effect,
        'main_se': main_se,
        'main_p': main_p,
        'chi2': lr_stat,
        'het_p': p_value
    })

# Summary across samples
print("\n" + "=" * 70)
print("SUMMARY ACROSS 5 SAMPLES")
print("=" * 70)

stats_df = pd.DataFrame(sample_stats)
print(f"\nMain effect of activity:")
print(f"  Mean: {stats_df['main_effect'].mean():.2f}%")
print(f"  Range: {stats_df['main_effect'].min():.2f}% to {stats_df['main_effect'].max():.2f}%")

print(f"\nHeterogeneity test (χ² statistic):")
print(f"  Mean: {stats_df['chi2'].mean():.2f}")
print(f"  Range: {stats_df['chi2'].min():.2f} to {stats_df['chi2'].max():.2f}")

print(f"\nHeterogeneity p-values:")
for _, row in stats_df.iterrows():
    sig = "***" if row['het_p'] < 0.001 else "**" if row['het_p'] < 0.01 else "*" if row['het_p'] < 0.05 else ""
    print(f"  Sample {int(row['sample'])}: p = {row['het_p']:.4f} {sig}")

n_significant = (stats_df['het_p'] < 0.05).sum()
print(f"\nSamples with significant heterogeneity (p < 0.05): {n_significant} / {N_SAMPLES}")

# Compare to women
print("\n" + "=" * 70)
print("COMPARISON TO WOMEN")
print("=" * 70)
print(f"\nWomen (n=138):")
print(f"  Heterogeneity: χ² = 22.52, p < 0.0001")
print(f"\nMen matched samples (n=138 each, averaged):")
print(f"  Heterogeneity: χ² = {stats_df['chi2'].mean():.2f}, mean p = {stats_df['het_p'].mean():.4f}")

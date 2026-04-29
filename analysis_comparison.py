import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

def run_analysis(df, outcome, phase_col, group_name, split_type):
    """Run mixed-effects analysis and return results."""

    model_df = df[['userid', 'active', phase_col, outcome]].dropna().copy()
    model_df['active'] = model_df['active'].astype(int)

    # Get unique phases
    phases = model_df[phase_col].unique()

    # Set reference category
    if 'early_luteal' in phases:
        ref_phase = 'early_luteal'
        phase_order = ['menstruation', 'follicular', 'early_luteal', 'late_luteal']
    else:
        ref_phase = 'week_1'
        phase_order = ['week_1', 'week_2', 'week_3', 'week_4']

    model_df[phase_col] = pd.Categorical(model_df[phase_col], categories=phase_order)

    n_obs = len(model_df)
    n_users = model_df['userid'].nunique()
    n_inactive = (model_df['active'] == 0).sum()
    n_active = (model_df['active'] == 1).sum()

    # Model 1: Main effects
    model1 = smf.mixedlm(
        f"{outcome} ~ C(active) + C({phase_col})",
        model_df,
        groups=model_df['userid']
    )
    result1 = model1.fit(reml=True)

    # Model 2: With interaction
    model2 = smf.mixedlm(
        f"{outcome} ~ C(active) * C({phase_col})",
        model_df,
        groups=model_df['userid']
    )
    result2 = model2.fit(reml=True)

    # Calculate phase-specific effects
    vcov = result2.cov_params()
    base_effect = result2.params['C(active)[T.1]']
    base_se = result2.bse['C(active)[T.1]']

    results = []

    # Reference phase
    ci_low = base_effect - 1.96 * base_se
    ci_high = base_effect + 1.96 * base_se
    z = base_effect / base_se
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    results.append({
        'phase': ref_phase,
        'effect': base_effect,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'p_value': p_val
    })

    # Other phases
    for phase in phase_order:
        if phase == ref_phase:
            continue
        interaction_term = f'C(active)[T.1]:C({phase_col})[T.{phase}]'
        if interaction_term in result2.params.index:
            effect = base_effect + result2.params[interaction_term]
            var_base = vcov.loc['C(active)[T.1]', 'C(active)[T.1]']
            var_int = vcov.loc[interaction_term, interaction_term]
            cov = vcov.loc['C(active)[T.1]', interaction_term]
            combined_se = np.sqrt(var_base + var_int + 2 * cov)
            ci_low = effect - 1.96 * combined_se
            ci_high = effect + 1.96 * combined_se
            z = effect / combined_se
            p_val = 2 * (1 - stats.norm.cdf(abs(z)))
            results.append({
                'phase': phase,
                'effect': effect,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'p_value': p_val
            })

    # Heterogeneity test
    ll1 = result1.llf
    ll2 = result2.llf
    lr_stat = 2 * (ll2 - ll1)
    p_lr = 1 - stats.chi2.cdf(lr_stat, 3)

    return {
        'group': group_name,
        'split': split_type,
        'n_obs': n_obs,
        'n_users': n_users,
        'n_inactive': n_inactive,
        'n_active': n_active,
        'phase_results': results,
        'lr_stat': lr_stat,
        'p_heterogeneity': p_lr
    }

def print_results(res):
    """Print results in formatted table."""
    print(f"\n{'='*60}")
    print(f"{res['group']} - {res['split']}")
    print(f"{'='*60}")
    print(f"N = {res['n_obs']} observations, {res['n_users']} users")
    print(f"Inactive: {res['n_inactive']} ({res['n_inactive']/res['n_obs']*100:.1f}%)")
    print(f"Active: {res['n_active']} ({res['n_active']/res['n_obs']*100:.1f}%)")
    print(f"\nPhase-specific effects:")
    for r in res['phase_results']:
        sig = ""
        if r['p_value'] < 0.001: sig = "***"
        elif r['p_value'] < 0.01: sig = "**"
        elif r['p_value'] < 0.05: sig = "*"
        print(f"  {r['phase']:15s}: {r['effect']:+5.2f}% [{r['ci_low']:+5.2f}%, {r['ci_high']:+5.2f}%] p={r['p_value']:.4f} {sig}")
    print(f"\nHeterogeneity: χ²={res['lr_stat']:.2f}, p={res['p_heterogeneity']:.6f}", end="")
    if res['p_heterogeneity'] < 0.001: print(" ***")
    elif res['p_heterogeneity'] < 0.01: print(" **")
    elif res['p_heterogeneity'] < 0.05: print(" *")
    else: print("")

# ============================================
# WOMEN - MEDIAN SPLIT (current data)
# ============================================
print("\n" + "#"*70)
print("WOMEN - MEDIAN SPLIT")
print("#"*70)

df_women_median = pd.read_csv('female_cycle_days_filtered.csv')
women_median = run_analysis(df_women_median, 'next_day_mood_pct', 'cycle_phase', 'WOMEN', 'Median Split')
print_results(women_median)

# ============================================
# Now prepare QUARTILE data for both
# ============================================

# Women - Quartile
print("\n" + "#"*70)
print("PREPARING QUARTILE SPLITS...")
print("#"*70)

# Re-run women with quartile split
import subprocess
subprocess.run(['/Users/kyraedwards/Documents/Oxford/DPhil/MCAnalysis/venv/bin/python',
                '/Users/kyraedwards/Documents/Oxford/DPhil/Mood_PA_MC/clean_female_cycles_v3.py'],
               capture_output=True)

df_women_quartile = pd.read_csv('female_cycle_days_filtered.csv')
women_quartile = run_analysis(df_women_quartile, 'next_day_mood_pct', 'cycle_phase', 'WOMEN', 'Quartile Split')
print_results(women_quartile)

# Men - Quartile (need to switch back)
# Update male script to use quartile
male_script = open('clean_male_pseudo_cycles.py', 'r').read()
male_script = male_script.replace(
    "user_median_activity = pseudo_cycles_with_activity.groupby('userid')['activity_minutes'].transform('median')",
    "user_q25_activity = pseudo_cycles_with_activity.groupby('userid')['activity_minutes'].transform(lambda x: x.quantile(0.25))"
)
male_script = male_script.replace(
    "pseudo_cycles_with_activity['user_median_activity'] = user_median_activity",
    "pseudo_cycles_with_activity['user_q25_activity'] = user_q25_activity"
)
male_script = male_script.replace(
    "pseudo_cycles_with_activity['activity_minutes'] > pseudo_cycles_with_activity['user_median_activity']",
    "pseudo_cycles_with_activity['activity_minutes'] > pseudo_cycles_with_activity['user_q25_activity']"
)

with open('clean_male_pseudo_cycles_quartile.py', 'w') as f:
    f.write(male_script)

subprocess.run(['/Users/kyraedwards/Documents/Oxford/DPhil/MCAnalysis/venv/bin/python',
                'clean_male_pseudo_cycles_quartile.py'],
               capture_output=True)

df_men_quartile = pd.read_csv('male_pseudo_cycles.csv')
men_quartile = run_analysis(df_men_quartile, 'next_day_mood_pct', 'pseudo_phase', 'MEN', 'Quartile Split')
print_results(men_quartile)

# Men - Median (already have this data, need to regenerate)
subprocess.run(['/Users/kyraedwards/Documents/Oxford/DPhil/MCAnalysis/venv/bin/python',
                'clean_male_pseudo_cycles.py'],
               capture_output=True)

df_men_median = pd.read_csv('male_pseudo_cycles.csv')
men_median = run_analysis(df_men_median, 'next_day_mood_pct', 'pseudo_phase', 'MEN', 'Median Split')
print_results(men_median)

# ============================================
# SUMMARY COMPARISON
# ============================================
print("\n" + "#"*70)
print("SUMMARY COMPARISON")
print("#"*70)

print("\n" + "="*80)
print("QUARTILE SPLIT (Bottom 25% vs Top 75%)")
print("="*80)
print(f"\n{'Phase':<20} {'Women':>15} {'Men':>15}")
print("-"*50)

women_q_dict = {r['phase']: r for r in women_quartile['phase_results']}
men_q_dict = {r['phase']: r for r in men_quartile['phase_results']}

phase_mapping = [
    ('menstruation', 'week_1'),
    ('follicular', 'week_2'),
    ('early_luteal', 'week_3'),
    ('late_luteal', 'week_4')
]

for w_phase, m_phase in phase_mapping:
    w = women_q_dict.get(w_phase, {})
    m = men_q_dict.get(m_phase, {})
    w_str = f"{w.get('effect', 0):+.1f}%{'*' if w.get('p_value', 1) < 0.05 else ''}" if w else "N/A"
    m_str = f"{m.get('effect', 0):+.1f}%{'*' if m.get('p_value', 1) < 0.05 else ''}" if m else "N/A"
    print(f"{w_phase:<20} {w_str:>15} {m_str:>15}")

print(f"\n{'Heterogeneity p':<20} {women_quartile['p_heterogeneity']:>15.6f} {men_quartile['p_heterogeneity']:>15.6f}")

print("\n" + "="*80)
print("MEDIAN SPLIT (Below vs Above Personal Median)")
print("="*80)
print(f"\n{'Phase':<20} {'Women':>15} {'Men':>15}")
print("-"*50)

women_m_dict = {r['phase']: r for r in women_median['phase_results']}
men_m_dict = {r['phase']: r for r in men_median['phase_results']}

for w_phase, m_phase in phase_mapping:
    w = women_m_dict.get(w_phase, {})
    m = men_m_dict.get(m_phase, {})
    w_str = f"{w.get('effect', 0):+.1f}%{'*' if w.get('p_value', 1) < 0.05 else ''}" if w else "N/A"
    m_str = f"{m.get('effect', 0):+.1f}%{'*' if m.get('p_value', 1) < 0.05 else ''}" if m else "N/A"
    print(f"{w_phase:<20} {w_str:>15} {m_str:>15}")

print(f"\n{'Heterogeneity p':<20} {women_median['p_heterogeneity']:>15.6f} {men_median['p_heterogeneity']:>15.6f}")

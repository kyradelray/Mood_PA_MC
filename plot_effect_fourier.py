import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================
# Load data
# ============================================
df_women = pd.read_csv('female_cycle_days_filtered.csv')
df_men = pd.read_csv('male_pseudo_cycles.csv')

# Filter women
plot_df_women = df_women[['userid', 'cycle_day', 'active', 'next_day_mood_pct', 'cycle_phase']].dropna().copy()
plot_df_women['active'] = plot_df_women['active'].astype(int)
plot_df_women = plot_df_women[(plot_df_women['cycle_day'] >= -14) & (plot_df_women['cycle_day'] <= 13)].copy()

# Filter men
plot_df_men = df_men[['userid', 'cycle_day', 'active', 'next_day_mood_pct', 'pseudo_phase']].dropna().copy()
plot_df_men['active'] = plot_df_men['active'].astype(int)
plot_df_men = plot_df_men[(plot_df_men['cycle_day'] >= 0) & (plot_df_men['cycle_day'] <= 27)].copy()

# Sample matched men (138 users, average over 5 samples)
TARGET_N = 138
N_SAMPLES = 5
all_male_users = plot_df_men['userid'].unique()

print(f"Women: {plot_df_women['userid'].nunique()} users")
print(f"Men: {len(all_male_users)} users, sampling {TARGET_N} x {N_SAMPLES}")

# ============================================
# Phase definitions
# ============================================
phase_colors = {
    'early_luteal': '#E8D4F0',
    'late_luteal': '#F0E4D4',
    'menstruation': '#FFD6D6',
    'follicular': '#D4F0D4',
    'phase_1': '#FFD6D6',
    'phase_2': '#D4F0D4',
    'phase_3': '#E8D4F0',
    'phase_4': '#F0E4D4'
}

phase_ranges_women = {
    'early_luteal': (-14, -8),
    'late_luteal': (-7, -1),
    'menstruation': (0, 5),
    'follicular': (6, 13)
}

phase_ranges_men = {
    'phase_1': (0, 5),
    'phase_2': (6, 13),
    'phase_3': (14, 20),
    'phase_4': (21, 27)
}

phase_labels_women = {
    'early_luteal': 'Early Luteal',
    'late_luteal': 'Late Luteal',
    'menstruation': 'Menstruation',
    'follicular': 'Follicular'
}

phase_labels_men = {
    'phase_1': 'Phase 1',
    'phase_2': 'Phase 2',
    'phase_3': 'Phase 3',
    'phase_4': 'Phase 4'
}

phase_effects_women = {
    'early_luteal': {'effect': 3.50, 'sig': '*'},
    'late_luteal': {'effect': 0.83, 'sig': ''},
    'menstruation': {'effect': 3.49, 'sig': '*'},
    'follicular': {'effect': 7.63, 'sig': '***'}
}

CI_MULTIPLIER = 1.645

# ============================================
# Fourier fitting functions
# ============================================
def fit_fourier(x, y, n_harmonics=3, period=28):
    x_norm = 2 * np.pi * (x - x.min()) / period
    n = len(x)
    X = np.ones((n, 1 + 2 * n_harmonics))
    for i in range(1, n_harmonics + 1):
        X[:, 2*i - 1] = np.cos(i * x_norm)
        X[:, 2*i] = np.sin(i * x_norm)
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return coeffs, x.min(), period

def eval_fourier(x_new, coeffs, x_min, period, n_harmonics=3):
    x_norm = 2 * np.pi * (x_new - x_min) / period
    y_pred = np.full_like(x_norm, coeffs[0])
    for i in range(1, n_harmonics + 1):
        y_pred += coeffs[2*i - 1] * np.cos(i * x_norm)
        y_pred += coeffs[2*i] * np.sin(i * x_norm)
    return y_pred

# ============================================
# Calculate daily effects for women
# ============================================
def calculate_daily_effects(df):
    """Calculate effect (active - inactive) by cycle day with CI."""
    means = df.groupby(['cycle_day', 'active'])['next_day_mood_pct'].agg(['mean', 'std', 'count', 'sem']).reset_index()
    means.columns = ['cycle_day', 'active', 'mean', 'std', 'count', 'sem']

    inactive = means[means['active'] == 0].set_index('cycle_day')
    active = means[means['active'] == 1].set_index('cycle_day')

    # Merge on cycle_day
    common_days = inactive.index.intersection(active.index)

    results = []
    for day in common_days:
        diff = active.loc[day, 'mean'] - inactive.loc[day, 'mean']
        se_diff = np.sqrt(active.loc[day, 'sem']**2 + inactive.loc[day, 'sem']**2)
        results.append({
            'cycle_day': day,
            'effect': diff,
            'se': se_diff,
            'ci_low': diff - CI_MULTIPLIER * se_diff,
            'ci_high': diff + CI_MULTIPLIER * se_diff
        })

    return pd.DataFrame(results).sort_values('cycle_day')

# Women's daily effects
women_effects = calculate_daily_effects(plot_df_women)
print(f"Women: {len(women_effects)} daily effect estimates")

# Men's daily effects (averaged over samples)
all_men_effects = []
for sample_i in range(N_SAMPLES):
    sampled_users = np.random.choice(all_male_users, size=TARGET_N, replace=False)
    sample_df = plot_df_men[plot_df_men['userid'].isin(sampled_users)].copy()
    sample_effects = calculate_daily_effects(sample_df)
    sample_effects['sample'] = sample_i
    all_men_effects.append(sample_effects)

# Average across samples
men_effects_combined = pd.concat(all_men_effects)
men_effects = men_effects_combined.groupby('cycle_day').agg({
    'effect': 'mean',
    'se': 'mean',
    'ci_low': 'mean',
    'ci_high': 'mean'
}).reset_index()
print(f"Men (matched): {len(men_effects)} daily effect estimates")

# Calculate matched men phase effects for labels
matched_phase_effects = {}
for phase, (start, end) in phase_ranges_men.items():
    phase_data = men_effects[(men_effects['cycle_day'] >= start) & (men_effects['cycle_day'] <= end)]
    if len(phase_data) > 0:
        avg_effect = phase_data['effect'].mean()
        matched_phase_effects[phase] = {'effect': avg_effect, 'sig': ''}

# ============================================
# Create figure
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
N_HARMONICS = 3

# --- WOMEN ---
ax1 = axes[0]

# Phase backgrounds
for phase, (start, end) in phase_ranges_women.items():
    ax1.axvspan(start - 0.5, end + 0.5, alpha=0.4, color=phase_colors[phase])

# Fourier fit
x_w = women_effects['cycle_day'].values
y_w = women_effects['effect'].values
ci_low_w = women_effects['ci_low'].values
ci_high_w = women_effects['ci_high'].values

sort_idx = np.argsort(x_w)
x_w, y_w = x_w[sort_idx], y_w[sort_idx]
ci_low_w, ci_high_w = ci_low_w[sort_idx], ci_high_w[sort_idx]

coeffs_y, x_min, period = fit_fourier(x_w, y_w, n_harmonics=N_HARMONICS, period=28)
coeffs_ci_low, _, _ = fit_fourier(x_w, ci_low_w, n_harmonics=N_HARMONICS, period=28)
coeffs_ci_high, _, _ = fit_fourier(x_w, ci_high_w, n_harmonics=N_HARMONICS, period=28)

x_smooth = np.linspace(x_w.min(), x_w.max(), 200)
y_smooth = eval_fourier(x_smooth, coeffs_y, x_min, period, N_HARMONICS)
ci_low_smooth = eval_fourier(x_smooth, coeffs_ci_low, x_min, period, N_HARMONICS)
ci_high_smooth = eval_fourier(x_smooth, coeffs_ci_high, x_min, period, N_HARMONICS)

ax1.fill_between(x_smooth, ci_low_smooth, ci_high_smooth, color='#1F77B4', alpha=0.25)
ax1.plot(x_smooth, y_smooth, color='#1F77B4', linewidth=3)
ax1.scatter(x_w, y_w, c='#1F77B4', s=50, alpha=0.7, zorder=3)

ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax1.set_xlim(-14.5, 13.5)
ax1.set_xlabel('Cycle Day', fontsize=11)
ax1.set_ylabel('Effect of Activity on Next-Day Mood\n(Active − Inactive, % points)', fontsize=11)
ax1.set_title('Women (n = 138)\nHeterogeneity: χ² = 22.52, p < 0.0001', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle=':')

# --- MEN ---
ax2 = axes[1]

# Phase backgrounds
for phase, (start, end) in phase_ranges_men.items():
    ax2.axvspan(start - 0.5, end + 0.5, alpha=0.4, color=phase_colors[phase])

# Fourier fit
x_m = men_effects['cycle_day'].values
y_m = men_effects['effect'].values
ci_low_m = men_effects['ci_low'].values
ci_high_m = men_effects['ci_high'].values

sort_idx = np.argsort(x_m)
x_m, y_m = x_m[sort_idx], y_m[sort_idx]
ci_low_m, ci_high_m = ci_low_m[sort_idx], ci_high_m[sort_idx]

coeffs_y_m, x_min_m, _ = fit_fourier(x_m, y_m, n_harmonics=N_HARMONICS, period=28)
coeffs_ci_low_m, _, _ = fit_fourier(x_m, ci_low_m, n_harmonics=N_HARMONICS, period=28)
coeffs_ci_high_m, _, _ = fit_fourier(x_m, ci_high_m, n_harmonics=N_HARMONICS, period=28)

x_smooth_m = np.linspace(x_m.min(), x_m.max(), 200)
y_smooth_m = eval_fourier(x_smooth_m, coeffs_y_m, x_min_m, 28, N_HARMONICS)
ci_low_smooth_m = eval_fourier(x_smooth_m, coeffs_ci_low_m, x_min_m, 28, N_HARMONICS)
ci_high_smooth_m = eval_fourier(x_smooth_m, coeffs_ci_high_m, x_min_m, 28, N_HARMONICS)

ax2.fill_between(x_smooth_m, ci_low_smooth_m, ci_high_smooth_m, color='#1F77B4', alpha=0.25)
ax2.plot(x_smooth_m, y_smooth_m, color='#1F77B4', linewidth=3)
ax2.scatter(x_m, y_m, c='#1F77B4', s=50, alpha=0.7, zorder=3)

ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax2.set_xlim(-0.5, 27.5)
ax2.set_xlabel('Day (from first Monday of month)', fontsize=11)
ax2.set_title(f'Men (n = 138, averaged over {N_SAMPLES} samples)\nMatched sample size comparison', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle=':')

# Match y-axis
y_min = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
ax1.set_ylim(y_min, y_max)
ax2.set_ylim(y_min, y_max)

# Phase labels
for phase, (start, end) in phase_ranges_women.items():
    mid = (start + end) / 2
    effect_info = phase_effects_women[phase]
    ax1.text(mid, y_max * 0.95, phase_labels_women[phase],
             ha='center', va='top', fontsize=10, fontweight='bold')
    ax1.text(mid, y_max * 0.82, f"{effect_info['effect']:+.1f}%{effect_info['sig']}",
             ha='center', va='top', fontsize=9, fontstyle='italic', color='#333333')

for phase, (start, end) in phase_ranges_men.items():
    mid = (start + end) / 2
    if phase in matched_phase_effects:
        effect_info = matched_phase_effects[phase]
        ax2.text(mid, y_max * 0.95, phase_labels_men[phase],
                 ha='center', va='top', fontsize=10, fontweight='bold')
        ax2.text(mid, y_max * 0.82, f"{effect_info['effect']:+.1f}%",
                 ha='center', va='top', fontsize=9, fontstyle='italic', color='#333333')

plt.tight_layout()
plt.savefig('effect_fourier_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('effect_fourier_comparison.pdf', bbox_inches='tight')
print("\nSaved: effect_fourier_comparison.png and effect_fourier_comparison.pdf")

plt.show()

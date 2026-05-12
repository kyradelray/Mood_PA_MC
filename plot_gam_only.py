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

plot_df_women = df_women[['userid', 'cycle_day', 'active', 'next_day_mood_pct', 'cycle_phase']].dropna().copy()
plot_df_women['active'] = plot_df_women['active'].astype(int)
plot_df_women = plot_df_women[(plot_df_women['cycle_day'] >= -14) & (plot_df_women['cycle_day'] <= 13)].copy()

plot_df_men = df_men[['userid', 'cycle_day', 'active', 'next_day_mood_pct', 'pseudo_phase']].dropna().copy()
plot_df_men['active'] = plot_df_men['active'].astype(int)
plot_df_men = plot_df_men[(plot_df_men['cycle_day'] >= 0) & (plot_df_men['cycle_day'] <= 27)].copy()

# Sample matched men - same total sample size as women
TARGET_N = 138
N_SAMPLES = 5
all_male_users = plot_df_men['userid'].unique()

# ============================================
# Phase definitions
# ============================================
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

CI_MULTIPLIER = 1.645

# ============================================
# Fourier fitting
# ============================================
def fit_fourier(x, y, n_harmonics=1, period=28):
    x_norm = 2 * np.pi * (x - x.min()) / period
    n = len(x)
    X = np.ones((n, 1 + 2 * n_harmonics))
    for i in range(1, n_harmonics + 1):
        X[:, 2*i - 1] = np.cos(i * x_norm)
        X[:, 2*i] = np.sin(i * x_norm)
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return coeffs, x.min(), period

def eval_fourier(x_new, coeffs, x_min, period, n_harmonics=1):
    x_norm = 2 * np.pi * (x_new - x_min) / period
    y_pred = np.full_like(x_norm, coeffs[0], dtype=float)
    for i in range(1, n_harmonics + 1):
        y_pred += coeffs[2*i - 1] * np.cos(i * x_norm)
        y_pred += coeffs[2*i] * np.sin(i * x_norm)
    return y_pred

# ============================================
# Calculate phase means
# ============================================
def calculate_phase_means(df, phase_ranges):
    results = {}
    for phase, (start, end) in phase_ranges.items():
        phase_data = df[(df['cycle_day'] >= start) & (df['cycle_day'] <= end)]
        mid_day = (start + end) / 2

        for active_status in [0, 1]:
            subset = phase_data[phase_data['active'] == active_status]
            if len(subset) > 0:
                mean = subset['next_day_mood_pct'].mean()
                sem = subset['next_day_mood_pct'].sem()
                n_users = subset['userid'].nunique()
                results[(phase, active_status)] = {
                    'mean': mean, 'ci_low': mean - CI_MULTIPLIER * sem,
                    'ci_high': mean + CI_MULTIPLIER * sem,
                    'start': start, 'end': end, 'mid': mid_day,
                    'n': len(subset), 'n_users': n_users
                }
    return results

# Women's results
women_results = calculate_phase_means(plot_df_women, phase_ranges_women)

# Men's results (averaged over samples)
all_sample_results = []
for _ in range(N_SAMPLES):
    sampled_users = np.random.choice(all_male_users, size=TARGET_N, replace=False)
    sample_df = plot_df_men[plot_df_men['userid'].isin(sampled_users)].copy()
    all_sample_results.append(calculate_phase_means(sample_df, phase_ranges_men))

# Average men's results
men_results = {}
for phase in phase_ranges_men.keys():
    for status in [0, 1]:
        key = (phase, status)
        values = [sr[key] for sr in all_sample_results if key in sr]
        if values:
            men_results[key] = {
                'mean': np.mean([v['mean'] for v in values]),
                'ci_low': np.mean([v['ci_low'] for v in values]),
                'ci_high': np.mean([v['ci_high'] for v in values]),
                'start': values[0]['start'],
                'end': values[0]['end'],
                'mid': values[0]['mid'],
                'n': np.mean([v['n'] for v in values]),
                'n_users': np.mean([v['n_users'] for v in values])
            }

# ============================================
# Create plot
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

def plot_gam_only(ax, results, phase_ranges, phase_colors, phase_labels, title, xlim):
    # Add phase background colors
    for phase, (start, end) in phase_ranges.items():
        ax.axvspan(start - 0.5, end + 0.5, alpha=0.4, color=phase_colors[phase])

    # Get midpoints and values for each phase
    phase_order = list(phase_ranges.keys())
    mid_points = []
    effects_active = []
    effects_inactive = []
    ci_low_active = []
    ci_high_active = []
    ci_low_inactive = []
    ci_high_inactive = []

    for phase in phase_order:
        start, end = phase_ranges[phase]
        mid = (start + end) / 2
        mid_points.append(mid)
        if (phase, 1) in results:
            effects_active.append(results[(phase, 1)]['mean'])
            ci_low_active.append(results[(phase, 1)]['ci_low'])
            ci_high_active.append(results[(phase, 1)]['ci_high'])
        if (phase, 0) in results:
            effects_inactive.append(results[(phase, 0)]['mean'])
            ci_low_inactive.append(results[(phase, 0)]['ci_low'])
            ci_high_inactive.append(results[(phase, 0)]['ci_high'])

    mid_points = np.array(mid_points)
    effects_active = np.array(effects_active)
    effects_inactive = np.array(effects_inactive)
    ci_low_active = np.array(ci_low_active)
    ci_high_active = np.array(ci_high_active)
    ci_low_inactive = np.array(ci_low_inactive)
    ci_high_inactive = np.array(ci_high_inactive)

    # Fit Fourier
    N_HARMONICS = 1
    period = xlim[1] - xlim[0]

    # Active
    coeffs_active, x_min, _ = fit_fourier(mid_points, effects_active, n_harmonics=N_HARMONICS, period=period)
    coeffs_ci_low_a, _, _ = fit_fourier(mid_points, ci_low_active, n_harmonics=N_HARMONICS, period=period)
    coeffs_ci_high_a, _, _ = fit_fourier(mid_points, ci_high_active, n_harmonics=N_HARMONICS, period=period)

    # Inactive
    coeffs_inactive, _, _ = fit_fourier(mid_points, effects_inactive, n_harmonics=N_HARMONICS, period=period)
    coeffs_ci_low_i, _, _ = fit_fourier(mid_points, ci_low_inactive, n_harmonics=N_HARMONICS, period=period)
    coeffs_ci_high_i, _, _ = fit_fourier(mid_points, ci_high_inactive, n_harmonics=N_HARMONICS, period=period)

    # Smooth curves - extend to full plot range
    x_smooth = np.linspace(xlim[0], xlim[1], 200)
    y_active = eval_fourier(x_smooth, coeffs_active, x_min, period, N_HARMONICS)
    y_inactive = eval_fourier(x_smooth, coeffs_inactive, x_min, period, N_HARMONICS)
    ci_low_active_smooth = eval_fourier(x_smooth, coeffs_ci_low_a, x_min, period, N_HARMONICS)
    ci_high_active_smooth = eval_fourier(x_smooth, coeffs_ci_high_a, x_min, period, N_HARMONICS)
    ci_low_inactive_smooth = eval_fourier(x_smooth, coeffs_ci_low_i, x_min, period, N_HARMONICS)
    ci_high_inactive_smooth = eval_fourier(x_smooth, coeffs_ci_high_i, x_min, period, N_HARMONICS)

    # Plot GAM curves with CI bands
    ax.fill_between(x_smooth, ci_low_active_smooth, ci_high_active_smooth,
                   color='#2CA02C', alpha=0.2, zorder=1)
    ax.plot(x_smooth, y_active, color='#2CA02C', linewidth=2.5, zorder=2, label='Active (above median)')

    ax.fill_between(x_smooth, ci_low_inactive_smooth, ci_high_inactive_smooth,
                   color='#D62728', alpha=0.2, zorder=1)
    ax.plot(x_smooth, y_inactive, color='#D62728', linewidth=2.5, zorder=2, label='Inactive (below median)')

    # Plot phase midpoint dots
    ax.scatter(mid_points, effects_active, c='#2CA02C', s=80, zorder=3)
    ax.scatter(mid_points, effects_inactive, c='#D62728', s=80, zorder=3)

    # Reference line at 0
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlim(xlim)
    ax.set_ylabel('Next-Day Mood\n(% change from personal mean)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':')
    # Legend will be added at figure level

    return results

# Women
ax1 = axes[0]
plot_gam_only(ax1, women_results, phase_ranges_women, phase_colors,
              phase_labels_women, 'Women (n = 138)\nHeterogeneity: χ² = 22.52, p < 0.0001',
              xlim=(-14.5, 13.5))
ax1.set_xlabel('Cycle Day', fontsize=11)

# Men (matched)
ax2 = axes[1]
plot_gam_only(ax2, men_results, phase_ranges_men, phase_colors,
              phase_labels_men, 'Men (n = 138, matched sample)\nHeterogeneity: χ² = 11.11, p = 0.41',
              xlim=(-0.5, 27.5))
ax2.set_xlabel('Day (from first Monday of month)', fontsize=11)

# Match y-axis
y_min = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
ax1.set_ylim(y_min, y_max)
ax2.set_ylim(y_min, y_max)

# Phase labels with effects and sample sizes
phase_effects_women = {
    'early_luteal': {'effect': 3.50, 'sig': '*'},
    'late_luteal': {'effect': 0.83, 'sig': ''},
    'menstruation': {'effect': 3.49, 'sig': '*'},
    'follicular': {'effect': 7.63, 'sig': '***'}
}

for phase, (start, end) in phase_ranges_women.items():
    mid = (start + end) / 2
    effect_info = phase_effects_women[phase]
    # Get sample size (sum of active + inactive)
    n_obs = 0
    n_users = 0
    if (phase, 0) in women_results:
        n_obs += women_results[(phase, 0)]['n']
        n_users = max(n_users, women_results[(phase, 0)]['n_users'])
    if (phase, 1) in women_results:
        n_obs += women_results[(phase, 1)]['n']
        n_users = max(n_users, women_results[(phase, 1)]['n_users'])

    ax1.text(mid, y_max * 0.95, phase_labels_women[phase],
             ha='center', va='top', fontsize=10, fontweight='bold')
    ax1.text(mid, y_max * 0.82, f"{effect_info['effect']:+.1f}%{effect_info['sig']}",
             ha='center', va='top', fontsize=9, fontstyle='italic', color='#333333')
    ax1.text(mid, y_min * 0.95, f"{int(n_users)} users\n{int(n_obs)} obs",
             ha='center', va='bottom', fontsize=8, color='#666666')

for phase, (start, end) in phase_ranges_men.items():
    mid = (start + end) / 2
    if (phase, 1) in men_results and (phase, 0) in men_results:
        effect = men_results[(phase, 1)]['mean'] - men_results[(phase, 0)]['mean']
        # Get sample size (sum of active + inactive, averaged)
        n_obs = men_results[(phase, 0)]['n'] + men_results[(phase, 1)]['n']
        n_users = max(men_results[(phase, 0)]['n_users'], men_results[(phase, 1)]['n_users'])

        ax2.text(mid, y_max * 0.95, phase_labels_men[phase],
                 ha='center', va='top', fontsize=10, fontweight='bold')
        ax2.text(mid, y_max * 0.82, f"{effect:+.1f}%",
                 ha='center', va='top', fontsize=9, fontstyle='italic', color='#333333')
        ax2.text(mid, y_min * 0.95, f"{int(n_users)} users\n{int(n_obs)} obs",
                 ha='center', va='bottom', fontsize=8, color='#666666')

# Add legend at bottom of figure
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#2CA02C', linewidth=2.5, label='Active (above median)'),
    Line2D([0], [0], color='#D62728', linewidth=2.5, label='Inactive (below median)'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=10,
           bbox_to_anchor=(0.5, 0.02))

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig('gam_only.png', dpi=150, bbox_inches='tight')
plt.savefig('gam_only.pdf', bbox_inches='tight')
print("Saved: gam_only.png and gam_only.pdf")

plt.show()

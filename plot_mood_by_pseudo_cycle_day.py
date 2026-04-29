import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('male_pseudo_cycles.csv')

# Filter to valid data
plot_df = df[['cycle_day', 'active', 'next_day_mood_pct', 'pseudo_phase']].dropna().copy()
plot_df['active'] = plot_df['active'].astype(int)

# Restrict to 28-day cycle: 0 to 27
plot_df = plot_df[(plot_df['cycle_day'] >= 0) & (plot_df['cycle_day'] <= 27)].copy()

print(f"Total observations: {len(plot_df)}")
print(f"Cycle day range: {plot_df['cycle_day'].min()} to {plot_df['cycle_day'].max()}")

# Calculate mean outcome by cycle day and activity status with 90% CI
means = plot_df.groupby(['cycle_day', 'active'])['next_day_mood_pct'].agg(['mean', 'std', 'count', 'sem']).reset_index()
means.columns = ['cycle_day', 'active', 'mean', 'std', 'count', 'sem']

# Calculate 90% confidence intervals
CI_MULTIPLIER = 1.645
means['ci_low'] = means['mean'] - CI_MULTIPLIER * means['sem']
means['ci_high'] = means['mean'] + CI_MULTIPLIER * means['sem']

# Separate active and inactive
inactive = means[means['active'] == 0].sort_values('cycle_day')
active = means[means['active'] == 1].sort_values('cycle_day')

# Define phase boundaries and colors (same structure as women)
phase_colors = {
    'phase_1': '#FFD6D6',     # Light red/pink (like menstruation)
    'phase_2': '#D4F0D4',     # Light green (like follicular)
    'phase_3': '#E8D4F0',     # Light purple (like early luteal)
    'phase_4': '#F0E4D4'      # Light tan (like late luteal)
}

phase_ranges = {
    'phase_1': (0, 5),      # 6 days, like menstruation
    'phase_2': (6, 13),     # 8 days, like follicular
    'phase_3': (14, 20),    # 7 days, like early luteal
    'phase_4': (21, 27)     # 7 days, like late luteal
}

# Phase-specific effects from analysis (median split)
phase_effects = {
    'phase_1': {'effect': 0.93, 'p': 0.383, 'sig': ''},
    'phase_2': {'effect': 2.12, 'p': 0.017, 'sig': '*'},
    'phase_3': {'effect': 0.54, 'p': 0.563, 'sig': ''},
    'phase_4': {'effect': 1.46, 'p': 0.129, 'sig': ''}
}

# Fourier fitting
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

def create_fourier_with_ci(data, color, label, ax, n_harmonics=3):
    x = data['cycle_day'].values
    y = data['mean'].values
    ci_low = data['ci_low'].values
    ci_high = data['ci_high'].values

    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    ci_low = ci_low[sort_idx]
    ci_high = ci_high[sort_idx]

    period = 28
    coeffs_y, x_min, _ = fit_fourier(x, y, n_harmonics=n_harmonics, period=period)
    coeffs_ci_low, _, _ = fit_fourier(x, ci_low, n_harmonics=n_harmonics, period=period)
    coeffs_ci_high, _, _ = fit_fourier(x, ci_high, n_harmonics=n_harmonics, period=period)

    x_smooth = np.linspace(x.min(), x.max(), 200)
    y_smooth = eval_fourier(x_smooth, coeffs_y, x_min, period, n_harmonics)
    ci_low_smooth = eval_fourier(x_smooth, coeffs_ci_low, x_min, period, n_harmonics)
    ci_high_smooth = eval_fourier(x_smooth, coeffs_ci_high, x_min, period, n_harmonics)

    ax.fill_between(x_smooth, ci_low_smooth, ci_high_smooth, color=color, alpha=0.2, zorder=1)
    ax.plot(x_smooth, y_smooth, color=color, linewidth=2.5, zorder=2, label=label)
    ax.scatter(x, y, c=color, s=60, alpha=0.7, zorder=3)

# Create figure
fig, ax = plt.subplots(figsize=(14, 8))

# Add phase background colors
for phase, (start, end) in phase_ranges.items():
    ax.axvspan(start - 0.5, end + 0.5, alpha=0.4, color=phase_colors[phase])

# Plot data
N_HARMONICS = 3
create_fourier_with_ci(inactive, '#D62728', 'Inactive (below median)', ax, n_harmonics=N_HARMONICS)
create_fourier_with_ci(active, '#2CA02C', 'Active (above median)', ax, n_harmonics=N_HARMONICS)

# Reference line
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# Labels
ax.set_xlabel('Day of Month (from 1st)', fontsize=12)
ax.set_ylabel('Next-Day Mood (% change from personal mean)', fontsize=12)
ax.set_title('Effect of Physical Activity on Next-Day Mood: MALES\n'
             '(Pseudo-phases from 1st of each month, same intervals as menstrual cycle)', fontsize=14)

ax.set_xlim(-0.5, 27.5)

# Phase labels
phase_labels_display = {
    'phase_1': 'Phase 1',
    'phase_2': 'Phase 2',
    'phase_3': 'Phase 3',
    'phase_4': 'Phase 4'
}

# Add phase labels with effects
for phase, (start, end) in phase_ranges.items():
    mid = (start + end) / 2
    effect_info = phase_effects[phase]
    effect_str = f"{effect_info['effect']:+.1f}%{effect_info['sig']}"

    ax.text(mid, ax.get_ylim()[1] * 0.98, phase_labels_display[phase],
            ha='center', va='top', fontsize=10, fontweight='bold')
    ax.text(mid, ax.get_ylim()[1] * 0.88, effect_str,
            ha='center', va='top', fontsize=9, fontstyle='italic', color='#333333')

# Legend
ax.legend(loc='lower right', framealpha=0.9, fontsize=9)
ax.grid(True, alpha=0.3, linestyle=':')

plt.tight_layout()
plt.savefig('mood_by_pseudo_cycle_day_males.png', dpi=150, bbox_inches='tight')
plt.savefig('mood_by_pseudo_cycle_day_males.pdf', bbox_inches='tight')
print("\nSaved: mood_by_pseudo_cycle_day_males.png")

plt.show()

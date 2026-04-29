import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('female_cycle_days_filtered.csv')

# Filter to valid data
plot_df = df[['cycle_day', 'active', 'next_day_mood_pct', 'cycle_phase']].dropna().copy()
plot_df['active'] = plot_df['active'].astype(int)

# Restrict to 28-day cycle: -14 to +13
plot_df = plot_df[(plot_df['cycle_day'] >= -14) & (plot_df['cycle_day'] <= 13)].copy()

print(f"Total observations: {len(plot_df)}")
print(f"Cycle day range: {plot_df['cycle_day'].min()} to {plot_df['cycle_day'].max()}")

# Calculate mean outcome by cycle day and activity status with 95% CI
means = plot_df.groupby(['cycle_day', 'active'])['next_day_mood_pct'].agg(['mean', 'std', 'count', 'sem']).reset_index()
means.columns = ['cycle_day', 'active', 'mean', 'std', 'count', 'sem']

# Calculate 90% confidence intervals (1.645 * SEM for thinner bands)
CI_MULTIPLIER = 1.645  # 1.96 for 95% CI, 1.645 for 90% CI, 1.0 for 68% CI
means['ci_low'] = means['mean'] - CI_MULTIPLIER * means['sem']
means['ci_high'] = means['mean'] + CI_MULTIPLIER * means['sem']

# Separate active and inactive
inactive = means[means['active'] == 0].sort_values('cycle_day')
active = means[means['active'] == 1].sort_values('cycle_day')

# Define phase boundaries and colors
phase_colors = {
    'early_luteal': '#E8D4F0',     # Light purple
    'late_luteal': '#F0E4D4',       # Light tan/beige
    'menstruation': '#FFD6D6',      # Light red/pink
    'follicular': '#D4F0D4'         # Light green
}

phase_ranges = {
    'early_luteal': (-14, -8),
    'late_luteal': (-7, -1),
    'menstruation': (0, 5),
    'follicular': (6, 13)  # Cap at day 13 for 28-day cycle
}

# Create figure
fig, ax = plt.subplots(figsize=(14, 8))

# Add phase background colors
for phase, (start, end) in phase_ranges.items():
    ax.axvspan(start - 0.5, end + 0.5, alpha=0.4, color=phase_colors[phase], label=f'{phase.replace("_", " ").title()}')

# Fourier series fitting for periodic/cyclical data
def fit_fourier(x, y, n_harmonics=3, period=28):
    """Fit truncated Fourier series to data for periodic smoothing."""
    # Normalize x to [0, 2*pi] for one period
    x_norm = 2 * np.pi * (x - x.min()) / period

    # Build design matrix for Fourier series
    # y = a0 + sum(a_n * cos(n*x) + b_n * sin(n*x))
    n = len(x)
    X = np.ones((n, 1 + 2 * n_harmonics))
    for i in range(1, n_harmonics + 1):
        X[:, 2*i - 1] = np.cos(i * x_norm)
        X[:, 2*i] = np.sin(i * x_norm)

    # Least squares fit
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    return coeffs, x.min(), period

def eval_fourier(x_new, coeffs, x_min, period, n_harmonics=3):
    """Evaluate Fourier series at new x values."""
    x_norm = 2 * np.pi * (x_new - x_min) / period

    y_pred = np.full_like(x_norm, coeffs[0])
    for i in range(1, n_harmonics + 1):
        y_pred += coeffs[2*i - 1] * np.cos(i * x_norm)
        y_pred += coeffs[2*i] * np.sin(i * x_norm)

    return y_pred

# Function to create Fourier-based curve with CI band
def create_fourier_with_ci(data, color, label, ax, n_harmonics=4):
    x = data['cycle_day'].values
    y = data['mean'].values
    ci_low = data['ci_low'].values
    ci_high = data['ci_high'].values

    # Sort by x
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    ci_low = ci_low[sort_idx]
    ci_high = ci_high[sort_idx]

    # Fit Fourier series (period = 28 days for menstrual cycle)
    period = 28
    coeffs_y, x_min, _ = fit_fourier(x, y, n_harmonics=n_harmonics, period=period)
    coeffs_ci_low, _, _ = fit_fourier(x, ci_low, n_harmonics=n_harmonics, period=period)
    coeffs_ci_high, _, _ = fit_fourier(x, ci_high, n_harmonics=n_harmonics, period=period)

    # Generate smooth x values for plotting
    x_smooth = np.linspace(x.min(), x.max(), 200)
    y_smooth = eval_fourier(x_smooth, coeffs_y, x_min, period, n_harmonics)
    ci_low_smooth = eval_fourier(x_smooth, coeffs_ci_low, x_min, period, n_harmonics)
    ci_high_smooth = eval_fourier(x_smooth, coeffs_ci_high, x_min, period, n_harmonics)

    # Plot CI band
    ax.fill_between(x_smooth, ci_low_smooth, ci_high_smooth, color=color, alpha=0.2, zorder=1)

    # Plot Fourier curve
    ax.plot(x_smooth, y_smooth, color=color, linewidth=2.5, zorder=2, label=label)

    # Plot dots
    ax.scatter(x, y, c=color, s=60, alpha=0.7, zorder=3)

    return coeffs_y, x_min, period

# Plot inactive (red) and active (green) with CIs using Fourier smoothing
# n_harmonics controls flexibility (higher = more flexible, lower = smoother)
N_HARMONICS = 3  # Number of Fourier harmonics (higher = more flexible)
create_fourier_with_ci(inactive, '#D62728', 'Inactive (bottom 25%)', ax, n_harmonics=N_HARMONICS)
create_fourier_with_ci(active, '#2CA02C', 'Active (top 75%)', ax, n_harmonics=N_HARMONICS)

# Add reference line at 0
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# Labels and formatting
ax.set_xlabel('Cycle Day', fontsize=12)
ax.set_ylabel('Next-Day Mood (% change from personal mean)', fontsize=12)
ax.set_title('Effect of Physical Activity on Next-Day Mood Across the Menstrual Cycle\n'
             '(Active = top 75% of personal activity, Inactive = bottom 25%)', fontsize=14)

# Set x-axis limits for 28-day cycle
ax.set_xlim(-14.5, 13.5)

# Phase-specific effects from mixed-effects analysis (quartile split)
phase_effects = {
    'early_luteal': {'effect': -1.56, 'p': 0.445, 'sig': ''},
    'late_luteal': {'effect': 2.99, 'p': 0.137, 'sig': ''},
    'menstruation': {'effect': 4.91, 'p': 0.016, 'sig': '*'},
    'follicular': {'effect': 8.55, 'p': 0.0001, 'sig': '***'}
}

# Add phase labels at top with effect sizes
for phase, (start, end) in phase_ranges.items():
    mid = (start + end) / 2
    phase_label = phase.replace('_', ' ').title()
    effect_info = phase_effects[phase]
    effect_str = f"{effect_info['effect']:+.1f}%{effect_info['sig']}"

    # Phase name
    ax.text(mid, ax.get_ylim()[1] * 0.98, phase_label,
            ha='center', va='top', fontsize=10, fontweight='bold')
    # Effect size below phase name
    ax.text(mid, ax.get_ylim()[1] * 0.88, effect_str,
            ha='center', va='top', fontsize=9, fontstyle='italic',
            color='#333333')

# Legend
handles, labels = ax.get_legend_handles_labels()
# Reorder: phases first, then data
phase_handles = handles[:4]
phase_labels = labels[:4]
data_handles = handles[4:]
data_labels = labels[4:]

ax.legend(data_handles + phase_handles, data_labels + phase_labels,
          loc='lower right', framealpha=0.9, fontsize=9)

# Grid
ax.grid(True, alpha=0.3, linestyle=':')

plt.tight_layout()
plt.savefig('mood_by_cycle_day.png', dpi=150, bbox_inches='tight')
plt.savefig('mood_by_cycle_day.pdf', bbox_inches='tight')
print("\nSaved: mood_by_cycle_day.png and mood_by_cycle_day.pdf")

# Also create a version showing the difference (active - inactive)
fig2, ax2 = plt.subplots(figsize=(14, 6))

# Calculate difference by cycle day with CI
merged = active.merge(inactive, on='cycle_day', suffixes=('_active', '_inactive'))
merged['diff'] = merged['mean_active'] - merged['mean_inactive']
# SE of difference (assuming independence)
merged['se_diff'] = np.sqrt(merged['sem_active']**2 + merged['sem_inactive']**2)
merged['diff_ci_low'] = merged['diff'] - CI_MULTIPLIER * merged['se_diff']
merged['diff_ci_high'] = merged['diff'] + CI_MULTIPLIER * merged['se_diff']

# Add phase backgrounds
for phase, (start, end) in phase_ranges.items():
    ax2.axvspan(start - 0.5, end + 0.5, alpha=0.4, color=phase_colors[phase])

# Sort and create spline
x = merged['cycle_day'].values
y = merged['diff'].values
ci_low = merged['diff_ci_low'].values
ci_high = merged['diff_ci_high'].values
sort_idx = np.argsort(x)
x = x[sort_idx]
y = y[sort_idx]
ci_low = ci_low[sort_idx]
ci_high = ci_high[sort_idx]

# Fit Fourier series for periodic smoothing
period = 28
coeffs_y, x_min, _ = fit_fourier(x, y, n_harmonics=N_HARMONICS, period=period)
coeffs_ci_low, _, _ = fit_fourier(x, ci_low, n_harmonics=N_HARMONICS, period=period)
coeffs_ci_high, _, _ = fit_fourier(x, ci_high, n_harmonics=N_HARMONICS, period=period)

# Generate smooth x values
x_smooth = np.linspace(x.min(), x.max(), 200)
y_smooth = eval_fourier(x_smooth, coeffs_y, x_min, period, N_HARMONICS)
ci_low_smooth = eval_fourier(x_smooth, coeffs_ci_low, x_min, period, N_HARMONICS)
ci_high_smooth = eval_fourier(x_smooth, coeffs_ci_high, x_min, period, N_HARMONICS)

# Plot CI band
ax2.fill_between(x_smooth, ci_low_smooth, ci_high_smooth, color='#1F77B4', alpha=0.2, zorder=1)

# Plot spline
ax2.plot(x_smooth, y_smooth, color='#1F77B4', linewidth=2.5, zorder=2)

# Plot dots
ax2.scatter(x, y, c='#1F77B4', s=60, alpha=0.7, zorder=3)

# Reference line at 0
ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1.5)

# Labels
ax2.set_xlabel('Cycle Day', fontsize=12)
ax2.set_ylabel('Mood Benefit of Activity\n(Active - Inactive, % points)', fontsize=12)
ax2.set_title('Difference in Next-Day Mood: Active vs Inactive Days Across the Menstrual Cycle', fontsize=14)
ax2.set_xlim(-14.5, 13.5)

# Phase labels with effect sizes
for phase, (start, end) in phase_ranges.items():
    mid = (start + end) / 2
    phase_label = phase.replace('_', ' ').title()
    effect_info = phase_effects[phase]
    effect_str = f"{effect_info['effect']:+.1f}%{effect_info['sig']}"

    # Phase name
    ax2.text(mid, ax2.get_ylim()[1] * 0.95, phase_label,
            ha='center', va='top', fontsize=10, fontweight='bold')
    # Effect size below phase name
    ax2.text(mid, ax2.get_ylim()[1] * 0.82, effect_str,
            ha='center', va='top', fontsize=9, fontstyle='italic',
            color='#333333')

ax2.grid(True, alpha=0.3, linestyle=':')

plt.tight_layout()
plt.savefig('mood_difference_by_cycle_day.png', dpi=150, bbox_inches='tight')
plt.savefig('mood_difference_by_cycle_day.pdf', bbox_inches='tight')
print("Saved: mood_difference_by_cycle_day.png and mood_difference_by_cycle_day.pdf")

plt.show()

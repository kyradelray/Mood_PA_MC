import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================
# WOMEN
# ============================================
df_women = pd.read_csv('female_cycle_days_filtered.csv')
plot_df_women = df_women[['cycle_day', 'active', 'next_day_mood_pct', 'cycle_phase']].dropna().copy()
plot_df_women['active'] = plot_df_women['active'].astype(int)
plot_df_women = plot_df_women[(plot_df_women['cycle_day'] >= -14) & (plot_df_women['cycle_day'] <= 13)].copy()

# ============================================
# MEN
# ============================================
df_men = pd.read_csv('male_pseudo_cycles.csv')
plot_df_men = df_men[['cycle_day', 'active', 'next_day_mood_pct', 'pseudo_phase']].dropna().copy()
plot_df_men['active'] = plot_df_men['active'].astype(int)
plot_df_men = plot_df_men[(plot_df_men['cycle_day'] >= 0) & (plot_df_men['cycle_day'] <= 27)].copy()

# Phase definitions
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

# CI multiplier (90% CI)
CI_MULTIPLIER = 1.645

def calculate_phase_means(df, phase_col, phase_ranges):
    """Calculate mean and CI for each phase and activity status."""
    results = {}
    for phase, (start, end) in phase_ranges.items():
        phase_data = df[(df['cycle_day'] >= start) & (df['cycle_day'] <= end)]

        for active_status in [0, 1]:
            subset = phase_data[phase_data['active'] == active_status]
            if len(subset) > 0:
                mean = subset['next_day_mood_pct'].mean()
                sem = subset['next_day_mood_pct'].sem()
                ci_low = mean - CI_MULTIPLIER * sem
                ci_high = mean + CI_MULTIPLIER * sem
                results[(phase, active_status)] = {
                    'mean': mean, 'ci_low': ci_low, 'ci_high': ci_high,
                    'start': start, 'end': end, 'n': len(subset)
                }
    return results

# Calculate phase means
women_results = calculate_phase_means(plot_df_women, 'cycle_phase', phase_ranges_women)
men_results = calculate_phase_means(plot_df_men, 'pseudo_phase', phase_ranges_men)

# Phase-specific effects for labels
phase_effects_women = {
    'early_luteal': {'effect': 3.50, 'p': 0.044, 'sig': '*'},
    'late_luteal': {'effect': 0.83, 'p': 0.636, 'sig': ''},
    'menstruation': {'effect': 3.49, 'p': 0.036, 'sig': '*'},
    'follicular': {'effect': 7.63, 'p': 0.0001, 'sig': '***'}
}

phase_effects_men = {
    'phase_1': {'effect': 1.67, 'p': 0.114, 'sig': ''},
    'phase_2': {'effect': 2.02, 'p': 0.022, 'sig': '*'},
    'phase_3': {'effect': 1.32, 'p': 0.159, 'sig': ''},
    'phase_4': {'effect': 1.35, 'p': 0.162, 'sig': ''}
}

def plot_phase_means(ax, results, phase_ranges, phase_colors, phase_labels, phase_effects, title, xlim):
    """Plot horizontal lines with CI bands for each phase."""

    # Add phase background colors
    for phase, (start, end) in phase_ranges.items():
        ax.axvspan(start - 0.5, end + 0.5, alpha=0.4, color=phase_colors[phase])

    # Plot horizontal lines and CI bands for each phase
    for phase, (start, end) in phase_ranges.items():
        # Inactive (red)
        if (phase, 0) in results:
            r = results[(phase, 0)]
            ax.fill_between([start - 0.5, end + 0.5], r['ci_low'], r['ci_high'],
                          color='#D62728', alpha=0.25, zorder=1)
            ax.hlines(r['mean'], start - 0.5, end + 0.5, colors='#D62728',
                     linewidth=3, zorder=2)

        # Active (green)
        if (phase, 1) in results:
            r = results[(phase, 1)]
            ax.fill_between([start - 0.5, end + 0.5], r['ci_low'], r['ci_high'],
                          color='#2CA02C', alpha=0.25, zorder=1)
            ax.hlines(r['mean'], start - 0.5, end + 0.5, colors='#2CA02C',
                     linewidth=3, zorder=2)

    # Add phase labels at top with effect sizes
    y_max = ax.get_ylim()[1]
    for phase, (start, end) in phase_ranges.items():
        mid = (start + end) / 2
        effect_info = phase_effects[phase]
        effect_str = f"{effect_info['effect']:+.1f}%{effect_info['sig']}"

        ax.text(mid, y_max * 0.95, phase_labels[phase],
                ha='center', va='top', fontsize=10, fontweight='bold')
        ax.text(mid, y_max * 0.82, effect_str,
                ha='center', va='top', fontsize=9, fontstyle='italic', color='#333333')

    # Reference line at 0
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlim(xlim)
    ax.set_ylabel('Next-Day Mood\n(% change from personal mean)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':')

# Create figure with two panels
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Women
ax1 = axes[0]
plot_phase_means(ax1, women_results, phase_ranges_women, phase_colors,
                 phase_labels_women, phase_effects_women,
                 'Women (n = 138)\nHeterogeneity: χ² = 22.52, p < 0.0001',
                 xlim=(-14.5, 13.5))
ax1.set_xlabel('Cycle Day', fontsize=11)

# Men
ax2 = axes[1]
plot_phase_means(ax2, men_results, phase_ranges_men, phase_colors,
                 phase_labels_men, phase_effects_men,
                 'Men (n = 966)\nHeterogeneity: χ² = 7.08, p = 0.07',
                 xlim=(-0.5, 27.5))
ax2.set_xlabel('Day (from first Monday of month)', fontsize=11)

# Match y-axis limits
y_min = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
ax1.set_ylim(y_min, y_max)
ax2.set_ylim(y_min, y_max)

# Re-add phase labels after setting y limits
for ax, results, phase_ranges, phase_labels, phase_effects in [
    (ax1, women_results, phase_ranges_women, phase_labels_women, phase_effects_women),
    (ax2, men_results, phase_ranges_men, phase_labels_men, phase_effects_men)
]:
    for phase, (start, end) in phase_ranges.items():
        mid = (start + end) / 2
        effect_info = phase_effects[phase]
        effect_str = f"{effect_info['effect']:+.1f}%{effect_info['sig']}"

        ax.text(mid, y_max * 0.95, phase_labels[phase],
                ha='center', va='top', fontsize=10, fontweight='bold')
        ax.text(mid, y_max * 0.82, effect_str,
                ha='center', va='top', fontsize=9, fontstyle='italic', color='#333333')

# Add legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#2CA02C', linewidth=3, label='Active (above median)'),
    Line2D([0], [0], color='#D62728', linewidth=3, label='Inactive (below median)'),
    Patch(facecolor='#2CA02C', alpha=0.25, label='90% CI (active)'),
    Patch(facecolor='#D62728', alpha=0.25, label='90% CI (inactive)')
]
fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10,
           bbox_to_anchor=(0.5, 0.02))

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig('phase_means_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('phase_means_comparison.pdf', bbox_inches='tight')
print("Saved: phase_means_comparison.png and phase_means_comparison.pdf")

plt.show()

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

# Filter to valid data
plot_df_women = df_women[['userid', 'cycle_day', 'active', 'next_day_mood_pct', 'cycle_phase']].dropna().copy()
plot_df_women['active'] = plot_df_women['active'].astype(int)
plot_df_women = plot_df_women[(plot_df_women['cycle_day'] >= -14) & (plot_df_women['cycle_day'] <= 13)].copy()

plot_df_men = df_men[['userid', 'cycle_day', 'active', 'next_day_mood_pct', 'pseudo_phase']].dropna().copy()
plot_df_men['active'] = plot_df_men['active'].astype(int)
plot_df_men = plot_df_men[(plot_df_men['cycle_day'] >= 0) & (plot_df_men['cycle_day'] <= 27)].copy()

print(f"Women: {plot_df_women['userid'].nunique()} users, {len(plot_df_women)} observations")
print(f"Men: {plot_df_men['userid'].nunique()} users, {len(plot_df_men)} observations")

# Target sample size (match women)
TARGET_N = plot_df_women['userid'].nunique()
N_SAMPLES = 5

print(f"\nSampling {TARGET_N} men, {N_SAMPLES} times...")

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

CI_MULTIPLIER = 1.645  # 90% CI

# ============================================
# Sample men and calculate effects
# ============================================
all_male_users = plot_df_men['userid'].unique()
print(f"Total male users available: {len(all_male_users)}")

# Store results from each sample
sample_results = []

for sample_i in range(N_SAMPLES):
    # Randomly sample TARGET_N men
    sampled_users = np.random.choice(all_male_users, size=TARGET_N, replace=False)
    sample_df = plot_df_men[plot_df_men['userid'].isin(sampled_users)].copy()

    print(f"  Sample {sample_i + 1}: {len(sampled_users)} users, {len(sample_df)} observations")

    # Calculate phase means for this sample
    sample_phase_results = {}
    for phase, (start, end) in phase_ranges_men.items():
        phase_data = sample_df[(sample_df['cycle_day'] >= start) & (sample_df['cycle_day'] <= end)]

        for active_status in [0, 1]:
            subset = phase_data[phase_data['active'] == active_status]
            if len(subset) > 0:
                mean = subset['next_day_mood_pct'].mean()
                std = subset['next_day_mood_pct'].std()
                n = len(subset)
                sem = std / np.sqrt(n)
                sample_phase_results[(phase, active_status)] = {
                    'mean': mean, 'sem': sem, 'n': n, 'start': start, 'end': end
                }

    sample_results.append(sample_phase_results)

# ============================================
# Average results across samples
# ============================================
print("\nAveraging across samples...")

averaged_results = {}
for phase, (start, end) in phase_ranges_men.items():
    for active_status in [0, 1]:
        means = []
        sems = []
        ns = []
        for sr in sample_results:
            if (phase, active_status) in sr:
                means.append(sr[(phase, active_status)]['mean'])
                sems.append(sr[(phase, active_status)]['sem'])
                ns.append(sr[(phase, active_status)]['n'])

        if means:
            avg_mean = np.mean(means)
            # Average SEM (approximation)
            avg_sem = np.mean(sems)
            avg_n = np.mean(ns)

            averaged_results[(phase, active_status)] = {
                'mean': avg_mean,
                'ci_low': avg_mean - CI_MULTIPLIER * avg_sem,
                'ci_high': avg_mean + CI_MULTIPLIER * avg_sem,
                'start': start,
                'end': end,
                'n': avg_n
            }

# Calculate phase effects (active - inactive) for matched men
print("\nPhase-specific effects (matched sample of men, averaged over 5 samples):")
print("-" * 60)
matched_phase_effects = {}
for phase in phase_ranges_men.keys():
    if (phase, 1) in averaged_results and (phase, 0) in averaged_results:
        active_mean = averaged_results[(phase, 1)]['mean']
        inactive_mean = averaged_results[(phase, 0)]['mean']
        effect = active_mean - inactive_mean

        # Approximate SE of difference
        active_sem = (averaged_results[(phase, 1)]['ci_high'] - averaged_results[(phase, 1)]['mean']) / CI_MULTIPLIER
        inactive_sem = (averaged_results[(phase, 0)]['ci_high'] - averaged_results[(phase, 0)]['mean']) / CI_MULTIPLIER
        se_diff = np.sqrt(active_sem**2 + inactive_sem**2)

        # Simple z-test for significance
        z = effect / se_diff if se_diff > 0 else 0
        p = 2 * (1 - 0.5 * (1 + np.sign(z) * (1 - np.exp(-0.5 * z**2))))  # Approximation

        sig = ''
        if abs(z) > 2.58: sig = '***'
        elif abs(z) > 1.96: sig = '**'
        elif abs(z) > 1.645: sig = '*'

        matched_phase_effects[phase] = {'effect': effect, 'sig': sig, 'p': p}
        print(f"  {phase_labels_men[phase]}: {effect:+.2f}% {sig}")

# ============================================
# Women's results (for comparison)
# ============================================
women_results = {}
for phase, (start, end) in phase_ranges_women.items():
    phase_data = plot_df_women[(plot_df_women['cycle_day'] >= start) & (plot_df_women['cycle_day'] <= end)]

    for active_status in [0, 1]:
        subset = phase_data[phase_data['active'] == active_status]
        if len(subset) > 0:
            mean = subset['next_day_mood_pct'].mean()
            sem = subset['next_day_mood_pct'].sem()
            women_results[(phase, active_status)] = {
                'mean': mean,
                'ci_low': mean - CI_MULTIPLIER * sem,
                'ci_high': mean + CI_MULTIPLIER * sem,
                'start': start,
                'end': end,
                'n': len(subset)
            }

phase_effects_women = {
    'early_luteal': {'effect': 3.50, 'sig': '*'},
    'late_luteal': {'effect': 0.83, 'sig': ''},
    'menstruation': {'effect': 3.49, 'sig': '*'},
    'follicular': {'effect': 7.63, 'sig': '***'}
}

# ============================================
# Create comparison plot
# ============================================
def plot_phase_means(ax, results, phase_ranges, phase_colors, phase_labels, phase_effects, title, xlim):
    """Plot horizontal lines with CI bands for each phase."""

    # Add phase background colors
    for phase, (start, end) in phase_ranges.items():
        ax.axvspan(start - 0.5, end + 0.5, alpha=0.4, color=phase_colors[phase])

    # Plot horizontal lines and CI bands
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

    # Reference line at 0
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlim(xlim)
    ax.set_ylabel('Next-Day Mood\n(% change from personal mean)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':')

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Women
ax1 = axes[0]
plot_phase_means(ax1, women_results, phase_ranges_women, phase_colors,
                 phase_labels_women, phase_effects_women,
                 f'Women (n = {TARGET_N})\nHeterogeneity: χ² = 22.52, p < 0.0001',
                 xlim=(-14.5, 13.5))
ax1.set_xlabel('Cycle Day', fontsize=11)

# Men (matched sample)
ax2 = axes[1]
plot_phase_means(ax2, averaged_results, phase_ranges_men, phase_colors,
                 phase_labels_men, matched_phase_effects,
                 f'Men (n = {TARGET_N}, averaged over {N_SAMPLES} samples)\nMatched sample size comparison',
                 xlim=(-0.5, 27.5))
ax2.set_xlabel('Day (from first Monday of month)', fontsize=11)

# Match y-axis limits
y_min = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
ax1.set_ylim(y_min, y_max)
ax2.set_ylim(y_min, y_max)

# Add phase labels
for ax, phase_ranges, phase_labels, phase_effects in [
    (ax1, phase_ranges_women, phase_labels_women, phase_effects_women),
    (ax2, phase_ranges_men, phase_labels_men, matched_phase_effects)
]:
    for phase, (start, end) in phase_ranges.items():
        mid = (start + end) / 2
        effect_info = phase_effects[phase]
        effect_str = f"{effect_info['effect']:+.1f}%{effect_info['sig']}"

        ax.text(mid, y_max * 0.95, phase_labels[phase],
                ha='center', va='top', fontsize=10, fontweight='bold')
        ax.text(mid, y_max * 0.82, effect_str,
                ha='center', va='top', fontsize=9, fontstyle='italic', color='#333333')

# Legend
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
plt.savefig('phase_means_matched_sample.png', dpi=150, bbox_inches='tight')
plt.savefig('phase_means_matched_sample.pdf', bbox_inches='tight')
print("\nSaved: phase_means_matched_sample.png and phase_means_matched_sample.pdf")

plt.show()

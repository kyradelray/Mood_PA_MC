import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Phase-specific effects from analysis (median split)
# Women
women_effects = {
    'Menstruation': {'effect': 3.5, 'ci_low': 0.2, 'ci_high': 6.8, 'p': 0.036},
    'Follicular': {'effect': 7.6, 'ci_low': 4.6, 'ci_high': 10.7, 'p': 0.001},
    'Early luteal': {'effect': 3.5, 'ci_low': 0.1, 'ci_high': 6.9, 'p': 0.044},
    'Late luteal': {'effect': 0.8, 'ci_low': -2.6, 'ci_high': 4.3, 'p': 0.64}
}

# Men
men_effects = {
    'Phase 1': {'effect': 1.7, 'ci_low': -0.4, 'ci_high': 3.7, 'p': 0.11},
    'Phase 2': {'effect': 2.0, 'ci_low': 0.3, 'ci_high': 3.8, 'p': 0.02},
    'Phase 3': {'effect': 1.3, 'ci_low': -0.5, 'ci_high': 3.2, 'p': 0.16},
    'Phase 4': {'effect': 1.4, 'ci_low': -0.5, 'ci_high': 3.3, 'p': 0.16}
}

# Phase colors
phase_colors = {
    'Menstruation': '#E57373',  # Red
    'Follicular': '#81C784',    # Green
    'Early luteal': '#BA68C8',  # Purple
    'Late luteal': '#FFB74D'    # Orange/tan
}

men_colors = {
    'Phase 1': '#E57373',
    'Phase 2': '#81C784',
    'Phase 3': '#BA68C8',
    'Phase 4': '#FFB74D'
}

# Create figure with two panels
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# --- WOMEN ---
ax1 = axes[0]
phases_w = list(women_effects.keys())
effects_w = [women_effects[p]['effect'] for p in phases_w]
ci_low_w = [women_effects[p]['ci_low'] for p in phases_w]
ci_high_w = [women_effects[p]['ci_high'] for p in phases_w]
errors_w = [[e - l for e, l in zip(effects_w, ci_low_w)],
            [h - e for e, h in zip(effects_w, ci_high_w)]]
colors_w = [phase_colors[p] for p in phases_w]

x_w = np.arange(len(phases_w))
bars_w = ax1.bar(x_w, effects_w, color=colors_w, edgecolor='black', linewidth=1.2, width=0.7)
ax1.errorbar(x_w, effects_w, yerr=errors_w, fmt='none', color='black', capsize=6, capthick=2, linewidth=2)

# Add significance markers
for i, phase in enumerate(phases_w):
    p_val = women_effects[phase]['p']
    y_pos = ci_high_w[i] + 0.5
    if p_val < 0.001:
        ax1.text(i, y_pos, '***', ha='center', va='bottom', fontsize=14, fontweight='bold')
    elif p_val < 0.01:
        ax1.text(i, y_pos, '**', ha='center', va='bottom', fontsize=14, fontweight='bold')
    elif p_val < 0.05:
        ax1.text(i, y_pos, '*', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax1.set_xticks(x_w)
ax1.set_xticklabels(phases_w, fontsize=11)
ax1.set_ylabel('Effect of Activity on Next-Day Mood\n(% change from personal mean)', fontsize=12)
ax1.set_title('Women (n = 138)\nHeterogeneity: χ² = 22.52, p < 0.0001', fontsize=13, fontweight='bold')
ax1.set_ylim(-5, 14)
ax1.grid(axis='y', alpha=0.3, linestyle=':')

# --- MEN ---
ax2 = axes[1]
phases_m = list(men_effects.keys())
effects_m = [men_effects[p]['effect'] for p in phases_m]
ci_low_m = [men_effects[p]['ci_low'] for p in phases_m]
ci_high_m = [men_effects[p]['ci_high'] for p in phases_m]
errors_m = [[e - l for e, l in zip(effects_m, ci_low_m)],
            [h - e for e, h in zip(effects_m, ci_high_m)]]
colors_m = [men_colors[p] for p in phases_m]

x_m = np.arange(len(phases_m))
bars_m = ax2.bar(x_m, effects_m, color=colors_m, edgecolor='black', linewidth=1.2, width=0.7)
ax2.errorbar(x_m, effects_m, yerr=errors_m, fmt='none', color='black', capsize=6, capthick=2, linewidth=2)

# Add significance markers for men
for i, phase in enumerate(phases_m):
    p_val = men_effects[phase]['p']
    y_pos = ci_high_m[i] + 0.5
    if p_val < 0.001:
        ax2.text(i, y_pos, '***', ha='center', va='bottom', fontsize=14, fontweight='bold')
    elif p_val < 0.01:
        ax2.text(i, y_pos, '**', ha='center', va='bottom', fontsize=14, fontweight='bold')
    elif p_val < 0.05:
        ax2.text(i, y_pos, '*', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax2.set_xticks(x_m)
ax2.set_xticklabels(phases_m, fontsize=11)
ax2.set_title('Men (n = 966)\nHeterogeneity: χ² = 7.08, p = 0.07', fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3, linestyle=':')

# Add note
fig.text(0.5, 0.02, '* p < 0.05, ** p < 0.01, *** p < 0.001. Error bars show 95% confidence intervals.',
         ha='center', fontsize=10, style='italic')

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('phase_effects_bars.png', dpi=150, bbox_inches='tight')
plt.savefig('phase_effects_bars.pdf', bbox_inches='tight')
print("Saved: phase_effects_bars.png and phase_effects_bars.pdf")

plt.show()

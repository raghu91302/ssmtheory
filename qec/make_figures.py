"""Generate the central threshold-scaling figure."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data from simulations
p_vals = np.array([0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02])
L4_per_lq = np.array([1.5e-5, 8.3e-5, 2.2e-4, 5.8e-4, 1.1e-3, 2.0e-3, 3.6e-3, 5.0e-3])
L6_per_lq = np.array([2.0e-5, 8.2e-5, 1.7e-4, 4.2e-4, 7.5e-4, 1.2e-3, 1.9e-3, 2.2e-3])
L8_per_lq = np.array([1.7e-5, 7.2e-5, 1.5e-4, 3.7e-4, 5.6e-4, 7.9e-4, 9.6e-4, 9.8e-4])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: log-log per-LQ failure rate vs p
ax = axes[0]
ax.loglog(p_vals, L4_per_lq, 'o-', label=r'$L=4$ ($k=130$)', linewidth=1.6, markersize=7, color='#1f77b4')
ax.loglog(p_vals, L6_per_lq, 's-', label=r'$L=6$ ($k=434$)', linewidth=1.6, markersize=7, color='#ff7f0e')
ax.loglog(p_vals, L8_per_lq, '^-', label=r'$L=8$ ($k=1026$)', linewidth=1.6, markersize=7, color='#2ca02c')
p_ref = np.linspace(0.001, 0.02, 50)
ax.loglog(p_ref, 23.2 * p_ref**2, 'k--', alpha=0.5, linewidth=1, label=r'$\propto p^2$ guide')
ax.set_xlabel(r'physical error rate $p$', fontsize=11)
ax.set_ylabel(r'per-logical-qubit failure rate $f_{LQ}$', fontsize=11)
ax.set_title('(a) Per-logical-qubit failure rate', fontsize=11)
ax.grid(True, which='both', alpha=0.3)
ax.legend(loc='upper left', fontsize=10)

# Right: ratio plot — improvement factor vs L
ax = axes[1]
ratio_L8_L4 = L8_per_lq / L4_per_lq
ratio_L6_L4 = L6_per_lq / L4_per_lq
ax.plot(p_vals, np.ones_like(p_vals), 'k--', alpha=0.5, label=r'baseline ($L=4$)', linewidth=1)
ax.plot(p_vals, ratio_L6_L4, 's-', color='#ff7f0e', label=r'$L=6$ / $L=4$', linewidth=1.6, markersize=7)
ax.plot(p_vals, ratio_L8_L4, '^-', color='#2ca02c', label=r'$L=8$ / $L=4$', linewidth=1.6, markersize=7)
ax.set_xlabel(r'physical error rate $p$', fontsize=11)
ax.set_ylabel(r'per-LQ failure rate ratio', fontsize=11)
ax.set_title('(b) Improvement factor with system size', fontsize=11)
ax.grid(True, alpha=0.3)
ax.legend(loc='lower left', fontsize=10)
ax.set_ylim(0, 1.5)

plt.tight_layout()
plt.savefig('/home/claude/lorentz_paper/fig_threshold.pdf', bbox_inches='tight', dpi=150)
plt.savefig('/home/claude/lorentz_paper/fig_threshold.png', bbox_inches='tight', dpi=150)
print("Saved fig_threshold.pdf")

# Now the R-distribution figure
import json
fig, ax = plt.subplots(figsize=(7.5, 4.5))
# Open-boundary R distribution data (from build_open.py)
R_dist = {
    4: {14: 12, 15: 12, 18: 24, 20: 12, 21: 12, 22: 6, 25: 24, 30: 6},
    6: {14: 12, 15: 12, 18: 72, 20: 24, 21: 24, 22: 54, 25: 144, 30: 108},
    8: {14: 12, 15: 12, 18: 120, 20: 36, 21: 36, 22: 150, 25: 360, 30: 450}
}
totals = {4: 108, 6: 450, 8: 1176}
all_R = sorted(set().union(*[d.keys() for d in R_dist.values()]))
width = 0.27
x_pos = np.arange(len(all_R))
colors = {4: '#1f77b4', 6: '#ff7f0e', 8: '#2ca02c'}
for i, L in enumerate([4, 6, 8]):
    fracs = [R_dist[L].get(r, 0) / totals[L] * 100 for r in all_R]
    ax.bar(x_pos + (i - 1) * width, fracs, width, label=f'$L = {L}$', color=colors[L])
ax.set_xticks(x_pos)
ax.set_xticklabels([str(r) for r in all_R])
ax.set_xlabel(r'informative-stabilizer count $R$ per edge', fontsize=11)
ax.set_ylabel(r'fraction of edges (%)', fontsize=11)
ax.set_title(r'Open-boundary FCC: $R$ distribution ($R_{bulk}=30$ at full interior)', fontsize=11)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, axis='y', alpha=0.3)
ax.axvline(x=x_pos[-1], color='red', linestyle=':', alpha=0.5)
ax.text(x_pos[-1] + 0.05, 38, r'bulk-saturated', color='red', fontsize=9, style='italic')

plt.tight_layout()
plt.savefig('/home/claude/lorentz_paper/fig_R_distribution.pdf', bbox_inches='tight', dpi=150)
plt.savefig('/home/claude/lorentz_paper/fig_R_distribution.png', bbox_inches='tight', dpi=150)
print("Saved fig_R_distribution.pdf")

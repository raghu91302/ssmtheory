import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json

# Load data
with open('rex_sweep_data.json') as f:
    data = json.load(f)

rex = [d[0] for d in data]
nodes = [d[1] for d in data]
kmax = [d[2] for d in data]
kmean = [d[3] for d in data]

# Critical values
rex_crit = 1/np.sqrt(3)  # 0.5774

fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

# ---- Background shading for three regimes ----
# Lattice breaking zone (Rex < 1/√3)
ax1.axvspan(0.49, rex_crit, alpha=0.12, color='#e74c3c', zorder=0)
# Valid FCC zone (1/√3 < Rex < 1.0)
ax1.axvspan(rex_crit, 0.995, alpha=0.08, color='#27ae60', zorder=0)
# Jamming zone (Rex >= 1.0)
ax1.axvspan(0.995, 1.03, alpha=0.12, color='#7f8c8d', zorder=0)

# ---- Plot K_max ----
ax1.plot(rex, kmax, 'o-', color='#2c3e50', markersize=7, linewidth=2.2,
         markerfacecolor='#3498db', markeredgecolor='#2c3e50', markeredgewidth=1.2,
         label=r'$K_{\mathrm{max}}$', zorder=5)

# ---- Plot K_mean on secondary axis ----
ax2 = ax1.twinx()
ax2.plot(rex, kmean, 's--', color='#8e44ad', markersize=5, linewidth=1.5,
         markerfacecolor='#9b59b6', markeredgecolor='#8e44ad', markeredgewidth=1,
         alpha=0.85, label=r'$\bar{K}$', zorder=4)

# ---- Critical line: 1/√3 ----
ax1.axvline(x=rex_crit, color='#e74c3c', linestyle='-', linewidth=2.5, zorder=6)
ax1.annotate(r'$R_{\mathrm{ex}} = \frac{1}{\sqrt{3}} \approx 0.577$',
             xy=(rex_crit, 12), xytext=(0.66, 21.5),
             fontsize=13, fontweight='bold', color='#c0392b',
             arrowprops=dict(arrowstyle='->', color='#c0392b', lw=2),
             ha='center', zorder=10)

# ---- K=12 reference line ----
ax1.axhline(y=12, color='#27ae60', linestyle=':', linewidth=1.5, alpha=0.7, zorder=3)
ax1.text(0.88, 12.6, r'$K = 12$ (FCC saturation)', fontsize=10,
         color='#27ae60', ha='center', style='italic')

# ---- Jamming annotation ----
ax1.annotate('Total\nJamming',
             xy=(1.0, 2), xytext=(0.93, 6),
             fontsize=11, fontweight='bold', color='#7f8c8d',
             arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.5),
             ha='center', zorder=10)

# ---- Zone labels ----
ax1.text(0.535, 25.5, 'LATTICE BREAKING $(K > 12)$',
         fontsize=10, ha='center', color='#c0392b', fontweight='bold',
         style='italic', zorder=10)

ax1.text(0.79, 15.5, 'STABLE FCC TOLERANCE BAND',
         fontsize=10, ha='center', color='#1e8449', fontweight='bold',
         style='italic', zorder=10)
ax1.annotate('', xy=(0.59, 14.8), xytext=(0.99, 14.8),
             arrowprops=dict(arrowstyle='<->', color='#1e8449', lw=1.8), zorder=10)

# ---- Axes formatting ----
ax1.set_xlabel(r'Exclusion Radius  $R_{\mathrm{ex}}$  (units of $L$)', fontsize=13, labelpad=8)
ax1.set_ylabel(r'Maximum Coordination Number  $K_{\mathrm{max}}$', fontsize=13, color='#2c3e50', labelpad=8)
ax2.set_ylabel(r'Mean Coordination  $\bar{K}$', fontsize=13, color='#8e44ad', labelpad=8)

ax1.set_xlim(0.49, 1.03)
ax1.set_ylim(0, 27)
ax2.set_ylim(0, 12)

ax1.set_xticks(np.arange(0.50, 1.05, 0.05))
ax1.tick_params(axis='both', labelsize=11)
ax2.tick_params(axis='y', labelsize=11, colors='#8e44ad')
ax2.spines['right'].set_color('#8e44ad')

# ---- Title ----
ax1.set_title(r'Holographic Lattice Sensitivity Sweep  ($N = 500$,  Lift $= 5\%$,  Seed $= 42$)',
              fontsize=14, fontweight='bold', pad=12)

# ---- Legend ----
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
           fontsize=11, framealpha=0.9, edgecolor='#bdc3c7')

# ---- Grid ----
ax1.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig('rex_sensitivity_sweep.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('rex_sensitivity_sweep.pdf', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved: rex_sensitivity_sweep.png and .pdf")

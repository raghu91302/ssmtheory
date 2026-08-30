"""Figures for: Black holes as incomplete crystallization."""
import numpy as np, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fig 1: one defect spectrum across scales (log mass in grams)
fig,ax=plt.subplots(figsize=(7.2,2.6))
ax.set_xlim(-24,42); ax.set_ylim(0,1); ax.set_yticks([])
ax.axvspan(-24.5,-22,color='0.75',alpha=0.9); ax.text(-23.2,0.55,'particles\n(paper 2)',ha='center',fontsize=7.5)
ax.axvspan(-22,16.5,color='0.9',alpha=0.9); ax.text(-3,0.55,'uncrystallized regions below the cutoff:\nevaporated on the epoch map (companion, §8)',ha='center',fontsize=7.5)
ax.axvspan(16.5,42,color='#cfe0f5',alpha=0.9); ax.text(29,0.62,'uncrystallized regions above the cutoff:\nsurvive; the primordial black-hole population',ha='center',fontsize=7.5,color='navy')
ax.axvline(16.5,color='k',lw=1.4); ax.text(16.8,0.12,r'$M_{\rm cut}\simeq10^{16.5}$ g',fontsize=7.5)
ax.axvline(math.log10(1e5*1.989e33),color='r',lw=1.2,ls='--'); ax.text(math.log10(1e5*1.989e33)+0.4,0.12,r'MoM-BH*-1, $\sim10^5 M_\odot$',fontsize=7.5,color='r')
ax.axvline(math.log10(1.673e-24),color='k',lw=0.8,ls=':'); ax.text(-23.6,0.1,'proton',fontsize=7)
ax.set_xlabel(r'$\log_{10}$ mass [g]   —   one incompleteness, forty orders of magnitude')
fig.tight_layout(); fig.savefig("figS1_spectrum.pdf")

print("figure written")

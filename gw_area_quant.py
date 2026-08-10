"""Numbers and figures for: Horizon Area Quantization from Bond-Counted Entropy.
alpha = 4 ln 2 from S = N k ln2 = A/(4 lP^2)."""
import numpy as np, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

G,c,hbar,Msun = 6.674e-11, 2.998e8, 1.055e-34, 1.989e30
alpha = 4*math.log(2)
print(f"alpha = 4 ln2 = {alpha:.4f}   (Bekenstein-Mukhanov value, here derived)")
print(f"area quantum dA = {alpha:.4f} lP^2")

def f1(M): return alpha/(32*math.pi)*c**3/(G*M*Msun)/(2*math.pi)   # n=1 line spacing [Hz]
def f220(M): return 0.3737*c**3/(G*M*Msun)/(2*math.pi)             # fundamental QNM, spin 0

for M in (10,30,65):
    print(f"M={M:3d} Msun: line spacing f1 = {f1(M):7.1f} Hz | f220 = {f220(M):7.1f} Hz | ratio = {100*f1(M)/f220(M):.2f}%")

# ---- Figure 1: ladder spacing vs mass, with detector bands ----
M = np.logspace(0.3, 3, 200)
fig, ax = plt.subplots(figsize=(5.2,3.6))
ax.loglog(M, [f1(m) for m in M], 'b-', lw=1.8, label=r'line spacing $f_1=\alpha c^3/(64\pi^2 GM)$,  $\alpha=4\ln2$')
ax.loglog(M, [f220(m) for m in M], 'k--', lw=1.2, label=r'fundamental ringdown $f_{220}$ (spin 0)')
ax.axhspan(20, 2000, color='orange', alpha=0.15, label='LIGO-Virgo band')
ax.axhspan(3, 20, color='green', alpha=0.12, label='Einstein Telescope extension')
ax.set_xlabel(r'remnant mass $M\ [M_\odot]$'); ax.set_ylabel('frequency [Hz]')
ax.set_xlim(2,1000); ax.set_ylim(1,3e4)
ax.legend(fontsize=7, loc='upper right'); ax.grid(alpha=0.25, which='both')
fig.tight_layout(); fig.savefig("figA_ladder.pdf")

# ---- Figure 2: discriminating alpha at one remnant ----
M0=30.0
fig, ax = plt.subplots(figsize=(5.2,3.6))
f0=f220(M0)
fr=np.linspace(0, 2.2*f0, 1000)
Q=2.2
lor = 1/(1+((fr-f0)/(f0/(2*Q)))**2)
ax.plot(fr, lor, 'k-', lw=1.2, label=r'classical $\ell{=}2$ ringdown resonance ($Q\simeq2.2$)')
for a,cn,lab in [(4*math.log(2),'b',r'$\alpha=4\ln2$ (this work, derived)'),
                 (4*math.log(3),'r',r'$\alpha=4\ln3$ (Hod)'),
                 (8*math.pi,'g',r'$\alpha=8\pi$ (Bekenstein 1974)')]:
    fa = a/(32*math.pi)*c**3/(G*M0*Msun)/(2*math.pi)
    lines = np.arange(fa, 2.2*f0, fa)
    ax.vlines(lines, 0, {'b':0.95,'r':0.62,'g':0.32}[cn], colors=cn, lw=1.0, alpha=0.8)
    ax.plot([],[],color=cn,label=lab)
ax.set_xlabel(r'frequency [Hz]  (remnant $M=30\,M_\odot$, spin 0)')
ax.set_ylabel('arbitrary amplitude')
ax.set_ylim(0,1.05); ax.legend(fontsize=7); ax.grid(alpha=0.2)
fig.tight_layout(); fig.savefig("figB_alpha.pdf")
print("figures written")

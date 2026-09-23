"""Order-of-magnitude clustering and pair statistics for Poisson-seeded LRD hosts.
Seeds uncorrelated with the primordial density field (separate-universe argument
for Gaussian adiabatic initial conditions) populate halos in proportion to mass.
LRDs are seeds visible in halos above a cocoon threshold M_min.
"""
import numpy as np
from colossus.cosmology import cosmology
from colossus.lss import mass_function, bias
from colossus.cosmology import cosmology as cc
cosmo = cosmology.setCosmology('planck18')
h = cosmo.h
z = 5.5
rho_m = cosmo.rho_m(0)*1e9          # M_sun h^2 / Mpc^3 comoving  (colossus: Msun h^2 / kpc^3)
lnM = np.linspace(np.log(1e7), np.log(1e14), 4000)   # M in Msun/h
M = np.exp(lnM)
# dn/dlnM in (Mpc/h)^-3
dndlnM = mass_function.massFunction(M, z, mdef='200m', model='tinker08', q_out='dndlnM')
b = bias.haloBias(M, z=z, mdef='200m', model='tinker10')

def stats(Mmin):
    s = M >= Mmin
    wN = dndlnM[s]; wM = dndlnM[s]*M[s]
    b_num = np.trapezoid(b[s]*wN, lnM[s])/np.trapezoid(wN, lnM[s])          # step occupation
    b_mass = np.trapezoid(b[s]*wM, lnM[s])/np.trapezoid(wM, lnM[s])         # occupation ∝ M
    F = np.trapezoid(wM, lnM[s])/rho_m                                  # mass fraction above Mmin
    M2 = np.trapezoid(wM*M[s], lnM[s])/np.trapezoid(wM, lnM[s])             # <M> mass-weighted (Msun/h)
    return b_num, b_mass, F, M2

# LRD comoving density from Tanaka+ sample: N=829 in 0.54 deg^2 over 5<z<8
Dc5 = cosmo.comovingDistance(0.0, 5.0)/h; Dc8 = cosmo.comovingDistance(0.0, 8.0)/h   # Mpc
V = 4/3*np.pi*(Dc8**3-Dc5**3)*(0.54/41252.96)
nLRD = 829/V                                                         # Mpc^-3
print(f"LRD comoving density (Tanaka+ sample): n = {nLRD:.2e} Mpc^-3  (volume {V:.2e} Mpc^3)")
rho_dm = cosmo.Om0*(1-cosmo.Ob0/cosmo.Om0)*cosmo.rho_c(0)*1e9*h**2    # Msun/Mpc^3
for Mseed in [1e5,1e6]:
    print(f"  seed mass fraction of DM, M_seed={Mseed:.0e}: f = {nLRD*Mseed/rho_dm:.1e} / d   (d = visible duty fraction)")
print()
print(f"{'log Mmin[Msun]':>14} {'b_step':>7} {'b_Poisson':>9} {'F(>Mmin)':>9} {'f_pair':>8}")
nLRD_h = nLRD/h**3                                                   # (Mpc/h)^-3
for lm in [10.0,10.5,11.0,11.5]:
    Mmin = 10**lm*h
    bn,bm,F,M2 = stats(Mmin)
    fpair = nLRD_h*M2/(rho_m*F)       # expected companions per LRD in same halo (duty-cycle independent)
    print(f"{lm:14.1f} {bn:7.2f} {bm:9.2f} {F:9.2e} {fpair:8.2e}")
print("\nobserved: halo mass log M ~ 11-11.5 (Arita+25, Lin+26); dual fraction 0.5 (+0.4,-0.2)% (Tanaka+)")

# ---------------------------------------------------------------- figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
lms = np.linspace(9.5, 11.6, 60)
bs, fs = [], []
for lm in lms:
    bn, bm, F, M2 = stats(10**lm*h)
    bs.append(bm); fs.append(nLRD_h*M2/(rho_m*F))
bs = np.array(bs); fs = np.array(fs)*100.0     # per cent

fig, ax = plt.subplots(figsize=(5.4, 4.0))
ax.plot(bs, fs, color='k', lw=2, zorder=3, label=r'Poisson seeding, occupation $\propto M$')
for lm, mk in [(10.0,'o'), (10.5,'s'), (11.0,'^')]:
    bn, bm, F, M2 = stats(10**lm*h)
    ax.plot(bm, nLRD_h*M2/(rho_m*F)*100, mk, color='k', ms=6, zorder=4)
    ax.annotate(rf'$10^{{{lm:.1f}}}$', (bm, nLRD_h*M2/(rho_m*F)*100),
                textcoords='offset points', xytext=(8,-10), fontsize=8)
ax.axhspan(0.3, 0.9, color='tab:red', alpha=0.18, zorder=1)
ax.axhline(0.5, color='tab:red', lw=1.2, ls='--', zorder=2)
ax.axvspan(4.7, 7.6, color='tab:blue', alpha=0.15, zorder=1)
ax.text(3.08, 1.25, 'observed dual fraction', color='tab:red', fontsize=8)
ax.text(4.85, 40, 'measured host bias', color='tab:blue', fontsize=8)
ax.text(3.6, 4.5, r'labels: $M_{\min}/M_\odot$', fontsize=8, color='0.35')
ax.set_yscale('log'); ax.set_xlabel(r'large-scale bias $b$ at $z=5.5$')
ax.set_ylabel(r'close companions per little red dot [\%]' if False else 'close companions per little red dot [%]')
ax.set_xlim(3.0, 8.0); ax.set_ylim(0.1, 100)
ax.legend(fontsize=8, loc='upper left', frameon=False)
ax.set_title('One threshold cannot fit both', fontsize=10)
fig.tight_layout(); fig.savefig('figS2_pairbias.pdf')
print("\nfigure written: figS2_pairbias.pdf")
print("curve endpoints: b=%.2f f=%.2f%%  ...  b=%.2f f=%.1f%%" % (bs[0], fs[0], bs[-1], fs[-1]))

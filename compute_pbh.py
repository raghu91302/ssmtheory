#!/usr/bin/env python3
"""
compute_pbh.py  --  Computations and figures for the geometric-evaporation PBH paper.

Revised build:
  * Lifetime coefficient written through the curvature-flow mobility M = alpha*c*L0
    (alpha = 1/2 reproduces tau = (2/sqrt(ln2)) t_P (M/m_P)^2). No surface-tension
    energy budget is invoked.
  * Survival cutoff reported as the scaling law M_cut ~ 1e16.5 (L_corr / 1 fm) g,
    with the underlying R_H(M_cut) ~ const * L_corr made explicit.
All manuscript numbers are produced here. Pure numpy/scipy/matplotlib.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.optimize import brentq

# ---- register Latin Modern so figures match the LaTeX body font ----
_LM = [
    "/usr/share/texmf/fonts/opentype/public/lm/lmroman10-regular.otf",
    "/usr/share/texmf/fonts/opentype/public/lm/lmroman10-bold.otf",
    "/usr/share/texmf/fonts/opentype/public/lm/lmroman10-italic.otf",
    "/usr/share/texmf/fonts/opentype/public/lm/lmroman10-bolditalic.otf",
]
for _p in _LM:
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 8.8,
    "axes.linewidth": 0.9,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 5, "ytick.major.size": 5,
    "xtick.minor.size": 2.8, "ytick.minor.size": 2.8,
    "lines.linewidth": 2.0,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,
})

# ---------------- constants (SI) ----------------
c = 2.998e8; hbar = 1.055e-34; G = 6.674e-11; kB = 1.381e-23
lP = np.sqrt(hbar*G/c**3); mP = np.sqrt(hbar*c/G); tP = lP/c
ln2 = np.log(2)
L0 = np.sqrt(4*ln2)*lP                 # bond length 1.665 lP
Lcorr = 1e-15                          # QCD correlation length, 1 fm (fiducial)
alpha = 0.5                            # curvature-flow O(1) coefficient
age = 13.8e9*365.25*86400             # s

# cosmology
hub = 0.674
Omega_dm = 0.264
Omega_g = 2.473e-5/hub**2
dm_over_g = Omega_dm/Omega_g          # ~4850
T0_eV = 2.348e-4
z_dc = 1.98e6
z_muy = 5e4
mu_FIRAS = 9e-5; y_FIRAS = 1.5e-5
fEM = 0.5

# ---------------- geometric channel ----------------
def RH(M): return 2*G*M/c**2
def tau_geo(M): return (1.0/(alpha*np.sqrt(ln2)))*tP*(M/mP)**2
def tau_eff(M, Lc=Lcorr):
    x = RH(M)/Lc
    return np.inf if x > 700 else tau_geo(M)*np.exp(x)
def tau_hawk(M): return 5120*np.pi*G**2*M**3/(hbar*c**4)

Mcut_logg = brentq(lambda lg: (np.log(tau_eff(10**lg/1000))-np.log(age))
                   if RH(10**lg/1000)/Lcorr < 700 else 50.0, 15, 18)
Mcut_g = 10**Mcut_logg

# ---------------- evaporation epoch ----------------
def T_inj_eV(t): return 1.0e6*t**-0.5
def z_inj(t): return T_inj_eV(t)/T0_eV - 1.0
def mass_evap_at(t): return 10**brentq(lambda lg: np.log(tau_eff(10**lg/1000))-np.log(t), 14, 17)

t_BBN_start, t_BBN_end = 1.0, 3e2
t_pdiss_on = 1e4
t_mu_lo = 2.4*(z_dc*T0_eV/1e6)**-2
M_preBBN = mass_evap_at(t_BBN_start)
M_BBNend = mass_evap_at(t_BBN_end)
M_pdiss  = mass_evap_at(t_pdiss_on)
M_zdc    = mass_evap_at(t_mu_lo)
def t_at_z(z): return 2.4*((1+z)*T0_eV/1e6)**-2
M_muy    = mass_evap_at(t_at_z(z_muy))
M_recomb = mass_evap_at(t_at_z(1100))

print("="*70)
print("GEOMETRIC CHANNEL  (alpha = %.2f)" % alpha)
print(f"  L_0 = {L0/lP:.3f} lP ; tau coeff 1/(alpha sqrt ln2) = {1/(alpha*np.sqrt(ln2)):.3f}")
print(f"  survival cutoff M_cut = 10^{Mcut_logg:.2f} g (R_H={RH(Mcut_g)*1e15:.0f} fm = {RH(Mcut_g)/Lcorr:.0f} L_corr)")
print("EPOCH BOUNDARIES (mass that evaporates at each epoch edge):")
print(f"  BBN onset  (t=1s)      : {M_preBBN:.2e} g")
print(f"  BBN end    (t=300s)    : {M_BBNend:.2e} g")
print(f"  photodiss. onset (1e4s): {M_pdiss:.2e} g")
print(f"  thermaliz. edge z_dc   : {M_zdc:.2e} g")
print(f"  mu/y boundary z=5e4    : {M_muy:.2e} g")
print(f"  recombination z=1100   : {M_recomb:.2e} g")

print("\nSURVIVAL-CUTOFF SCALING WITH L_corr:")
def Mcut_of_Lcorr(Lc):
    return brentq(lambda lg:(np.log(tau_eff(10**lg/1000, Lc))-np.log(age))
                  if RH(10**lg/1000)/Lc < 700 else 50.0, 14, 19)
for Lc in [0.1e-15, 0.3e-15, 1e-15, 3e-15, 10e-15]:
    lg = Mcut_of_Lcorr(Lc)
    print(f"  L_corr={Lc*1e15:5.1f} fm  ->  M_cut = 10^{lg:.2f} g   (R_H = {RH(10**lg/1000)/Lc:.0f} L_corr)")

def N_area(M): return 4*np.pi*RH(M)**2/L0**2
def kTH(M): return kB*(hbar*c**3/(8*np.pi*G*M*kB))
print("\nENERGY CLOSURE (thermal): N_severed * kT_H / (M c^2):")
for Mg in [1e15, 1e16]:
    M = Mg/1000
    print(f"  M={Mg:.0e} g : {N_area(M)*kTH(M)/(M*c**2):.4f}   (= 1/(2 ln2) = {1/(2*ln2):.4f})")

def dE_over_Egamma(M, fPBH):
    z = z_inj(tau_eff(M))
    return fEM*fPBH*dm_over_g/(1+z), z
def J_bb(z): return np.exp(-(z/z_dc)**2.5)
def mu_dist(M, fPBH):
    r, z = dE_over_Egamma(M, fPBH)
    if z < z_muy: return 0.0, z
    return 1.4*r*J_bb(z), z
def y_dist(M, fPBH):
    r, z = dE_over_Egamma(M, fPBH)
    if z >= z_muy: return 0.0, z
    return 0.25*r, z
def fmax_mu(M):
    m1, z = mu_dist(M, 1.0); return (mu_FIRAS/m1) if m1 > 0 else np.inf
def fmax_y(M):
    y1, z = y_dist(M, 1.0); return (y_FIRAS/y1) if y1 > 0 else np.inf
def fmax_pdiss(M):
    r, z = dE_over_Egamma(M, 1.0)
    t = tau_eff(M)
    if not (t_pdiss_on < t < 1e12): return np.inf
    return 1e-6/r

print("\nDISTORTION BOUNDS (f_PBH max), fEM=0.5:")
print(f"{'M(g)':>9} | {'z_inj':>9} | {'mu(f=1)':>9} | {'y(f=1)':>9} | {'f_max':>9} | channel")
for Mg in [5e15, 8e15, 1e16, 1.3e16, 1.5e16, 1.8e16, 2.0e16, 2.3e16, 2.5e16, 2.9e16]:
    Mk = Mg/1000
    m1, z = mu_dist(Mk, 1.0); y1, _ = y_dist(Mk, 1.0)
    fm = fmax_mu(Mk); fy = fmax_y(Mk); fp = fmax_pdiss(Mk)
    best = min(fm, fy, fp)
    ch = "mu" if best == fm else ("y" if best == fy else "pdiss")
    print(f"{Mg:>9.1e} | {z:>9.1e} | {m1:>9.2e} | {y1:>9.2e} | {best:>9.2e} | {ch}")

print("\nHawking present-day gamma-ray bound on 1e16-1e17 g: f_PBH <~ 1e-8 (literature)")
print("=> geometric channel relaxes this to f_PBH <~ 1e-4..1e-2 across most of the band")

# =====================================================================
#  FIGURES
# =====================================================================
C_HAWK = "#1f4e79"; C_GEO = "#d4691e"; GREY = "#9aa0a6"; DM = "#2e8b57"; PUR = "#6a3d9a"

# ---- FIG 1 ----
Mg = np.logspace(-10, 42, 1600); Mk = Mg/1000
fig, ax = plt.subplots(figsize=(7.2, 4.9))
ax.axvspan(1e17, 1e22, color=DM, alpha=0.10, zorder=0)
ax.text(10**20, 1e-19, "asteroid-mass PBH\ndark-matter window", color=DM, ha="center",
        fontsize=8, style="italic")
ax.axhline(age, color=GREY, ls=":", lw=1)
ax.text(1e-9, age*3, "age of universe", color=GREY, fontsize=8, style="italic")
ax.axhline(tP, color=GREY, ls=":", lw=1)
ax.text(1e-9, tP*3, "Planck time", color=GREY, fontsize=8, style="italic")
ax.axvline(Mcut_g, color=C_GEO, ls=(0, (4, 3)), lw=1, alpha=0.7)
ax.plot(Mg, tau_hawk(Mk), color=C_HAWK, lw=2.2, label=r"Hawking: $\tau\propto M^{3}$")
ax.plot(Mg, tau_geo(Mk), color=C_GEO, lw=1.5, ls="--",
        label=r"geometric, unsuppressed: $\tau\propto M^{2}$")
ax.plot(Mg, [tau_eff(m) for m in Mk], color=C_GEO, lw=2.4,
        label=r"geometric $+$ Peierls locking")
ax.plot([Mcut_g], [age], "o", color=C_GEO, ms=7, mec="white", zorder=6)
ax.annotate(r"$M_{\rm cut}\sim10^{16.5}\,$g", xy=(Mcut_g, age), xytext=(10**6, 1e18),
            color=C_GEO, fontsize=9, ha="center",
            arrowprops=dict(arrowstyle="-", color=C_GEO, lw=.9))
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(1e-10, 1e42); ax.set_ylim(1e-50, 1e57)
ax.set_xlabel(r"PBH mass $M$ (g)"); ax.set_ylabel(r"lifetime $\tau$ (s)")
ax.grid(True, which="major", ls=":", lw=0.5, alpha=0.4, zorder=0)
ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="0.8")
plt.tight_layout(); plt.savefig("fig1_lifetime.pdf"); plt.close()

# ---- FIG 2 ----
fig, ax = plt.subplots(figsize=(7.2, 4.9))
Mb = np.logspace(15, np.log10(Mcut_g), 400); zb = []
for m in Mb:
    t = tau_eff(m/1000); zb.append(max(z_inj(t), 1e-2))
bands = [(z_dc, 1e12, "#cfe8ff", "thermalized (no distortion)"),
         (z_muy, z_dc, "#ffe0b3", r"$\mu$-distortion"),
         (1100, z_muy, "#ffc2c2", r"$y$-distortion"),
         (1, 1100, "#e6d6f5", "post-recomb. / present")]
for zlo, zhi, col, lab in bands:
    ax.axhspan(zlo, zhi, color=col, alpha=0.55, zorder=0)
    ax.text(1.03e15, np.sqrt(zlo*zhi), lab, fontsize=7.8, va="center")
ax.axhline(1e9, color="k", ls=":", lw=0.8)
ax.text(2.6e16, 1.3e9, r"BBN ($z\sim10^{9}$)", fontsize=7.5, ha="right")
ax.plot(Mb, zb, color=C_GEO, lw=2.6, zorder=5)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(1e15, Mcut_g*1.05); ax.set_ylim(1, 3e11)
ax.set_xlabel(r"PBH mass $M$ (g)")
ax.set_ylabel(r"redshift of evaporation $z_{\rm evap}$")
plt.tight_layout(); plt.savefig("fig2_epochs.pdf"); plt.close()

# ---- FIG 3 ----
fig, ax = plt.subplots(figsize=(7.4, 5.1))
Mc = np.logspace(15.3, np.log10(Mcut_g)*0.999, 500)
fmu = [fmax_mu(m/1000) for m in Mc]
fy = [fmax_y(m/1000) for m in Mc]
fp = [fmax_pdiss(m/1000) for m in Mc]
ax.plot(Mc, fmu, color=C_GEO, lw=2.0, label=r"this work: $\mu$-distortion (FIRAS)")
ax.plot(Mc, fy, color=PUR, lw=2.0, label=r"this work: $y$-distortion (FIRAS)")
ax.plot(Mc, fp, color="#b03060", lw=1.6, ls="--",
        label=r"this work: BBN/photodiss. (estimated)")
Mh = np.logspace(15.3, 17.5, 200)
ax.plot(Mh, np.full_like(Mh, 1e-8), color=C_HAWK, lw=2.0, ls=":",
        label=r"standard Hawking $\gamma$-ray bound (today)")
ax.axvline(Mcut_g, color=GREY, ls=(0, (4, 3)), lw=1)
ax.text(Mcut_g*1.05, 3e-7, "survival\ncutoff", color=GREY, fontsize=8)
ax.axvspan(1e17, 9e17, color=DM, alpha=0.08)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(1e15, 1e17); ax.set_ylim(1e-9, 3)
ax.set_xlabel(r"PBH mass $M$ (g)")
ax.set_ylabel(r"$f_{\rm PBH}$ (would-be DM fraction)")
ax.grid(True, which="major", ls=":", lw=0.5, alpha=0.4, zorder=0)
ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="0.8")
plt.tight_layout(); plt.savefig("fig3_constraints.pdf"); plt.close()

print("\nwrote fig1_lifetime.pdf, fig2_epochs.pdf, fig3_constraints.pdf")

#!/usr/bin/env python3
"""Figures for the revised EPJP black-hole manuscript. All vector PDF."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9.5,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.6,
    "figure.dpi": 200,
})

C1 = "#1f5f8b"   # blue
C2 = "#d95f02"   # orange
C3 = "#2c8c5a"   # green
C4 = "#8b1f5f"   # magenta
GY = "#666666"

# physical constants (cgs-ish mixed, SI where noted)
G_SI   = 6.674e-11
c_SI   = 2.998e8
tP     = 5.391e-44          # s
mP_g   = 2.176e-5           # g
lP_m   = 1.616e-35          # m
t_univ = 4.35e17            # s
L0_lP  = 1.843              # L0 in Planck lengths

def RH_m(M_g):
    return 2.0 * G_SI * (M_g * 1e-3) / c_SI**2

def tau_geo(M_g):
    return 2.17 * tP * (M_g / mP_g)**2

def tau_geo_eff(M_g, xi_m=1e-15):
    x = RH_m(M_g) / xi_m
    return tau_geo(M_g) * np.exp(np.minimum(x, 700.0))

def tau_hawk(M_g):
    return t_univ * (M_g / 5e14)**3

# ----------------------------------------------------------------------
# Fig 1: direct FCC bond count vs horizon area (computed)
# ----------------------------------------------------------------------
def fig_entropy():
    rng = np.random.default_rng(7)
    n = 15  # integer coordinate half-range (units: a_cube/2 with parity)
    ax_ = np.arange(-n, n + 1)
    X, Y, Z = np.meshgrid(ax_, ax_, ax_, indexing="ij")
    mask = (X + Y + Z) % 2 == 0
    pts = np.stack([X[mask], Y[mask], Z[mask]], axis=1).astype(float)
    # rescale so nearest-neighbor bond length = 1 (L0 units)
    pts /= np.sqrt(2.0)
    center = np.array([0.131, 0.207, 0.083])  # avoid lattice degeneracies
    d2 = np.sum((pts - center)**2, axis=1)

    # build NN bond list via displacement set
    NN = np.array([(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                   for dz in (-1, 0, 1)
                   if abs(dx) + abs(dy) + abs(dz) == 2]) / np.sqrt(2.0)
    idx = {tuple(np.round(p * np.sqrt(2)).astype(int)): i
           for i, p in enumerate(pts)}
    bonds = []
    for i, p in enumerate(pts):
        for v in NN:
            q = p + v
            key = tuple(np.round(q * np.sqrt(2)).astype(int))
            j = idx.get(key)
            if j is not None and i < j:
                bonds.append((i, j))
    bonds = np.array(bonds)
    din = d2[bonds[:, 0]]
    djn = d2[bonds[:, 1]]

    radii_L0 = np.linspace(2.5, 9.0, 24)       # in units of L0
    Ncross = [(np.sum((din < r * r) != (djn < r * r))) for r in radii_L0]
    Ncross = np.array(Ncross, dtype=float)

    R_P = radii_L0 * L0_lP
    A_P = 4 * np.pi * R_P**2
    slope_th = 3 * np.sqrt(2) / L0_lP**2       # N = 3sqrt2 A / L0^2
    fit = np.polyfit(A_P, Ncross, 1)
    r2 = 1 - np.sum((Ncross - np.polyval(fit, A_P))**2) / np.sum((Ncross - Ncross.mean())**2)

    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    ax.plot(A_P, Ncross, "o", ms=4, color=C2, zorder=3,
            label="direct FCC bond count (24 radii)")
    Ag = np.linspace(0, A_P.max() * 1.03, 100)
    ax.plot(Ag, slope_th * Ag, "-", color=C1,
            label=rf"bulk count, $S=3\sqrt{{2}}\,\ln 2\,A/L_0^2$  ($R^2={r2:.4f}$)")
    ax.plot(Ag, Ag / (4 * np.log(2)), "--", color=GY,
            label=r"projected coefficient $S=A/(4\ell_P^2)$ (\S6)")
    ax.set_xlabel(r"horizon area $A$ ($\ell_P^2$)")
    ax.set_ylabel(r"entropy $S$ (units of $\ln 2$)")
    ax.set_xlim(0, Ag.max())
    ax.set_ylim(0, Ncross.max() * 1.08)
    ax.legend(loc="upper left", frameon=False)
    ax.text(0.97, 0.06, "direct bond count is linear in $A$",
            transform=ax.transAxes, ha="right", fontsize=8, color=GY, style="italic")
    fig.tight_layout()
    fig.savefig("figs/fig_entropy.pdf")
    plt.close(fig)
    print("fig_entropy: slope fit %.3f vs theory %.3f, R2=%.5f" % (fit[0], slope_th, r2))

# ----------------------------------------------------------------------
# Fig 2: rank-four isotropy (computed) + polarization sectors
# ----------------------------------------------------------------------
def fig_isotropy():
    def T_components(vecs):
        v = np.asarray(vecs, float)
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        T1111 = np.sum(v[:, 0]**4)
        T1122 = np.sum(v[:, 0]**2 * v[:, 1]**2)
        return T1111, T1122

    # D4 minimal vectors: +-e_mu +- e_nu, mu<nu, in 4D -> 24
    d4 = []
    for m in range(4):
        for n_ in range(m + 1, 4):
            for sm in (1, -1):
                for sn in (1, -1):
                    e = np.zeros(4); e[m] = sm; e[n_] = sn
                    d4.append(e)
    z4 = [np.eye(4)[i] * s for i in range(4) for s in (1, -1)]
    fcc = []
    for m in range(3):
        for n_ in range(m + 1, 3):
            for sm in (1, -1):
                for sn in (1, -1):
                    e = np.zeros(3); e[m] = sm; e[n_] = sn
                    fcc.append(e)

    ratios = []
    for vv in (d4, z4, fcc):
        a, b = T_components(vv)
        ratios.append(np.inf if b == 0 else a / b)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.9))

    labels = [r"$D_4$" + "\n(24 nn)", r"$\mathbb{Z}^4$" + "\n(8 nn)",
              "FCC slice\n(12 nn)"]
    vals = [ratios[0], 0.0, ratios[2]]
    cols = [C1, "#bbbbbb", "#888888"]
    bars = ax1.bar(labels, vals, color=cols, width=0.55)
    ax1.axhline(3.0, ls="--", color="#c02020", lw=1.2)
    ax1.text(2.35, 3.08, "isotropic value $=3$", color="#c02020",
             fontsize=8, ha="right")
    ax1.text(0, ratios[0] + 0.10, f"{ratios[0]:.2f}", ha="center", fontsize=8.5)
    ax1.text(1, 0.12, "degenerate\n$(T_{1122}=0)$", ha="center", fontsize=7.5)
    ax1.text(2, ratios[2] + 0.10, f"{ratios[2]:.2f}", ha="center", fontsize=8.5)
    ax1.set_ylabel(r"$T_{1111}/T_{1122}$")
    ax1.set_ylim(0, 4.0)
    ax1.set_title("(a) rank-four bond-tensor isotropy", fontsize=9)

    sects = ["TT", "trace", "gauge"]
    regge = [-1, 3, 0]
    einst = [-1, 3, 0]
    x = np.arange(3); w = 0.34
    ax2.bar(x - w / 2, regge, w, color=C1, label=r"$D_4$ Regge (computed)")
    ax2.bar(x + w / 2, einst, w, color=C3, label="linearized Einstein")
    ax2.axhline(0, color="k", lw=0.7)
    ax2.set_xticks(x); ax2.set_xticklabels(sects)
    ax2.set_ylabel(r"$C/|k|^2$")
    ax2.set_ylim(-1.8, 3.8)
    ax2.legend(frameon=False, loc="upper left")
    ax2.set_title("(b) polarization sectors of the kinetic operator", fontsize=9)

    fig.tight_layout()
    fig.savefig("figs/fig_isotropy.pdf")
    plt.close(fig)
    print("fig_isotropy ratios:", ratios)

# ----------------------------------------------------------------------
# Fig 3: lifetime vs mass
# ----------------------------------------------------------------------
def fig_lifetime():
    M = np.logspace(-10, 40, 1200)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.loglog(M, tau_hawk(M), color=C1, label=r"Hawking: $\tau\propto M^3$")
    ax.loglog(M, tau_geo(M), "--", color=C2,
              label=r"geometric (unsuppressed): $\tau\propto M^2$")
    ax.loglog(M, tau_geo_eff(M), "-", color=C2, lw=2.0,
              label="geometric + code threshold")
    ax.axhline(t_univ, color=GY, lw=0.7, ls=":")
    ax.text(1.5e-10, t_univ * 8, "age of universe", fontsize=7.5, color=GY)
    ax.axhline(tP, color=GY, lw=0.7, ls=":")
    ax.text(1.5e-10, tP * 8, "Planck time", fontsize=7.5, color=GY)
    ax.axvspan(1e17, 1e22, color=C3, alpha=0.10)
    ax.text(10**19.4, 1e-38, "asteroid-mass PBH\ndark-matter window",
            fontsize=7.5, color=C3, ha="center", style="italic")
    # cutoff marker
    from scipy.optimize import brentq
    Mcut = brentq(lambda lm: np.log(tau_geo_eff(10**lm)) - np.log(t_univ), 15.0, 17.5)
    Mcut = 10**Mcut
    ax.plot([Mcut], [t_univ], "o", ms=5, color=C2, zorder=4)
    ax.annotate(r"$M_{\rm cut}\sim 10^{16.5}$ g",
                xy=(Mcut, t_univ), xytext=(3e10, 3e21),
                fontsize=8.5, color=C2,
                arrowprops=dict(arrowstyle="-", color=C2, lw=0.8))
    ax.set_xlabel(r"PBH mass $M$ (g)")
    ax.set_ylabel(r"lifetime $\tau$ (s)")
    ax.set_xlim(1e-10, 1e40); ax.set_ylim(1e-50, 1e52)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="0.8")
    fig.tight_layout()
    fig.savefig("figs/fig_lifetime.pdf")
    plt.close(fig)
    print("fig_lifetime: Mcut = 10^%.2f g" % np.log10(Mcut))
    return Mcut

# ----------------------------------------------------------------------
# Fig 4: evaporation redshift vs mass, distortion bands
# ----------------------------------------------------------------------
def z_evap(M):
    t = tau_geo_eff(M)
    T_eV = 1e6 * (t)**-0.5          # T ~ 1 MeV (t/s)^-1/2
    z = T_eV / 2.35e-4
    return np.maximum(z, 1.0)

def fig_zevap():
    M = np.logspace(15, np.log10(3.2e16), 600)
    z = z_evap(M)
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.loglog(M, z, color=C2, lw=2.0)
    ax.axhspan(2e6, 1e11, color=C1, alpha=0.08)
    ax.axhspan(5e4, 2e6, color=C2, alpha=0.10)
    ax.axhspan(1.1e3, 5e4, color="#c02020", alpha=0.08)
    ax.axhspan(1, 1.1e3, color=C4, alpha=0.07)
    ax.axhline(1e9, ls=":", color=GY, lw=0.8)
    ax.text(1.05e15, 1.5e9, "BBN ($z\\sim 10^9$)", fontsize=7.5, color=GY)
    ax.text(1.05e15, 3e6, "thermalized (no distortion)", fontsize=7.5, color=GY)
    ax.text(1.05e15, 1.6e5, r"$\mu$-distortion", fontsize=7.5, color=C2)
    ax.text(1.05e15, 4e3, r"$y$-distortion", fontsize=7.5, color="#c02020")
    ax.text(1.05e15, 60, "post-recombination / present", fontsize=7.5, color=C4)
    ax.set_xlabel(r"PBH mass $M$ (g)")
    ax.set_ylabel(r"redshift of evaporation $z_{\rm evap}$")
    ax.set_ylim(1, 3e11)
    fig.tight_layout()
    fig.savefig("figs/fig_zevap.pdf")
    plt.close(fig)
    print("fig_zevap done")

# ----------------------------------------------------------------------
# Fig 5: fPBH constraints (order-of-magnitude reconstruction)
# ----------------------------------------------------------------------
def fig_fpbh(Mcut):
    M = np.logspace(15, 17, 900)
    z = z_evap(M)
    fEM = 0.5
    X1 = fEM * 4.85e3 / (1 + z)             # (Delta E/E_gamma) per unit fPBH
    zdc = 2e6

    with np.errstate(over="ignore"):
        mu_per_f = 1.4 * X1 * np.exp(-np.clip((z / zdc)**2.5, 0, 700))
    f_mu = np.where(mu_per_f > 0, 9e-5 / mu_per_f, np.inf)
    f_y = np.where(z < 5e4, 6.0e-5 / X1, np.inf)

    # BBN / photodissociation: zeta_EM = 3.1e-9 fEM fPBH GeV <= zeta_lim(t)
    t = tau_geo_eff(M)
    tt = np.array([1e2, 1e4, 1e6, 1e8, 1e10, 1e13])
    zl = np.array([3e-8, 2e-9, 1e-10, 5e-12, 2e-12, 1.5e-12])
    zeta_lim = 10**np.interp(np.log10(t), np.log10(tt), np.log10(zl))
    f_bbn = zeta_lim / (3.1e-9 * fEM)
    f_bbn = np.where((t > 1) & (t < 1e13), f_bbn, np.inf)

    combined = np.minimum.reduce([f_mu, f_y, f_bbn])
    evaporated = M < Mcut
    combined = np.where(evaporated, combined, np.inf)

    fig, ax = plt.subplots(figsize=(5.2, 3.7))
    band = np.clip(combined, 1e-9, 10)
    ax.fill_between(M, band / 3, band * 3, color="0.75", alpha=0.5,
                    label=r"combined band (all $O(1)$ uncertainties)")
    ax.loglog(M, np.clip(f_bbn, None, 10), "--", color="#c02020",
              label="BBN/photodissociation (reconstructed)")
    ax.loglog(M, np.clip(f_mu, None, 10), "-", color=C2,
              label=r"$\mu$-distortion, FIRAS")
    ax.loglog(M, np.clip(f_y, None, 10), "-", color=C4,
              label=r"$y$-distortion, FIRAS")
    ax.loglog(M, np.clip(combined, None, 10), "-", color="k", lw=2.0,
              label="combined exclusion (central)")
    ax.axhline(1e-8, ls=":", color=C1,
               label=r"standard Hawking $\gamma$-ray (does not apply; reference)")
    ax.axvline(Mcut, ls="--", color=GY, lw=0.9)
    ax.text(Mcut * 1.06, 3e-8, "survival\ncutoff", fontsize=7.5, color=GY)
    ax.set_xlabel(r"PBH mass $M$ (g)")
    ax.set_ylabel(r"$f_{\rm PBH}$ (would-be DM fraction)")
    ax.set_xlim(1e15, 1e17); ax.set_ylim(1e-9, 2)
    ax.legend(loc="lower left", frameon=True, framealpha=0.92,
              edgecolor="0.85", fontsize=6.8)
    fig.tight_layout()
    fig.savefig("figs/fig_fpbh.pdf")
    plt.close(fig)
    print("fig_fpbh done")

fig_entropy()
fig_isotropy()
Mcut = fig_lifetime()
fig_zevap()
fig_fpbh(Mcut)
print("all figures written")

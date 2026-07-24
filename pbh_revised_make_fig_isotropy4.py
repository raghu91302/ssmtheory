#!/usr/bin/env python3
"""Rebuild fig_isotropy as a 2x2 panel using the author's D4Regge class:
(a) rank-4 ratios (direct enumeration), (b) sector coefficients (computed),
(c) isotropy sweep of C(khat)/|k|^2 (computed), (d) C_Regge vs -q_FP scatter
(computed)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9.5,
    "axes.labelsize": 9.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.7, "lines.linewidth": 1.5, "figure.dpi": 200,
})
C1, C2, C3, C4, GY = "#1f5f8b", "#d95f02", "#2c8c5a", "#8b1f5f", "#666666"

# ---- load the author's script up to the analytic-proof block ----
src = open("linearized_gravity_verify.py").read()
cut = src.index("#  ANALYTIC proof")
# also silence its prints
import io, contextlib
ns = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(src[:cut], ns)
D4Regge, TTpol, _n = ns["D4Regge"], ns["TTpol"], ns["_n"]

with contextlib.redirect_stdout(io.StringIO()):
    r = D4Regge()
    r.hessian()

def qFP(kh, eps):
    kh = kh / np.linalg.norm(kh)
    a = kh @ eps; s = kh @ eps @ kh; t = np.trace(eps)
    return 0.5 * np.sum(eps * eps) - a @ a + t * s - 0.5 * t * t

# (c) sweep from (0,0,0,1) axis to (1,1,1,0)/sqrt3 body-type direction
angles = np.linspace(0, 90, 31)
e_axis = np.array([0, 0, 0, 1.0])
e_diag = _n(np.array([1, 1, 1, 0.0]))
Cp, Cx = [], []
for th in np.deg2rad(angles):
    k = np.cos(th) * e_axis + np.sin(th) * e_diag
    Cp.append(r.coeff(k, TTpol(k, 0)))
    Cx.append(r.coeff(k, TTpol(k, 1)))
Cp, Cx = np.array(Cp), np.array(Cx)
spread = (max(Cp.max(), Cx.max()) - min(Cp.min(), Cx.min())) / abs(np.mean(Cp))

# (d) 60 random generic polarizations
rng = np.random.default_rng(0)
xs, ys = [], []
for _ in range(60):
    k = rng.normal(size=4)
    e = rng.normal(size=(4, 4)); eps = e + e.T
    xs.append(-qFP(k, eps))
    ys.append(r.coeff(k, eps))
xs, ys = np.array(xs), np.array(ys)
resid = np.max(np.abs(ys - xs) / (np.abs(xs) + 1e-30))

# sector coefficients (computed)
kh = np.array([0, 0, 0, 1.0])
sect_regge = [r.coeff(kh, TTpol(kh, 0)),
              r.coeff(kh, np.eye(4)),
              r.coeff(kh, np.outer(kh, [1, 0, 0, 0]) + np.outer([1, 0, 0, 0], kh))]
sect_einst = [-1, 3, 0]

# (a) rank-4 ratios by direct enumeration
def ratio(vecs):
    v = np.asarray(vecs, float)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    T1111 = np.sum(v[:, 0]**4); T1122 = np.sum(v[:, 0]**2 * v[:, 1]**2)
    return None if T1122 == 0 else T1111 / T1122
d4v = []
for m in range(4):
    for n_ in range(m + 1, 4):
        for sm in (1, -1):
            for sn in (1, -1):
                e = np.zeros(4); e[m] = sm; e[n_] = sn; d4v.append(e)
fccv = []
for m in range(3):
    for n_ in range(m + 1, 3):
        for sm in (1, -1):
            for sn in (1, -1):
                e = np.zeros(3); e[m] = sm; e[n_] = sn; fccv.append(e)
r_d4, r_fcc = ratio(d4v), ratio(fccv)

fig, axs = plt.subplots(2, 2, figsize=(6.9, 5.4))
ax1, ax2, ax3, ax4 = axs.ravel()

labels = [r"$D_4$" + "\n(24 nn)", r"$\mathbb{Z}^4$" + "\n(8 nn)", "FCC slice\n(12 nn)"]
ax1.bar(labels, [r_d4, 0.0, r_fcc], color=[C1, "#bbbbbb", "#888888"], width=0.55)
ax1.axhline(3.0, ls="--", color="#c02020", lw=1.1)
ax1.text(2.35, 3.08, "isotropic value $=3$", color="#c02020", fontsize=7.5, ha="right")
ax1.text(0, r_d4 + 0.10, f"{r_d4:.2f}", ha="center", fontsize=8)
ax1.text(1, 0.12, "degenerate\n$(T_{1122}=0)$", ha="center", fontsize=7)
ax1.text(2, r_fcc + 0.10, f"{r_fcc:.2f}", ha="center", fontsize=8)
ax1.set_ylabel(r"$T_{1111}/T_{1122}$"); ax1.set_ylim(0, 4.0)
ax1.set_title("(a) rank-four bond-tensor isotropy", fontsize=9)

x = np.arange(3); w = 0.34
ax2.bar(x - w / 2, sect_regge, w, color=C1, label=r"$D_4$ Regge (computed)")
ax2.bar(x + w / 2, sect_einst, w, color=C3, label="linearized Einstein")
ax2.axhline(0, color="k", lw=0.7)
ax2.set_xticks(x); ax2.set_xticklabels(["TT", "trace", "gauge"])
ax2.set_ylabel(r"$C/|k|^2$"); ax2.set_ylim(-1.8, 3.8)
ax2.legend(frameon=False, loc="upper left")
ax2.set_title("(b) polarization sectors", fontsize=9)

ax3.plot(angles, Cp, "-", color=C1, marker="o", ms=2.5, label=r"TT $+$")
ax3.plot(angles, Cx, "--", color="#c02020", marker="s", ms=2.5, label=r"TT $\times$")
ax3.set_ylim(-1.1, -0.9)
ax3.set_xlabel(r"angle of $\mathbf{k}$: $(0001)\to(1110)$ [deg]")
ax3.set_ylabel(r"$C(\hat k)/|\mathbf{k}|^2$")
ax3.legend(frameon=False, loc="upper right")
ax3.text(0.04, 0.06, rf"spread $= {spread:.1e}$", transform=ax3.transAxes,
         fontsize=7.5, bbox=dict(fc="white", ec="0.8", lw=0.6))
ax3.set_title("(c) isotropy of the TT kinetic coefficient", fontsize=9)

lim = 1.06 * max(np.abs(xs).max(), np.abs(ys).max())
ax4.plot([-lim, lim], [-lim, lim], "-", color="0.7", lw=0.8, zorder=1)
ax4.plot(xs, ys, "o", ms=3.2, color="#c02020", zorder=2)
ax4.set_xlim(-lim, lim); ax4.set_ylim(-lim, lim)
ax4.set_xlabel(r"$-q_{\mathrm{FP}}(\varepsilon,\hat k)$  (lin. Einstein)")
ax4.set_ylabel(r"$C_{\mathrm{Regge}}(\varepsilon,\hat k)$")
ax4.text(0.04, 0.88, "60 random $(\\varepsilon,k)$\nmax rel. dev. "
         rf"$={resid:.0e}$", transform=ax4.transAxes, fontsize=7.5,
         bbox=dict(fc="white", ec="0.8", lw=0.6))
ax4.set_title("(d) generic polarizations", fontsize=9)

fig.tight_layout()
fig.savefig("figs/fig_isotropy.pdf")
print(f"ratios: D4={r_d4}, FCC={r_fcc}")
print(f"sectors: {[f'{v:.4f}' for v in sect_regge]}")
print(f"sweep spread = {spread:.2e}, scatter max rel dev = {resid:.1e}")

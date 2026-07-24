#!/usr/bin/env python3
"""Final vacancy dataset on wrap-free radii + boundary-localization check on
the Z-sector artifacts + figure."""
import numpy as np
import itertools as it
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pbh_revised_vacancy_analysis import (FCCCode, gf2_rank, gf2_in_rowspace, puncture,
                              exterior_triangle_logical, conversion_step_stats,
                              torus_dist2, NN)

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9.5,
    "axes.labelsize": 9.5, "legend.fontsize": 7.3,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.7, "lines.linewidth": 1.5, "figure.dpi": 200,
})
C1, C2, C3, GY = "#1f5f8b", "#d95f02", "#2c8c5a", "#666666"

code = FCCCode(8)
n0, k0, _, _ = code.params()
radii = [1.2, 1.5, 2.0, 2.5]
rows = []
for RL0 in radii:
    R_int2 = 2.0 * RL0**2
    HZp, HXp, keep, rn, re, severed = puncture(code, R_int2)
    npr = len(keep)
    kp = npr - gf2_rank(HZp) - gf2_rank(HXp)
    # X-sector low-weight exhaustive
    zero_cols = np.where(~HZp.any(axis=0))[0]
    w1x = sum(1 for c in zero_cols if not gf2_in_rowspace(
        HXp, np.eye(1, HZp.shape[1], c, dtype=np.uint8)[0]))
    colkeys, w2x = {}, 0
    for c in range(HZp.shape[1]):
        key = HZp[:, c].tobytes()
        if key in colkeys:
            v = np.zeros(HZp.shape[1], dtype=np.uint8)
            v[c] = 1; v[colkeys[key]] = 1
            if not gf2_in_rowspace(HXp, v):
                w2x += 1
        else:
            colkeys[key] = c
    # Z-sector artifacts + localization: distance of each artifact edge to vacancy
    zc = np.where(~HXp.any(axis=0))[0]
    dists = []
    for c in zc:
        a, b = code.edges[keep[c]]
        d = min(np.sqrt(torus_dist2(a, (0, 0, 0), 8) / 2.0),
                np.sqrt(torus_dist2(b, (0, 0, 0), 8) / 2.0))
        dists.append(d - RL0)
    zloc = (max(dists) if dists else 0.0)
    tri, td = exterior_triangle_logical(code, keep, rn, HXp)
    (m_e, m_z, m_x), (M_e, M_z, M_x), nb = conversion_step_stats(code, rn)
    A = 4 * np.pi * RL0**2
    rows.append(dict(R=RL0, rmn=len(rn), rme=len(re), sev=severed, npr=npr,
                     kp=kp, deficit=k0 - kp, w1x=w1x, w2x=w2x, nZart=len(zc),
                     zloc=zloc, tri=tri, m_e=m_e, M_z=M_z, M_x=M_x,
                     sevA=severed / A))
    print(rows[-1])

# figure: 3 panels
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.0, 2.6))
R = [r["R"] for r in rows]
ax1.plot(R, [3] * len(R), "o-", color=C1, label=r"min. bulk logical weight $d'(R)$")
ax1.plot(R, [r["M_z"] + r["M_x"] for r in rows], "s-", color=C2,
         label="checks per conversion step (max)")
ax1.plot(np.array(R), 12 * np.array(R) / R[0], ":", color=GY,
         label=r"linear law $\propto R$, for contrast")
ax1.set_xlabel(r"vacancy radius $R$ ($L_0$)"); ax1.set_ylabel("weight / count")
ax1.set_ylim(0, 40); ax1.legend(frameon=False, fontsize=6.3)
ax1.set_title("(a) no static growth with $R$", fontsize=8.5)

ax2.plot([r["rme"] for r in rows], [r["deficit"] for r in rows], "o-", color=C1)
xx = np.linspace(0, max(r["rme"] for r in rows) * 1.05, 10)
ax2.plot(xx, (2 / 3) * xx, "--", color=GY, label="code-rate slope $2/3$")
ax2.set_xlabel("edges removed"); ax2.set_ylabel(r"logicals lost $k_0-k'$")
ax2.legend(frameon=False); ax2.set_title("(b) logical deficit", fontsize=8.5)

ax3.plot([4 * np.pi * r["R"]**2 for r in rows], [r["sev"] for r in rows],
         "o-", color=C1, label="severed bonds")
aa = np.linspace(0, 4 * np.pi * max(R)**2 * 1.05, 10)
ax3.plot(aa, 3 * np.sqrt(2) * aa, "--", color=GY,
         label=r"$3\sqrt{2}\,A/L_0^2$ (App. A)")
ax3.set_xlabel(r"vacancy area $A$ ($L_0^2$)"); ax3.set_ylabel("severed bonds")
ax3.legend(frameon=False); ax3.set_title("(c) severed bonds vs.\\ area", fontsize=8.5)

fig.tight_layout()
fig.savefig("figs/fig_vacancy.pdf")
print("figure written")

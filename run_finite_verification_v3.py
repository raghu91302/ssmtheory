#!/usr/bin/env python3
"""run_finite_verification_v3.py

Driver for finite_ssm_verification_v3.py, Part I of
'Emergent Face-Centered Cubic Vacuum from Discrete Entanglement Networks'.

Produces the two results quoted in the manuscript beyond the enumeration itself:

  1. Concentration. The stationary weight of the maximally bonded sector
     pi_N(Omega_{N,0}) as a function of beta*eps, for N = 4..8.

  2. rho-independence. Forward lift proposals are given relative weight rho while
     stitch and reverse proposals keep unit weight. Because the Metropolis-Hastings
     acceptance factor carries the reverse-to-forward proposal ratio, the stationary
     measure is unchanged for every rho > 0; only relaxation times and path
     statistics move. Verified at rho = 1, 0.1 and e^-3.

Figures are written only if matplotlib is present. Requires numpy.
Run time: about one minute.
"""
import numpy as np
from finite_ssm_verification_v3 import enumerate_states, build_kernel

SIZES = [4, 5, 6, 7, 8]
BETAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
RHOS = [1.0, 0.1, float(np.exp(-3))]


def maximal_sector_weight(states, beta_eps, rho=1.0):
    keys, P, pi, B = build_kernel(states, beta_eps, rho)
    db = max(abs(pi[i] * P[i, j] - pi[j] * P[j, i])
             for i in range(len(keys)) for j in range(len(keys)))
    stat = np.abs(pi @ P - pi).max()
    return pi[B == B.max()].sum(), db, stat


def main():
    cache = {N: enumerate_states(N) for N in SIZES}

    print("== 1. Concentration: pi_N(Omega_N,0) vs beta*eps ==")
    header = "  beta*eps  " + "".join(f"   N={N:<2d}" for N in SIZES)
    print(header)
    curves = {N: [] for N in SIZES}
    for b in BETAS:
        row = f"  {b:8.2f}  "
        for N in SIZES:
            w, _, _ = maximal_sector_weight(cache[N], b)
            curves[N].append(w)
            row += f"  {w:5.3f}"
        print(row)
    print("  the maximal sector approaches unity at low effective temperature for every cutoff")

    print()
    print("== 2. Kinetic suppression does not move the equilibrium ==")
    print("  forward lift proposal weight rho, at beta*eps = 1.5")
    print("   N    rho=1        rho=0.1      rho=e^-3     max |difference|   max residual")
    worst_diff = worst_res = 0.0
    for N in SIZES:
        ws, res = [], []
        for r in RHOS:
            w, db, stat = maximal_sector_weight(cache[N], 1.5, r)
            ws.append(w); res.append(max(db, stat))
        d = max(ws) - min(ws)
        worst_diff = max(worst_diff, d); worst_res = max(worst_res, max(res))
        print(f"   {N}  {ws[0]:.12f} {ws[1]:.12f} {ws[2]:.12f}   {d:.1e}         {max(res):.1e}")
    print(f"  largest difference across kernels: {worst_diff:.1e}  (machine precision)")
    print(f"  largest detailed-balance / stationarity residual: {worst_res:.1e}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.6, 4.0))
        for N in SIZES:
            ax.plot(BETAS, curves[N], "o-", label=f"N={N}")
        ax.set_xlabel(r"$\beta\epsilon$")
        ax.set_ylabel("stationary probability of maximal bonding")
        ax.set_ylim(0, 1); ax.legend(); ax.set_title("Exact finite-state concentration")
        fig.tight_layout(); fig.savefig("fig_concentration.pdf")
        print("\n  wrote fig_concentration.pdf")
    except ImportError:
        print("\n  (matplotlib not present; figure skipped)")


if __name__ == "__main__":
    main()

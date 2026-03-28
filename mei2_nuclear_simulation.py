#!/usr/bin/env python3
"""
Mass-Energy-Information Equivalence II:
Nuclear Binding as Max-Cut Deduplication on the FCC Lattice Code

Raghu Kulkarni, SSMTheory Group, IDrive Inc.
raghu@idrive.com

This script reproduces all results from MEI Paper II.
It builds FCC polycrystal clusters, optimizes the proton-neutron
Max-Cut, applies Coulomb and Shannon asymmetry penalties, and
predicts the optimal proton number Z for 21 benchmark nuclei.

Model:
  S = alpha * N_pn - 23.2 * (A-2Z)^2/A - 0.72 * Z(Z-1)/A^(1/3)

  alpha = 4.5 MeV/bond  (emergent, self-consistent)
  23.2 MeV              (standard Weizsacker asymmetry coefficient)
  0.72 MeV              (standard Coulomb coefficient)

Requirements: numpy, scipy
Usage: python3 mei2_nuclear_simulation.py

Reference: arXiv:2603.20294 (FCC QEC code)
           doi:10.5281/zenodo.19248556 (MEI Paper I)
"""
import numpy as np
from scipy.spatial.distance import pdist


# =============================================================
# FCC LATTICE CONSTRUCTION
# =============================================================

def fcc_cluster(A):
    """
    Build a compact spherical FCC cluster of A nodes.

    FCC sites are integer points (x,y,z) with x+y+z even.
    Nodes are sorted by distance from origin and the closest
    A are selected, forming a roughly spherical polycrystal.
    """
    pts = []
    r = 1
    while len(pts) < A:
        for x in range(-r, r + 1):
            for y in range(-r, r + 1):
                for z in range(-r, r + 1):
                    if (x + y + z) % 2 == 0 and (x, y, z) not in pts:
                        pts.append((x, y, z))
        r += 1
    pts.sort(key=lambda p: p[0]**2 + p[1]**2 + p[2]**2)
    return pts[:A]


def adj_list(coords):
    """
    Build FCC nearest-neighbor adjacency list.

    FCC nearest neighbors are at distance sqrt(2), connected
    by the 12 displacement vectors (+-1,+-1,0) and permutations.
    Returns adjacency list and total edge count.
    """
    cs = {c: i for i, c in enumerate(coords)}
    NN = [
        (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
        (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
        (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
    ]
    adj = [[] for _ in range(len(coords))]
    total_edges = 0
    for i, (x, y, z) in enumerate(coords):
        for dx, dy, dz in NN:
            nb = (x + dx, y + dy, z + dz)
            if nb in cs:
                j = cs[nb]
                adj[i].append(j)
                if j > i:
                    total_edges += 1
    return adj, total_edges


# =============================================================
# MAX-CUT OPTIMIZER
# =============================================================

def maxcut(adj, A, Z, restarts=4, iters=2500):
    """
    Multi-restart greedy Max-Cut for p-n bond maximization.

    Assigns Z nodes as protons (1) and A-Z as neutrons (0).
    Greedy swap: pick a random proton and neutron, swap if
    the cut increases. Accepts neutral swaps with probability
    0.2 to escape local optima.

    Returns the best cut value (number of p-n bonds).
    """
    best = 0
    for _ in range(restarts):
        s = np.zeros(A, dtype=np.int8)
        s[np.random.choice(A, Z, replace=False)] = 1

        # Count initial cut
        cut = sum(
            1 for i in range(A) for j in adj[i]
            if j > i and s[i] != s[j]
        )

        for _ in range(iters):
            p = np.where(s == 1)[0]
            n = np.where(s == 0)[0]
            if len(p) == 0 or len(n) == 0:
                break
            pi = p[np.random.randint(len(p))]
            ni = n[np.random.randint(len(n))]

            # Delta: change in cut if we swap pi <-> ni
            d = 0
            for j in adj[pi]:
                if j != ni:
                    d += 1 if s[j] == 1 else -1
            for j in adj[ni]:
                if j != pi:
                    d += 1 if s[j] == 0 else -1

            if d > 0:
                s[pi], s[ni] = 0, 1
                cut += d
            elif d == 0 and np.random.random() < 0.2:
                s[pi], s[ni] = 0, 1

        if cut > best:
            best = cut
    return best


# =============================================================
# EXPERIMENTAL DATA
# =============================================================

# Format: A -> (Z_experimental, BE_per_A in MeV, name)
# Sources: AME2020, CODATA 2022
EXPERIMENTAL = {
    2:   (1,   1.112, "2H"),
    4:   (2,   7.074, "4He"),
    6:   (3,   5.333, "6Li"),
    7:   (3,   5.606, "7Li"),
    9:   (4,   6.463, "9Be"),
    12:  (6,   7.680, "12C"),
    14:  (7,   7.476, "14N"),
    16:  (8,   7.976, "16O"),
    20:  (10,  8.032, "20Ne"),
    24:  (12,  8.261, "24Mg"),
    28:  (14,  8.448, "28Si"),
    32:  (16,  8.493, "32S"),
    40:  (20,  8.551, "40Ca"),
    56:  (26,  8.790, "56Fe"),
    58:  (28,  8.732, "58Ni"),
    80:  (34,  8.711, "80Se"),
    90:  (40,  8.710, "90Zr"),
    120: (50,  8.505, "120Sn"),
    150: (62,  8.278, "150Sm"),
    208: (82,  7.868, "208Pb"),
    238: (92,  7.570, "238U"),
}


# =============================================================
# MODEL PARAMETERS
# =============================================================

ALPHA = 4.5     # MeV per p-n bond (emergent, self-consistent)
GAMMA = 23.2    # MeV, standard Weizsacker asymmetry coefficient
COULOMB = 0.72  # MeV, standard Coulomb coefficient


# =============================================================
# MAIN SIMULATION
# =============================================================

def predict_Z(A, adj, mc_cache, alpha=ALPHA, gamma=GAMMA):
    """
    Predict optimal proton number Z for mass number A.

    Score = alpha * N_pn - gamma * (A-2Z)^2/A - 0.72 * Z(Z-1)/A^(1/3)

    Returns (best_Z, best_score, best_pn_bonds).
    """
    z_min = max(1, int(A * 0.20))
    z_max = min(A - 1, int(A * 0.55))
    if A < 10:
        z_min, z_max = 1, A - 1

    best_Z, best_S, best_pn = z_min, -1e9, 0
    for Z in range(z_min, z_max + 1):
        if (A, Z) not in mc_cache:
            continue
        pn = mc_cache[(A, Z)]
        asym = gamma * (A - 2 * Z)**2 / A
        coul = COULOMB * Z * (Z - 1) / (A**(1/3)) if A > 1 else 0
        score = alpha * pn - asym - coul
        if score > best_S:
            best_S, best_Z, best_pn = score, Z, pn
    return best_Z, best_S, best_pn


def compute_alpha_eff(A, Z_exp, BE_exp, pn_bonds, gamma=GAMMA):
    """
    Compute the effective deduplication energy per bond.

    alpha_eff = (B_exp + E_Coulomb + E_asym) / N_pn

    If alpha_eff is constant across all A, the model is
    self-consistent.
    """
    if pn_bonds == 0:
        return 0.0
    coul = COULOMB * Z_exp * (Z_exp - 1) / (A**(1/3)) if A > 1 else 0
    asym = gamma * (A - 2 * Z_exp)**2 / A
    gross = BE_exp * A + coul + asym
    return gross / pn_bonds


# =============================================================
# RUN
# =============================================================

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 65)
    print("MEI Paper II: Nuclear Binding from FCC Max-Cut Deduplication")
    print("=" * 65)
    print(f"alpha   = {ALPHA} MeV/bond (emergent)")
    print(f"gamma   = {GAMMA} MeV (standard Weizsacker a_a)")
    print(f"Coulomb = {COULOMB} MeV (standard a_c)")
    print()

    # --- Phase 1: Pre-compute Max-Cut for all (A, Z) ---
    print("Phase 1: Pre-computing Max-Cut values...")
    mc_cache = {}
    cluster_info = {}
    for A in sorted(EXPERIMENTAL.keys()):
        coords = fcc_cluster(A)
        adj, total_edges = adj_list(coords)
        cluster_info[A] = (adj, total_edges)

        z_min = max(1, int(A * 0.20))
        z_max = min(A - 1, int(A * 0.55))
        if A < 10:
            z_min, z_max = 1, A - 1

        for Z in range(z_min, z_max + 1):
            r = 5 if A <= 60 else 3
            it = min(A * 12, 3500)
            mc_cache[(A, Z)] = maxcut(adj, A, Z, restarts=r, iters=it)
        print(f"  A={A:>3} done ({z_max - z_min + 1} Z values, "
              f"{total_edges} edges)")

    # --- Phase 2: Predict Z for each nucleus ---
    print()
    print("=" * 65)
    print("Phase 2: Z Predictions")
    print("=" * 65)
    header = (f"{'A':>4} {'Name':<8} {'Z_pred':>6} {'Z_exp':>6} "
              f"{'dZ':>5} {'N_pn':>6} {'BE/A':>7} {'Exp':>7}")
    print(header)
    print("-" * 65)

    exact = 0
    within_1 = 0
    within_3 = 0
    total = 0

    for A in sorted(EXPERIMENTAL.keys()):
        Z_exp, BE_exp, name = EXPERIMENTAL[A]
        adj, _ = cluster_info[A]
        Z_pred, score, pn = predict_Z(A, adj, mc_cache)

        dz = Z_pred - Z_exp
        mark = "ok" if dz == 0 else f"{dz:+d}"
        if dz == 0:
            exact += 1
        if abs(dz) <= 1:
            within_1 += 1
        if abs(dz) <= 3:
            within_3 += 1
        total += 1

        print(f"{A:>4} {name:<8} {Z_pred:>6} {Z_exp:>6} "
              f"{mark:>5} {pn:>6} {score/A:>7.2f} {BE_exp:>7.3f}")

    print()
    print(f"Exact Z:    {exact}/{total} ({exact/total*100:.0f}%)")
    print(f"Within +/-1:  {within_1}/{total} ({within_1/total*100:.0f}%)")
    print(f"Within +/-3:  {within_3}/{total} ({within_3/total*100:.0f}%)")

    # --- Phase 3: Self-consistent alpha ---
    print()
    print("=" * 65)
    print("Phase 3: Self-Consistent alpha_eff (MeV/bond)")
    print("=" * 65)

    alphas = []
    for A in sorted(EXPERIMENTAL.keys()):
        Z_exp, BE_exp, name = EXPERIMENTAL[A]
        adj, _ = cluster_info[A]
        # Compute Max-Cut at EXPERIMENTAL Z
        pn_at_exp = maxcut(
            adj, A, Z_exp,
            restarts=5,
            iters=min(A * 15, 5000)
        )
        a_eff = compute_alpha_eff(A, Z_exp, BE_exp, pn_at_exp)
        alphas.append((A, name, a_eff, pn_at_exp))
        print(f"  A={A:>3} {name:<8}: alpha_eff = {a_eff:.2f} MeV/bond "
              f"(N_pn = {pn_at_exp})")

    # Exclude outliers (A=2 surface, A=4 magic)
    bulk_alphas = [a for A, _, a, _ in alphas if A > 5]
    print()
    print(f"Bulk alpha (A > 5): {np.mean(bulk_alphas):.2f} "
          f"+/- {np.std(bulk_alphas):.2f} MeV/bond")
    print(f"Range: {min(bulk_alphas):.2f} to {max(bulk_alphas):.2f}")

    # --- Summary ---
    print()
    print("=" * 65)
    print("SUMMARY: All Four Weizsacker Terms")
    print("=" * 65)
    print(f"  Volume:    Interior K=12 x alpha={ALPHA} MeV/bond  [EMERGES]")
    print(f"  Surface:   Boundary K<12, automatic              [EMERGES]")
    print(f"  Coulomb:   {COULOMB} x Z(Z-1)/A^(1/3) MeV        [APPLIED]")
    print(f"  Asymmetry: {GAMMA} x (A-2Z)^2/A MeV              "
          f"[FORM DERIVED, COEFF APPLIED]")
    print()
    print("Done.")

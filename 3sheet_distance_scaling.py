"""
3sheet_distance_scaling.py: Distance scaling for the 3-sheet multiplexed
configuration of Paper 3 — the version that runs on existing hardware.

Configuration:
  - All three triad sheets (xy, xz, yz) live on the same FCC chip
  - Total data qubits: 3L³  (192 at L=4, 648 at L=6, 1536 at L=8)
  - Ancillas (vertex Z-stabs, oct-void X-stabs) are SHARED across sheets:
    L³/2 of each = 32 + 32 = 64 at L=4
  - Total physical qubit count: 3L³ + L³ = 4L³  → 256 at L=4, 864 at L=6, 2048 at L=8
  - Encoding: 6L logical qubits at distance L

Time-multiplexing protocol:
  - Round k (k = 1, 2, 3) measures sheet k's stabilizers
  - Sheets not currently measured idle for one round (2p effective per
    active round from the active sheet's POV, since two idle rounds
    pass between consecutive active measurements)
  - Modeled in Stim with before_round_data_depolarization = 2p

Hardware feasibility (256 qubits at L=4):
  - IBM Osprey (433 qubits, 2022)         ✓
  - IBM Condor (1121 qubits, 2023)         ✓  
  - IBM Heron R2 (156 qubits, 2024)        ✗ (192 data alone exceeds)
  - Google Willow (105 qubits)             ✗
  - Quantinuum H2-1 (56 qubits)            ✗
  - Atom Computing (1180+ neutral atoms)   ✓

Approach:
  Each sheet is independent and identically distributed. We simulate ONE
  sheet's circuit-level memory experiment with the multiplex idle penalty,
  then combine: 3-sheet block error = 1 − (1 − single-sheet block)³.
  Per-logical error rate is the same for every logical (within statistical
  fluctuation) since each sheet's logicals are equivalent under the symmetry.
"""

from __future__ import annotations
import numpy as np
import stim
import pymatching
import time
import os
import sys

from fcc_code import (
    build_fcc_lattice, gf2_rref, gf2_kernel,
    build_sheet_code_xy, to_matrix, find_logical_Z_basis as find_logical_basis,
)

OUT = "/home/claude/3sheet_scaling"
os.makedirs(OUT, exist_ok=True)


# ============================================================
# Single-sheet circuit WITH multiplex idle penalty
# ============================================================
def build_sheet_circuit_multiplex(L, p, rounds, multiplex_factor=2.0):
    """Build a sheet-code memory circuit with idle-decoherence penalty
    on every data qubit before each round of stabilizer extraction.
    
    multiplex_factor: 0.0 for single-sheet, 2.0 for 3-sheet multiplexing.
    """
    n_data, Z_stabs, X_stabs = build_sheet_code_xy(L)
    HX = to_matrix(X_stabs, n_data)
    HZ = to_matrix(Z_stabs, n_data)
    L_Z = find_logical_basis(HX, HZ)
    
    n_z = len(Z_stabs)
    n_x = len(X_stabs)
    z_anc = list(range(n_data, n_data + n_z))
    x_anc = list(range(n_data + n_z, n_data + n_z + n_x))
    
    p_idle = multiplex_factor * p
    
    c = stim.Circuit()
    c.append("R", list(range(n_data + n_z + n_x)))
    if p > 0:
        c.append("X_ERROR", list(range(n_data)), p)
    
    for r in range(rounds):
        # Multiplex idle penalty: data qubits decohere while OTHER sheets
        # are being measured
        if p_idle > 0:
            c.append("DEPOLARIZE1", list(range(n_data)), p_idle)
        
        # Z-stab extraction
        c.append("R", z_anc)
        if p > 0:
            c.append("X_ERROR", z_anc, p)
        for i, stab in enumerate(Z_stabs):
            for d in stab:
                c.append("CX", [d, z_anc[i]])
                if p > 0:
                    c.append("DEPOLARIZE2", [d, z_anc[i]], p)
        if p > 0:
            c.append("X_ERROR", z_anc, p)
        c.append("M", z_anc)
        
        # X-stab extraction
        c.append("R", x_anc)
        if p > 0:
            c.append("X_ERROR", x_anc, p)
        c.append("H", x_anc)
        if p > 0:
            c.append("DEPOLARIZE1", x_anc, p)
        for i, stab in enumerate(X_stabs):
            for d in stab:
                c.append("CX", [x_anc[i], d])
                if p > 0:
                    c.append("DEPOLARIZE2", [x_anc[i], d], p)
        c.append("H", x_anc)
        if p > 0:
            c.append("DEPOLARIZE1", x_anc, p)
            c.append("X_ERROR", x_anc, p)
        c.append("M", x_anc)
        
        # Detectors on Z-stabs
        n_per_round = n_z + n_x
        if r == 0:
            for i in range(n_z):
                offset = -(n_per_round) + i
                c.append("DETECTOR", [stim.target_rec(offset)])
        else:
            for i in range(n_z):
                off_now = -(n_per_round) + i
                off_prev = off_now - n_per_round
                c.append("DETECTOR",
                         [stim.target_rec(off_now),
                          stim.target_rec(off_prev)])
    
    # Final destructive measurement
    if p > 0:
        c.append("X_ERROR", list(range(n_data)), p)
    c.append("M", list(range(n_data)))
    
    for i, stab in enumerate(Z_stabs):
        z_offset = -n_data - (n_z + n_x) + i
        targets = [stim.target_rec(z_offset)]
        for d in stab:
            targets.append(stim.target_rec(-n_data + d))
        c.append("DETECTOR", targets)
    
    for k, lz in enumerate(L_Z):
        targets = [stim.target_rec(-n_data + d) for d in range(n_data) if lz[d]]
        c.append("OBSERVABLE_INCLUDE", targets, k)
    
    return c, n_data, L_Z


def run_with_pymatching(circ, num_shots):
    dem = circ.detector_error_model(decompose_errors=True,
                                     approximate_disjoint_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    sampler = circ.compile_detector_sampler()
    det, obs = sampler.sample(num_shots, separate_observables=True)
    pred = matcher.decode_batch(det)
    block_err = np.any(pred != obs, axis=1)
    per_obs_err = (pred != obs).mean(axis=0)
    return block_err.mean(), per_obs_err, int(block_err.sum()), num_shots


# ============================================================
# Run at L = 4, 6, 8 with multiplex penalty
# ============================================================
print("=" * 76)
print("  3-Sheet Multiplexed Distance Scaling")
print("  Idle decoherence rate = 2p   (Paper 3 §9.3 model)")
print("=" * 76)

LATTICES = [4, 6, 8]
P_VALUES = [0.0001, 0.0003, 0.0005, 0.001, 0.002, 0.003, 0.005]
SHOTS_BY_L = {4: 30_000, 6: 30_000, 8: 20_000}

# Verify constructions
print(f"\n{'L':>3s}  {'1-sheet':>14s}  {'3-sheet':>14s}  {'phys. qubits':>14s}  "
      f"{'rate':>6s}")
print("-" * 76)
for L in LATTICES:
    n_data, Z_stabs, X_stabs = build_sheet_code_xy(L)
    n_data_3sheet = 3 * n_data
    n_phys = 4 * L**3   # 3L³ data + L³ ancillas (32+32 each shared) at L=4 = 192+64=256
    k_per_sheet = 2 * L
    k_3sheet = 6 * L
    rate = 100 * k_3sheet / n_data_3sheet
    print(f"{L:>3d}  [[{n_data},{k_per_sheet},{L}]]   "
          f"[[{n_data_3sheet},{k_3sheet},{L}]]   {n_phys:>5d} (data+anc)   {rate:>5.1f}%")


print(f"\n{'-' * 76}")
print(f"Memory experiment with multiplex penalty: DEPOLARIZE1(2p) before each round")
print(f"{'-' * 76}")

results = {L: {} for L in LATTICES}

for L in LATTICES:
    n_per_sheet = 2 * L
    print(f"\n  L = {L}:  per sheet [[{L**3}, {n_per_sheet}, {L}]],  "
          f"3-sheet [[{3*L**3}, {3*n_per_sheet}, {L}]]")
    print(f"  shots: {SHOTS_BY_L[L]}, rounds: {L}")
    print(f"  {'p':>8s}  {'1-sheet block':>14s}  {'1-sheet per-log':>16s}  "
          f"{'3-sheet block':>14s}  {'time':>6s}")
    print(f"  {'-' * 70}")
    for p in P_VALUES:
        t0 = time.time()
        circ, _, _ = build_sheet_circuit_multiplex(
            L=L, p=p, rounds=L, multiplex_factor=2.0
        )
        s_block, perlog_arr, n_block, n_total = run_with_pymatching(
            circ, SHOTS_BY_L[L]
        )
        # 3-sheet block: at least one of three independent sheets fails
        three_sheet_block = 1 - (1 - s_block) ** 3
        elapsed = time.time() - t0
        results[L][p] = {
            "1sheet_block": s_block,
            "1sheet_perlog_mean": perlog_arr.mean(),
            "1sheet_perlog_std": perlog_arr.std(),
            "1sheet_perlog_arr": perlog_arr.tolist(),
            "3sheet_block": three_sheet_block,
            "n_block": n_block,
            "n_shots": n_total,
        }
        print(f"  {p:>8.4f}  {s_block:>14.3e}  {perlog_arr.mean():>16.3e}  "
              f"{three_sheet_block:>14.3e}  {elapsed:>5.1f}s")


# ============================================================
# Suppression analysis
# ============================================================
print(f"\n{'=' * 76}")
print("Suppression factor Λ for the 3-sheet multiplexed configuration")
print(f"{'=' * 76}")

print(f"\n  {'p':>8s}  {'Λ(4→6)':>10s}  {'Λ(6→8)':>10s}    "
      f"per-log @L=4    per-log @L=6    per-log @L=8")
print(f"  {'-' * 80}")

for p in P_VALUES:
    pl4 = results[4][p]["1sheet_perlog_mean"]
    pl6 = results[6][p]["1sheet_perlog_mean"]
    pl8 = results[8][p]["1sheet_perlog_mean"]
    
    if pl6 > 0.5 / SHOTS_BY_L[6]:
        l46 = pl4 / pl6
        l46_s = f"{l46:.2f}"
    else:
        l46_s = "—"
    if pl8 > 0.5 / SHOTS_BY_L[8] and pl6 > 0.5 / SHOTS_BY_L[6]:
        l68 = pl6 / pl8
        l68_s = f"{l68:.2f}"
    else:
        l68_s = "—"
    
    print(f"  {p:>8.4f}  {l46_s:>10s}  {l68_s:>10s}    "
          f"  {pl4:>9.2e}      {pl6:>9.2e}      {pl8:>9.2e}")


# ============================================================
# Save and plot
# ============================================================
with open(f"{OUT}/results.txt", "w") as f:
    f.write("# 3-Sheet multiplexed distance scaling\n")
    f.write("# Idle decoherence: DEPOLARIZE1(2p) per round (Paper 3 §9.3)\n")
    f.write("# d=L rounds, MWPM via PyMatching\n\n")
    f.write(f"{'p':>10s}")
    for L in LATTICES:
        f.write(f"  {'L=' + str(L) + '_block_1s':>16s}  "
                f"{'L=' + str(L) + '_perlog':>15s}  "
                f"{'L=' + str(L) + '_block_3s':>16s}")
    f.write("\n")
    for p in P_VALUES:
        f.write(f"{p:>10.5f}")
        for L in LATTICES:
            r = results[L][p]
            f.write(f"  {r['1sheet_block']:>16.4e}  "
                    f"{r['1sheet_perlog_mean']:>15.4e}  "
                    f"{r['3sheet_block']:>16.4e}")
        f.write("\n")
print(f"\nResults: {OUT}/results.txt")


import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 12.5,
    "lines.linewidth": 2.0, "lines.markersize": 8,
    "axes.grid": True, "grid.alpha": 0.25,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))
colors = {4: "#1d3a6c", 6: "#2f7d8c", 8: "#a83232"}
markers = {4: "s", 6: "o", 8: "^"}

# Left: per-logical (single sheet) and 3-sheet block under multiplex
for L in LATTICES:
    p_show = []
    perlog = []
    for p in P_VALUES:
        r = results[L][p]
        if r["1sheet_perlog_mean"] > 0.5 / r["n_shots"]:
            p_show.append(p)
            perlog.append(r["1sheet_perlog_mean"])
    n_3 = 3 * L**3
    k_3 = 6 * L
    rate_pct = 100 * k_3 / n_3
    label = f"L={L}: 3-sheet [[{n_3},{k_3},{L}]] ({rate_pct:.1f}%)"
    ax1.loglog(p_show, perlog, marker=markers[L], color=colors[L], label=label)

ax1.set_xlabel("Physical depolarizing error rate, $p$")
ax1.set_ylabel("Per-logical error rate")
ax1.set_title("3-sheet multiplexed: per-logical scaling with $L$")
ax1.legend(loc="lower right", fontsize=9.5)
ax1.set_xlim(8e-5, 7e-3)
ax1.set_ylim(5e-6, 1)

# Right: Λ values
lambdas_46_p = []
lambdas_46_v = []
lambdas_68_p = []
lambdas_68_v = []
for p in P_VALUES:
    pl4 = results[4][p]["1sheet_perlog_mean"]
    pl6 = results[6][p]["1sheet_perlog_mean"]
    pl8 = results[8][p]["1sheet_perlog_mean"]
    if pl6 > 0.5 / SHOTS_BY_L[6]:
        lambdas_46_p.append(p)
        lambdas_46_v.append(pl4 / pl6 if pl6 > 0 else 0)
    if pl8 > 0.5 / SHOTS_BY_L[8] and pl6 > 0:
        lambdas_68_p.append(p)
        lambdas_68_v.append(pl6 / pl8)

ax2.semilogx(lambdas_46_p, lambdas_46_v, "o-", color="#2f7d8c",
             label="$\\Lambda(4\\to 6)$")
ax2.semilogx(lambdas_68_p, lambdas_68_v, "s-", color="#a83232",
             label="$\\Lambda(6\\to 8)$")
ax2.axhline(1, ls=":", color="black", alpha=0.4)
ax2.text(2e-3, 1.05, "$\\Lambda = 1$ (no improvement)",
         color="black", fontsize=9)
ax2.axhline(2.14, ls="--", color="#888", alpha=0.5)
ax2.text(2e-3, 2.25, "Google Willow: $\\Lambda \\approx 2.14$",
         color="#444", fontsize=9)
ax2.set_xlabel("Physical depolarizing error rate, $p$")
ax2.set_ylabel("Suppression factor $\\Lambda$")
ax2.set_title("Distance suppression (multiplex regime)")
ax2.legend(loc="upper right", fontsize=10)
ax2.set_xlim(8e-5, 7e-3)

fig.suptitle("FCC 3-sheet multiplexed code: distance scaling under "
             "Paper 3's idle-noise model (2p)",
             fontsize=12.5, y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/3sheet_distance_scaling.png")
fig.savefig(f"{OUT}/3sheet_distance_scaling.pdf")
plt.close(fig)
print(f"Plot: {OUT}/3sheet_distance_scaling.{{png,pdf}}")

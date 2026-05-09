"""Quick L=8 threshold simulation to add a third data point."""
import numpy as np
import pymatching
import time
import json
import sys
sys.path.insert(0, '/home/claude/lorentz_paper')
from build_code import build_fcc_lattice, build_z_stabilizers, build_x_stabilizers, f2_rank
from threshold_sim import compute_logical_operators


def run_L8():
    L = 8
    print(f"=== L = {L} ===")
    t0 = time.time()
    vertices, voids, edges = build_fcc_lattice(L)
    H_Z = build_z_stabilizers(vertices, edges)
    H_X = build_x_stabilizers(vertices, voids, edges, L)
    n_e = len(edges)

    print(f"  Built code: n_e={n_e}, n_v={len(vertices)}, n_o={len(voids)}, took {time.time()-t0:.1f}s")
    assert (H_X @ H_Z.T % 2).max() == 0

    rk_Z = f2_rank(H_Z)
    rk_X = f2_rank(H_X)
    k = n_e - rk_Z - rk_X
    expected_k = 2 * L**3 + 2
    print(f"  k = {k} (expected {expected_k})")
    assert k == expected_k

    print(f"  Computing logical operators...")
    t0 = time.time()
    L_X = compute_logical_operators(H_Z, H_X)
    L_Z = compute_logical_operators(H_X, H_Z)
    print(f"    L_X: {L_X.shape}, L_Z: {L_Z.shape}, took {time.time()-t0:.1f}s")

    matcher_X = pymatching.Matching.from_check_matrix(H_Z, faults_matrix=L_X)
    matcher_Z = pymatching.Matching.from_check_matrix(H_X, faults_matrix=L_Z)

    ps = [0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02]
    n_trials = 1000  # reduced for L=8

    results = []
    print(f"\n  {'p':>8} | {'X fail':>8} | {'X rate':>10} | {'per-LQ':>12} | time")
    print("  " + "-" * 55)
    for p in ps:
        t0 = time.time()
        rng = np.random.default_rng(42 + L)
        n_X = 0
        bs = 500
        for batch in range((n_trials + bs - 1) // bs):
            cur = min(bs, n_trials - batch * bs)
            E = (rng.random((cur, n_e)) < p).astype(np.uint8)
            S = (E @ H_Z.T) % 2
            actual = (E @ L_X.T) % 2
            for i in range(cur):
                pred = matcher_X.decode(S[i])
                if (pred != actual[i]).any():
                    n_X += 1
        rX = n_X / n_trials
        per_lq = rX / k
        el = time.time() - t0
        results.append({'L': L, 'p': p, 'n_trials': n_trials, 'n_e': n_e, 'k': k,
                        'X_fail': n_X, 'rX': rX, 'per_lq': per_lq, 't': el})
        print(f"  {p:>8.4f} | {n_X:>8} | {rX:>10.4f} | {per_lq:>12.2e} | {el:.1f}s")
    return results


if __name__ == "__main__":
    r8 = run_L8()
    with open('/home/claude/lorentz_paper/threshold_L8.json', 'w') as f:
        json.dump(r8, f, indent=2)
    print("\nSaved to threshold_L8.json")

"""Build periodic-torus L=6 code and run threshold sim, comparing to L=4."""
import numpy as np
import itertools
import pymatching
import time
import json
import sys
sys.path.insert(0, '/home/claude/lorentz_paper')
from build_code import build_fcc_lattice, build_z_stabilizers, build_x_stabilizers, f2_rank
from threshold_sim import compute_logical_operators


def run_threshold(L, n_trials_per_p, ps, seed_base=42):
    """Build code at given L (periodic torus) and run threshold sweep."""
    print(f"\n=== L = {L} ===")
    vertices, voids, edges = build_fcc_lattice(L)
    H_Z = build_z_stabilizers(vertices, edges)
    H_X = build_x_stabilizers(vertices, voids, edges, L)

    n_e = len(edges)
    n_v = len(vertices)
    n_o = len(voids)

    # Verify code params
    assert (H_X @ H_Z.T % 2).max() == 0, "CSS validity failed"
    assert all(H_Z.sum(axis=1) == 12)
    assert all(H_X.sum(axis=1) == 12)
    assert all(H_X.sum(axis=0) == 2)

    rk_Z = f2_rank(H_Z)
    rk_X = f2_rank(H_X)
    k = n_e - rk_Z - rk_X
    expected_k = (3 - 2) * (3 + 1) // 2 * L**3 + 2
    print(f"  n = {n_e}, k = {k} (expected {expected_k})")
    print(f"  Code: [[{n_e}, {k}, 3]]")
    assert k == expected_k

    # Compute logical operators (this can be slow for large L; takes O(n_e^2) work)
    print(f"  Computing logical operators...")
    t0 = time.time()
    L_X = compute_logical_operators(H_Z, H_X)
    L_Z = compute_logical_operators(H_X, H_Z)
    print(f"    L_X: {L_X.shape}, L_Z: {L_Z.shape}, took {time.time()-t0:.1f}s")
    assert L_X.shape == (k, n_e) and L_Z.shape == (k, n_e)

    matcher_X = pymatching.Matching.from_check_matrix(H_Z, faults_matrix=L_X)
    matcher_Z = pymatching.Matching.from_check_matrix(H_X, faults_matrix=L_Z)

    print(f"\n  {'p':>8} | {'trials':>7} | {'X fail':>8} | {'X rate':>10} | {'Z fail':>8} | {'Z rate':>10} | time")
    print("  " + "-" * 75)

    results = []
    for p in ps:
        t0 = time.time()
        rng_X = np.random.default_rng(seed_base + L)
        rng_Z = np.random.default_rng(seed_base + L + 100)
        n_X = 0
        n_Z = 0
        bs = 1000
        for batch in range((n_trials_per_p + bs - 1) // bs):
            cur = min(bs, n_trials_per_p - batch * bs)
            E_X = (rng_X.random((cur, n_e)) < p).astype(np.uint8)
            S_X = (E_X @ H_Z.T) % 2
            actual_X = (E_X @ L_X.T) % 2
            for i in range(cur):
                pred = matcher_X.decode(S_X[i])
                if (pred != actual_X[i]).any():
                    n_X += 1
            E_Z = (rng_Z.random((cur, n_e)) < p).astype(np.uint8)
            S_Z = (E_Z @ H_X.T) % 2
            actual_Z = (E_Z @ L_Z.T) % 2
            for i in range(cur):
                pred = matcher_Z.decode(S_Z[i])
                if (pred != actual_Z[i]).any():
                    n_Z += 1

        rX = n_X / n_trials_per_p
        rZ = n_Z / n_trials_per_p
        el = time.time() - t0
        results.append({'L': L, 'p': p, 'n_trials': n_trials_per_p, 'n_e': n_e, 'k': k,
                        'X_fail': n_X, 'Z_fail': n_Z, 'rX': rX, 'rZ': rZ, 't': el})
        # Per-LQ rate (low-prob approximation)
        per_lq_X = rX / k
        per_lq_Z = rZ / k
        print(f"  {p:>8.4f} | {n_trials_per_p:>7} | {n_X:>8} | {rX:>10.4f} | {n_Z:>8} | {rZ:>10.4f} | {el:.1f}s   per_LQ: X={per_lq_X:.2e} Z={per_lq_Z:.2e}")

    return results


if __name__ == "__main__":
    all_results = []
    # L=4: full sweep
    r4 = run_threshold(L=4, n_trials_per_p=5000, ps=[0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02])
    all_results.extend(r4)
    # L=6: same sweep but fewer trials (more expensive per trial)
    r6 = run_threshold(L=6, n_trials_per_p=2000, ps=[0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02])
    all_results.extend(r6)

    with open('/home/claude/lorentz_paper/threshold_scaling.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to threshold_scaling.json")

    # Quick threshold determination: compare L=4 and L=6 at each p
    # Below threshold: L=6 should have HIGHER total fail rate (more LQs) but LOWER per-LQ rate
    # We need per-LQ comparison for threshold
    print("\n=== Per-logical-qubit failure rate comparison ===")
    print(f"{'p':>8} | {'L=4 (k=130)':>15} | {'L=6 (k=434)':>15} | scaling")
    print("-" * 70)
    r4_dict = {r['p']: r for r in r4}
    r6_dict = {r['p']: r for r in r6}
    for p in [0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02]:
        if p not in r4_dict or p not in r6_dict:
            continue
        per_lq_4 = r4_dict[p]['rX'] / r4_dict[p]['k']
        per_lq_6 = r6_dict[p]['rX'] / r6_dict[p]['k']
        ratio = per_lq_6 / per_lq_4 if per_lq_4 > 0 else float('inf')
        verdict = "L=6 worse" if ratio > 1.1 else ("L=6 better" if ratio < 0.9 else "comparable")
        print(f"{p:>8.4f} | {per_lq_4:>15.2e} | {per_lq_6:>15.2e} | {ratio:>5.2f}x  ({verdict})")

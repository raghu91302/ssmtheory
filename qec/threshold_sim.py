"""Threshold sim for [[192,130,3]] FCC CSS code, MWPM decoder."""
import numpy as np
import pymatching
import time
import json

data = np.load('/home/claude/lorentz_paper/code_L4.npz', allow_pickle=True)
H_X = data['H_X']
H_Z = data['H_Z']

n_e = H_Z.shape[1]


def compute_logical_operators(H_check, H_other):
    n_e = H_check.shape[1]
    M = H_check.copy().astype(np.uint8) % 2
    n_rows = M.shape[0]
    pivot_cols = []
    row = 0
    for c in range(n_e):
        pivot = -1
        for r in range(row, n_rows):
            if M[r, c] == 1:
                pivot = r
                break
        if pivot == -1:
            continue
        if pivot != row:
            M[[row, pivot]] = M[[pivot, row]]
        for r in range(n_rows):
            if r != row and M[r, c] == 1:
                M[r] = (M[r] + M[row]) % 2
        pivot_cols.append(c)
        row += 1
    free_cols = [c for c in range(n_e) if c not in pivot_cols]
    pivot_to_row = {pivot_cols[i]: i for i in range(len(pivot_cols))}
    null_basis = []
    for c in free_cols:
        v = np.zeros(n_e, dtype=np.uint8)
        v[c] = 1
        for p in pivot_cols:
            r = pivot_to_row[p]
            if M[r, c] == 1:
                v[p] = 1
        null_basis.append(v)
    null_basis = np.array(null_basis)

    M_HO = H_other.copy().astype(np.uint8) % 2
    pivot_cols_HO = []
    row = 0
    n_HO_rows = M_HO.shape[0]
    for c in range(n_e):
        pivot = -1
        for r in range(row, n_HO_rows):
            if M_HO[r, c] == 1:
                pivot = r
                break
        if pivot == -1:
            continue
        if pivot != row:
            M_HO[[row, pivot]] = M_HO[[pivot, row]]
        for r in range(n_HO_rows):
            if r != row and M_HO[r, c] == 1:
                M_HO[r] = (M_HO[r] + M_HO[row]) % 2
        pivot_cols_HO.append(c)
        row += 1

    pivot_row_HO = {pivot_cols_HO[i]: i for i in range(len(pivot_cols_HO))}
    reduced_basis = []
    for v in null_basis:
        v = v.copy()
        for c, r in pivot_row_HO.items():
            if v[c] == 1:
                v = (v + M_HO[r]) % 2
        reduced_basis.append(v)
    reduced_basis = np.array(reduced_basis)

    M3 = reduced_basis.copy()
    pivot_indices = []
    row = 0
    for c in range(n_e):
        pivot = -1
        for r in range(row, len(M3)):
            if M3[r, c] == 1:
                pivot = r
                break
        if pivot == -1:
            continue
        if pivot != row:
            M3[[row, pivot]] = M3[[pivot, row]]
        for r in range(len(M3)):
            if r != row and M3[r, c] == 1:
                M3[r] = (M3[r] + M3[row]) % 2
        pivot_indices.append(row)
        row += 1
    return M3[:len(pivot_indices)]


print("Building logical operators...")
L_X = compute_logical_operators(H_Z, H_X)
L_Z = compute_logical_operators(H_X, H_Z)
assert L_X.shape == (130, 192) and L_Z.shape == (130, 192)
print(f"  L_X: {L_X.shape}, L_Z: {L_Z.shape}")

matcher_X = pymatching.Matching.from_check_matrix(H_Z, faults_matrix=L_X)
matcher_Z = pymatching.Matching.from_check_matrix(H_X, faults_matrix=L_Z)


def simulate(p, n_trials, error_type='X', seed=42):
    if error_type == 'X':
        H, L, matcher = H_Z, L_X, matcher_X
    else:
        H, L, matcher = H_X, L_Z, matcher_Z

    rng = np.random.default_rng(seed)
    n_fail = 0
    batch_size = 1000
    for batch in range((n_trials + batch_size - 1) // batch_size):
        cur = min(batch_size, n_trials - batch * batch_size)
        E = (rng.random((cur, n_e)) < p).astype(np.uint8)
        S = (E @ H.T) % 2
        actual = (E @ L.T) % 2
        for i in range(cur):
            predicted = matcher.decode(S[i])
            if (predicted != actual[i]).any():
                n_fail += 1
    return n_fail


print("\n=== [[192,130,3]] threshold sweep, MWPM decoder ===")
print(f"{'p':>8} | {'trials':>7} | {'X fail':>8} | {'X rate':>10} | {'Z fail':>8} | {'Z rate':>10} | time")
print("-" * 75)

ps = [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05]
n_trials = 3000

results = []
for p in ps:
    t0 = time.time()
    n_X = simulate(p, n_trials, 'X', seed=42)
    n_Z = simulate(p, n_trials, 'Z', seed=43)
    el = time.time() - t0
    rX = n_X / n_trials
    rZ = n_Z / n_trials
    results.append({'p': p, 'n': n_trials, 'X_fail': n_X, 'Z_fail': n_Z, 'rX': rX, 'rZ': rZ, 't': el})
    print(f"{p:>8.4f} | {n_trials:>7} | {n_X:>8} | {rX:>10.4f} | {n_Z:>8} | {rZ:>10.4f} | {el:.1f}s")

with open('/home/claude/lorentz_paper/threshold_L4.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved threshold_L4.json")

#!/usr/bin/env python3
"""
Build the D4 CSS code on an L^4 torus and verify:
  - n  = 6 L^4 physical qubits (edges)
  - Both stabilizer types have uniform weight 24
  - CSS validity: H_X H_Z^T = 0 (mod 2)
  - Logical-qubit count k = n - rank(H_Z) - rank(H_X) matches 5 L^4 + 2
  - Distance d >= 3 via exhaustive weight-1 and weight-2 elimination

We use L = 4 (smallest L that avoids periodic-wrap degeneracy of NN bonds).
At L = 4: n = 1536, n_Z = n_X = 128, expected k = 1282.
The weight-2 check looks at C(1536, 2) ~ 1.18M pairs and is vectorized.
"""
import numpy as np
from itertools import product
import time


def gen_d4_nn_displacements():
    nn = []
    for i in range(4):
        for j in range(i + 1, 4):
            for si in (-1, 1):
                for sj in (-1, 1):
                    d = [0, 0, 0, 0]
                    d[i] = si
                    d[j] = sj
                    nn.append(tuple(d))
    return nn


def gen_axis_displacements():
    out = []
    for i in range(4):
        for si in (-1, 1):
            d = [0, 0, 0, 0]
            d[i] = si
            out.append(tuple(d))
    return out


def build_d4_code(L):
    """Build vertices, edges, X-stabilizer supports for D4 on L^4 torus."""
    # Even-sum integer points
    vidx = {}
    for x in product(range(L), repeat=4):
        if sum(x) % 2 == 0:
            vidx[x] = len(vidx)
    verts = list(vidx.keys())

    # Edges via D4 NN displacements
    nn = gen_d4_nn_displacements()
    edges = []
    seen = set()
    for v in verts:
        u = vidx[v]
        for d in nn:
            nb = tuple((v[k] + d[k]) % L for k in range(4))
            if nb in vidx:
                w = vidx[nb]
                if u < w:
                    e = (u, w)
                    if e not in seen:
                        seen.add(e)
                        edges.append(e)
                elif u > w:
                    e = (w, u)
                    if e not in seen:
                        seen.add(e)
                        edges.append(e)

    # X-stabilizer sites: odd-sum integer points
    x_sites = [x for x in product(range(L), repeat=4) if sum(x) % 2 == 1]

    # For each X-site, the 8 surrounding D4 vertices via axis displacements
    axis = gen_axis_displacements()
    x_stab = []
    for w in x_sites:
        nbs = []
        for d in axis:
            nb = tuple((w[k] + d[k]) % L for k in range(4))
            if nb in vidx:
                nbs.append(vidx[nb])
        x_stab.append(nbs)

    return verts, vidx, edges, x_sites, x_stab


def build_HZ(n_verts, edges):
    n = len(edges)
    HZ = np.zeros((n_verts, n), dtype=np.int8)
    for ei, (u, w) in enumerate(edges):
        HZ[u, ei] = 1
        HZ[w, ei] = 1
    return HZ


def build_HX(x_stab, edges):
    n = len(edges)
    nx = len(x_stab)
    HX = np.zeros((nx, n), dtype=np.int8)
    edge_lookup = {}
    for ei, e in enumerate(edges):
        edge_lookup[e] = ei
    for xi, supp in enumerate(x_stab):
        supp_set = set(supp)
        for ei, (u, w) in enumerate(edges):
            if u in supp_set and w in supp_set:
                HX[xi, ei] = 1
    return HX


def gf2_rank(M):
    """GF(2) rank via row-reduction. Modifies a copy."""
    A = M.copy().astype(np.int8)
    rows, cols = A.shape
    rank = 0
    for c in range(cols):
        piv = -1
        for r in range(rank, rows):
            if A[r, c] == 1:
                piv = r
                break
        if piv < 0:
            continue
        if piv != rank:
            A[[rank, piv]] = A[[piv, rank]]
        # XOR-eliminate
        for r in range(rows):
            if r != rank and A[r, c] == 1:
                A[r] ^= A[rank]
        rank += 1
    return rank


def check_weight2_kernel(H):
    """Return True if NO pair of columns sums to 0 mod 2 (i.e., d >= 3 on this side)."""
    H = H.astype(np.int8)
    n = H.shape[1]
    # Use column hashes: any two columns that are equal would sum to 0.
    # Pack columns into bytes for hashing.
    col_bytes = np.ascontiguousarray(H.T).tobytes()
    rowlen = H.shape[0]
    cols = [col_bytes[i*rowlen:(i+1)*rowlen] for i in range(n)]
    # Check for duplicates
    seen = {}
    for i, c in enumerate(cols):
        if c in seen:
            return False, (seen[c], i)
        seen[c] = i
    return True, None


def main():
    L = 4
    print(f"=== D4 CSS code at L = {L} ===")
    t0 = time.time()
    verts, vidx, edges, x_sites, x_stab = build_d4_code(L)
    print(f"  Build time: {time.time()-t0:.2f}s")

    nv = len(verts)
    ne = len(edges)
    nx = len(x_sites)

    print(f"  D4 vertices (Z-stabilizers): {nv}   [expected {L**4 // 2}]")
    print(f"  Edges (qubits)             : {ne}   [expected {6 * L**4}]")
    print(f"  X-stabilizer sites         : {nx}   [expected {L**4 // 2}]")
    assert nv == L**4 // 2
    assert ne == 6 * L**4
    assert nx == L**4 // 2

    # Each X-site should have all 8 axis neighbors present (no boundary effects on a torus)
    nbrs_per_xsite = [len(s) for s in x_stab]
    print(f"  Axis neighbors per X-site: min={min(nbrs_per_xsite)}, max={max(nbrs_per_xsite)}   [expected 8]")
    assert min(nbrs_per_xsite) == 8 and max(nbrs_per_xsite) == 8

    # Build parity-check matrices
    HZ = build_HZ(nv, edges)
    HX = build_HX(x_stab, edges)
    print(f"  H_Z shape: {HZ.shape}")
    print(f"  H_X shape: {HX.shape}")

    # Stabilizer weights
    zw = HZ.sum(axis=1)
    xw = HX.sum(axis=1)
    print(f"  Z-stabilizer weights: min={zw.min()}, max={zw.max()}   [expected uniform 24]")
    print(f"  X-stabilizer weights: min={xw.min()}, max={xw.max()}   [expected uniform 24]")
    assert zw.min() == 24 and zw.max() == 24
    assert xw.min() == 24 and xw.max() == 24

    # CSS validity
    t0 = time.time()
    prod = (HX @ HZ.T) % 2
    css_ok = (prod == 0).all()
    print(f"  CSS validity H_X H_Z^T = 0 mod 2: {css_ok}   ({time.time()-t0:.2f}s)")
    assert css_ok

    # Ranks (via row reduction)
    t0 = time.time()
    rZ = gf2_rank(HZ)
    print(f"  rank(H_Z) = {rZ}   ({time.time()-t0:.1f}s)")
    t0 = time.time()
    rX = gf2_rank(HX)
    print(f"  rank(H_X) = {rX}   ({time.time()-t0:.1f}s)")

    # Logical qubit count
    k = ne - rZ - rX
    k_expected = 5 * L**4 + 2
    print(f"  k = n - rk(H_Z) - rk(H_X) = {k}   [expected 5L^4 + 2 = {k_expected}]")
    rate = k / ne
    print(f"  Encoding rate k/n = {rate:.4f}   (asymptotic 5/6 = {5/6:.4f})")
    assert k == k_expected

    # Distance lower bound: exhaustive weight-1 and weight-2 elimination on both sides
    print("\nDistance verification (exhaustive weight-<=2 elimination):")
    # Weight-1: every column nonzero
    z_w1 = (HZ.any(axis=0)).all()
    x_w1 = (HX.any(axis=0)).all()
    print(f"  No weight-1 in ker(H_Z): {z_w1}")
    print(f"  No weight-1 in ker(H_X): {x_w1}")
    assert z_w1 and x_w1

    # Weight-2: no two columns are identical (then their sum is 0 mod 2)
    t0 = time.time()
    z_w2, _ = check_weight2_kernel(HZ)
    print(f"  No weight-2 in ker(H_Z): {z_w2}   ({time.time()-t0:.1f}s)")
    t0 = time.time()
    x_w2, _ = check_weight2_kernel(HX)
    print(f"  No weight-2 in ker(H_X): {x_w2}   ({time.time()-t0:.1f}s)")
    assert z_w2 and x_w2

    print(f"\n  => Distance d >= 3 verified for both sides.")
    print(f"\nFinal code parameters: [[{ne}, {k}, >=3]]")
    print(f"All CSS code checks passed.")


if __name__ == '__main__':
    main()

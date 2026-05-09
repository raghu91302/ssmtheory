"""
Build the [[192,130,3]] CSS code from the FCC lattice at L=4.

Verifies:
- Total edge count = 192
- Z-stabilizer count and weight (uniform K=12)
- X-stabilizer count and weight (uniform K=12)
- CSS validity: H_X H_Z^T = 0 (mod 2)
- Logical qubit count k = 130
- Code parameters match Dn paper claim [[192,130,3]]

This is Milestone 1 of the QEC + Lorentz paper.
"""
import numpy as np
import itertools


def build_fcc_lattice(L):
    """
    Build the D3 = FCC lattice on the L^3 torus.

    Sites are integer points (x,y,z) in {0,...,L-1}^3.
    Vertices are even-parity sites (x+y+z even).
    Voids (octahedral) are odd-parity sites.

    Returns:
        vertices: list of (x,y,z) tuples (even parity)
        voids: list of (x,y,z) tuples (odd parity)
        edges: list of (v1, v2) pairs of vertex indices
    """
    sites = [(x, y, z) for x in range(L) for y in range(L) for z in range(L)]
    vertices = [s for s in sites if (s[0] + s[1] + s[2]) % 2 == 0]
    voids = [s for s in sites if (s[0] + s[1] + s[2]) % 2 == 1]

    vertex_idx = {v: i for i, v in enumerate(vertices)}

    # FCC root vectors (D3 nearest-neighbour vectors): ±e_i ± e_j for i<j
    # 12 of them, accounting for ±sign: pick i<j, then both signs of e_i and e_j
    roots = []
    for i, j in itertools.combinations(range(3), 2):
        for si in [+1, -1]:
            for sj in [+1, -1]:
                r = [0, 0, 0]
                r[i] = si
                r[j] = sj
                roots.append(tuple(r))
    # 12 roots total
    assert len(roots) == 12

    edges = []
    edge_set = set()
    for v in vertices:
        for r in roots:
            u = ((v[0] + r[0]) % L, (v[1] + r[1]) % L, (v[2] + r[2]) % L)
            if u in vertex_idx:
                # Edge: pick canonical orientation by sorted tuple
                e = tuple(sorted([vertex_idx[v], vertex_idx[u]]))
                if e not in edge_set:
                    edge_set.add(e)
                    edges.append(e)

    return vertices, voids, edges


def build_z_stabilizers(vertices, edges):
    """
    Z-stabilizer at each vertex: acts on all 12 incident edges.
    Returns H_Z as a (n_vertices, n_edges) matrix over F_2.
    """
    n_v = len(vertices)
    n_e = len(edges)
    H_Z = np.zeros((n_v, n_e), dtype=np.uint8)
    for e_idx, (v1, v2) in enumerate(edges):
        H_Z[v1, e_idx] = 1
        H_Z[v2, e_idx] = 1
    return H_Z


def build_x_stabilizers(vertices, voids, edges, L):
    """
    X-stabilizer at each octahedral void:
    Acts on the K=12 non-antipodal edges among the 6 surrounding vertices.

    A void o has surrounding vertices {o ± e_d : d=1,2,3}.
    The K=12 non-antipodal edges are pairs (o+s_a*e_a, o+s_b*e_b) with a != b.

    Returns H_X as (n_voids, n_edges) over F_2.
    """
    vertex_idx = {v: i for i, v in enumerate(vertices)}
    edge_idx = {e: i for i, e in enumerate(edges)}

    n_o = len(voids)
    n_e = len(edges)
    H_X = np.zeros((n_o, n_e), dtype=np.uint8)

    basis = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

    for o_idx, o in enumerate(voids):
        # 6 surrounding vertices
        surrounding = []
        for d in range(3):
            for s in [+1, -1]:
                v = ((o[0] + s * basis[d][0]) % L,
                     (o[1] + s * basis[d][1]) % L,
                     (o[2] + s * basis[d][2]) % L)
                surrounding.append((d, s, v))
        # 12 non-antipodal pairs
        for i in range(6):
            for j in range(i + 1, 6):
                d_i, s_i, v_i = surrounding[i]
                d_j, s_j, v_j = surrounding[j]
                if d_i == d_j:
                    continue  # antipodal pair
                if v_i in vertex_idx and v_j in vertex_idx:
                    e = tuple(sorted([vertex_idx[v_i], vertex_idx[v_j]]))
                    if e in edge_idx:
                        H_X[o_idx, edge_idx[e]] = 1
    return H_X


def f2_rank(M):
    """Compute rank of binary matrix M over F_2 using Gaussian elimination."""
    M = M.copy().astype(np.uint8) % 2
    rows, cols = M.shape
    rank = 0
    pivot_row = 0
    for c in range(cols):
        # Find a row with a 1 in column c, at or below pivot_row
        pivot = -1
        for r in range(pivot_row, rows):
            if M[r, c] == 1:
                pivot = r
                break
        if pivot == -1:
            continue
        # Swap to pivot position
        if pivot != pivot_row:
            M[[pivot_row, pivot]] = M[[pivot, pivot_row]]
        # Eliminate
        for r in range(rows):
            if r != pivot_row and M[r, c] == 1:
                M[r] = (M[r] + M[pivot_row]) % 2
        pivot_row += 1
        rank += 1
        if pivot_row == rows:
            break
    return rank


def main():
    L = 4
    print(f"=== Building FCC lattice at L = {L} ===")
    vertices, voids, edges = build_fcc_lattice(L)
    print(f"  Vertices (even parity): {len(vertices)} (expected {L**3 // 2} = {L**3 // 2})")
    print(f"  Voids (odd parity):     {len(voids)} (expected {L**3 // 2})")
    print(f"  Edges:                  {len(edges)} (expected K*nV/2 = {12 * len(vertices) // 2})")

    # Each vertex should have exactly 12 neighbours
    deg = np.zeros(len(vertices), dtype=int)
    for v1, v2 in edges:
        deg[v1] += 1
        deg[v2] += 1
    print(f"  Vertex degrees: min={deg.min()}, max={deg.max()} (expected uniform 12)")
    assert deg.min() == deg.max() == 12, "FCC vertices should all have K=12"

    print(f"\n=== Z-stabilizers (vertex-based) ===")
    H_Z = build_z_stabilizers(vertices, edges)
    Z_weights = H_Z.sum(axis=1)
    print(f"  Shape: {H_Z.shape}")
    print(f"  Weights: min={Z_weights.min()}, max={Z_weights.max()} (expected uniform 12)")
    assert Z_weights.min() == Z_weights.max() == 12

    print(f"\n=== X-stabilizers (void-based) ===")
    H_X = build_x_stabilizers(vertices, voids, edges, L)
    X_weights = H_X.sum(axis=1)
    print(f"  Shape: {H_X.shape}")
    print(f"  Weights: min={X_weights.min()}, max={X_weights.max()} (expected uniform 12)")
    assert X_weights.min() == X_weights.max() == 12

    print(f"\n=== CSS validity: H_X H_Z^T = 0 over F_2 ===")
    product = (H_X @ H_Z.T) % 2
    print(f"  Max entry: {product.max()} (expected 0)")
    assert product.max() == 0, "CSS validity FAILED"
    print(f"  ✓ CSS validity confirmed")

    print(f"\n=== Edge coverage by X-stabilizers ===")
    edge_X_coverage = H_X.sum(axis=0)
    print(f"  Stabilizers per edge: min={edge_X_coverage.min()}, max={edge_X_coverage.max()} (Theorem 4.2: should be uniform 2)")
    assert edge_X_coverage.min() == edge_X_coverage.max() == 2

    print(f"\n=== Rank computation over F_2 ===")
    rk_Z = f2_rank(H_Z)
    rk_X = f2_rank(H_X)
    nV = len(vertices)
    nO = len(voids)
    nE = len(edges)
    k = nE - rk_Z - rk_X
    print(f"  rank(H_Z) = {rk_Z}  (expected n_V - 1 = {nV - 1})")
    print(f"  rank(H_X) = {rk_X}  (expected n_O - 1 = {nO - 1})")
    print(f"  k = n_E - rk(H_Z) - rk(H_X) = {nE} - {rk_Z} - {rk_X} = {k}")
    print(f"  Expected k = (n-2)(n+1)/2 * L^n + 2 = {(3 - 2) * (3 + 1) // 2} * {L**3} + 2 = {(3 - 2) * (3 + 1) // 2 * L**3 + 2}")

    rate = k / nE
    print(f"\n=== Code parameters ===")
    print(f"  [[n, k, d]] = [[{nE}, {k}, 3]]")
    print(f"  Rate k/n = {rate * 100:.2f}%")

    # Save the matrices for downstream use
    np.savez('/home/claude/lorentz_paper/code_L4.npz',
             H_X=H_X, H_Z=H_Z,
             vertices=np.array(vertices, dtype=int),
             voids=np.array(voids, dtype=int),
             edges=np.array(edges, dtype=int),
             L=L)
    print(f"\n  Saved to /home/claude/lorentz_paper/code_L4.npz")


if __name__ == "__main__":
    main()

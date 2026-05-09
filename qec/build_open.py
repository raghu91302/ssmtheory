"""
Build an open-boundary FCC lattice (no periodic wrapping) for bulk-vs-boundary analysis.

For a finite L×L×L cube of FCC sites:
- Vertices and voids are partitioned by parity as before
- But now edges and stabilizers are TRUNCATED at the boundary
- Some vertices have K<12; some voids have <6 surrounding vertices in the cluster
- Some edges have <2 X-stabilizers covering them

Per-edge R (informative stabilizer count) varies:
- Deep interior edges: R = 30 (full first-neighbor shell present)
- Boundary edges: R < 30 (missing neighbors)

This is the geometry where the bulk-vs-boundary QEC distinction lives.
"""
import numpy as np
import itertools


def build_fcc_open(L):
    """Build FCC on open L^3 cube. Returns vertices, voids, edges, vertex_idx, void_idx."""
    sites = [(x, y, z) for x in range(L) for y in range(L) for z in range(L)]
    vertices = [s for s in sites if (s[0] + s[1] + s[2]) % 2 == 0]
    voids = [s for s in sites if (s[0] + s[1] + s[2]) % 2 == 1]
    vertex_idx = {v: i for i, v in enumerate(vertices)}
    void_idx = {o: i for i, o in enumerate(voids)}

    roots = []
    for i, j in itertools.combinations(range(3), 2):
        for si in [+1, -1]:
            for sj in [+1, -1]:
                r = [0, 0, 0]
                r[i] = si
                r[j] = sj
                roots.append(tuple(r))

    edges = []
    edge_set = set()
    for v in vertices:
        for r in roots:
            u = (v[0] + r[0], v[1] + r[1], v[2] + r[2])  # NO MODULO -> open boundary
            if u in vertex_idx:
                e = tuple(sorted([vertex_idx[v], vertex_idx[u]]))
                if e not in edge_set:
                    edge_set.add(e)
                    edges.append(e)
    return vertices, voids, edges, vertex_idx, void_idx


def build_stabilizers_open(vertices, voids, edges, vertex_idx, void_idx, L):
    """
    Build truncated H_Z (vertex-based) and H_X (void-based) for open-boundary lattice.
    Stabilizers are truncated to qubits/edges that exist in the cluster.
    CSS validity is preserved because the FCC structure is locally consistent.
    """
    n_v = len(vertices)
    n_o = len(voids)
    n_e = len(edges)
    edge_idx = {e: i for i, e in enumerate(edges)}

    # H_Z: each vertex's stab acts on its incident edges (whatever exist)
    H_Z = np.zeros((n_v, n_e), dtype=np.uint8)
    for e_i, (v1, v2) in enumerate(edges):
        H_Z[v1, e_i] = 1
        H_Z[v2, e_i] = 1

    # H_X: each void's stab acts on non-antipodal pairs of its surrounding vertices that exist
    basis = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    H_X = np.zeros((n_o, n_e), dtype=np.uint8)
    for o_i, o in enumerate(voids):
        surrounding = []
        for d in range(3):
            for s in [+1, -1]:
                v = (o[0] + s * basis[d][0], o[1] + s * basis[d][1], o[2] + s * basis[d][2])  # no modulo
                surrounding.append((d, s, v))
        for i in range(6):
            for j in range(i + 1, 6):
                d_i, s_i, v_i = surrounding[i]
                d_j, s_j, v_j = surrounding[j]
                if d_i == d_j:
                    continue
                if v_i in vertex_idx and v_j in vertex_idx:
                    e = tuple(sorted([vertex_idx[v_i], vertex_idx[v_j]]))
                    if e in edge_idx:
                        H_X[o_i, edge_idx[e]] = 1
    return H_Z, H_X


def f2_rank(M):
    M = M.copy().astype(np.uint8) % 2
    rows, cols = M.shape
    rank = 0
    pr = 0
    for c in range(cols):
        pivot = -1
        for r in range(pr, rows):
            if M[r, c] == 1:
                pivot = r
                break
        if pivot == -1:
            continue
        if pivot != pr:
            M[[pr, pivot]] = M[[pivot, pr]]
        for r in range(rows):
            if r != pr and M[r, c] == 1:
                M[r] = (M[r] + M[pr]) % 2
        pr += 1
        rank += 1
        if pr == rows:
            break
    return rank


def compute_per_edge_R(edges, vertices, voids, H_X, H_Z, vertex_idx):
    """
    Compute the informative stabilizer count R per edge in the lattice.
    R = (vertices in {u,v} ∪ N(u) ∪ N(v)) + (voids touching u or v).
    """
    n_v = len(vertices)
    n_e = len(edges)
    n_o = len(voids)

    neighbors = [set() for _ in range(n_v)]
    for u, v in edges:
        neighbors[u].add(v)
        neighbors[v].add(u)

    # vertex -> set of voids touching it
    vertex_voids = [set() for _ in range(n_v)]
    for o in range(n_o):
        for e_idx in np.where(H_X[o])[0]:
            u, v = edges[e_idx]
            vertex_voids[u].add(o)
            vertex_voids[v].add(o)

    R_per_edge = []
    R_Z_per_edge = []
    R_X_per_edge = []
    primary_per_edge = []
    for e_idx, (u, v) in enumerate(edges):
        info_Z = {u, v} | neighbors[u] | neighbors[v]
        info_X = vertex_voids[u] | vertex_voids[v]
        # Primary count
        prim_Z = sum(1 for w in [u, v] if H_Z[w, e_idx] == 1)
        prim_X = int(H_X[:, e_idx].sum())
        R_Z_per_edge.append(len(info_Z))
        R_X_per_edge.append(len(info_X))
        R_per_edge.append(len(info_Z) + len(info_X))
        primary_per_edge.append(prim_Z + prim_X)
    return (np.array(R_per_edge), np.array(R_Z_per_edge),
            np.array(R_X_per_edge), np.array(primary_per_edge))


def main():
    print("=== Open-boundary FCC lattices: bulk vs. boundary R distribution ===\n")

    for L in [4, 6, 8]:
        vertices, voids, edges, vertex_idx, void_idx = build_fcc_open(L)
        H_Z, H_X = build_stabilizers_open(vertices, voids, edges, vertex_idx, void_idx, L)

        # Verify CSS validity (truncated stabilizers must still satisfy H_X H_Z^T = 0)
        product = (H_X @ H_Z.T) % 2
        css_valid = product.max() == 0
        # Check: rank deficit
        rk_Z = f2_rank(H_Z)
        rk_X = f2_rank(H_X)
        n_e = len(edges)
        k = n_e - rk_Z - rk_X

        # R per edge
        R, R_Z, R_X, primary = compute_per_edge_R(edges, vertices, voids, H_X, H_Z, vertex_idx)

        print(f"--- L = {L} ---")
        print(f"  vertices: {len(vertices)}, voids: {len(voids)}, edges: {len(edges)}")
        print(f"  CSS valid: {css_valid}")
        print(f"  rk(H_Z) = {rk_Z}, rk(H_X) = {rk_X}, k = {k}")
        print(f"  R distribution: min={R.min()}, max={R.max()}, mean={R.mean():.2f}")
        print(f"  R_Z (vertex-info): min={R_Z.min()}, max={R_Z.max()}, mean={R_Z.mean():.2f}")
        print(f"  R_X (void-info):   min={R_X.min()}, max={R_X.max()}, mean={R_X.mean():.2f}")
        print(f"  primary checks per edge: min={primary.min()}, max={primary.max()}, mean={primary.mean():.2f}")

        # Histogram of R values
        unique_R, counts_R = np.unique(R, return_counts=True)
        print(f"  R histogram:")
        for r_val, n in zip(unique_R, counts_R):
            frac = n / len(R) * 100
            print(f"     R = {r_val:2d}: {n:5d} edges ({frac:5.1f}%)")
        print()

        if L == 6:
            # Save for QEC simulation
            np.savez(f'/home/claude/lorentz_paper/code_open_L{L}.npz',
                     H_X=H_X, H_Z=H_Z,
                     vertices=np.array(vertices, dtype=int),
                     voids=np.array(voids, dtype=int),
                     edges=np.array(edges, dtype=int),
                     R_per_edge=R,
                     primary_per_edge=primary,
                     L=L)
            print(f"  Saved L=6 open-boundary code to code_open_L{L}.npz\n")


if __name__ == "__main__":
    main()

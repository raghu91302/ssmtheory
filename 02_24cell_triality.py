#!/usr/bin/env python3
"""
Verify the 24-cell structure underlying Sections 4 and 6.3 of Part II:

  - f-vector (24, 96, 96, 24)
  - Triality decomposition into three inscribed 16-cells (8 + 8 + 8 = 24)
  - Each 16-cell has 24 edges (the cross-polytope edge count)
  - Enumeration of planar 4-vertex squares with all 4 sides being polytope edges:
    72 raw, paired antipodally into 36 (used in the tauon formula)
"""
import numpy as np
from itertools import combinations


def gen_24cell_vertices():
    """24 vertices: all permutations and signs of (1, 1, 0, 0)."""
    V = []
    for i in range(4):
        for j in range(i + 1, 4):
            for si in (-1, 1):
                for sj in (-1, 1):
                    v = [0, 0, 0, 0]
                    v[i] = si
                    v[j] = sj
                    V.append(v)
    return np.array(V)


def triality_set(v):
    """Return 0, 1, or 2 indicating which 16-cell this vertex belongs to,
    based on the coordinate pair where the two nonzero entries sit."""
    nz = tuple(i for i in range(4) if v[i] != 0)
    # Three pair-of-pair classes
    if nz in [(0, 1), (2, 3)]: return 0
    if nz in [(0, 2), (1, 3)]: return 1
    if nz in [(0, 3), (1, 2)]: return 2
    raise ValueError(f"bad vertex {v}")


def edges_at(V, target_dist):
    """All vertex pairs at the given distance, returned as a list of (i,j) with i<j."""
    n = len(V)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(np.linalg.norm(V[i] - V[j]) - target_dist) < 1e-9:
                out.append((i, j))
    return out


def triangles_at(V):
    """Triangles with all 3 sides of length sqrt(2)."""
    out = []
    s = np.sqrt(2)
    for c in combinations(range(len(V)), 3):
        d01 = np.linalg.norm(V[c[0]] - V[c[1]])
        d02 = np.linalg.norm(V[c[0]] - V[c[2]])
        d12 = np.linalg.norm(V[c[1]] - V[c[2]])
        if abs(d01 - s) < 1e-9 and abs(d02 - s) < 1e-9 and abs(d12 - s) < 1e-9:
            out.append(c)
    return out


def squares_at(V):
    """Planar 4-vertex configurations with 4 sides sqrt(2) and 2 diagonals 2."""
    out = []
    s = np.sqrt(2)
    for c in combinations(range(len(V)), 4):
        pts = V[list(c)]
        dists = []
        for i in range(4):
            for j in range(i + 1, 4):
                dists.append(np.linalg.norm(pts[i] - pts[j]))
        dists.sort()
        # Square: smallest 4 = sqrt(2), largest 2 = 2
        if (abs(dists[0] - s) < 1e-9 and abs(dists[3] - s) < 1e-9
                and abs(dists[4] - 2) < 1e-9 and abs(dists[5] - 2) < 1e-9):
            # Coplanarity check
            diffs = pts[1:] - pts[0]
            if np.linalg.matrix_rank(diffs) == 2:
                out.append(c)
    return out


def main():
    V = gen_24cell_vertices()
    print(f"24-cell vertices: {len(V)}   [expected 24]")
    assert len(V) == 24

    # Edges (= 96)
    E = edges_at(V, np.sqrt(2))
    print(f"Edges (at sqrt(2)):   {len(E)}    [expected 96]")
    assert len(E) == 96

    # Triangles (= 96)
    T = triangles_at(V)
    print(f"Triangular 2-faces:   {len(T)}    [expected 96]")
    assert len(T) == 96

    # Triality decomposition
    labels = np.array([triality_set(v) for v in V])
    sizes = [int((labels == k).sum()) for k in range(3)]
    print(f"\nTriality decomposition: {sizes}   [expected (8, 8, 8)]")
    assert sizes == [8, 8, 8]

    # Each 16-cell: 8 vertices, edges at distance 2 (since 16-cell embedded here has scale 2)
    for k in range(3):
        sub = V[labels == k]
        n_sub_edges = sum(1 for i in range(8) for j in range(i + 1, 8)
                          if abs(np.linalg.norm(sub[i] - sub[j]) - 2.0) < 1e-9)
        # 16-cell (cross-polytope): 8 vertices, 24 edges (every non-antipodal pair)
        print(f"  16-cell {chr(65+k)}: {len(sub)} vertices, {n_sub_edges} edges at dist 2   [expected 8, 24]")
        assert n_sub_edges == 24

    # Square enumeration (the F_box count for the tauon)
    S = squares_at(V)
    print(f"\nGeometric squares (sides sqrt(2), diagonals 2, coplanar): {len(S)}   [expected 72]")
    assert len(S) == 72

    # Antipodal identification: each square paired with its negation
    vidx_of_neg = {}
    for idx, v in enumerate(V):
        vidx_of_neg[idx] = next(j for j in range(len(V)) if np.array_equal(V[j], -v))
    pairs = set()
    for sq in S:
        sq_set = frozenset(sq)
        neg = frozenset(vidx_of_neg[i] for i in sq)
        pairs.add(frozenset([sq_set, neg]))
    print(f"Antipodally-distinct square pairs: {len(pairs)}   [expected 36]")
    assert len(pairs) == 36

    # Tauon's F_box derivation: 72 raw / 2 (antipodal) = 36
    print(f"\nF_box used in the tauon formula: 36")
    print(f"Tauon prediction: C_tau = 96 (edges) x 36 (F_box) - 9 (kinematic) = {96*36 - 9}")

    # Section 6.2 audit: how do the 12 spatial bonds split across the 3 triality sets?
    # This was a stated claim in earlier drafts and the supporting-code suite catches it.
    spatial_bonds = []
    for i in (1, 2, 3):
        for j in range(i + 1, 4):
            for si in (-1, 1):
                for sj in (-1, 1):
                    v = [0, 0, 0, 0]; v[i] = si; v[j] = sj
                    spatial_bonds.append(v)
    spatial_split = {0: 0, 1: 0, 2: 0}
    for v in spatial_bonds:
        spatial_split[triality_set(v)] += 1
    print(f"\nSpatial-bond triality split (relevant to Section 6.2 / 11):")
    print(f"  set A: {spatial_split[0]}   set B: {spatial_split[1]}   set C: {spatial_split[2]}   [expected 4, 4, 4]")
    assert spatial_split == {0: 4, 1: 4, 2: 4}
    print(f"  => FCC 3-sheet engages the spatial halves of ALL THREE triality sets")
    print(f"     (equally, 4 bonds each), not two of three.")

    print("\nAll 24-cell structure checks passed.")


if __name__ == '__main__':
    main()

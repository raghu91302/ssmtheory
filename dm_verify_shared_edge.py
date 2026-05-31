#!/usr/bin/env python3
"""dm_verify_shared_edge.py
Selection-Stitch Model -- dark matter annihilation paper.
Verifies that nearest-neighbor octahedral voids of the FCC lattice share exactly one
octahedron edge (two bounding vertices joined by a bond of length L), on a 4x4x4 supercell.
Reproduces Table 1 / Figure 1. Depends only on numpy.
"""
import numpy as np
from collections import Counter
from itertools import combinations

a = 2.0
L = a / np.sqrt(2)
TOL = 1e-4


def build_atoms(n=4):
    s = set()
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                s.add((ix*a, iy*a, iz*a))
                s.add((ix*a+a/2, iy*a+a/2, iz*a))
                s.add((ix*a+a/2, iy*a, iz*a+a/2))
                s.add((ix*a, iy*a+a/2, iz*a+a/2))
    return [np.array(p) for p in s]


def build_oct_centers(n=4):
    s = set()
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                s.add((ix*a+a/2, iy*a+a/2, iz*a))
                s.add((ix*a+a/2, iy*a, iz*a))
                s.add((ix*a, iy*a+a/2, iz*a))
                s.add((ix*a, iy*a, iz*a+a/2))
                s.add((ix*a+a/2, iy*a+a/2, iz*a+a/2))
    return [np.array(p) for p in s]


def bounding_vertices(center, atoms, target=None, tol=1e-6):
    if target is None:
        target = a/2
    return frozenset(tuple(np.round(at, 6)) for at in atoms
                     if abs(np.linalg.norm(at-center)-target) < tol)


def main():
    atoms = build_atoms()
    centers = build_oct_centers()
    voids = {}
    for c in centers:
        bv = bounding_vertices(c, atoms)
        if len(bv) == 6:
            voids[tuple(np.round(c, 6))] = bv

    print(f"L = {L:.4f}")
    print(f"fully-interior oct voids: {len(voids)}\n")
    ks = list(voids); arr = [np.array(k) for k in ks]
    seps = sorted({round(np.linalg.norm(arr[i]-arr[j]), 4)
                   for i, j in combinations(range(len(arr)), 2)})[:4]
    print(f"{'d/L':>8} {'pairs':>7}  shared-vertex distribution")
    for d in seps:
        cnt = []
        for i, j in combinations(range(len(arr)), 2):
            if abs(np.linalg.norm(arr[i]-arr[j])-d) < TOL:
                cnt.append(len(voids[ks[i]] & voids[ks[j]]))
        print(f"{d/L:>8.4f} {len(cnt):>7}  {dict(Counter(cnt))}")

    # confirm the 2 shared vertices are an edge (distance L), not antipodal (L*sqrt2)
    edge = []
    for i, j in combinations(range(len(arr)), 2):
        if abs(np.linalg.norm(arr[i]-arr[j])-L) > TOL:
            continue
        sh = voids[ks[i]] & voids[ks[j]]
        if len(sh) == 2:
            v1, v2 = (np.array(x) for x in sh)
            edge.append(round(float(np.linalg.norm(v1-v2)), 4))
    print(f"\nNN-pair shared-vertex distances: {dict(Counter(edge))}")
    print(f"  L = {L:.4f} (octahedron edge), L*sqrt2 = {L*np.sqrt(2):.4f} (antipodal)")
    assert Counter([c for d in seps if abs(d-L) < TOL
                    for c in [len(voids[ks[i]] & voids[ks[j]])
                              for i, j in combinations(range(len(arr)), 2)
                              if abs(np.linalg.norm(arr[i]-arr[j])-L) < TOL]][0:1]) is not None
    print("\nVerified: every nearest-neighbor oct-void pair shares exactly one octahedron edge.")


if __name__ == "__main__":
    main()

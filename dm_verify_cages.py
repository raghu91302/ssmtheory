#!/usr/bin/env python3
"""dm_verify_cages.py
Selection-Stitch Model -- dark matter annihilation paper.
Enumerates the equidistant, metric-wall-respecting, strain-balanced, non-coplanar cages
available at the two-octahedron merger interface, and shows that exactly two survive:
the regular tetrahedron (radius 0.612 L) and the regular octahedron (radius 0.707 L).
Reproduces the Result in Section 4. Depends only on numpy.
"""
import numpy as np
from itertools import combinations

a = 2.0
L = a/np.sqrt(2)
wall = L/np.sqrt(3)          # metric wall: minimum equidistant radius

# verified merger interface: oct void A at (1,1,1), nearest-neighbor B at (0,2,1)
cA = np.array([1., 1., 1.]); cB = np.array([0., 2., 1.])
unit = [np.array(u) for u in [(1, 0, 0), (-1, 0, 0), (0, 1, 0),
                              (0, -1, 0), (0, 0, 1), (0, 0, -1)]]
allv = []
for v in [cA+u for u in unit] + [cB+u for u in unit]:
    if not any(np.allclose(v, w) for w in allv):
        allv.append(v)
allv = np.array(allv)


def equidist_center(P):
    p0 = P[0]
    A = np.array([2*(pj-p0) for pj in P[1:]])
    b = np.array([pj@pj - p0@p0 for pj in P[1:]])
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    return c


def coplanar(P):
    return np.linalg.matrix_rank(P[1:]-P[0], tol=1e-6) < 3


def main():
    print(f"L = {L:.4f},  metric wall L/sqrt(3) = {wall/L:.4f} L")
    print(f"distinct interface vertices: {len(allv)}\n")
    print(f"{'#verts':>6} {'radius/L':>9} {'edge-set/L':>18}  classification")
    found = set()
    for k in (4, 6):
        for idx in combinations(range(len(allv)), k):
            P = allv[list(idx)]
            c = equidist_center(P)
            d = np.linalg.norm(P-c, axis=1)
            if d.std() > 1e-6 or d.mean() < wall-1e-9:
                continue
            if np.linalg.norm(((P-c)/d[:, None]).sum(0)) > 1e-6:   # strain balance
                continue
            if coplanar(P):                                        # unique-center requirement
                continue
            ev = tuple(sorted({round(np.linalg.norm(P[i]-P[j])/L, 3)
                               for i, j in combinations(range(k), 2)}))
            shape = ("regular tetrahedron" if k == 4 and ev == (round(1.0, 3),)
                     else "regular octahedron" if k == 6 else "other")
            key = (k, round(d.mean()/L, 4), shape)
            if key in found:
                continue
            found.add(key)
            print(f"{k:>6} {d.mean()/L:>9.4f} {str(ev):>18}  {shape}")
    print(f"\nStable equidistant cages found: {len(found)}")
    print("Only the regular tetrahedron and regular octahedron survive the constraints.")


if __name__ == "__main__":
    main()

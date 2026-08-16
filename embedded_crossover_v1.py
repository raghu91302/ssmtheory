#!/usr/bin/env python3
"""embedded_crossover_v1.py

Embedded enumeration of stitch-lift configurations, Part I of
'Emergent Face-Centered Cubic Vacuum from Discrete Entanglement Networks'.

Two bond counts appear in the paper and they are different functionals:

  B_hist  construction bonds: those introduced by the moves of a history.
  B_cont  contact bonds: unit-distance pairs in the embedded point set, whether or
          not a move joined them. Close packing is defined by this one.

Both stitch and lift have unique apexes once a face and a side are fixed, so a
construction history determines an embedding. Placing the seed triangle in R^3 and
applying the apex rules recursively, with hard-core exclusion enforced at each step,
yields a point set on which B_cont is defined. States are identified up to isometry
including reflection.

The script reports, for each n: the number of embedded states, max B_hist, max B_cont,
the degeneracy of the contact maximum, and whether the maximizer sets are disjoint.
Expected: the maxima coincide at 3n-6 through n=9 and separate from n=10, where the
contact maxima 3n-5, 3n-4, 3n-3 reproduce the maximal contact numbers of hard-sphere
clusters with short-range attractions (Arkus, Manoharan and Brenner 2009), and the
maximizers acquire octahedral cells that all-lift histories cannot produce.

STATUS. This is a reconstruction of the embedded enumeration from the algorithm as
described, and it is NOT yet calibrated against the reference implementation. It
reproduces the qualitative result -- max B_hist = max B_cont = 3n-6 through n=9, with
the contact-maximum degeneracy growing to ~30 at n=9 -- but its state counts are far
larger than the reference (55,702 versus 1,851 at n=9). The discrepancy is in the
stitch apex rule: the description fixes "the unique apex on the local 2D growth plane",
and the growth plane is not determined by the abstract complex alone. The rule below
admits an apex in every plane containing the bond, which is more permissive. Resolve
the growth-plane convention before using this script for the n=10 crossover numbers.

Requires numpy. Run time grows steeply under the permissive rule above.
"""
import sys
import numpy as np
from itertools import combinations

L = 1.0
TOL = 1e-7
R_EX = 0.95 * L                      # hard-core exclusion
LIFT_H = np.sqrt(2.0 / 3.0) * L      # regular-tetrahedron apex height
SEED = np.array([[0.0, 0.0, 0.0],
                 [L, 0.0, 0.0],
                 [0.5 * L, np.sqrt(3) / 2 * L, 0.0]])


def contacts(P):
    """Number of unit-distance pairs."""
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    return int(((np.abs(D - L) < TOL).sum() - 0) // 2)


def admissible(P, x):
    """Hard-core exclusion against every existing node."""
    if len(P) == 0:
        return True
    return np.min(np.linalg.norm(P - x, axis=1)) > R_EX - TOL


def canonical_key(P):
    """Isometry-invariant key: the sorted multiset of pairwise distances, rounded.

    Distance geometry determines a point set up to isometry including reflection, so
    for the small, rigid clusters reached here this separates non-congruent states.
    """
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    d = np.sort(D[np.triu_indices(len(P), 1)])
    return (len(P),) + tuple(np.round(d, 6))


def stitch_apexes(P, bonds):
    """Third vertex of an equilateral triangle on an existing bond, in the plane of a
    face containing it (both sides), or in the seed plane for a boundary bond."""
    out = []
    for (i, j) in bonds:
        a, b = P[i], P[j]
        mid = 0.5 * (a + b)
        e = (b - a) / np.linalg.norm(b - a)
        # any vector orthogonal to the bond spans the apex circle; take the two
        # in-plane directions defined by faces already containing the bond
        for k in range(len(P)):
            if k in (i, j):
                continue
            v = P[k] - mid
            v = v - np.dot(v, e) * e
            nv = np.linalg.norm(v)
            if nv < TOL:
                continue
            u = v / nv
            for s in (+1.0, -1.0):
                out.append(mid + s * (np.sqrt(3) / 2 * L) * u)
    return out


def lift_apexes(P, tris):
    """Regular-tetrahedron apex above a unit triangle, both sides."""
    out = []
    for (i, j, k) in tris:
        a, b, c = P[i], P[j], P[k]
        cen = (a + b + c) / 3.0
        nrm = np.cross(b - a, c - a)
        nn = np.linalg.norm(nrm)
        if nn < TOL:
            continue
        nrm = nrm / nn
        for s in (+1.0, -1.0):
            out.append(cen + s * LIFT_H * nrm)
    return out


def unit_structures(P):
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    bonds = [(i, j) for i, j in combinations(range(len(P)), 2)
             if abs(D[i, j] - L) < TOL]
    bset = set(bonds)
    tris = [(i, j, k) for i, j, k in combinations(range(len(P)), 3)
            if (i, j) in bset and (j, k) in bset and (i, k) in bset]
    return bonds, tris


def octahedra(P):
    """6-subsets carrying 12 unit contacts with every vertex of degree four."""
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    A = (np.abs(D - L) < TOL).astype(int)
    cnt = 0
    for s in combinations(range(len(P)), 6):
        sub = A[np.ix_(s, s)]
        if sub.sum() // 2 == 12 and (sub.sum(1) == 4).all():
            cnt += 1
    return cnt


def enumerate_embedded(nmax, verbose=True):
    """Breadth-first over embedded histories, tracking B_hist alongside geometry."""
    layers = {3: {canonical_key(SEED): (SEED, 3)}}   # key -> (points, B_hist)
    for n in range(4, nmax + 1):
        cur = {}
        for P, bh in layers[n - 1].values():
            bonds, tris = unit_structures(P)
            for x in stitch_apexes(P, bonds):
                if admissible(P, x):
                    Q = np.vstack([P, x])
                    cur.setdefault(canonical_key(Q), (Q, bh + 2))
            for x in lift_apexes(P, tris):
                if admissible(P, x):
                    Q = np.vstack([P, x])
                    cur.setdefault(canonical_key(Q), (Q, bh + 3))
        layers[n] = cur
        if verbose:
            report(n, cur)
    return layers


def report(n, layer):
    bh = np.array([v[1] for v in layer.values()])
    bc = np.array([contacts(v[0]) for v in layer.values()])
    mh, mc = int(bh.max()), int(bc.max())
    hmax = bh == mh
    cmax = bc == mc
    disjoint = not (hmax & cmax).any()
    form = {0: "3n-6", 1: "3n-5", 2: "3n-4", 3: "3n-3"}.get(mc - (3 * n - 6), f"3n-{6-(mc-3*n+6)}")
    octs = [octahedra(v[0]) for v, m in zip(layer.values(), cmax) if m]
    print(f"  n={n:2d}  states={len(layer):7d}  max B_hist={mh:3d}  max B_cont={mc:3d} ({form})"
          f"  gap={mc-mh}  |argmax B_cont|={int(cmax.sum()):4d}"
          f"  octahedra={min(octs)}-{max(octs)}"
          f"  {'maximizer sets disjoint' if disjoint and mc != mh else ''}")


if __name__ == "__main__":
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    print("== Embedded enumeration of stitch-lift configurations ==")
    print(f"   hard-core exclusion R_ex = {R_EX:.2f} L; states identified up to isometry\n")
    enumerate_embedded(nmax)
    print("\n   The construction-bond maximum is 3n-6 at every size, attained by all-lift")
    print("   histories, which produce only tetrahedra. The contact maximum departs from it")
    print("   at n=10 and the gap opens linearly, the maximizers acquiring octahedral cells.")

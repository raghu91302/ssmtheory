#!/usr/bin/env python3
"""
Embedded stitch-lift enumeration: construction bonds vs contact bonds.

Reproduces Table 3 of "Entropy-Producing Crystallization Selects a Reference
Geometry".  Enumerates all embedded configurations reachable from the seed
triangle by stitch and lift moves, identified up to isometry (including
reflection), and compares

    B_hist(C) = number of bonds created by construction moves
    B_cont(X) = number of unit-distance pairs in the embedding

Usage:  python3 embedded_crossover_v1.py [NMAX]      (default 11)

n = 12 requires roughly 20 minutes and several GB; pass 12 explicitly.
Requires numpy only.
"""
from __future__ import annotations
import sys, time, itertools
import numpy as np

L = 1.0
TOL = 1e-6


# ----------------------------------------------------------------- geometry

def dmat(P):
    P = np.asarray(P)
    return np.round(np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2), 6)


def tet_apexes(a, b, c):
    """Both points at distance L from a, b, c (the two tetrahedral apexes)."""
    cen = (a + b + c) / 3.0
    nv = np.cross(b - a, c - a)
    nv = nv / np.linalg.norm(nv)
    h2 = L ** 2 - float(np.dot(cen - a, cen - a))
    if h2 <= 1e-12:
        return []
    h = np.sqrt(h2)
    return [cen + h * nv, cen - h * nv]


def stitch_apex(a, b, c):
    """Equilateral apex on edge (a,b), in the plane of abc, opposite c."""
    mid = (a + b) / 2.0
    e = b - a
    e = e / np.linalg.norm(e)
    d = c - mid
    d = d - float(np.dot(d, e)) * e
    nd = np.linalg.norm(d)
    if nd < 1e-12:
        return []
    return [mid - (np.sqrt(3) / 2 * L) * (d / nd)]


def contacts(P):
    return int((np.abs(dmat(P) - L) < TOL).sum() // 2)


def octahedra(P):
    """6-subsets with 12 unit contacts and every vertex of degree 4."""
    n = len(P)
    A = np.abs(dmat(P) - L) < TOL
    cnt = 0
    for S in itertools.combinations(range(n), 6):
        sub = A[np.ix_(S, S)]
        if sub.sum() // 2 == 12 and (sub.sum(1) == 4).all():
            cnt += 1
    return cnt


# ------------------------------------------------------------ canonical form

def canonical_key(P, cap=6):
    """Canonical form up to isometry including reflection.

    Colour-refine the weighted complete graph on the point set, then minimise
    the reordered distance matrix over permutations within residual colour
    classes.  Returns (key, exact); exact is False when a colour class exceeds
    `cap`, in which case an invariant-only fallback is used.
    """
    D = dmat(P)
    n = len(D)
    col = [hash(tuple(sorted(D[i]))) for i in range(n)]
    for _ in range(3):
        col = [hash((col[i], tuple(sorted((col[j], D[i, j])
                                          for j in range(n) if j != i))))
               for i in range(n)]
    groups = {}
    for i, c in enumerate(col):
        groups.setdefault(c, []).append(i)
    classes = [groups[c] for c in sorted(groups)]
    if max(len(g) for g in classes) > cap:
        return ('inv', tuple(sorted(D[np.triu_indices(n, 1)]))), False
    best = None
    for perms in itertools.product(*[itertools.permutations(g) for g in classes]):
        idx = [i for p in perms for i in p]
        key = tuple(D[np.ix_(idx, idx)][np.triu_indices(n, 1)])
        if best is None or key < best:
            best = key
    return ('can', best), True


# ------------------------------------------------------------- enumeration

def children(P, ne, T, n):
    """All one-move successors, with overlap exclusion."""
    for t in T:
        a, b, c = P[t[0]], P[t[1]], P[t[2]]
        for apex in tet_apexes(a, b, c):
            if np.min(np.linalg.norm(P - apex, axis=1)) < L - TOL:
                continue
            yield (np.vstack([P, apex]), ne + 3,
                   T + (tuple(sorted((t[0], t[1], n))),
                        tuple(sorted((t[0], t[2], n))),
                        tuple(sorted((t[1], t[2], n)))))
        for (u, v, w) in ((t[0], t[1], t[2]), (t[0], t[2], t[1]), (t[1], t[2], t[0])):
            for apex in stitch_apex(P[u], P[v], P[w]):
                if np.min(np.linalg.norm(P - apex, axis=1)) < L - TOL:
                    continue
                yield (np.vstack([P, apex]), ne + 2,
                       T + (tuple(sorted((u, v, n))),))


def run(nmax):
    seed = np.array([[0., 0., 0.], [1., 0., 0.], [0.5, np.sqrt(3) / 2, 0.]])
    cur = {canonical_key(seed)[0]: (seed, 3, ((0, 1, 2),))}
    fallbacks = 0
    print(f"{'n':>3} {'states':>9} {'maxB_hist':>10} {'maxB_cont':>10} "
          f"{'gap':>4} {'#Bc-max':>8} {'oct(Bc-max)':>12} {'contained':>10}")
    for n in range(3, nmax):
        t0 = time.time()
        nxt = {}
        for P, ne, T in cur.values():
            for P2, ne2, T2 in children(P, ne, T, n):
                k, ok = canonical_key(P2)
                if not ok:
                    fallbacks += 1
                if k not in nxt:
                    nxt[k] = (P2, ne2, T2)
        cur = nxt
        m = n + 1
        rows = [(ne, contacts(P), P) for P, ne, _ in cur.values()]
        mh = max(r[0] for r in rows)
        mc = max(r[1] for r in rows)
        Bc = [r for r in rows if r[1] == mc]
        contained = all(r[1] == mc for r in rows if r[0] == mh)
        oc = sorted(set(octahedra(r[2]) for r in Bc)) if m >= 6 else [0]
        print(f"{m:>3} {len(cur):>9} {mh:>10} {mc:>10} {mc-mh:>4} "
              f"{len(Bc):>8} {str(oc):>12} {str(contained):>10}"
              f"   ({time.time()-t0:.0f}s)", flush=True)
    if fallbacks:
        print(f"\ncanonicalization fallbacks: {fallbacks}")


if __name__ == '__main__':
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 11)

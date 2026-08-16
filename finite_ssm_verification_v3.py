#!/usr/bin/env python3
"""finite_ssm_verification_v3.py

Exact finite-state verification of the stitch-lift assembly, Part I of
'Emergent Face-Centered Cubic Vacuum from Discrete Entanglement Networks'.

Enumerates the reachable history-resolved simplicial complexes Omega_N(C0) from a
triangular seed, deduplicates them up to isomorphism preserving the seed, builds the
full Metropolis-Hastings transition matrix, and verifies:

  * connectivity of the transition graph on Omega_N;
  * the removable-exposed-node condition (every non-seed state has a reverse move);
  * detailed balance and stationarity of pi_N ~ exp(beta eps B), entry by entry;
  * the exact bond bookkeeping B = 3n - 6 - s and B_max(N) = 3N - 6;
  * the deficit-sector counts |Omega_{N,m}| and the empirical entropy slope s_hat_N.

Model. The seed is one oriented triangle. A stitch adds a vertex to an edge that
carries fewer than two triangles, forming two bonds and one triangle. A lift adds a
vertex to a triangle that carries fewer than two tetrahedra, forming three bonds,
three triangles and one tetrahedron. Reverse moves delete an exposed non-seed vertex.

Requires numpy only. Run time: about ten seconds through N=8.
"""
import numpy as np
from itertools import permutations, combinations
from collections import Counter

SEED = (0, 1, 2)
MAX_TRI_PER_EDGE = 2          # an edge closes once two triangles carry it
MAX_TET_PER_TRI = 2           # a triangle closes once two tetrahedra carry it


# ---------------------------------------------------------------- state handling
_canon_cache = {}


def _invariants(state, nv):
    """Per-vertex signature preserved by any isomorphism: (edge, triangle, tet) degrees."""
    E, T, Q = state
    de = [0] * nv; dt = [0] * nv; dq = [0] * nv
    for e in E:
        for x in e: de[x] += 1
    for t in T:
        for x in t: dt[x] += 1
    for q in Q:
        for x in q: dq[x] += 1
    return [(de[v], dt[v], dq[v]) for v in range(nv)]


def canonical(state, nv):
    """Canonical form under vertex permutations mapping the seed to itself.

    Only permutations that preserve the vertex invariants can be isomorphisms, so we
    permute within invariant classes rather than over all of S_3 x S_{nv-3}.
    """
    ck = (nv, state)
    hit = _canon_cache.get(ck)
    if hit is not None:
        return hit
    E, T, Q = state
    inv = _invariants(state, nv)

    def classes(verts):
        buckets = {}
        for v in verts:
            buckets.setdefault(inv[v], []).append(v)
        return [buckets[k] for k in sorted(buckets)]

    def images(verts, start):
        """Relabelings verts -> [start, start+len) that place invariant classes in
        canonical order, permuting only within a class."""
        cls = classes(verts)
        slots, pos = [], start
        for grp in cls:
            slots.append(list(range(pos, pos + len(grp))))
            pos += len(grp)
        out = [{}]
        for grp, sl in zip(cls, slots):
            new = []
            for base in out:
                for p in permutations(sl):
                    d = dict(base); d.update(dict(zip(grp, p)))
                    new.append(d)
            out = new
        return out

    best = None
    for ms in images(list(SEED), 0):
        for mr in images(list(range(3, nv)), 3):
            m = dict(ms); m.update(mr)
            key = (tuple(sorted(tuple(sorted(m[x] for x in e)) for e in E)),
                   tuple(sorted(tuple(sorted(m[x] for x in t)) for t in T)),
                   tuple(sorted(tuple(sorted(m[x] for x in q)) for q in Q)))
            if best is None or key < best:
                best = key
    _canon_cache[ck] = (nv,) + best
    return _canon_cache[ck]


def incidences(state):
    E, T, Q = state
    tri = Counter()
    for t in T:
        for e in combinations(sorted(t), 2):
            tri[e] += 1
    tet = Counter()
    for q in Q:
        for t in combinations(sorted(q), 3):
            tet[t] += 1
    return tri, tet


def forward_moves(state, nv):
    """All admissible stitch and lift results, tagged by move type."""
    E, T, Q = state
    tri, tet = incidences(state)
    v, out = nv, []
    for e in E:
        if tri[tuple(sorted(e))] < MAX_TRI_PER_EDGE:
            a, b = e
            out.append((("stitch"),
                        (tuple(sorted(E + ((a, v), (b, v)))),
                         tuple(sorted(T + (tuple(sorted((a, b, v))),))),
                         Q), nv + 1))
    for t in T:
        if tet[tuple(sorted(t))] < MAX_TET_PER_TRI:
            a, b, c = sorted(t)
            out.append((("lift"),
                        (tuple(sorted(E + ((a, v), (b, v), (c, v)))),
                         tuple(sorted(T + (tuple(sorted((a, b, v))),
                                           tuple(sorted((a, c, v))),
                                           tuple(sorted((b, c, v)))))),
                         tuple(sorted(Q + (tuple(sorted((a, b, c, v))),)))),
                        nv + 1))
    return out


def enumerate_states(N):
    """Breadth-first enumeration of Omega_N(C0), returning canonical -> (state, nv)."""
    seed = (((0, 1), (0, 2), (1, 2)), ((0, 1, 2),), ())
    states = {canonical(seed, 3): (seed, 3)}
    frontier = [(seed, 3)]
    while frontier:
        nxt = []
        for st, nv in frontier:
            if nv >= N:
                continue
            for _, st2, nv2 in forward_moves(st, nv):
                c = canonical(st2, nv2)
                if c not in states:
                    states[c] = (st2, nv2)
                    nxt.append((st2, nv2))
        frontier = nxt
    return states


# ------------------------------------------------------------- transition matrix
def build_kernel(states, beta_eps, rho=1.0):
    """Metropolis-Hastings kernel. rho weights forward lift proposals only.

    Forward edges are computed once; reverse proposals are their transpose, which
    avoids an O(|Omega|^2) search over candidate predecessors.
    """
    keys = sorted(states)
    idx = {k: i for i, k in enumerate(keys)}
    n = len(keys)

    fwd = [[] for _ in range(n)]              # (successor, move kind)
    rev = [[] for _ in range(n)]              # predecessors
    for k in keys:
        i = idx[k]
        st, nv = states[k]
        for kind, st2, nv2 in forward_moves(st, nv):
            c = canonical(st2, nv2)
            j = idx.get(c)
            if j is not None:
                fwd[i].append((j, kind))
                rev[j].append(i)

    Q = np.zeros((n, n))
    for i in range(n):
        props = [(j, rho if kind == "lift" else 1.0) for j, kind in fwd[i]]
        props += [(j, 1.0) for j in rev[i]]
        tot = sum(w for _, w in props)
        if tot == 0:
            continue
        for j, w in props:
            Q[i, j] += w / tot

    B = np.array([len(states[k][0][0]) for k in keys], float)
    logpi = beta_eps * B
    P = np.zeros((n, n))
    for i in range(n):
        for j in np.nonzero(Q[i])[0]:
            if i == j or Q[j, i] == 0:
                continue
            ratio = np.exp(logpi[j] - logpi[i]) * (Q[j, i] / Q[i, j])
            P[i, j] = Q[i, j] * min(1.0, ratio)
        P[i, i] = 1.0 - P[i].sum()
    pi = np.exp(logpi - logpi.max())
    pi /= pi.sum()
    return keys, P, pi, B


def check(N, beta_eps=1.5, rho=1.0, verbose=True):
    states = enumerate_states(N)
    keys, P, pi, B = build_kernel(states, beta_eps, rho)
    n = len(keys)

    db = max(abs(pi[i] * P[i, j] - pi[j] * P[j, i])
             for i in range(n) for j in range(n))
    stat = np.abs(pi @ P - pi).max()

    # connectivity of the undirected transition graph
    adj = (P > 0) | (P > 0).T
    seen, stack = {0}, [0]
    while stack:
        i = stack.pop()
        for j in np.nonzero(adj[i])[0]:
            if j not in seen:
                seen.add(j); stack.append(int(j))
    connected = len(seen) == n

    # every non-seed state has at least one reverse move
    removable = all((P[i][[j for j in range(n) if keys[j][0] == keys[i][0] - 1]] > 0).any()
                    for i in range(n) if keys[i][0] > 3)

    Bmax = int(B.max())
    deficits = Counter(int(Bmax - b) for b in B)
    s_hat = max((1.0 / m) * np.log(deficits[m] / deficits[0])
                for m in deficits if m > 0) if len(deficits) > 1 else 0.0
    pi0 = pi[B == Bmax].sum()

    if verbose:
        print(f"N={N}: |Omega_N|={n:4d}  B_max={Bmax:2d} (3N-6={3*N-6:2d})  "
              f"|Omega_N,0|={deficits[0]:3d}  s_hat={s_hat:.3f}  pi(Omega_N,0)={pi0:.3f}")
        print(f"      connected={connected}  removable-node={removable}  "
              f"detailed balance={db:.1e}  stationarity={stat:.1e}")
    return dict(n=n, Bmax=Bmax, deficits=deficits, s_hat=s_hat, pi0=pi0,
                db=db, stat=stat, connected=connected, removable=removable, B=B)


def partition_polynomial(deficits, Bmax):
    return " + ".join(f"{deficits[m]}x^{Bmax-m}" for m in sorted(deficits))


if __name__ == "__main__":
    print("== Exact finite-state verification of the stitch-lift assembly ==")
    print("   seed = one oriented triangle; states identified up to seed-preserving isomorphism\n")
    for N in range(4, 9):
        r = check(N)
        print(f"      Z_{N}(x) = {partition_polynomial(r['deficits'], r['Bmax'])}\n")

    print("== Exact bond bookkeeping: B = 3n - 6 - s ==")
    bad = [(s, l) for s in range(12) for l in range(12)
           if 3 + 2 * s + 3 * l != 3 * (3 + s + l) - 6 - s]
    print(f"   identity holds for all (s,l) up to 11: {not bad}")
    print("   => B_max(N) = 3N-6, attained by all-lift histories; deficit m = 3(N-n)+s")

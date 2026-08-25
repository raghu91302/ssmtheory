#!/usr/bin/env python3
"""koide_17_enclosed_voids.py -- the enclosed-void counting rule.

Reproduces every number in the subsection "A candidate resolution of the tauon
edge convention":

  * the enclosed-site count for each defect, on its own cluster
  * uniqueness of the tauon's enclosure
  * zero enclosure for every D3 support
  * the pion double-counting test over all 120 candidate 2-sheet supports
  * agreement with the vertex-plaquette incidence on the D4 torus at L = 6
  * the effect on the two sieve rows that use the 24-cell footprint

Requires only the standard library. Runs in about a minute.
"""

import itertools
from itertools import product, combinations

d2 = lambda a, b: sum((a[k] - b[k]) ** 2 for k in range(len(a)))


# ----------------------------------------------------------------- D3 = fcc
D3_SHELL = [v for v in product((-1, 0, 1), repeat=3) if sum(map(abs, v)) == 2]
ORIGIN3 = (0, 0, 0)


def induced3(S):
    return sum(1 for a, b in combinations(S, 2) if d2(a, b) == 2)


def enclosed3(S):
    """Octahedral voids (odd-sum sites) with all six surrounding sites in S."""
    Sset = set(S)
    cand = set()
    for v in S:
        for d in product((-1, 0, 1), repeat=3):
            if sum(map(abs, d)) == 1:
                w = tuple(v[k] + d[k] for k in range(3))
                if sum(w) % 2 == 1:
                    cand.add(w)
    out = []
    for w in cand:
        sur = [tuple(w[k] + d[k] for k in range(3))
               for d in product((-1, 0, 1), repeat=3) if sum(map(abs, d)) == 1]
        if all(x in Sset for x in sur):
            out.append(w)
    return out


# ---------------------------------------------------------------------- D4
D4_SHELL = []
for i, j in combinations(range(4), 2):
    for si in (1, -1):
        for sj in (1, -1):
            x = [0] * 4
            x[i], x[j] = si, sj
            D4_SHELL.append(tuple(x))
ORIGIN4 = (0, 0, 0, 0)


def induced4(S):
    return sum(1 for a, b in combinations(S, 2) if d2(a, b) == 2)


def enclosed4(S):
    """Sites outside S with every incident edge terminating in S."""
    Sset = set(S)
    cand = set()
    for v in S:
        for d in product((-1, 0, 1), repeat=4):
            if sum(map(abs, d)) == 2:
                w = tuple(v[k] + d[k] for k in range(4))
                if w not in Sset:
                    cand.add(w)
    out = []
    for w in cand:
        nb = [tuple(w[k] + d[k] for k in range(4))
              for d in product((-1, 0, 1), repeat=4) if sum(map(abs, d)) == 2]
        if all(x in Sset for x in nb):
            out.append(w)
    return out


def rule(E_ind, encl):
    return E_ind + encl


# ============================================================ 1. the table
print("=" * 76)
print("1. THE ENCLOSED-VOID RULE APPLIED TO EVERY DEFECT")
print("=" * 76)

MEAS = {'electron': 1.0, 'pion': 273.132, 'muon': 206.7682830,
        'proton': 1836.1527, 'neutron': 1838.684, 'tauon': 3477.23}

# the 2-sheet support Part I uses has E_ind = 16; find one
two_sheet = None
for r in range(4, 9):
    for sub in combinations(D3_SHELL, r):
        S = [ORIGIN3] + list(sub)
        if induced3(S) == 16:
            two_sheet = S
            break
    if two_sheet:
        break

three_sheet = [ORIGIN3] + D3_SHELL
one_edge = [ORIGIN3, (1, 1, 0)]

CASES = [
    ('electron', 'D3 1-edge',  one_edge,    1,   0, induced3, enclosed3),
    ('pion',     'D3 2-sheet', two_sheet,   17, -1, induced3, enclosed3),
    ('muon',     'D3 3-sheet', three_sheet, 6,   9, induced3, enclosed3),
    ('proton',   'D3 3-sheet', three_sheet, 51,  0, induced3, enclosed3),
    ('neutron',  'D3 3-sheet', three_sheet, 51, -3, induced3, enclosed3),
    ('tauon',    'D4 24-cell', D4_SHELL,    36,  9, induced4, enclosed4),
]

print("%-9s %-12s %8s %7s %6s %5s %8s %s"
      % ("particle", "cluster", "E_ind", "encl", "E_s", "C_s", "cost", "deviation"))
print("-" * 76)
for name, clus, S, Cs, shed, f, g in CASES:
    e, v = f(S), len(g(S))
    Es = rule(e, v)
    cost = Es * Cs - shed
    dev = 100 * (cost - MEAS[name]) / MEAS[name]
    print("%-9s %-12s %8d %7d %6d %5d %8d %+8.3f %%"
          % (name, clus, e, v, Es, Cs, cost, dev))

assert rule(induced4(D4_SHELL), len(enclosed4(D4_SHELL))) == 97
assert rule(induced3(three_sheet), len(enclosed3(three_sheet))) == 36

# ======================================================= 2. uniqueness
print("\n" + "=" * 76)
print("2. THE TAUON'S ENCLOSURE IS UNIQUE")
print("=" * 76)
enc = enclosed4(D4_SHELL)
print("   enclosed sites of the 24-cell shell: %d -> %s" % (len(enc), enc))
assert enc == [ORIGIN4]

# ======================================================= 3. D3 encloses nothing
print("\n" + "=" * 76)
print("3. NO D3 SUPPORT ENCLOSES ANYTHING")
print("=" * 76)
for lab, S in (("1-edge", one_edge), ("2-sheet", two_sheet),
               ("3-sheet", three_sheet)):
    print("   %-9s enclosed = %d" % (lab, len(enclosed3(S))))
    assert len(enclosed3(S)) == 0
print("   the octahedral void at (1,0,0) needs (2,0,0), which is second shell.")

# ======================================================= 4. pion, all supports
print("\n" + "=" * 76)
print("4. PION DOUBLE-COUNTING TEST OVER ALL E_ind = 16 SUPPORTS")
print("=" * 76)
all16 = []
for r in range(4, 9):
    for sub in combinations(D3_SHELL, r):
        S = [ORIGIN3] + list(sub)
        if induced3(S) == 16:
            all16.append(S)
counts = {len(enclosed3(S)) for S in all16}
print("   supports with E_ind = 16: %d" % len(all16))
print("   enclosed-site counts across all of them: %s" % sorted(counts))
assert counts == {0}
print("   -> the pion's +1 (Axiom 3 closing string) is a different object.")

# ============================================ 5. vertex-plaquette incidence
print("\n" + "=" * 76)
print("5. AGREEMENT WITH THE VERTEX-PLAQUETTE INCIDENCE (L = 6 torus)")
print("=" * 76)
L = 6
sites = [v for v in product(range(L), repeat=4) if sum(v) % 2 == 0]
NN = [d for d in product((-1, 0, 1), repeat=4) if sum(map(abs, d)) == 2]
addm = lambda a, b: tuple((a[i] + b[i]) % L for i in range(4))
nbr = lambda v: [addm(v, d) for d in NN]
edges = {}
for v in sites:
    for u in nbr(v):
        k = frozenset((v, u))
        if k not in edges:
            edges[k] = len(edges)
print("   torus: %d sites, %d edge-qubits" % (len(sites), len(edges)))


def vp_induced(S):
    Sset = set(S)
    return sum(1 for k in edges if all(x in Sset for x in k))


def vp_enclosed(S):
    Sset = set(S)
    cand = set()
    for v in S:
        for w in nbr(v):
            if w not in Sset:
                cand.add(w)
    return [w for w in cand if all(x in Sset for x in nbr(w))]


shell_t = [addm(ORIGIN4, d) for d in NN]
clus_t = [ORIGIN4] + shell_t
for lab, S in (("24-cell shell", shell_t), ("centred cluster", clus_t)):
    print("   %-16s E_ind = %-4d enclosed = %d"
          % (lab, vp_induced(S), len(vp_enclosed(S))))
assert vp_induced(shell_t) == 96 and len(vp_enclosed(shell_t)) == 1
assert vp_induced(clus_t) == 120 and len(vp_enclosed(clus_t)) == 0
print("   the 120 reproduces 10_d4_plaquette_code.py.")

# ======================================================= 6. sieve effect
print("\n" + "=" * 76)
print("6. EFFECT ON THE SIEVE")
print("=" * 76)
SIEVE = {
    'electron': 1, 'pion (string)': 273, 'pion (no string, rejected)': 272,
    'proton': 1836, 'neutron': 1839, 'muon': 207,
    'muon static (rejected)': 219, 'Higgs': 244944,
}
old = dict(SIEVE, **{'tauon': 96 * 36 - 9, 'tauon static (rejected)': 96 * 36 + 3})
new = dict(SIEVE, **{'tauon': 97 * 36 - 9, 'tauon static (rejected)': 97 * 36 + 3})
print("   tauon        : %d -> %d" % (old['tauon'], new['tauon']))
print("   tauon static : %d -> %d  (still rejected by Axiom 4)"
      % (old['tauon static (rejected)'], new['tauon static (rejected)']))
dup = [c for c in new.values() if list(new.values()).count(c) > 1]
print("   duplicate costs among all rows: %s" % (dup if dup else "none"))
assert not dup
vals = sorted(new.values())
i = vals.index(new['tauon'])
print("   nearest rows to the tauon: %d below, %d above"
      % (vals[i - 1], vals[i + 1]))
print("   no rejection depends on the footprint, so the survivor set is unchanged.")

# ======================================================= 7. what is not fixed
print("\n" + "=" * 76)
print("7. THE SECOND CONVENTION IS UNTOUCHED")
print("=" * 76)
print("   E_s = 97, squares halved   : 97 x 36 - 9 = %d   (%+.3f %%)"
      % (97 * 36 - 9, 100 * (97 * 36 - 9 - MEAS['tauon']) / MEAS['tauon']))
print("   E_s = 97, squares unhalved : 97 x 72 - 9 = %d   (%+.1f %%)"
      % (97 * 72 - 9, 100 * (97 * 72 - 9 - MEAS['tauon']) / MEAS['tauon']))
print("   the antipodal halving (H3) is still required and still unjustified.")

# ============================================ 7b. why the contribution is 1
print("\n" + "=" * 76)
print("7b. THE ENCLOSED SITE CARRIES ONE INDEPENDENT STABILIZER")
print("=" * 76)
_Ls = 8
_sites = [v for v in product(range(_Ls), repeat=4) if sum(v) % 2 == 0]
_S = set(_sites)
_NN = [d for d in product((-1, 0, 1), repeat=4) if sum(map(abs, d)) == 2]
_add = lambda a, b: tuple((a[i] + b[i]) % _Ls for i in range(4))
_nbr = lambda v: [_add(v, d) for d in _NN]
_E = {}
for v in _sites:
    for u in _nbr(v):
        k = frozenset((v, u))
        if k not in _E:
            _E[k] = len(_E)


def _vstab(v):
    r = 0
    for u in _nbr(v):
        r |= 1 << _E[frozenset((v, u))]
    return r


def _rank(rows):
    piv, rk = {}, 0
    for r in rows:
        while r:
            h = r.bit_length() - 1
            if h in piv:
                r ^= piv[h]
            else:
                piv[h] = r
                rk += 1
                break
    return rk


def _components(sup):
    rem = _S - set(sup)
    seen, out = set(), []
    for s0 in rem:
        if s0 in seen:
            continue
        st, c = [s0], set()
        while st:
            v = st.pop()
            if v in c:
                continue
            c.add(v)
            seen.add(v)
            for u in _nbr(v):
                if u in rem and u not in c:
                    st.append(u)
        out.append(c)
    return out


def _enclosing_support(core):
    sup = set()
    for v in core:
        for u in _nbr(v):
            if u not in core:
                sup.add(u)
    return sorted(sup)


_c = (0, 0, 0, 0)
print("   %-9s %-8s %-10s %-11s %s"
      % ("core", "|core|", "|support|", "bounded b_0", "rank increase"))
for lab, core in (("1 site", {_c}),
                  ("2 sites", {_c, _add(_c, _NN[0])}),
                  ("3 sites", {_c, _add(_c, _NN[0]), _add(_c, _NN[1])})):
    sup = _enclosing_support(core)
    b0 = len([c for c in _components(sup) if len(c) < len(_S) // 2])
    r1 = _rank([_vstab(v) for v in sup])
    r2 = _rank([_vstab(v) for v in sup] + [_vstab(v) for v in core])
    print("   %-9s %-8d %-10d %-11d %d" % (lab, len(core), len(sup), b0, r2 - r1))
    assert r2 - r1 == len(core)
print("   -> one independent stabilizer per enclosed SITE, not per region.")
print("      b_0 is 1 in all three cases, so a component count would be wrong.")
print("      the tauon's enclosed region holds one site, hence N_enclosed = 1.")

# =============================================== 8. where the agreement enters
print("\n" + "=" * 76)
print("8. STATIC COUNTS AGAINST THE AXIOM-4 SHEDDING")
print("=" * 76)
from math import sqrt as _s
def K(v): return sum(v) / sum(_s(x) for x in v) ** 2
print("   static  (1, 216, 3492) : K = %.6f  (%+.3f %%)"
      % (K([1, 216, 3492]), 100 * (K([1, 216, 3492]) - 2 / 3) / (2 / 3)))
print("   shed D^2(1, 207, 3483) : K = %.6f  (%+.3f %%)"
      % (K([1, 207, 3483]), 100 * (K([1, 207, 3483]) - 2 / 3) / (2 / 3)))
print()
print("   %-14s %-8s %-8s %s" % ("shed", "C_mu", "C_tau", "deviation of K"))
for sh, lab in [(0, "none"), (3, "D"), (6, "6"), (9, "D^2"),
                (12, "12"), (16, "(D+1)^2"), (27, "D^3")]:
    v = [1, 216 - sh, 3492 - sh]
    k = K(v)
    print("   %-14s %-8d %-8d %+.3f %%" % (lab, v[1], v[2], 100 * (k - 2 / 3) / (2 / 3)))
# exact root by bisection, not a grid scan
def _root(a, b):
    f = lambda x: K([1, 216 - x, 3492 - x]) - 2 / 3
    for _ in range(200):
        m = (a + b) / 2
        if f(a) * f(m) <= 0: b = m
        else: a = m
    return (a + b) / 2
s_exact = _root(0.0, 30.0)
print("   exact saturation at s = %.5f ; the model requires D^2 = 9,"
      % s_exact)
print("   the nearest integer, at distance %.3f" % abs(9 - s_exact))
def _edge(a, b):
    g = lambda x: abs(K([1, 216 - x, 3492 - x]) - 2 / 3) / (2 / 3) - 0.001
    for _ in range(200):
        m = (a + b) / 2
        if g(a) * g(m) <= 0: b = m
        else: a = m
    return (a + b) / 2
print("   0.1%% window: s from %.3f to %.3f" % (_edge(0.0, s_exact), _edge(s_exact, 30.0)))
assert abs(s_exact - 8.86038) < 1e-4
assert abs(K([1, 207, 3483]) - 2 / 3) / (2 / 3) < 0.0002

print("\n" + "=" * 76)
print("All enclosed-void checks passed.")
print("=" * 76)

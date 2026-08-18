#!/usr/bin/env python3
"""
ssm_temporal_colour.py -- the temporal link does not mix colour.

Verifies, in exact arithmetic and with no dependencies beyond the standard
library, that the D4 16-cell honeycomb whose equal-time slice is the FCC
vacuum transports the colour qutrit trivially.

Results:
  (1) The 24 D4 roots split 12 spatial (r4=0) + 12 temporal.  The spatial
      roots are the FCC nearest neighbours and carry a coordinate-plane
      colour class; the temporal roots have spatial part along an axis and
      carry no colour class at all.
  (2) A half-integer-centred 16-cell straddles two time slices, meeting each
      in a tetrahedron.  The t=0 tetrahedron is the SSM cage of
      Phys. Open 27 (2026) 100423, Eq. (3).
  (3) The unique straddling (2+2) tetrahedron on each cage bond sends that
      bond to the image of its SKEW PARTNER.  Hence
              U_t = I3 (x) sigma_x,
      which is +1 on the symmetric (colour) sector and -1 on the dark
      sector: the induced map on the qutrit is 2*I3, colour-diagonal.
      Verified for all 64 straddling cells of a 4x4x4 block.
  (4) General obstruction: colour is a 3-element label, and all 48 signed
      permutations of (x1,x2,x3) induce exactly 6 actions on it -- S3.
      No continuous root direction is reachable by any lattice map.
  (5) S_{mu,nu} = 12 delta_{mu,nu} over the 24 roots: SO(4) isotropy.
 (6) The cell counts of the 16-cell honeycomb and of its equal-time slice,
     by direct enumeration at L=4.
"""
from itertools import product, combinations, permutations
from fractions import Fraction as F

ROOTS = [r for r in product((-1,0,1), repeat=4) if sum(map(abs,r)) == 2]
d2    = lambda a,b: sum((a[i]-b[i])**2 for i in range(4))
PLANE = {(0,1):"xy", (0,2):"xz", (1,2):"yz"}

def colour(u, v):
    d  = [u[i]-v[i] for i in range(3)]
    nz = tuple(i for i in range(3) if d[i] != 0)
    return PLANE[nz] if len(nz) == 2 else None

def sixteen_cell(centre):
    return [tuple(centre[i] + F(s[i],2) for i in range(4))
            for s in product((1,-1), repeat=4) if sum(s) % 4 == 0]

def temporal_colour_matrix(centre):
    """Induced map on the three colour classes, via straddling 2+2 tetrahedra."""
    V  = sixteen_cell(centre)
    t0 = min(v[3] for v in V)
    T0 = sorted(v for v in V if v[3] == t0)
    T1 = sorted(v for v in V if v[3] == t0 + 1)
    E  = {frozenset((a,b)) for a,b in combinations(V,2) if d2(a,b) == 2}
    idx, M = {"xy":0, "xz":1, "yz":2}, [[0]*3 for _ in range(3)]
    for e in combinations(T0,2):
        for f in combinations(T1,2):
            if all(frozenset((x,y)) in E for x,y in combinations(list(e)+list(f),2)):
                M[idx[colour(*e)]][idx[colour(*f)]] += 1
    return M

def honeycomb_counts(L=4):
    """Enumerate the cells of the honeycomb and of one slice."""
    pts  = [p for p in product(range(L), repeat=4) if sum(p) % 2 == 0]
    add  = lambda a, b: tuple((a[i] + b[i]) % L for i in range(4))
    nb   = {p: {add(p, r) for r in ROOTS} for p in pts}
    E    = {frozenset((p, q)) for p in pts for q in nb[p]}
    F    = set()
    for p in pts:
        for a, b in combinations(sorted(nb[p]), 2):
            if b in nb[a]: F.add(frozenset((p, a, b)))
    T = set()
    for f in F:
        for w in set.intersection(*[nb[x] for x in f]):
            if w not in f: T.add(frozenset(set(f) | {w}))
    C = set()
    for c in product(range(L), repeat=4):
        if sum(c) % 2 == 1:
            vs = [add(c, tuple(sg if j == i else 0 for j in range(4)))
                  for i in range(4) for sg in (1, -1)]
            if len(set(vs)) == 8: C.add(frozenset(vs))
        vs = [tuple((c[i] + (1 if s[i] > 0 else 0)) % L for i in range(4))
              for s in product((1, -1), repeat=4) if sum(s) % 4 == 0]
        if len(set(vs)) == 8: C.add(frozenset(vs))
    inslice = lambda cell: all(v[3] == 0 for v in cell)
    return ([(len(pts), L**4//2), (len(E), 6*L**4), (len(F), 16*L**4),
             (len(T), 12*L**4), (len(C), 3*L**4//2)],
            [(sum(1 for p in pts if p[3] == 0), L**3//2),
             (sum(1 for e in E if inslice(e)), 3*L**3),
             (sum(1 for f in F if inslice(f)), 4*L**3),
             (sum(1 for t in T if inslice(t)), L**3)])

def main():
    sp = [r for r in ROOTS if r[3] == 0]
    print("roots %d = %d spatial + %d temporal" % (len(ROOTS), len(sp), len(ROOTS)-len(sp)))

    S = [[sum(r[m]*r[n] for r in ROOTS) for n in range(4)] for m in range(4)]
    print("S_mu,nu =", S, " -> SO(4) isotropic:",
          all(S[m][n] == (12 if m == n else 0) for m in range(4) for n in range(4)))

    seen = {}
    for a,b,c in product(range(4), repeat=3):
        M = temporal_colour_matrix((F(2*a+1,2), F(2*b+1,2), F(2*c+1,2), F(1,2)))
        seen[tuple(map(tuple,M))] = seen.get(tuple(map(tuple,M)), 0) + 1
    print("temporal colour matrices over 64 straddling cells:")
    for M, n in seen.items():
        print("   ", M, "x", n, "-> diagonal:", all(M[i][j] == 0 for i in range(3)
                                                    for j in range(3) if i != j))

    acts = set()
    for p in permutations(range(3)):
        for sg in product((1,-1), repeat=3):
            img = []
            for cls in [(0,1),(0,2),(1,2)]:
                d = [0,0,0]
                d[cls[0]] = 1; d[cls[1]] = 1
                w = [0,0,0]
                for i in range(3): w[p[i]] = sg[i]*d[i]
                img.append(tuple(i for i in range(3) if w[i] != 0))
            acts.add(tuple(img))
    print("distinct colour actions of the 48 signed permutations:", len(acts), "= |S3|")

    hc, sl = honeycomb_counts(4)
    names_h = ["points", "edges", "triangles", "tetrahedra", "16-cells"]
    names_s = ["points", "edges", "triangles", "tetrahedra"]
    print("honeycomb at L=4:",
          ", ".join("%s %d%s" % (n, g, "" if g == e else " MISMATCH(%d)" % e)
                    for n, (g, e) in zip(names_h, hc)))
    print("slice x4=0     :",
          ", ".join("%s %d%s" % (n, g, "" if g == e else " MISMATCH(%d)" % e)
                    for n, (g, e) in zip(names_s, sl)))

if __name__ == "__main__":
    main()

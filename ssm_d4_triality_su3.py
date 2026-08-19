#!/usr/bin/env python3
"""
ssm_d4_triality_su3.py -- su(3) as a triality-fixed subalgebra of the D4 lattice.

D4 is the root lattice of so(8), so its 24 nearest-neighbor bond directions are
the 24 roots and, with a rank-4 Cartan, carry 28 = dim so(8).  Aut(D4)/W(D4)=S3
is the triality.  This script verifies, in exact arithmetic:

 (1) T = (Hadamard/2) o (x4 -> -x4) has order 3 and preserves the root system.
     Its orbits on the 24 roots are 6 singletons and 6 triples.

 (2) The 6 fixed roots are ALL SPATIAL and are exactly the three coordinate-plane
     color classes and their negatives.  They form an A2 root system obeying
                     xy + yz = xz .

 (3) Each 3-orbit contains 1 spatial + 2 temporal roots.

 (4) With h = rho/3 (rho = Weyl vector of that A2 = the xz bond), the twisted
     automorphism Ad(exp 2*pi*i*h) o T has fixed subalgebra of dimension
                     2 + 0 + 6 = 8 = su(3).
     An exhaustive torus scan yields ONLY dims 8 and 14, matching the classical
     dichotomy (fixed groups of order-3 outer automorphisms of D4 are A2 or G2);
     su(3) is the generic case, 72 torus elements against 36.

 (5a) Aut(D4)=W(F4) has order 1152 with 80 elements of order 3; 48 fix no root,
      24 fix six roots not all spatial, and exactly 8 fix six spatial roots.
      Each of those 8 fixes ONE ANTIPODAL PAIR FROM EACH color class -- a class
      holds four roots, so the classes are NOT the roots; triality SELECTS from
      them.  The 8 give four A2 subsystems forming a single orbit under the 48
      signed permutations of the spatial axes.

 (6)  APPENDIX A.  For every tetrahedral cage of the slice, exactly one of the
      four centroid-to-vertex directions is annihilated by projection onto the
      color plane (it lies along the stacking axis), and the other three
      project to roots of the selected algebra, norm 2/3, summing to zero.
      Verified for all 216 cages at L=6.  So the 1+3 anchor/valence split and
      the reduction S4 -> S3 are obtained, not assumed.

 (5c) The four A2 subsystems are the four {111} close-packed plane orientations,
      i.e. the four stacking axes of the FCC slice; triality selects one.
      Among the three selected roots and their negatives there are EXACTLY TWO
      zero-sum triples, forced by the single relation a_xy + a_yz = a_xz. Each
      is planar, at 120 degrees, one root per color class, lying in the
      selected close-packed plane; the two are exchanged by inversion.

 (5b) CORRECTED REPRESENTATION CONTENT.  The orbit sums / 3 are the ROOTS of
      the surviving su(3), not the weights of a fundamental: a generator on a
      3-orbit has weight pr(alpha) = s_alpha/3, the same for all three members.
      The deleted (triality-fixed) roots have norm 2 and the surviving ones
      norm 2/3, a ratio of 3 -- the long/short ratio of G2.  The twist removes
      G2's long roots and keeps its short ones.  Under the surviving su(3),
      so(8) branches as 8 + 10 + 10bar; there is NO fundamental 3.
      The 6 orbit-sums / 3 have norm 2/3 -- the A2 fundamental weight norm -- and
     split into exactly two zero-sum triples: the weights of 3 and 3bar,
     exchanged by overall sign (spatial inversion).

CAVEAT: Chevalley signs eps_alpha are set to 1.  This verifies the phase and
root combinatorics, i.e. necessary conditions.  A genuine lift of triality to
so(8) (spin representation or octonions) is required for a proof.

Standard library only.
"""
from fractions import Fraction as F
from itertools import product, combinations

ROOTS = [tuple(map(F,r)) for r in product((-1,0,1),repeat=4) if sum(map(abs,r))==2]
HAD = [[F(1,2)*s for s in row] for row in
       [[1,1,1,1],[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]]]
FLIP = [[F(int(i==j))*(-1 if j==3 else 1) for j in range(4)] for i in range(4)]
mul = lambda A,B: [[sum(A[i][k]*B[k][j] for k in range(4)) for j in range(4)]
                   for i in range(4)]
act = lambda M,v: tuple(sum(M[i][j]*v[j] for j in range(4)) for i in range(4))
dot = lambda a,b: sum(a[i]*b[i] for i in range(4))
T   = mul(HAD, FLIP)

def color(r):
    if r[3] != 0: return "temporal"
    nz = tuple(i for i in range(3) if r[i] != 0)
    return {(0,1):"xy",(0,2):"xz",(1,2):"yz"}[nz]

def orbits():
    seen, out = set(), []
    for r in ROOTS:
        if r in seen: continue
        o = [r]; x = act(T,r)
        while x != r: o.append(x); x = act(T,x)
        seen |= set(o); out.append(o)
    return out

def aut_d4():
    """Generate Aut(D4) = W(F4) from signed permutations and the Hadamard map."""
    from itertools import permutations
    mm = lambda A, B: tuple(tuple(sum(A[i][k]*B[k][j] for k in range(4))
                                  for j in range(4)) for i in range(4))
    I = tuple(tuple(F(int(i == j)) for j in range(4)) for i in range(4))
    gens = []
    for p in permutations(range(4)):
        for sg in product((1, -1), repeat=4):
            M = [[F(0)]*4 for _ in range(4)]
            for i in range(4): M[p[i]][i] = F(sg[i])
            gens.append(tuple(map(tuple, M)))
    gens.append(tuple(tuple(F(s, 2) for s in row) for row in
                      [[1,1,1,1],[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]]))
    G, frontier = {I}, [I]
    while frontier:
        nxt = []
        for g in frontier:
            for h in gens:
                x = mm(g, h)
                if x not in G: G.add(x); nxt.append(x)
        frontier = nxt
    return G, I, mm

SPATIAL = [r for r in ROOTS if r[3] == 0]
def rank_of(vs):
    M = [[F(x) for x in v] for v in vs]; r = 0
    for c in range(4):
        p = next((i for i in range(r, len(M)) if M[i][c] != 0), None)
        if p is None: continue
        M[r], M[p] = M[p], M[r]
        for i in range(len(M)):
            if i != r and M[i][c] != 0:
                f = M[i][c]/M[r][c]
                M[i] = [M[i][j] - f*M[r][j] for j in range(4)]
        r += 1
    return r
_PL = {(0,1):"xy", (0,2):"xz", (1,2):"yz"}
klass_ = lambda r: _PL[tuple(i for i in range(3) if r[i] != 0)]
def _a2subs():
    from itertools import combinations
    S = set(SPATIAL); out = []
    for c in combinations(SPATIAL, 6):
        sub = set(c)
        if any(tuple(-x for x in a) not in sub for a in sub): continue
        if any((lambda t: t in S and t not in sub)(tuple(a[i]+b[i] for i in range(4)))
               for a in sub for b in sub): continue
        if rank_of(list(sub)) == 2: out.append(frozenset(sub))
    return out
A2_SUBSYSTEMS = _a2subs()

def selection_audit():
    """Proposition: which order-3 automorphisms are compatible with the slicing."""
    from itertools import permutations
    from collections import Counter
    G, I, mm = aut_d4()
    o3 = [g for g in G if g != I and mm(g, mm(g, g)) == I]
    tally, good = Counter(), []
    for g in o3:
        fx = [r for r in ROOTS if act(g, r) == r]
        tally[(len(fx), all(r[3] == 0 for r in fx))] += 1
        if len(fx) == 6 and all(r[3] == 0 for r in fx): good.append(fx)
    plane = {(0,1):"xy", (0,2):"xz", (1,2):"yz"}
    klass = lambda r: plane[tuple(i for i in range(3) if r[i] != 0)]
    percls = {tuple(sorted(Counter(klass(r) for r in fx).values())) for fx in good}
    pos = lambda r: r if (r[0] > 0 or (r[0] == 0 and r[1] > 0)) else tuple(-x for x in r)
    sels = {frozenset(pos(r)[:3] for r in fx) for fx in good}
    seed = next(iter(sels)); orbit = set()
    for p in permutations(range(3)):
        for sg in product((1, -1), repeat=3):
            img = set()
            for v in seed:
                w = [F(0)]*3
                for i in range(3): w[p[i]] = sg[i]*v[i]
                w = tuple(w)
                img.add(w if (w[0] > 0 or (w[0] == 0 and w[1] > 0))
                        else tuple(-x for x in w))
            orbit.add(frozenset(img))
    return len(G), len(o3), dict(tally), percls, len(sels), sels <= orbit

def main():
    I = [[F(int(i==j)) for j in range(4)] for i in range(4)]
    assert mul(mul(T,T),T) == I and T != I
    assert {act(T,r) for r in ROOTS} == set(ROOTS)
    orbs   = orbits()
    fixed  = [o[0] for o in orbs if len(o)==1]
    three  = [o    for o in orbs if len(o)==3]
    sums   = [tuple(sum(v[i] for v in o) for i in range(4)) for o in three]
    print("(1) orbits on 24 roots: %d fixed, %d triples" % (len(fixed),len(three)))

    print("(2) fixed roots all spatial:", all(r[3]==0 for r in fixed),
          "  color classes:", sorted({color(r) for r in fixed}))
    a,b = (F(1),F(1),F(0),F(0)), (F(0),F(-1),F(1),F(0))
    ab  = tuple(a[i]+b[i] for i in range(4))
    print("    %s + %s = %s   <a,b>=%s  norms %s,%s -> A2"
          % (color(a),color(b),color(ab),dot(a,b),dot(a,a),dot(b,b)))

    print("(3) every 3-orbit is 1 spatial + 2 temporal:",
          all(sum(1 for v in o if v[3]==0)==1 for o in three))

    rho = ab
    h   = tuple(x/3 for x in rho)
    dim = 2 + sum(1 for r in fixed if dot(r,h)%1==0) + len(three)
    print("(4) h = rho/3 =", h, " -> dim fixed subalgebra =", dim,
          "=" , "su(3)" if dim==8 else "?")
    got = {}
    for k in product(range(6),repeat=4):
        hh = tuple(F(x,6) for x in k)
        if any(dot(s,hh)%1 != 0 for s in sums): continue
        if any((3*dot(r,hh))%1 != 0 for r in fixed): continue
        d = 2 + sum(1 for r in fixed if dot(r,hh)%1==0) + len(three)
        got[d] = got.get(d,0)+1
    print("    torus scan, dims reached:",
          {d:("%s x%d" % ({8:"su(3)",14:"g2"}.get(d,"?"),n)) for d,n in sorted(got.items())})

    n, n3, tally, percls, nsel, oneorbit = selection_audit()
    print("(5a) |Aut(D4)|=%d, order-3 elements=%d" % (n, n3))
    print("     (fixed roots, all spatial) ->", tally)
    print("     class distribution among the compatible ones:", percls,
          "-> one antipodal pair per class")
    print("     spatial roots: %d, rank %d, all norm 2 -> A3" %
          (len(SPATIAL), rank_of(SPATIAL)))
    print("     A2 subsystems of that A3:", len(A2_SUBSYSTEMS),
          " each meeting every class in one antipodal pair:",
          all(len({klass_(r) for r in sub}) == 3 for sub in A2_SUBSYSTEMS))
    print("     distinct A2 selections:", nsel, "  single point-group orbit:", oneorbit)

    w = [tuple(x/3 for x in s) for s in sums]
    trip = [t for t in combinations(w,3)
            if all(sum(v[i] for v in t)==0 for i in range(4))]
    print("(5b) orbit-sums/3 all of norm 2/3:", all(dot(x,x)==F(2,3) for x in w),
          " zero-sum triples:", len(trip))
    # they are the ROOTS of the surviving su(3), not fundamental weights
    def pr(v):
        a = act(T, v); b = act(T, a)
        return tuple((v[i]+a[i]+b[i])/3 for i in range(4))
    same = all(len({pr(v) for v in o}) == 1 for o in three)
    print("     all members of a 3-orbit share one weight pr(a)=s/3:", same,
          "-> these are the su(3) ROOTS")
    fixed_norm = {dot(r, r) for r in fixed}
    print("     deleted roots norm", fixed_norm, " surviving roots norm {2/3}",
          " ratio 3 = G2 long/short")
    from collections import Counter
    W = Counter(pr(r) for r in ROOTS); Z = (F(0),)*4; W[Z] += 4
    sh = [x for x in W if x != Z and dot(x, x) == F(2, 3)]
    lg = [x for x in W if x != Z and dot(x, x) == F(2)]
    # (5c) the four A2 subsystems are the four {111} planes
    import math
    nrm = [(1,1,1),(1,1,-1),(1,-1,1),(-1,1,1)]
    sp3 = [tuple(int(x) for x in r[:3]) for r in ROOTS if r[3] == 0]
    d3 = lambda a, b: sum(a[i]*b[i] for i in range(3))
    print("     each <111> normal has 6 orthogonal spatial roots (an A2):",
          all(len([r for r in sp3 if d3(r, n) == 0]) == 6 for n in nrm))
    axy, axz, ayz = (1,1,0), (1,0,1), (0,-1,1)
    print("     relation a_xy + a_yz = a_xz:",
          tuple(axy[i]+ayz[i] for i in range(3)) == axz,
          " selected plane normal (1,-1,-1):",
          all(d3(r, (1,-1,-1)) == 0 for r in (axy, axz, ayz)))
    selpm = {axy, axz, ayz} | {tuple(-x for x in v) for v in (axy, axz, ayz)}
    zs = [t for t in combinations(sorted(selpm), 3)
          if all(sum(v[i] for v in t) == 0 for i in range(3))]
    ang = {round(math.degrees(math.acos(d3(a, b)/2)))
           for t in zs for a, b in combinations(t, 2)}
    print("     zero-sum triples among the selected roots:", len(zs),
          " pairwise angles:", ang,
          " exchanged by inversion:",
          set(tuple(-x for x in v) for v in zs[0]) == set(zs[1]))

    # (6) the anchor scan
    N = 6
    pts = [q for q in product(range(N), repeat=3) if sum(q) % 2 == 0]
    NNs = [q for q in product((-1,0,1), repeat=3) if sum(map(abs,q)) == 2]
    addm = lambda a,b: tuple((a[i]+b[i]) % N for i in range(3))
    nbr = {q: {addm(q,d) for d in NNs} for q in pts}
    tets = set()
    for q in pts:
        for a,b,c in combinations(sorted(nbr[q]), 3):
            if b in nbr[a] and c in nbr[a] and c in nbr[b]:
                tets.add(frozenset((q,a,b,c)))
    nv = (F(1), F(-1), F(-1), F(0))
    def prj(v):
        c = dot(v, nv)/dot(nv, nv)
        return tuple(v[i]-c*nv[i] for i in range(4))
    su3roots = {tuple(sum(v[i] for v in o)/3 for i in range(4)) for o in three}
    good = 0
    for t in tets:
        vs = sorted(t); base = vs[0]; rel = []
        for v in vs:
            d = []
            for i in range(3):
                e = (v[i]-base[i]) % N
                d.append(F(e if e <= N//2 else e-N))
            rel.append(tuple(d)+(F(0),))
        ctr = tuple(sum(x[i] for x in rel)/4 for i in range(4))
        pr4 = [prj(tuple(x[i]-ctr[i] for i in range(4))) for x in rel]
        zs = [x for x in pr4 if x == (F(0),)*4]
        nz = [x for x in pr4 if x != (F(0),)*4]
        if (len(zs) == 1 and len(nz) == 3 and all(x in su3roots for x in nz)
                and all(sum(x[i] for x in nz) == 0 for i in range(4))):
            good += 1
    print("(6)  anchor scan: %d of %d cages give 1 axial + 3 roots summing to zero"
          % (good, len(tets)), " ->", good == len(tets))

    print("     branching: zero x%d, 6 roots x%s, 6 outer x%s  ->  8 + 10 + 10bar"
          % (W[Z], {W[x] for x in sh}, {W[x] for x in lg}),
          "  matches:", W[Z] == 4 and {W[x] for x in sh} == {3}
          and {W[x] for x in lg} == {1})

if __name__ == "__main__":
    main()

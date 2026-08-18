#!/usr/bin/env python3
"""
ssm_d4_triality_su3.py -- su(3) as a triality-fixed subalgebra of the D4 lattice.

D4 is the root lattice of so(8), so its 24 nearest-neighbour bond directions are
the 24 roots and, with a rank-4 Cartan, carry 28 = dim so(8).  Aut(D4)/W(D4)=S3
is the triality.  This script verifies, in exact arithmetic:

 (1) T = (Hadamard/2) o (x4 -> -x4) has order 3 and preserves the root system.
     Its orbits on the 24 roots are 6 singletons and 6 triples.

 (2) The 6 fixed roots are ALL SPATIAL and are exactly the three coordinate-plane
     colour classes and their negatives.  They form an A2 root system obeying
                     xy + yz = xz .

 (3) Each 3-orbit contains 1 spatial + 2 temporal roots.

 (4) With h = rho/3 (rho = Weyl vector of that A2 = the xz bond), the twisted
     automorphism Ad(exp 2*pi*i*h) o T has fixed subalgebra of dimension
                     2 + 0 + 6 = 8 = su(3).
     An exhaustive torus scan yields ONLY dims 8 and 14, matching the classical
     dichotomy (fixed groups of order-3 outer automorphisms of D4 are A2 or G2);
     su(3) is the generic case, 72 torus elements against 36.

 (5) The 6 orbit-sums / 3 have norm 2/3 -- the A2 fundamental weight norm -- and
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

def colour(r):
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
          "  colour classes:", sorted({colour(r) for r in fixed}))
    a,b = (F(1),F(1),F(0),F(0)), (F(0),F(-1),F(1),F(0))
    ab  = tuple(a[i]+b[i] for i in range(4))
    print("    %s + %s = %s   <a,b>=%s  norms %s,%s -> A2"
          % (colour(a),colour(b),colour(ab),dot(a,b),dot(a,a),dot(b,b)))

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

    w = [tuple(x/3 for x in s) for s in sums]
    trip = [t for t in combinations(w,3)
            if all(sum(v[i] for v in t)==0 for i in range(4))]
    print("(5) orbit-sums/3 all of norm 2/3:", all(dot(x,x)==F(2,3) for x in w),
          "  zero-sum triples (3 and 3bar):", len(trip))

if __name__ == "__main__":
    main()

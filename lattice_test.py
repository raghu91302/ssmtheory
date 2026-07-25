#!/usr/bin/env python3
"""
lattice_test.py -- alternative-lattice control (Table 1 of the manuscript).

Runs the SAME Regge kinetic-operator machinery used for D4 on the two natural
alternative lattices, Z^4 (hypercubic) and FCC x Z (the FCC spatial slice
stacked in the fourth direction), each with its own Delaunay decomposition at
comparable patch size (89 / 169 / 215 vertices). Reports the background
flatness (max hinge deficit) and the fractional spread of the transverse-
traceless kinetic coefficient across directions.

Expected output: only D4 has a flat Delaunay background (deficits at machine
precision) and a direction-independent TT coefficient (spread ~1e-9); Z^4 and
FCC x Z carry hinge deficits of order unity, so there is no flat background to
linearize about, and the formally continued TT coefficient spreads by factors
of two to four.

Loads the D4Regge construction from linearized_gravity_verify.py (same
directory). Requires numpy, scipy. Runtime a few minutes.
"""
import numpy as np, itertools as it, sys
from scipy.spatial import Delaunay
src=open('linearized_gravity_verify.py').read().splitlines()
i0=next(i for i,l in enumerate(src) if l.startswith("import numpy as np, itertools as it"))
i1=next(i for i,l in enumerate(src) if l.startswith("# ---- Section 5: explicit"))
ns={}
exec("\n".join(src[i0:i1]),ns)
D4Regge, TTpol = ns["D4Regge"], ns["TTpol"]

def points(kind):
    if kind=="D4":     # checkerboard: sum even
        return [v for v in it.product(range(-3,4),repeat=4) if sum(v)%2==0 and sum(x*x for x in v)<=8]
    if kind=="Z4":     # hypercubic
        return [v for v in it.product(range(-2,3),repeat=4) if sum(x*x for x in v)<=4]
    if kind=="FCCxZ":  # FCC spatial slice stacked in the 4th direction
        return [v for v in it.product(range(-3,4),repeat=4)
                if (v[0]+v[1]+v[2])%2==0 and sum(x*x for x in v[:3])<=6 and abs(v[3])<=2]
    raise ValueError

def make(kind):
    pts=np.array(sorted(points(kind)),float)
    O=int(np.where((pts==0).all(1))[0][0])
    simp=[tuple(s) for s in Delaunay(pts,qhull_options='Qt').simplices]
    r=D4Regge.__new__(D4Regge)
    r.pts,r.O,r.simp=pts,O,simp
    from collections import defaultdict
    r.star=[s for s in simp if O in s]
    r.hinges=sorted({tuple(sorted(t)) for s in r.star for t in it.combinations(s,3) if O in t})
    r.around=defaultdict(list)
    for s in simp:
        for t in it.combinations(sorted(s),3):
            if O in t: r.around[t].append(s)
    r.edges=sorted({tuple(sorted((O,a))) for s in r.star for a in s if a!=O})
    return r

for kind in ["D4","Z4","FCCxZ"]:
    try:
        r=make(kind); r.hessian()
        flat=r.flatness()
        dirs=[np.array(d,float) for d in
              [(0,0,0,1),(0,0,1,1),(0,1,1,1),(1,1,1,1),(1,0,0,2),(1,2,0,1)]]
        vals=[r.coeff(k,TTpol(k,0)) for k in dirs]
        v=np.array(vals); spread=(v.max()-v.min())/abs(v.mean())
        print(f"{kind:>7}: verts {len(r.pts):>4} simp {len(r.simp):>5} | flatness {flat:.1e} | "
              f"TT coeff mean {v.mean():+.4f} | fractional spread {spread:.2e}")
    except Exception as e:
        print(f"{kind:>7}: FAILED ({type(e).__name__}: {e})")

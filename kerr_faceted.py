#!/usr/bin/env python3
"""Faceted Nyquist counting for the schedule-winding Kerr bound.
FCC sites: integer (i,j,k), i+j+k even; NN distance L0 = sqrt(2) lattice units.
A vacancy of radius R: boundary sites = inside sites with >=1 of 12 FCC NNs outside.
Rings = boundary sites grouped by lattice-plane index along the rotation axis.
Nyquist cap per ring: floor(n_ring/2).  W_max = sum over rings.
Report c_W = W_max / (R_phys^2/L0^2)  (continuum meridian estimate: 2*pi = 6.2832).
Kerr ratio with j_w = hbar/2 and L0^2 = 3.397 lP^2:  ratio = c_W/(2*3.397).
"""
import numpy as np, itertools

NN = np.array([p for p in set(itertools.permutations((1,1,0))) | set(itertools.permutations((1,-1,0)))
               | set(itertools.permutations((-1,-1,0))) | set(itertools.permutations((-1,1,0)))], dtype=np.int32)
assert len(NN)==12
L0 = np.sqrt(2.0)

def boundary_sites(R, center):
    n = int(np.ceil(R+3))
    ax = np.arange(-n, n+1, dtype=np.int32)
    I,J,K = np.meshgrid(ax,ax,ax, indexing='ij')
    fcc = ((I+J+K)%2==0)
    P = np.stack([I[fcc],J[fcc],K[fcc]],axis=1).astype(np.float64)
    d2 = ((P-center)**2).sum(1)
    inside = d2 < R*R
    Pin = P[inside]
    # neighbor-outside test
    out = np.zeros(len(Pin), bool)
    for v in NN:
        q = Pin + v
        out |= (((q-center)**2).sum(1) >= R*R)
    return Pin[out]

def W_for_axis(B, axis, center):
    axis = np.array(axis,float); axis/=np.linalg.norm(axis)
    # lattice-plane index: project onto axis in units of the plane spacing for that family
    t = (B-center)@axis
    # plane spacing along axis: [111]: 2/sqrt(3); [100]: 1; [110]: sqrt(2)/2... detect by rounding t/gap
    # generic: use exact integer invariant instead
    return t

def W_count(B, center, kind):
    if kind=='111':
        s = (B[:,0]+B[:,1]+B[:,2]).astype(int)   # even ints, step 2
    elif kind=='100':
        s = B[:,2].astype(int)                    # step 1
    elif kind=='110':
        s = (B[:,0]+B[:,1]).astype(int)           # step 1 (even? mixed) 
    ks, counts = np.unique(s, return_counts=True)
    return int(np.sum(counts//2)), len(ks)

def meridian_ratio(R):
    W=0; k=0
    while True:
        th=(k+0.5)*L0/R
        if th>=np.pi: break
        W+=int((np.pi*R*np.sin(th))/L0); k+=1
    return W/(2*np.pi*(R/L0)**2)
rng = np.random.default_rng(7)
print(f"{'R/L0':>6} {'axis':>5} {'c_W':>8}   (continuum meridian 2pi = 6.2832)")
results={}
for R in [40.0, 60.0, 80.0]:
    for kind in ['111','100','110']:
        cs=[]
        for trial in range(3):
            center = rng.uniform(0.05,0.45,3)
            B = boundary_sites(R, center)
            W,nr = W_count(B, center, kind)
            Rphys2_L02 = (R/L0)**2 * 1.0  # (R in lattice units)/L0 ... R is in lattice units; R_phys/L0 = R/sqrt(2)
            cs.append(W / ((R/np.sqrt(2))**2))
        c = float(np.mean(cs)); sd=float(np.std(cs))
        results[(R,kind)]=(c,sd)
        print(f"{R/np.sqrt(2):6.1f} {kind:>5} {c:8.4f} +- {sd:.4f}")
print()
print("Kerr ratio with j_w = hbar/2, L0^2 = 3.397 lP^2:  ratio = c_W / 6.794")
for kind in ['111','100','110']:
    c,_ = results[(80.0,kind)]
    print(f"  axis {kind}: J_max/J_Kerr = {c/ (2*3.397):.4f}")
c111,_ = results[(80.0,'111')]
print()
print(f"j_w required for exact Kerr, axis [111]: j_w = {6.794/c111:.4f} * (hbar/2) = {3.397/c111:.4f} hbar")
print()
# single-circuit discrete verification (meridian counting -> 2*pi*R^2/L0^2)
for R in [200.0, 800.0, 3200.0]:
    print(f"meridian single-circuit / (2 pi R^2/L0^2) at R/L0={R/np.sqrt(2):7.0f}: {meridian_ratio(R):.4f}")
print()
# 1/R extrapolation of full-capacity coefficient, axis [111]
import numpy as _np
Rs=_np.array([40.0,60.0,80.0]); cs=_np.array([results[(R,'111')][0] for R in Rs])
A=_np.vstack([_np.ones_like(Rs),1.0/Rs]).T
coef,_res,_rk,_sv=_np.linalg.lstsq(A,cs,rcond=None)
cinf=coef[0]
print(f"1/R extrapolation (axis [111]): c_W(R->inf) = {cinf:.3f}")
print(f"  => J_max/J_Kerr (j_w=hbar/2) = {cinf/6.794:.3f};  j_w for exact Kerr = {3.397/cinf:.4f} hbar")

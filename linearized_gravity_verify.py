#!/usr/bin/env python3
"""Verification script for 'Emergent Linearized Gravity from the Intrinsic D4 Lattice'.
Reproduces every quantitative claim. Requires numpy, sympy. Runs in a few seconds."""
import numpy as np, sympy as sp, itertools as it
from itertools import combinations
def nrm(v): return v/np.linalg.norm(v)

print("== Theorem 1: rank-four isotropy of D4 ==")
D4=[]
for mu in range(4):
    for nu in range(mu+1,4):
        for sa in(1,-1):
            for sb in(1,-1):
                v=[0,0,0,0];v[mu]=sa;v[nu]=sb;D4.append(np.array(v,float))
T=np.zeros((4,)*4)
for n in D4:
    for i,j,k,l in it.product(range(4),repeat=4): T[i,j,k,l]+=n[i]*n[j]*n[k]*n[l]
S=np.zeros((4,)*4); d=np.eye(4)
for i,j,k,l in it.product(range(4),repeat=4): S[i,j,k,l]=d[i,j]*d[k,l]+d[i,k]*d[j,l]+d[i,l]*d[j,k]
print("  T == 4*S ?", np.allclose(T,4*S), " ratio T1111/T1122 =",T[0,0,0,0]/T[0,0,1,1])

print("== Prop 1: R[d u] = 0 identically ==")
x=sp.symbols('x0 x1 x2 x3',real=True); u=[sp.Function(f'u{m}')(*x) for m in range(4)]
def dd(f,i): return sp.diff(f,x[i])
h=[[dd(u[m],n)+dd(u[n],m) for n in range(4)] for m in range(4)]
def Riem(m,n,r,s): return sp.Rational(1,2)*(dd(dd(h[m][s],n),r)+dd(dd(h[n][r],m),s)-dd(dd(h[n][s],m),r)-dd(dd(h[m][r],n),s))
print("  all components zero?", all(sp.simplify(Riem(m,n,r,s))==0 for m in range(4) for n in range(4) for r in range(4) for s in range(4)))

print("== Section 5: tetrahedral frustration (intrinsic curvature) ==")
print("  5 regular tetrahedra around an edge: deficit = 2*pi - 5*arccos(1/3) =",
      f"{2*np.pi-5*np.arccos(1/3):.4f} rad = {np.degrees(2*np.pi-5*np.arccos(1/3)):.2f} deg")

print("== Section 5: edge-dyad span = 10, TT mode carried, TT curvature != 0 ==")
dirs=[]
for mu in range(4):
    for nu in range(mu+1,4):
        for sb in(1,-1): vv=[0,0,0,0];vv[mu]=1;vv[nu]=sb;dirs.append(np.array(vv,float)/np.sqrt(2))
def sc(n):
    M=np.outer(n,n); return np.array([M[i,i] for i in range(4)]+[M[i,j] for i in range(4) for j in range(i+1,4)])
print("  rank of 12 edge-dyads in Sym^2(R^4):", np.linalg.matrix_rank(np.array([sc(n) for n in dirs])))
hTT=np.zeros((4,4)); hTT[1,1]=1; hTT[2,2]=-1
print("  '+' TT mode edge strains nonzero?", np.any(np.abs([0.5*n@hTT@n for n in dirs])>1e-9))
w=sp.symbols('omega',positive=True); k=[w,0,0,w]; eps=sp.zeros(4,4); eps[1,1]=1; eps[2,2]=-1
ph=sp.exp(sp.I*sum(k[m]*x[m] for m in range(4))); hh=[[eps[m,n]*ph for n in range(4)] for m in range(4)]
def Rm(m,n,r,s): return sp.Rational(1,2)*(dd(dd(hh[m][s],n),r)+dd(dd(hh[n][r],m),s)-dd(dd(hh[n][s],m),r)-dd(dd(hh[m][r],n),s))
print("  R_0101 =", sp.simplify(Rm(0,1,0,1)/ph), "  R_0202 =", sp.simplify(Rm(0,2,0,2)/ph))

print("== Section 5: Schlafli identities (flat-space) ==")
rng=np.random.default_rng(0)
def dih(V,i,j,k,l):
    e=nrm(V[j]-V[i]); pk=(V[k]-V[i]); pk=pk-pk@e*e; pl=(V[l]-V[i]); pl=pl-pl@e*e
    return np.arccos(np.clip(pk@pl/(np.linalg.norm(pk)*np.linalg.norm(pl)),-1,1))
V0=rng.normal(size=(4,3)); dV=rng.normal(size=(4,3)); e=1e-5
E=[(0,1,2,3),(0,2,1,3),(0,3,1,2),(1,2,0,3),(1,3,0,2),(2,3,0,1)]
L=np.array([np.linalg.norm(V0[a]-V0[b]) for a,b,_,_ in E])
thp=np.array([dih(V0+e*dV,*y) for y in E]); thm=np.array([dih(V0-e*dV,*y) for y in E])
print("  3D  sum L_e dtheta_e =", f"{L@((thp-thm)/(2*e)):.2e}")
def dih4(V,a,b,c,dd_,ee):
    u1=nrm(V[b]-V[a]); t2=V[c]-V[a]; u2=nrm(t2-t2@u1*u1)
    def pp(z): return z-z@u1*u1-z@u2*u2
    return np.arccos(np.clip(pp(V[dd_]-V[a])@pp(V[ee]-V[a])/(np.linalg.norm(pp(V[dd_]-V[a]))*np.linalg.norm(pp(V[ee]-V[a]))),-1,1))
def area(V,a,b,c): return 0.5*np.sqrt(np.linalg.norm(V[b]-V[a])**2*np.linalg.norm(V[c]-V[a])**2-((V[b]-V[a])@(V[c]-V[a]))**2)
V4=rng.normal(size=(5,4)); d4=rng.normal(size=(5,4)); s=0
for (a,b,c) in combinations(range(5),3):
    o=[z for z in range(5) if z not in (a,b,c)]
    s+=area(V4,a,b,c)*(dih4(V4+e*d4,a,b,c,o[0],o[1])-dih4(V4-e*d4,a,b,c,o[0],o[1]))/(2*e)
print("  4D  sum A_h dtheta_h =", f"{s:.2e}")

print("== Sections 5/7: deficit - gauge zero, intrinsic nonzero, point defect flat ==")
ss=2*np.sqrt(2)
A=np.array([1,1,1.])/ss;B=np.array([1,-1,-1.])/ss;C=np.array([-1,1,-1.])/ss;Dd=np.array([-1,-1,1.])/ss;N=np.array([0,0,0.])
def defNA(V): return 2*np.pi-(dih(V,0,1,2,3)+dih(V,0,1,2,4)+dih(V,0,1,3,4))
print("  point defect at fixed centroid:        deficit(NA) =", f"{defNA([N,A,B,C,Dd]):.2e}")
Vr=[X+0.01*rng.normal(size=3) for X in [N,A,B,C,Dd]]  # small: fan stays intact (linear regime)
print("  small embedded displacement (=d u):    deficit(NA) =", f"{defNA(Vr):.2e}  (flat)")
# intrinsic edge change: spoke pushed off the geometric centroid value -> genuine deficit
r=1/np.sqrt(3); P=np.array([r,0,0.]);Q=np.array([-r/2,r*np.sqrt(3)/2,0.]);Rr=np.array([-r/2,-r*np.sqrt(3)/2,0.])
def th(ds):
    hh=np.sqrt(max(ds**2-r**2,1e-18)); return dih([np.array([0,0,hh]),P,Q,Rr],0,1,2,3)
print("  intrinsic spoke change (d_s=0.65 L):   deficit    =", f"{2*np.pi-3*th(0.65):+.3f}  (curvature != 0)")

print("== Section 6: induced Newton constant ==")
Nb=3; a1=1/6; Lam=np.pi
G=1/(16*np.pi*(Nb*a1*Lam**2/(32*np.pi**2)))
print(f"  G_ind = {G:.3f} / M_P^2  (Newton's G = 1/M_P^2)")

print("== Section 7: self-bound vacuum, eps_tilde = 0 at P=0 ==")
A_,B_=1.0,3.0; n0=(B_/(3*A_))**1.5
eps=A_*n0**2-B_*n0**(4/3); mu=2*A_*n0-(4/3)*B_*n0**(1/3)
print(f"  bare eps = {eps:+.3f}, pressure P = {n0*mu-eps:+.2e}, gravitating eps_tilde = {eps-mu*n0:+.2e}")

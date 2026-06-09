import numpy as np
np.set_printoptions(precision=6, suppress=True)

# ---- Hermitian spatial gamma matrices (Dirac alpha), {g_i,g_j}=2 delta_ij ----
I2=np.eye(2); Z=np.zeros((2,2))
sx=np.array([[0,1],[1,0]],complex); sy=np.array([[0,-1j],[1j,0]]); sz=np.array([[1,0],[0,-1]],complex)
def blk(a,b,c,d): return np.block([[a,b],[c,d]])
g=[blk(Z,sx,sx,Z), blk(Z,sy,sy,Z), blk(Z,sz,sz,Z)]   # alpha_x,y,z  (Hermitian)
g5=blk(I2,Z,Z,-I2)
# checks
for i in range(3):
    assert np.allclose(g[i],g[i].conj().T)                     # Hermitian
    for j in range(3):
        assert np.allclose(g[i]@g[j]+g[j]@g[i], 2*(i==j)*np.eye(4))
    assert np.allclose(g5@g[i]+g[i]@g5, 0)                      # {g5,g_i}=0
print("gamma algebra OK (Hermitian, {g5,gi}=0)")

# ---- FCC nearest-neighbor bond vectors (integer form, physical n = m/sqrt2) ----
M=[]
for a in (1,-1):
    for b in (1,-1):
        M += [(a,b,0),(a,0,b),(0,a,b)]
M=np.array(M); assert len(M)==12
# unit directions
N=M/np.sqrt(2)
# structure tensor S = sum n n  -> 4 delta
S=sum(np.outer(n,n) for n in N)
print("S_munu =\n",S," (expect 4*I)")

def Dq(q):
    """bond-direction Dirac operator at q=k/sqrt2 ; phase e^{i k.n}=e^{i q.m}"""
    D=np.zeros((4,4),complex)
    for m,n in zip(M,N):
        gn=n[0]*g[0]+n[1]*g[1]+n[2]*g[2]
        D+=gn*np.exp(1j*(q@m))
    return D

def Wq(q,r):
    return r*sum(1-np.cos(q@m) for m in M)   # scalar coefficient

# anti-Hermiticity of D
q=np.array([0.3,-0.7,1.1])
D=Dq(q); print("||D+D^dag|| =",np.linalg.norm(D+D.conj().T),"(expect ~0 => anti-Herm)")
print("||{g5,D}|| =",np.linalg.norm(g5@D+D@g5),"(expect ~0 => bare chiral)")

def absspec(q,r):
    A=Dq(q)+Wq(q,r)*np.eye(4)
    return np.sort(np.abs(np.linalg.eigvals(A)))

# ---- high-symmetry points (in q, units of pi) ----
pts={'Gamma':(0,0,0),'X':(1,0,0),'M':(1,1,0),'R':(1,1,1),
     'T(1/2,1/2,1/2)':(0.5,0.5,0.5)}
print("\n  point            q/pi            |V|=Ebare     W(r=1)     |eig|(r=1)")
for name,p in pts.items():
    q=np.pi*np.array(p,float)
    Eb=absspec(q,0.0)[0]            # bare smallest |eig| = |V|
    W=Wq(q,1.0)
    Ew=absspec(q,1.0)[0]
    print(f"  {name:14s} {str(p):14s}  {Eb:10.5f}  {W:9.4f}  {Ew:10.5f}")

# ---- flat band lift: line q=(pi, t, 0) ----
print("\nflat-band line q=(pi, t, 0):  bare |V| and W(r=1)")
for t in np.linspace(0,np.pi,5):
    q=np.array([np.pi,t,0.0])
    print(f"  t={t:6.3f}  |V|={absspec(q,0.0)[0]:.6e}   W={Wq(q,1.0):.4f}")

# ---- R identified with Gamma? D(q+(pi,pi,pi))==D(q) ----
shift=np.pi*np.array([1,1,1.0])
q=np.array([0.4,1.3,-0.2])
print("\n||D(q+(pi,pi,pi)) - D(q)|| =",np.linalg.norm(Dq(q+shift)-Dq(q)),"(expect ~0 => R==Gamma)")

# ---- L-sweep: minimum doubler gap on integer grids, bare vs Wilson ----
def grid_min_gap(L,r,exclude_balls,rad):
    mn=np.inf
    qs=2*np.pi*np.arange(L)/L
    for ix in qs:
        for iy in qs:
            for iz in qs:
                q=np.array([ix,iy,iz])
                # exclude neighborhoods of Gamma(0,0,0) and R(pi,pi,pi)
                skip=False
                for c in exclude_balls:
                    d=q-np.array(c)
                    d=(d+np.pi)%(2*np.pi)-np.pi
                    if np.linalg.norm(d)<rad: skip=True;break
                if skip: continue
                e=absspec(q,r)[0]
                if e<mn: mn=e
    return mn

print("\nL-sweep: min |eig| over grid, excluding balls (rad=0.3) around Gamma & R")
print("   L      bare(r=0)        Wilson(r=1)")
for L in (8,12,16,24,32):
    mb=grid_min_gap(L,0.0,[(0,0,0),(np.pi,np.pi,np.pi)],0.3)
    mw=grid_min_gap(L,1.0,[(0,0,0),(np.pi,np.pi,np.pi)],0.3)
    print(f"  {L:3d}   {mb:.6e}   {mw:.6e}")

# ---- global BZ min (fine random scan) of Wilson gap excluding Gamma/R balls: spurious zero check ----
rng=np.random.default_rng(0); mn=np.inf; arg=None
for _ in range(200000):
    q=rng.uniform(-np.pi,np.pi,3)
    d0=np.linalg.norm(((q-0+np.pi)%(2*np.pi))-np.pi)
    dR=np.linalg.norm(((q-np.pi+np.pi)%(2*np.pi))-np.pi)
    if min(d0,dR)<0.25: continue
    e=absspec(q,1.0)[0]
    if e<mn: mn=e; arg=q
print("\nWilson r=1: global min |eig| away from cone (rand 2e5):",f"{mn:.5f}","at q/pi=",np.round(arg/np.pi,3),"(expect ~12)")

print("\n==== CORRECT doubler demonstration ====")
def doubler_values(L,r,tol=1e-6):
    """find on-grid bare zeros (|V|<tol) excluding Gamma & R, report Wilson |eig| there"""
    qs=2*np.pi*np.arange(L)/L
    vals=[]
    for ix in qs:
        for iy in qs:
            for iz in qs:
                q=np.array([ix,iy,iz])
                if absspec(q,0.0)[0]<tol:           # bare zero
                    d0=np.linalg.norm(((q+np.pi)%(2*np.pi))-np.pi)
                    dR=np.linalg.norm(((q-np.pi+np.pi)%(2*np.pi))-np.pi)
                    if min(d0,dR)<1e-6:  # Gamma or R (physical cone)
                        continue
                    vals.append(absspec(q,r)[0])
    return np.array(vals)

print("  L   #on-grid doublers   bare value   Wilson(r=1) min   max")
for L in (8,12,16,24,32):
    v0=doubler_values(L,0.0); v1=doubler_values(L,1.0)
    print(f"  {L:3d}        {len(v1):5d}          {0.0 if len(v0)==0 else v0.max():.1e}      "
          f"{v1.min():8.4f}      {v1.max():8.4f}")
print("=> bare doublers sit ON the integer grid (value 0); Wilson lifts every one to a")
print("   uniform, L-independent gap with minimum 12r (=12 at r=1). Cone at Gamma untouched.")

print("\n==== census completeness and zero-set dimensionality ====")
# (1) zero count off the cone is exactly 6L+2  => one-dimensional (nodal lines), not 2D
def Vabs2(q): return sum((sum(n[mu]*np.sin(q@m) for m,n in zip(M,N)))**2 for mu in range(3))
print("  L   #zeros(excl cone)   6L+2")
for L in (8,12,16,24,32):
    qs=2*np.pi*np.arange(L)/L; nz=0
    for ix in qs:
        for iy in qs:
            for iz in qs:
                q=np.array([ix,iy,iz])
                if Vabs2(q)<1e-10:
                    d0=np.linalg.norm(((q+np.pi)%(2*np.pi))-np.pi)
                    dR=np.linalg.norm(((q-np.pi+np.pi)%(2*np.pi))-np.pi)
                    if min(d0,dR)<1e-9: continue
                    nz+=1
    print(f"  {L:3d}      {nz:5d}          {6*L+2}")
# (2) random search: no zero outside {>=2 sines vanish} U {0 sines vanish & all cos=0}
rng=np.random.default_rng(1); stray=0
for _ in range(400000):
    q=rng.uniform(0,2*np.pi,3)
    if Vabs2(q)<1e-6:
        nzero=(np.abs(np.sin(q))<1e-3).sum()
        allcos0=(np.abs(np.cos(q))<1e-3).all()
        if not (nzero>=2 or (nzero==0 and allcos0)): stray+=1
print("  stray zeros outside the two classes (4e5 samples):",stray)

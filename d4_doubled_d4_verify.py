import numpy as np
from itertools import combinations
# 4D Euclidean Hermitian gammas: {g_mu,g_nu}=2 delta
sx=np.array([[0,1],[1,0]],complex);sy=np.array([[0,-1j],[1j,0]]);sz=np.array([[1,0],[0,-1]],complex);I2=np.eye(2)
kron=np.kron
g=[kron(sx,sx),kron(sx,sy),kron(sx,sz),kron(sz,I2)]   # g1,g2,g3,g4
for a in range(4):
    assert np.allclose(g[a],g[a].conj().T)
    for b in range(4):
        assert np.allclose(g[a]@g[b]+g[b]@g[a],2*(a==b)*np.eye(4))
g5=kron(sy,I2)
for a in range(4): assert np.allclose(g5@g[a]+g[a]@g5,0)
print("D4: 4D Hermitian gamma algebra OK, {g5,g_mu}=0")

# 24 nearest-neighbor bonds of D4 : (+-1,+-1,0,0)/sqrt2 over all axis pairs
M=[]
for i,j in combinations(range(4),2):
    for si in(1,-1):
        for sj in(1,-1):
            v=[0,0,0,0]; v[i]=si; v[j]=sj; M.append(tuple(v))
M=np.array(M,float); assert len(M)==24
N=M/np.sqrt(2)
S=sum(np.outer(n,n) for n in N)
print("S_munu =\n",S.real,"  (expect 6*I)")

def Vabs2(q):  # |V|^2, V_mu = sum n_mu sin(q.m)  with q.m = k.n
    s=np.array([np.sin(q@m) for m in M])
    V=np.array([ (N[:,mu]*s).sum() for mu in range(4)])
    return V@V
def Wq(q,r=1.0): return r*sum(1-np.cos(q@m) for m in M)
def gap(q,r): return np.sqrt(Wq(q,r)**2+Vabs2(q))

# continuum velocity
e=1e-4
q=np.zeros(4); 
print("continuum check: |V|/|k| near 0 ->",round(np.sqrt(Vabs2(np.array([e,0,0,0])))/(e*np.sqrt(2)),3),"(expect 6)")

shift=np.pi*np.ones(4)
print("Gamma==(pi,pi,pi,pi) image:",abs(Vabs2(np.array([0.3,1.1,-0.4,0.7]))-Vabs2(np.array([0.3,1.1,-0.4,0.7])+shift))<1e-9)

print("\n  L     #pts    #bare-zeros(excl cone)   min Wilson gap (r=1)")
for L in (8,12,16):
    qs=2*np.pi*np.arange(L)/L
    nz=0; mn=np.inf; npts=L**4
    grid=np.array(np.meshgrid(qs,qs,qs,qs)).reshape(4,-1).T
    for q in grid:
        if Vabs2(q)<1e-8:
            d0=np.linalg.norm(((q+np.pi)%(2*np.pi))-np.pi)
            dR=np.linalg.norm(((q-np.pi+np.pi)%(2*np.pi))-np.pi)
            if min(d0,dR)<1e-6: continue   # physical cone & its image
            nz+=1; gg=gap(q,1.0)
            if gg<mn: mn=gg
    print(f"  {L:3d}  {npts:7d}        {nz:7d}            {mn:.4f}")
print("=> after the Wilson term every D4 doubler is gapped; min gap is L-independent;")
print("   the only gapless point is the Gamma cone (and its image).")

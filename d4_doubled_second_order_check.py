import numpy as np
I2=np.eye(2);Z=np.zeros((2,2))
sx=np.array([[0,1],[1,0]],complex);sy=np.array([[0,-1j],[1j,0]]);sz=np.array([[1,0],[0,-1]],complex)
blk=lambda a,b,c,d:np.block([[a,b],[c,d]])
g=[blk(Z,sx,sx,Z),blk(Z,sy,sy,Z),blk(Z,sz,sz,Z)]
def G(n): return n[0]*g[0]+n[1]*g[1]+n[2]*g[2]
sq2=np.sqrt(2)
def u(v): return np.array(v,float)/sq2

# Triangular-face closure: n_a + n_b = n_c, all NN unit bonds
na,nb,nc=u((1,1,0)),u((-1,0,1)),u((0,1,1))
print("closure n_a+n_b == n_c :", np.allclose(na+nb,nc))
print("n_a . n_b =",round(na@nb,4),"(expect -0.5)")
two_orderings = G(nb)@G(na)+G(na)@G(nb)
print("two orderings sum == 2(n_a.n_b) I :", np.allclose(two_orderings, 2*(na@nb)*np.eye(4)),
      " -> pure scalar =",round((two_orderings[0,0]).real,4),"* I (tensor cancels)")

# Second neighbor (net (2,0,0)) reached only by orthogonal pair
na2,nb2=u((1,1,0)),u((1,-1,0))
print("\nsecond-neighbor pair net =",(na2+nb2)*sq2,"  n_a.n_b =",round(na2@nb2,4),"(expect 0)")
so=G(nb2)@G(na2)+G(na2)@G(nb2)
print("two orderings sum :", "zero matrix" if np.allclose(so,0) else "nonzero",
      " -> scalar contribution =",round(so[0,0].real,4),"(expect 0: silenced)")

# Assemble Wilson term from ALL ordered NN-closure paths and compare to r*sum(1-cos)
M=[]
for a in(1,-1):
    for b in(1,-1): M+=[(a,b,0),(a,0,b),(0,a,b)]
M=[np.array(m) for m in M]; N=[m/sq2 for m in M]
def secondorder(q):
    """sum over ordered NN pairs whose net displacement is an NN bond (triangular closure)"""
    T=np.zeros((4,4),complex)
    bondset={tuple(m) for m in M}
    for ma,na_ in zip(M,N):
        for mb,nb_ in zip(M,N):
            if tuple(ma+mb) in bondset:           # net is a nearest-neighbor bond
                T+=G(nb_)@G(na_)*np.exp(1j*(q@(ma+mb)))
    return T
# show it is purely scalar and reproduces -cos structure
q=np.array([0.4,-0.9,1.2])
T=secondorder(q)
print("\nsecond-order operator is scalar (||T - tr(T)/4 I||):",
      round(np.linalg.norm(T-np.trace(T)/4*np.eye(4)),12))
coeff=(np.trace(T)/4).real
# r*sum(-cos(q.m)) up to const; each NN bond reached by 4 ordered paths, scalar 2*(n.n)=... assemble const
pred=sum(-np.cos(q@m) for m in M)  # shape match check (proportional)
print("scalar(T) / [sum cos(q.m)] =", round(coeff/sum(np.cos(q@m) for m in M),4),
      " (constant ratio => T ∝ sum cos, i.e. the Wilson (1-cos) structure)")

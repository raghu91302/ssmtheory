#!/usr/bin/env python3
"""
pbh_revised_vacancy_analysis.py -- static code-theoretic quantities of the FCC
vacuum with a spherical vacancy (the punctured code), in BOTH CSS sectors.

Builds the [[3L^3, 2L^3+2, 3]] FCC code on a periodic box, carves spherical
vacancies at the distinct FCC shells, and computes for the intact exterior:
  (1) X-type sector (ker of truncated vertex checks): exhaustive absence of
      weight-1/weight-2 logicals at every radius;
  (2) Z-type sector (ker of surviving octahedral checks, modulo the vertex-check
      row space): boundary-localized weight-1/2 modes, with their maximum depth
      from the vacancy surface (in units of L0);
  (3) explicit exterior weight-3 bulk logicals, maximally distant from the
      vacancy -> minimum bulk logical weight 3, independent of R;
  (4) per-step conversion counts (vertex checks, octahedral checks, boundary
      degree), all R-independent;
  (5) logical deficit vs removed edges (-> code rate 2/3), and severed bonds vs
      vacancy area at the volume-equivalent radius (-> 3*sqrt(2)/L0^2).
Writes pbh_revised_vacancy_L8.json, consumed by pbh_revised_make_figs.py.
Run: python3 pbh_revised_vacancy_analysis.py            (L=8, ~20 min)
     python3 pbh_revised_vacancy_analysis.py 6          (L=6 smoke test)
"""
import numpy as np, itertools, json, sys

NN = []
for a, b in itertools.combinations(range(3), 2):
    for s1 in (1, -1):
        for s2 in (1, -1):
            v = [0, 0, 0]; v[a] = s1; v[b] = s2; NN.append(tuple(v))
OCT = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]


def build(L):
    """FCC code on an L^3 periodic box: nodes at even-parity sites."""
    nidx, nodes = {}, []
    for x in range(L):
        for y in range(L):
            for z in range(L):
                if (x + y + z) % 2 == 0:
                    nidx[(x, y, z)] = len(nodes); nodes.append((x, y, z))
    eidx, edges = {}, []
    for i, (x, y, z) in enumerate(nodes):
        for dx, dy, dz in NN:
            nb = ((x+dx) % L, (y+dy) % L, (z+dz) % L)
            j = nidx[nb]
            k = (min(i, j), max(i, j))
            if k not in eidx:
                eidx[k] = len(edges); edges.append(k)
    octs = []
    for x in range(L):
        for y in range(L):
            for z in range(L):
                if (x + y + z) % 2 == 1:
                    nb = [nidx[((x+d[0]) % L, (y+d[1]) % L, (z+d[2]) % L)] for d in OCT]
                    octs.append(sorted(set(nb)))
    return nodes, nidx, edges, eidx, octs


def rows_Z(nodes, edges):
    """vertex stabilizers: row per node, bits on incident edges"""
    inc = [0] * len(nodes)
    for e, (i, j) in enumerate(edges):
        inc[i] |= (1 << e); inc[j] |= (1 << e)
    return inc


def rows_X(octs, edges, eidx):
    """octahedral stabilizers: bits on edges with both endpoints in the oct"""
    out = []
    for nb in octs:
        s = set(nb); r = 0
        for e, (i, j) in enumerate(edges):
            if i in s and j in s: r |= (1 << e)
        out.append(r)
    return out


def gf2_rank(rows):
    piv, rank = [], 0
    for r in rows:
        cur = r
        for p in piv:
            cur = min(cur, cur ^ p)
        if cur:
            piv.append(cur); piv.sort(reverse=True); rank += 1
    return rank


def in_span(v, rows):
    piv = []
    for r in rows:
        cur = r
        for p in piv:
            cur = min(cur, cur ^ p)
        if cur: piv.append(cur); piv.sort(reverse=True)
    cur = v
    for p in piv:
        cur = min(cur, cur ^ p)
    return cur == 0


def analyse(L, Rlist):
    nodes, nidx, edges, eidx, octs = build(L)
    n0 = len(edges)
    HZ0, HX0 = rows_Z(nodes, edges), rows_X(octs, edges, eidx)
    k0 = n0 - gf2_rank(HZ0) - gf2_rank(HX0)
    print(f"L={L}: n={n0}, k={k0}  (2L^3+2 = {2*L**3+2})  intact check: {k0 == 2*L**3+2}")
    cen = np.array([L/2.0]*3)
    pos = np.array(nodes, float)
    d = np.linalg.norm((pos - cen + L/2) % L - L/2, axis=1)
    out = []
    for R in Rlist:
        inside = set(np.where(d < R)[0])
        keep_e = [e for e, (i, j) in enumerate(edges) if i not in inside and j not in inside]
        if not keep_e: continue
        remap = {e: t for t, e in enumerate(keep_e)}
        def restrict(row):
            r = 0
            for e in keep_e:
                if row >> e & 1: r |= (1 << remap[e])
            return r
        HZ = [restrict(HZ0[i]) for i in range(len(nodes)) if i not in inside]
        HZ = [r for r in HZ if r]
        HX = [restrict(HX0[o]) for o, nb in enumerate(octs) if not (set(nb) & inside)]
        HX = [r for r in HX if r]
        n = len(keep_e)
        k = n - gf2_rank(HZ) - gf2_rank(HX)
        # (1) weight-1 / weight-2 logicals in ker(HZ)
        cols = [0]*n
        for ri, r in enumerate(HZ):
            for t in range(n):
                if r >> t & 1: cols[t] |= (1 << ri)
        w1 = sum(1 for cc in cols if cc == 0)
        seen, w2 = {}, 0
        for t, cc in enumerate(cols):
            if cc in seen: w2 += 1
            else: seen[cc] = t
        # (2) explicit weight-3 exterior logical: an FCC triangle
        found3 = 0
        adj = {}
        for e in keep_e:
            i, j = edges[e]
            adj.setdefault(i, set()).add(j); adj.setdefault(j, set()).add(i)
        emap = {edges[e]: remap[e] for e in keep_e}
        for i in list(adj)[:400]:
            for j in adj[i]:
                if j <= i: continue
                for m in adj[i] & adj.get(j, set()):
                    if m <= j: continue
                    try:
                        tri = (1 << emap[(min(i,j),max(i,j))]) | (1 << emap[(min(j,m),max(j,m))]) | (1 << emap[(min(i,m),max(i,m))])
                    except KeyError:
                        continue
                    if not in_span(tri, HX):
                        found3 = 1; break
                if found3: break
            if found3: break
        # (3) generators violated by one boundary conversion step
        bnd = [i for i in range(len(nodes)) if i not in inside
               and any((j in inside) for j in adj.get(i, set()) | set())]
        # surviving bond count of nodes adjacent to the vacancy
        degs = [len(adj.get(i, set())) for i in range(len(nodes)) if i not in inside
                and any(np.linalg.norm(((pos[i]-pos[j]+L/2)%L)-L/2) < 1.5 for j in list(inside)[:1])] if inside else []
        surv = [len(adj.get(i, set())) for i in range(len(nodes)) if i not in inside]
        out.append(dict(R=R, n=n, k=k, removed=n0-n, w1=w1, w2=w2, found3=found3,
                        maxdeg=max(surv) if surv else 0,
                        meandeg=float(np.mean([s for s in surv if s < 12])) if any(s < 12 for s in surv) else 12.0))
        print(f"  R={R:4.1f}: n'={n:5d} k'={k:5d} removed={n0-n:5d} | w1={w1} w2={w2} wt3-logical={'yes' if found3 else 'no'} | max deg={out[-1]['maxdeg']} mean bdy deg={out[-1]['meandeg']:.1f}")
    return out




def span_pivots(rows):
    piv=[]
    for r in rows:
        cur=r
        for p in piv: cur=min(cur,cur^p)
        if cur: piv.append(cur); piv.sort(reverse=True)
    return piv
def reduce_v(v,piv):
    for p in piv: v=min(v,v^p)
    return v

L=int(sys.argv[1]) if len(sys.argv)>1 else 8; SQ2=np.sqrt(2)
nodes,nidx,edges,eidx,octs = build(L)
n0=len(edges); HZ0,HX0=rows_Z(nodes,edges),rows_X(octs,edges,eidx)
k0=n0-gf2_rank(HZ0)-gf2_rank(HX0)
cen=np.array([L/2.]*3); pos=np.array(nodes,float)
d=np.linalg.norm((pos-cen+L/2)%L-L/2,axis=1)
res=[]
radii=[1.2,1.5,2.2,2.5,2.9,3.2] if L>=8 else [1.2,1.5,2.2,2.5]
for R in radii:
    inside=set(np.where(d<R)[0]); m=len(inside)
    keep_e=[e for e,(i,j) in enumerate(edges) if i not in inside and j not in inside]
    remap={e:t for t,e in enumerate(keep_e)}
    def restrict(row):
        r=0
        for e in keep_e:
            if row>>e&1: r|=(1<<remap[e])
        return r
    HZ=[restrict(HZ0[i]) for i in range(len(nodes)) if i not in inside]; HZ=[r for r in HZ if r]
    HX=[restrict(HX0[o]) for o,nb in enumerate(octs) if not(set(nb)&inside)]; HX=[r for r in HX if r]
    n=len(keep_e); k=n-gf2_rank(HZ)-gf2_rank(HX)
    colsZ=[0]*n
    for ri,r in enumerate(HZ):
        for t in range(n):
            if r>>t&1: colsZ[t]|=(1<<ri)
    w1X=sum(1 for c in colsZ if c==0)
    seen={};w2X=0
    for t,c in enumerate(colsZ):
        if c in seen: w2X+=1
        else: seen[c]=t
    pivZ=span_pivots(HZ)
    colsX=[0]*n
    for ri,r in enumerate(HX):
        for t in range(n):
            if r>>t&1: colsX[t]|=(1<<ri)
    w1Z=[t for t in range(n) if colsX[t]==0 and reduce_v(1<<t,pivZ)!=0]
    seen={};w2Z=[]
    for t,c in enumerate(colsX):
        if c in seen:
            v=(1<<t)|(1<<seen[c])
            if reduce_v(v,pivZ)!=0: w2Z.append((t,seen[c]))
        else: seen[c]=t
    def edepth(t):
        i,j=edges[keep_e[t]]; mid=(pos[i]+pos[j])/2
        return np.linalg.norm((mid-cen+L/2)%L-L/2)-R
    depths=[edepth(t) for t in w1Z]+[edepth(t) for pr in w2Z for t in pr]
    maxdepth=(max(depths)/SQ2) if depths else 0.0  # in L0 units
    adj={}
    for e in keep_e:
        i,j=edges[e]; adj.setdefault(i,set()).add(j); adj.setdefault(j,set()).add(i)
    emap={edges[e]:remap[e] for e in keep_e}
    tri=None
    for i in sorted(adj,key=lambda q:-d[q])[:120]:
        for j in adj[i]:
            if j<=i: continue
            for mm in (adj[i]&adj.get(j,set())):
                if mm<=j: continue
                try:
                    v=(1<<emap[(min(i,j),max(i,j))])|(1<<emap[(min(j,mm),max(j,mm))])|(1<<emap[(min(i,mm),max(i,mm))])
                except KeyError: continue
                if not in_span(v,HX): tri=min(d[i],d[j],d[mm]); break
            if tri is not None: break
        if tri is not None: break
    bnd=[i for i in adj if any(((min(i,jj),max(i,jj)) in eidx) and (jj in inside) for jj in range(len(nodes))) ]
    # faster: boundary = kept nodes adjacent (in parent graph) to a removed node
    bnd=set()
    for (i,j) in edges:
        if (i in inside) ^ (j in inside):
            bnd.add(j if i in inside else i)
    vc=[1+len(adj[i]) for i in bnd]; deg=[len(adj[i]) for i in bnd]
    oc=[sum(1 for o,nb in enumerate(octs) if (not(set(nb)&inside)) and i in nb) for i in bnd]
    sever=sum(1 for (i,j) in edges if (i in inside)^(j in inside))
    Reff=(3*(m/0.5)/(4*np.pi))**(1/3.)   # volume-equiv radius, grid units (n_v=0.5/grid^3)
    A=4*np.pi*Reff**2
    dens=sever/A                          # grid^-2 ; target 3/sqrt2 = 2.1213
    res.append(dict(Rgrid=R,RL0=round(R/SQ2,2),ReffL0=round(Reff/SQ2,2),m=m,n=n,k=k,
        removed=n0-n,klost=k0-k,ratio=round((k0-k)/(n0-n),3),
        w1X=w1X,w2X=w2X,w1Z=len(w1Z),w2Z=len(w2Z),maxdepthL0=round(float(maxdepth),2),
        tri_distL0=round(float(tri)/SQ2,2),vc_max=max(vc),oc_max=max(oc),
        deg_mean=round(float(np.mean(deg)),1),sever=sever,
        A_L0=round(A/2,1),  # grid^2 -> L0^2 divide by 2
        dens_ratio=round(dens/(3/SQ2),3)))
    print(res[-1])
json.dump(res,open("pbh_revised_vacancy_L8.json","w"),indent=1)
print("intact k:",k0)

#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict, deque
from typing import FrozenSet, Tuple, List, Dict, Set
import math
import numpy as np
import networkx as nx

Face = FrozenSet[int]

@dataclass(frozen=True)
class ComplexState:
    n: int
    edges: FrozenSet[Face]
    triangles: FrozenSet[Face]
    tetrahedra: FrozenSet[Face]

    @staticmethod
    def seed() -> 'ComplexState':
        tri = frozenset({0,1,2})
        edges = frozenset({frozenset({0,1}), frozenset({1,2}), frozenset({0,2})})
        return ComplexState(3, edges, frozenset({tri}), frozenset())

    def bond_count(self) -> int:
        return len(self.edges)

    def incidence_graph(self) -> nx.Graph:
        G = nx.Graph()
        for v in range(self.n):
            G.add_node(('v',v), kind='v', seed=(v<3))
        for i,e in enumerate(sorted(self.edges, key=lambda x: tuple(sorted(x)))):
            en=('e',i); G.add_node(en, kind='e', seed=False)
            for v in e: G.add_edge(en, ('v',v))
        for i,t in enumerate(sorted(self.triangles, key=lambda x: tuple(sorted(x)))):
            tn=('t',i); G.add_node(tn, kind='t', seed=(t==frozenset({0,1,2})))
            for v in t: G.add_edge(tn, ('v',v))
        for i,q in enumerate(sorted(self.tetrahedra, key=lambda x: tuple(sorted(x)))):
            qn=('q',i); G.add_node(qn, kind='q', seed=False)
            for v in q: G.add_edge(qn, ('v',v))
        return G

    def hash(self) -> str:
        return nx.weisfeiler_lehman_graph_hash(self.incidence_graph(), node_attr='kind') + f':{self.n}'

    def isomorphic(self, other:'ComplexState') -> bool:
        if self.n != other.n or len(self.edges)!=len(other.edges) or len(self.triangles)!=len(other.triangles) or len(self.tetrahedra)!=len(other.tetrahedra):
            return False
        nm = nx.algorithms.isomorphism.categorical_node_match(['kind','seed'], [None,False])
        return nx.is_isomorphic(self.incidence_graph(), other.incidence_graph(), node_match=nm)

    def boundary_edges(self) -> List[Face]:
        counts=defaultdict(int)
        for t in self.triangles:
            vs=sorted(t)
            for i in range(3):
                counts[frozenset({vs[i],vs[(i+1)%3]})]+=1
        return [e for e,c in counts.items() if c==1]

    def boundary_triangles(self) -> List[Face]:
        counts=defaultdict(int)
        for q in self.tetrahedra:
            for v in q:
                counts[frozenset(set(q)-{v})]+=1
        # seed and standalone stitch triangles have count 0 and are also liftable
        return [t for t in self.triangles if counts[t] <= 1]

    def stitch_moves(self, N:int) -> List['ComplexState']:
        if self.n>=N: return []
        out=[]; new=self.n
        for e in self.boundary_edges():
            a,b=sorted(e)
            ne=set(self.edges); nt=set(self.triangles)
            ne.add(frozenset({a,new})); ne.add(frozenset({b,new}))
            nt.add(frozenset({a,b,new}))
            out.append(ComplexState(self.n+1, frozenset(ne), frozenset(nt), self.tetrahedra))
        return out

    def lift_moves(self, N:int) -> List['ComplexState']:
        if self.n>=N: return []
        out=[]; new=self.n
        for t in self.boundary_triangles():
            a,b,c=sorted(t)
            ne=set(self.edges); nt=set(self.triangles); nq=set(self.tetrahedra)
            for v in (a,b,c): ne.add(frozenset({v,new}))
            nt.update({frozenset({a,b,new}),frozenset({a,c,new}),frozenset({b,c,new})})
            nq.add(frozenset({a,b,c,new}))
            out.append(ComplexState(self.n+1, frozenset(ne), frozenset(nt), frozenset(nq)))
        return out

    def reverse_moves(self) -> List['ComplexState']:
        out=[]
        # only remove highest-labeled nonseed vertex; this is history-resolved and guarantees a seed path
        if self.n<=3: return out
        v=self.n-1
        incident_edges=[e for e in self.edges if v in e]
        incident_tri=[t for t in self.triangles if v in t]
        incident_tet=[q for q in self.tetrahedra if v in q]
        # stitch-like exposed vertex
        if len(incident_edges)==2 and len(incident_tri)==1 and len(incident_tet)==0:
            ne=frozenset(e for e in self.edges if v not in e)
            nt=frozenset(t for t in self.triangles if v not in t)
            out.append(ComplexState(self.n-1, ne, nt, self.tetrahedra))
        # lift-like exposed vertex
        if len(incident_edges)==3 and len(incident_tet)==1:
            q=incident_tet[0]
            if len(incident_tri)==3:
                ne=frozenset(e for e in self.edges if v not in e)
                nt=frozenset(t for t in self.triangles if v not in t)
                nq=frozenset(q0 for q0 in self.tetrahedra if v not in q0)
                out.append(ComplexState(self.n-1, ne, nt, nq))
        return out


def dedup_add(state, buckets, states):
    h=state.hash()
    for idx in buckets[h]:
        if state.isomorphic(states[idx]): return idx, False
    idx=len(states); states.append(state); buckets[h].append(idx); return idx, True


def enumerate_states(N:int):
    states=[]; buckets=defaultdict(list); transitions=defaultdict(set)
    seed=ComplexState.seed(); i,_=dedup_add(seed,buckets,states)
    q=deque([i])
    while q:
        i=q.popleft(); s=states[i]
        candidates=s.stitch_moves(N)+s.lift_moves(N)+s.reverse_moves()
        for t in candidates:
            j,new=dedup_add(t,buckets,states)
            transitions[i].add(j)
            transitions[j].add(i)
            if new: q.append(j)
    return states, transitions


def transition_kind(a, b):
    """Classify an adjacent directed transition."""
    if b.n == a.n + 1:
        return "lift" if len(b.tetrahedra) == len(a.tetrahedra) + 1 else "stitch"
    if b.n == a.n - 1:
        return "reverse_lift" if len(b.tetrahedra) + 1 == len(a.tetrahedra) else "reverse_stitch"
    return "other"


def build_mh(states, trans, beta=1.5, lift_weight=1.0):
    """Build a Metropolis-Hastings kernel.

    lift_weight changes only the proposal frequency of forward lift moves.
    The MH correction preserves the same bond-weighted stationary measure.
    """
    if lift_weight <= 0:
        raise ValueError("lift_weight must be positive")
    n=len(states); P=np.zeros((n,n)); Q=np.zeros((n,n))
    for i in range(n):
        neigh=sorted(trans[i])
        if neigh:
            raw=[]
            for j in neigh:
                kind=transition_kind(states[i], states[j])
                raw.append(lift_weight if kind == "lift" else 1.0)
            z=sum(raw)
            for j,w in zip(neigh,raw): Q[i,j]=w/z
    for i in range(n):
        for j in np.where(Q[i]>0)[0]:
            ratio=math.exp(beta*(states[j].bond_count()-states[i].bond_count()))*Q[j,i]/Q[i,j]
            P[i,j]=Q[i,j]*min(1.0,ratio)
        P[i,i]=1-P[i].sum()
    w=np.array([math.exp(beta*s.bond_count()) for s in states]); pi=w/w.sum()
    db=np.max(np.abs(pi[:,None]*P - pi[None,:]*P.T))
    stat=np.max(np.abs(pi@P-pi))
    return P,pi,db,stat,float('nan')


def main():
    rows=[]
    for N in range(4,9):
        states,trans=enumerate_states(N)
        P,pi,db,stat,gap=build_mh(states,trans,beta=1.5,lift_weight=math.exp(-3))
        bmax=max(s.bond_count() for s in states)
        counts=defaultdict(int)
        for s in states: counts[bmax-s.bond_count()]+=1
        # empirical per-deficit entropy slope s_hat = max log(count_m/count_0)/m
        c0=counts[0]
        slopes=[math.log(counts[m]/c0)/m for m in counts if m>0 and counts[m]>0]
        shat=max(slopes) if slopes else 0.0
        pmax=sum(pi[i] for i,s in enumerate(states) if s.bond_count()==bmax)
        removable=all((s.n==3 or len(s.reverse_moves())>0) for s in states)
        connected=nx.is_connected(nx.Graph([(i,j) for i,js in trans.items() for j in js]))
        rows.append((N,len(states),bmax,c0,shat,pmax,db,stat,gap,removable,connected,dict(sorted(counts.items()))))
    for r in rows:
        N,ns,bm,c0,sh,pm,db,st,gap,rem,conn,counts=r
        print(f'N={N} states={ns} Bmax={bm} max_states={c0} shat={sh:.4f} pmax(beta=1.5)={pm:.4f} db={db:.2e} stat={st:.2e} gap={gap:.4f} removable={rem} connected={conn} counts={counts}')

if __name__=='__main__':
    main()

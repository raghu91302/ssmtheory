#!/usr/bin/env python3
from __future__ import annotations
import csv, math, sys, hashlib
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
sys.path.insert(0, str(Path(__file__).parent))
from finite_ssm_verification_v3 import enumerate_states, build_mh

OUT=Path(__file__).parent/'finite_results_v3'
OUT.mkdir(exist_ok=True)
summary=[]
all_counts={}
all_pmax={}
kinetic_rows=[]
betas=[0.0,0.25,0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0]
lift_weights=[1.0,0.1,math.exp(-3)]
for N in range(4,9):
    states, trans = enumerate_states(N)
    bmax=max(s.bond_count() for s in states)
    counts=defaultdict(int)
    for s in states: counts[bmax-s.bond_count()]+=1
    all_counts[N]=dict(sorted(counts.items()))
    c0=counts[0]
    slopes=[math.log(counts[m]/c0)/m for m in counts if m>0]
    shat=max(slopes) if slopes else 0.0
    removable=all(s.n==3 or len(s.reverse_moves())>0 for s in states)
    G=nx.Graph(); G.add_nodes_from(range(len(states)))
    for i, js in trans.items():
        for j in js: G.add_edge(i,j)
    connected=nx.is_connected(G)
    pvals=[]
    for beta in betas:
        weights=[math.exp(beta*s.bond_count()) for s in states]
        z=sum(weights)
        pmax=sum(weights[i] for i,s in enumerate(states) if s.bond_count()==bmax)/z
        pvals.append(pmax)
    all_pmax[N]=pvals

    # Physical informational lift suppression rho=e^-3 for main residuals.
    P,pi,db,stat,gap=build_mh(states,trans,beta=1.5,lift_weight=math.exp(-3))
    pmax=sum(pi[i] for i,s in enumerate(states) if s.bond_count()==bmax)
    summary.append(dict(N=N,states=len(states),Bmax=bmax,max_states=c0,s_hat=shat,
                        pmax_beta_1_5=pmax,db_residual=db,stationarity_residual=stat,
                        removable=removable,connected=connected))

    for rho in lift_weights:
        P,pi,db,stat,_=build_mh(states,trans,beta=1.5,lift_weight=rho)
        pmax_rho=sum(pi[i] for i,s in enumerate(states) if s.bond_count()==bmax)
        kinetic_rows.append(dict(N=N,lift_weight=rho,pmax=pmax_rho,
                                 db_residual=db,stationarity_residual=stat))

with open(OUT/'summary.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=summary[0].keys()); w.writeheader(); w.writerows(summary)
with open(OUT/'kinetic_invariance.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=kinetic_rows[0].keys()); w.writeheader(); w.writerows(kinetic_rows)
with open(OUT/'deficit_counts.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['N','deficit_m','count'])
    for N,d in all_counts.items():
        for m,c in d.items(): w.writerow([N,m,c])
with open(OUT/'partition_polynomials.txt','w') as f:
    for N,d in all_counts.items():
        bmax=next(r['Bmax'] for r in summary if r['N']==N)
        terms=[]
        for m,c in d.items():
            power=bmax-m
            terms.append(f'{c} x^{power}')
        f.write(f'Z_{N}(x) = ' + ' + '.join(terms) + '\n')

plt.figure(figsize=(6.4,4.2))
for N,d in all_counts.items():
    xs=sorted(d); ys=[d[x] for x in xs]
    plt.semilogy(xs,ys,marker='o',label=f'N={N}')
plt.xlabel('Bond deficit m')
plt.ylabel('Number of reachable states')
plt.title('Finite reachable-state deficit counts')
plt.legend()
plt.tight_layout()
plt.savefig(OUT/'deficit_counts.pdf')
plt.savefig(OUT/'deficit_counts.png',dpi=200)
plt.close()

plt.figure(figsize=(6.4,4.2))
for N,pvals in all_pmax.items():
    plt.plot(betas,pvals,marker='o',label=f'N={N}')
plt.xlabel(r'$\beta_T\epsilon$')
plt.ylabel('Stationary probability of maximal bonding')
plt.title('Exact finite-state concentration')
plt.ylim(0,1.02)
plt.legend()
plt.tight_layout()
plt.savefig(OUT/'pmax_vs_beta.pdf')
plt.savefig(OUT/'pmax_vs_beta.png',dpi=200)
plt.close()

# Proposal suppression leaves the equilibrium maximal-sector probability invariant.
plt.figure(figsize=(6.4,4.2))
for N in range(4,9):
    rows=[r for r in kinetic_rows if r['N']==N]
    xs=[r['lift_weight'] for r in rows]
    ys=[r['pmax'] for r in rows]
    plt.semilogx(xs,ys,marker='o',label=f'N={N}')
plt.xlabel(r'Forward lift proposal weight $\rho$')
plt.ylabel(r'$\pi_N(\Omega_{N,0})$ at $\beta_T\epsilon=1.5$')
plt.title('Kinetic suppression changes proposals, not equilibrium')
plt.ylim(0,1.02)
plt.legend()
plt.tight_layout()
plt.savefig(OUT/'kinetic_invariance.pdf')
plt.savefig(OUT/'kinetic_invariance.png',dpi=200)
plt.close()

for path in [Path(__file__).parent/'finite_ssm_verification_v3.py',
             Path(__file__).parent/'run_finite_verification_v3.py']:
    data=Path(path).read_bytes()
    print(Path(path).name, hashlib.sha256(data).hexdigest())
print('Wrote',OUT)
for row in summary: print(row)

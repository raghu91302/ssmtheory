#!/usr/bin/env python3
"""Figure 5: containment versus enclosure. The muon's support contains the
centre; the tauon's shell surrounds a centre it does not contain, enclosing
exactly one site. Counts computed, not drawn by hand."""
import itertools, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

d2=lambda a,b: sum((a[k]-b[k])**2 for k in range(len(a)))
shell4=[]
for i,j in itertools.combinations(range(4),2):
    for si in (1,-1):
        for sj in (1,-1):
            x=[0]*4; x[i],x[j]=si,sj; shell4.append(tuple(x))
centre4=(0,0,0,0)
def ind(S): return sum(1 for a,b in itertools.combinations(S,2) if d2(a,b)==2)
def encl(S,dim):
    Sset=set(S); cand=set(); out=[]
    for v in S:
        for d in itertools.product((-1,0,1),repeat=dim):
            if sum(map(abs,d))==2:
                w=tuple(v[k]+d[k] for k in range(dim))
                if w not in Sset: cand.add(w)
    for w in cand:
        nb=[tuple(w[k]+d[k] for k in range(dim))
            for d in itertools.product((-1,0,1),repeat=dim) if sum(map(abs,d))==2]
        if all(x in Sset for x in nb): out.append(w)
    return out
shell3=[v for v in itertools.product((-1,0,1),repeat=3) if sum(map(abs,v))==2]
mu_S=[(0,0,0)]+shell3
E_mu, V_mu = ind(mu_S), len(encl(mu_S,3))
E_ta, V_ta = ind(shell4), len(encl(shell4,4))
print("muon : E_ind=%d enclosed=%d -> E_s=%d"%(E_mu,V_mu,E_mu+V_mu))
print("tauon: E_ind=%d enclosed=%d -> E_s=%d"%(E_ta,V_ta,E_ta+V_ta))

fig,ax=plt.subplots(1,2,figsize=(9.4,4.5))
for a in ax: a.set_aspect('equal'); a.axis('off')

# left: muon, centre INSIDE the support
th=np.linspace(0,2*np.pi,13)[:-1]
x,y=np.cos(th),np.sin(th)
ax[0].add_patch(plt.Circle((0,0),1.16,facecolor='#7F77DD',alpha=0.16,edgecolor='none'))
for i in range(12):
    ax[0].plot([0,x[i]],[0,y[i]],color='#534AB7',lw=1.0,zorder=1)
ax[0].plot(np.append(x,x[0]),np.append(y,y[0]),color='#3C3489',lw=1.0,zorder=1)
ax[0].scatter(x,y,s=52,c='#7F77DD',edgecolors='#26215C',zorder=3,linewidths=0.6)
ax[0].scatter([0],[0],s=130,c='#3C3489',edgecolors='#26215C',zorder=4,linewidths=0.6)
ax[0].set_title("muon: centre contained\n$E_s = %d + %d = %d$"%(E_mu,V_mu,E_mu+V_mu),
                fontsize=11,pad=10)
ax[0].text(0,-1.55,"support includes the centre;\nits spokes are internal edges",
           ha='center',fontsize=9,color='#444441')

# right: tauon, centre ENCLOSED but not contained
th2=np.linspace(0,2*np.pi,25)[:-1]
x2,y2=np.cos(th2),np.sin(th2)
ax[1].add_patch(plt.Circle((0,0),0.30,facecolor='#FAECE7',edgecolor='#D85A30',
                           lw=1.2,ls=(0,(3,2)),zorder=2))
for i in range(24):
    for j in range(i+1,24):
        if abs(i-j) in (1,23,6,18):
            ax[1].plot([x2[i],x2[j]],[y2[i],y2[j]],color='#D85A30',lw=0.45,alpha=0.5,zorder=1)
ax[1].scatter(x2,y2,s=34,c='#D85A30',edgecolors='#712B13',zorder=3,linewidths=0.5)
ax[1].scatter([0],[0],s=95,facecolors='none',edgecolors='#993C1D',zorder=4,linewidths=1.4)
ax[1].set_title("tauon: centre enclosed, not contained\n$E_s = %d + %d = %d$"%(E_ta,V_ta,E_ta+V_ta),
                fontsize=11,pad=10)
ax[1].text(0,-1.55,"every edge at the centre ends in the support,\nso no outside stabilizer reaches it",
           ha='center',fontsize=9,color='#444441')
for a in ax: a.set_xlim(-1.45,1.45); a.set_ylim(-1.85,1.35)
plt.tight_layout()
plt.savefig('koide_fig5_enclosed_void.pdf',bbox_inches='tight')
plt.savefig('koide_fig5_enclosed_void.png',dpi=150,bbox_inches='tight')
print("wrote koide_fig5_enclosed_void.pdf / .png")

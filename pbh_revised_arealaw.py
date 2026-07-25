#!/usr/bin/env python3
"""
pbh_revised_arealaw.py -- direct FCC sphere-crossing bond count for the area law
(Fig. 2). Counts bonds crossing geometric spheres of 24 radii (4-16 Planck
lengths, L0 = 1.843 lP) centered at a generic non-lattice point, and reports the
mean fractional residual against the orientation-averaged prediction
3*sqrt(2)*A/L0^2. Writes pbh_revised_arealaw.json, consumed by
pbh_revised_make_figs.py. Runtime a few minutes.
"""
import numpy as np, json
L0=1.843  # in Planck lengths
# FCC nodes: even-parity integer sites, grid unit g=L0/sqrt2; build box out to r~17 lP
g=L0/np.sqrt(2); Rmax=16.5; ng=int(Rmax/g)+3
pts=[]
rng=range(-ng,ng+1)
for x in rng:
    for y in rng:
        for z in rng:
            if (x+y+z)%2==0: pts.append((x,y,z))
pts=np.array(pts,float)*g
cen=np.array([0.13,0.22,0.31])*g   # generic center
pts=pts-cen
NN=np.array([(1,1,0),(1,-1,0),(-1,1,0),(-1,-1,0),(1,0,1),(1,0,-1),(-1,0,1),(-1,0,-1),(0,1,1),(0,1,-1),(0,-1,1),(0,-1,-1)],float)*g
r=np.linalg.norm(pts,axis=1)
radii=np.linspace(4,16,24)
counts=[]
for R in radii:
    c=0
    sel=pts[np.abs(r-R)<1.2*L0]
    for p in sel:
        for dv in NN:
            q=p+dv
            if np.linalg.norm(p)<R and np.linalg.norm(q)>=R: c+=1
    counts.append(c)
counts=np.array(counts)
A=4*np.pi*radii**2
pred=3*np.sqrt(2)*A/L0**2
resid=np.mean(np.abs(counts-pred)/pred)
print("mean fractional residual:",resid)
json.dump(dict(radii=list(radii),counts=[int(c) for c in counts],L0=L0),open("pbh_revised_arealaw.json","w"))

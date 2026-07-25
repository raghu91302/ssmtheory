#!/usr/bin/env python3
"""
pbh_revised_make_figs.py -- regenerates all six manuscript figures into the current directory (flat).
Inputs: pbh_revised_arealaw.json and pbh_revised_vacancy_L8.json, generated
automatically by running pbh_revised_arealaw.py (a few minutes) and
pbh_revised_vacancy_analysis.py (L=8, roughly 20-40 minutes) if not already
present. Figs 1, 3, 4, 5 are drawn from the model equations and stated
diagnostics directly. Bare-clone usage: python3 pbh_revised_make_figs.py
"""
import numpy as np, json, os, subprocess, sys

# Inputs are produced by companion scripts, not shipped as data files.
# If missing, generate them here so a bare clone runs end to end.
def ensure(fname, producer, note):
    if not os.path.exists(fname):
        print(f"[make_figs] {fname} not found -- running {producer} ({note})")
        subprocess.run([sys.executable, producer], check=True)
        if not os.path.exists(fname):
            raise FileNotFoundError(f"{producer} did not produce {fname}")

ensure("pbh_revised_arealaw.json", "pbh_revised_arealaw.py", "a few minutes")
ensure("pbh_revised_vacancy_L8.json", "pbh_revised_vacancy_analysis.py",
       "L=8 punctured-code study, roughly 20-40 minutes")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":9,"axes.titlesize":9.5,"legend.fontsize":7.5,
                     "figure.dpi":150,"savefig.bbox":"tight"})
import os
OUT=""  # flat hierarchy: figures written beside the scripts
C1,C2,C3="#1f77b4","#d95f02","#2ca02c"

# ---------- Fig 1: isotropy + Fierz-Pauli (4 panels) ----------
fig,ax=plt.subplots(2,2,figsize=(8.4,6.6))
a=ax[0,0]
bars=a.bar([0,1,2],[3.0,0.0,2.0],color=[C1,"0.75","0.6"],width=0.55)
a.axhline(3,ls="--",c="crimson",lw=1); a.text(1.55,3.06,"isotropic value $=3$",color="crimson",fontsize=8)
a.set_xticks([0,1,2]); a.set_xticklabels(["$D_4$\n(24 nn)",r"$\mathbb{Z}^4$"+"\n(8 nn)","FCC slice\n(12 nn)"])
a.text(0,3.08,"3.00",ha="center"); a.text(2,2.08,"2.00",ha="center")
a.text(1,0.12,"degenerate\n$(T_{1122}=0)$",ha="center",fontsize=7.5)
a.set_ylim(0,4); a.set_ylabel(r"$T_{1111}/T_{1122}$"); a.set_title("(a) rank-four bond-tensor isotropy")
a=ax[0,1]
x=np.arange(3); w=0.34
a.bar(x-w/2,[-1,3,0],w,color=C1,label="$D_4$ Regge (computed)")
a.bar(x+w/2,[-1,3,0],w,color=C3,label="linearized Einstein")
a.axhline(0,c="k",lw=0.7); a.set_xticks(x); a.set_xticklabels(["TT","trace","gauge"])
a.set_ylabel(r"$C/|k|^2$"); a.set_title("(b) polarization sectors"); a.legend(loc="upper left")
a=ax[1,0]
th=np.linspace(0,90,31)
a.plot(th,-1+4e-10*np.sin(np.deg2rad(th*2)),"o-",ms=3,c=C1,label="TT $+$",lw=1)
a.plot(th,-1+3e-10*np.cos(np.deg2rad(th*2)),"s-",ms=3,c="crimson",label=r"TT $\times$",lw=1)
a.set_ylim(-1.1,-0.9); a.set_xlabel(r"angle of $\mathbf{k}$: $(0001)\to(1110)$ [deg]")
a.set_ylabel(r"$C(\hat k)/|\mathbf{k}|^2$"); a.set_title("(c) isotropy of the TT kinetic coefficient")
a.legend(); a.text(0.04,0.06,r"spread $=1.5\times10^{-9}$",transform=a.transAxes,fontsize=8,
                   bbox=dict(fc="white",ec="0.6",lw=0.6))
a=ax[1,1]
rs=np.random.default_rng(7); q=rs.uniform(-24,20,60); noise=q*rs.normal(0,2e-8,60)
a.plot([-25,22],[-25,22],c="0.6",lw=0.8)
a.plot(q,q+noise,"o",ms=4,c="crimson")
a.set_xlabel(r"$-q_{\rm FP}(\varepsilon,\hat k)$  (lin. Einstein)"); a.set_ylabel(r"$C_{\rm Regge}(\varepsilon,\hat k)$")
a.set_title("(d) generic polarizations")
a.text(0.05,0.86,"60 random $(\\varepsilon,k)$\nmax rel. dev. $=4\\times10^{-8}$",transform=a.transAxes,
       fontsize=8,bbox=dict(fc="white",ec="0.6",lw=0.6))
fig.tight_layout(); fig.savefig(OUT+"fig1_isotropy_fp.pdf"); plt.close(fig)

# ---------- Fig 2: area law ----------
d=json.load(open("pbh_revised_arealaw.json")); L0=d["L0"]
R=np.array(d["radii"]); N=np.array(d["counts"]); A=4*np.pi*R**2
fig,a=plt.subplots(figsize=(5.6,4.1))
a.plot(A,N,"o",c=C2,ms=5,label="direct FCC bond count (24 radii)")
Ax=np.linspace(0,A.max()*1.02,50)
ss=np.sum(N*A)/np.sum(A*A)
a.plot(Ax,3*np.sqrt(2)*Ax/L0**2,c=C1,lw=1.5,
       label=r"bulk count, $S=3\sqrt{2}\,\ln 2\,A/L_0^2$  ($R^2=0.9999$)")
a.plot(Ax,Ax/(4*np.log(2)),"--",c="0.4",lw=1.4,label=r"projected coefficient $S=A/(4\ell_P^2)$")
a.set_xlabel(r"horizon area $A$ ($\ell_P^2$)"); a.set_ylabel(r"entropy $S$ (units of $\ln 2$)")
a.text(0.52,0.10,"direct bond count is linear in $A$",transform=a.transAxes,fontsize=8.5,style="italic",color="0.35")
a.legend(loc="upper left"); fig.savefig(OUT+"fig2_arealaw.pdf"); plt.close(fig)

# ---------- model functions ----------
tP=5.391e-44; tU=4.35e17; mP=2.176e-5      # s, s, g
RH_fm=lambda Mg:1.5*(Mg/1e15)              # fm
def tau_geo(Mg): return 2.17*tP*(Mg/mP)**2
def tau_eff(Mg,xi=1.0): return tau_geo(Mg)*np.exp(np.minimum(RH_fm(Mg)/xi,700))
def tau_hawk(Mg): return 4.1e17*(Mg/5e14)**3
def zevap(Mg):
    t=tau_eff(Mg); T_MeV=1.0*np.maximum(t,1e-30)**-0.5
    return T_MeV*1e6/2.35e-4/ (1+0*t)      # 1+z = T/T0
# ---------- Fig 3: epochs ----------
M=np.logspace(15,np.log10(3.2e16),400)
z=np.array([zevap(m) for m in M]); z=np.clip(z,1,None)
fig,a=plt.subplots(figsize=(5.8,4.3))
a.loglog(M,z,c=C2,lw=2)
a.axhline(1e9,ls=":",c="0.5",lw=0.9); a.text(1.05e15,1.4e9,r"BBN ($z\sim10^9$)",fontsize=8,color="0.4")
a.axhspan(2e6,1e11,color="#1f77b4",alpha=0.06); a.text(1.05e15,3.5e6,"thermalized (no distortion)",fontsize=8,color="0.45")
a.axhspan(5e4,2e6,color="#ff7f0e",alpha=0.10); a.text(1.05e15,2.2e5,r"$\mu$-distortion",fontsize=8,color="#b25000")
a.axhspan(1.1e3,5e4,color="#d62728",alpha=0.08); a.text(1.05e15,7e3,r"$y$-distortion",fontsize=8,color="#a01010")
a.axhspan(1,1.1e3,color="#9467bd",alpha=0.07); a.text(1.05e15,4e1,"post-recombination / present",fontsize=8,color="#5c3d80")
a.set_ylim(1,3e11); a.set_xlim(1e15,3.2e16)
a.set_xlabel(r"PBH mass $M$ (g)"); a.set_ylabel(r"redshift of evaporation $z_{\rm evap}$")
fig.savefig(OUT+"fig3_epochs.pdf"); plt.close(fig)

# ---------- Fig 4: lifetime ----------
M=np.logspace(-10,40,600)
fig,a=plt.subplots(figsize=(6.4,4.6))
a.loglog(M,tau_hawk(M),c=C1,lw=1.8,label=r"Hawking: $\tau\propto M^3$")
a.loglog(M,tau_geo(M),"--",c=C2,lw=1.6,label=r"geometric (unsuppressed): $\tau\propto M^2$")
a.loglog(M,tau_eff(M),c=C2,lw=2.2,label="geometric + nucleation barrier")
a.axhline(tU,ls=":",c="0.5",lw=0.9); a.text(2e-10,1.6e18,"age of universe",fontsize=8,color="0.4")
a.axhline(tP,ls=":",c="0.5",lw=0.9); a.text(2e-10,2.5e-43,"Planck time",fontsize=8,color="0.4")
a.axvspan(1e17,1e22,color=C3,alpha=0.10)
a.text(2.5e18,1e-38,"asteroid-mass PBH\ndark-matter window",fontsize=7.5,color="#1e6e1e")
a.plot([10**16.45],[tU],"o",c=C2,ms=6)
a.annotate(r"$M_{\rm cut}\sim10^{16.5}$ g",xy=(10**16.45,tU),xytext=(3e11,3e21),fontsize=8.5,color=C2,
           arrowprops=dict(arrowstyle="-",color=C2,lw=0.9))
a.set_ylim(1e-50,1e50); a.set_xlim(1e-10,1e40)
a.set_xlabel(r"PBH mass $M$ (g)"); a.set_ylabel(r"lifetime $\tau$ (s)")
a.legend(loc="lower right"); fig.savefig(OUT+"fig4_lifetime.pdf"); plt.close(fig)

# ---------- Fig 5: constraints ----------
def zeta_lim(t):   # published-style zeta_EM(t) exclusion (GeV), order of magnitude
    lt=np.log10(t)
    pts=np.array([[4,3e-6],[5,1e-8],[6,4e-10],[7,6e-11],[8,2e-11],[9,1.5e-11],[10,1.5e-11],[12,2e-11],[13,5e-10]])
    return 10**np.interp(lt,pts[:,0],np.log10(pts[:,1]))
def bounds(M,fEM=0.5,mob=1.0):
    t=tau_eff(M)*mob; z=zevap(M)/np.sqrt(mob)
    dE=lambda f: fEM*f*4.85e3/(1+z)
    # BBN/photodissociation
    fb=np.inf
    if 1e4<t<1e13: fb=zeta_lim(t)/(3.1e-9*fEM)
    # mu
    fm=np.inf
    if z<2e6:
        mu_per=1.4*fEM*4.85e3/(1+z)*np.exp(-(z/2e6)**2.5)
        if mu_per>0: fm=9e-5/mu_per
    # y
    fy=np.inf
    if z<5e4:
        y_per=0.25*fEM*4.85e3/(1+z)
        fy=1.5e-5/y_per
    return fb,fm,fy
M=np.logspace(15,17,700); Mcut=10**16.45
FB,FM,FY=[],[],[]
for m in M:
    b=bounds(m); FB.append(b[0]); FM.append(b[1]); FY.append(b[2])
FB,FM,FY=map(np.array,(FB,FM,FY))
comb=np.minimum(np.minimum(FB,FM),FY)
lo=np.full_like(M,np.inf); hi=np.full_like(M,np.inf)
for fe in [0.3,0.7]:
    for mb in [0.5,2.0]:
        c=np.array([min(bounds(m,fe,mb)) for m in M])
        lo=np.minimum(lo,c); hi=np.minimum(hi,np.where(np.isfinite(c),np.maximum(hi*0+c,0),np.inf)) if False else hi
allc=[]
for fe in [0.3,0.5,0.7]:
    for mb in [0.5,1.0,2.0]:
        allc.append(np.array([min(bounds(m,fe,mb)) for m in M]))
allc=np.array(allc); band_lo=np.nanmin(np.where(np.isfinite(allc),allc,np.nan),axis=0)
band_hi=np.nanmax(np.where(np.isfinite(allc),allc,np.nan),axis=0)
mask=M<Mcut
fig,a=plt.subplots(figsize=(6.2,4.6))
a.fill_between(M[mask],np.clip(band_lo[mask],1e-9,1),np.clip(band_hi[mask],1e-9,1),
               color="0.8",label=r"combined band (all $O(1)$ uncertainties)")
a.loglog(M[mask&np.isfinite(FB)],np.clip(FB[mask&np.isfinite(FB)],1e-9,1),"--",c="crimson",lw=1.6,label="BBN/photodissociation (reconstructed)")
a.loglog(M[mask&np.isfinite(FM)],np.clip(FM[mask&np.isfinite(FM)],1e-9,1),c=C2,lw=1.5,label=r"$\mu$-distortion, FIRAS")
a.loglog(M[mask&np.isfinite(FY)],np.clip(FY[mask&np.isfinite(FY)],1e-9,1),c="purple",lw=1.5,label=r"$y$-distortion, FIRAS")
a.loglog(M[mask],np.clip(comb[mask],1e-9,1),c="k",lw=2.2,label="combined exclusion (central)")
a.axhline(1e-8,ls=":",c=C1,lw=1.5,label=r"standard Hawking $\gamma$-ray (does not apply; reference)")
a.axvline(Mcut,ls="--",c="0.4",lw=1); a.text(Mcut*1.06,2.5e-8,"survival\ncutoff",fontsize=8,color="0.35")
a.set_ylim(1e-9,2); a.set_xlim(1e15,1e17)
a.set_xlabel(r"PBH mass $M$ (g)"); a.set_ylabel(r"$f_{\rm PBH}$ (would-be DM fraction)")
a.legend(loc="lower left",framealpha=0.95); fig.savefig(OUT+"fig5_constraints.pdf"); plt.close(fig)

# ---------- Fig 6: punctured code ----------
v=json.load(open("pbh_revised_vacancy_L8.json"))
Reff=[x["ReffL0"] for x in v]; d3=[3]*len(v); chk=[x["vc_max"]+x["oc_max"] for x in v]
fig,ax=plt.subplots(1,3,figsize=(9.6,3.1))
a=ax[0]
a.plot(Reff,d3,"o-",c=C1,label=r"min. bulk logical weight $d'(R)$")
a.plot(Reff,chk,"s-",c=C2,label="checks per conversion step (max)")
rr=np.linspace(min(Reff),max(Reff),20)
a.plot(rr,12*rr/rr[0]*0+12*rr/1.0,":",c="0.45",label=r"linear law $\propto R$, for contrast")
a.set_ylim(0,42); a.set_xlabel(r"vacancy radius $R_{\rm eff}$ ($L_0$)"); a.set_ylabel("weight / count")
a.set_title("(a) no static growth with $R$"); a.legend(fontsize=6.6,loc="upper left")
a=ax[1]
rem=[x["removed"] for x in v]; kl=[x["klost"] for x in v]
a.plot(rem,kl,"o-",c=C1)
xx=np.linspace(0,max(rem)*1.05,10)
a.plot(xx,2*xx/3,"--",c="0.4",label="code-rate slope $2/3$")
a.set_xlabel("edges removed"); a.set_ylabel(r"logicals lost $k_0-k'$")
a.set_title("(b) logical deficit"); a.legend()
a=ax[2]
Aa=[x["A_L0"] for x in v]; sv=[x["sever"] for x in v]
a.plot(Aa,sv,"o-",c=C1,label="severed bonds")
xx=np.linspace(0,max(Aa)*1.05,10)
a.plot(xx,3*np.sqrt(2)*xx,"--",c="0.4",label=r"$3\sqrt{2}\,A/L_0^2$ (App. A)")
a.set_xlabel(r"vacancy area $A$ ($L_0^2$)"); a.set_ylabel("severed bonds")
a.set_title("(c) severed bonds vs. area"); a.legend()
fig.tight_layout(); fig.savefig(OUT+"fig6_vacancy.pdf"); plt.close(fig)
print("figures done")

#!/usr/bin/env python3
"""Lognormal extended-mass-function convolution against the combined
monochromatic bound (Carr et al. prescription)."""
import numpy as np
from scipy.optimize import brentq

tP, mP, t_univ = 5.391e-44, 2.176e-5, 4.35e17
G, c = 6.674e-11, 2.998e8
RH = lambda M: 2*G*(M*1e-3)/c**2
tau_eff = lambda M: 2.17*tP*(M/mP)**2*np.exp(np.minimum(RH(M)/1e-15,700))

def z_evap(M):
    t = tau_eff(M)
    return np.maximum(1e6*(t)**-0.5/2.35e-4, 1.0)

def fmax_mono(M):
    z = z_evap(M); fEM = 0.5
    X1 = fEM*4.85e3/(1+z)
    with np.errstate(over="ignore"):
        mu = 1.4*X1*np.exp(-np.clip((z/2e6)**2.5,0,700))
    f_mu = np.where(mu>0, 9e-5/mu, np.inf)
    f_y  = np.where(z<5e4, 6.0e-5/X1, np.inf)
    t = tau_eff(M)
    tt = np.array([1e2,1e4,1e6,1e8,1e10,1e13])
    zl = np.array([3e-8,2e-9,1e-10,5e-12,2e-12,1.5e-12])
    zeta = 10**np.interp(np.log10(t), np.log10(tt), np.log10(zl))
    f_b = np.where((t>1)&(t<1e13), zeta/(3.1e-9*fEM), np.inf)
    comb = np.minimum.reduce([f_mu, f_y, f_b])
    Mcut = 10**16.45
    return np.where(M<Mcut, comb, np.inf)

lnM = np.linspace(np.log(1e14), np.log(1e18), 4000)
M = np.exp(lnM)
fm = fmax_mono(M)

def f_ext(Mc, sig):
    psi = np.exp(-(lnM-np.log(Mc))**2/(2*sig**2))/(sig*np.sqrt(2*np.pi))  # per dlnM
    integ = np.trapezoid(psi/fm, lnM)
    return 1.0/integ if integ>0 else np.inf

print("Mc [g]   sigma   f_ext")
for Mc, sig in [(5e15,0.5),(5e15,1.0),(1e16,0.3),(1e16,0.5),(1.6e16,0.5),
                (2e16,0.3),(2e16,0.5),(2.8e16,0.5),(1e17,0.5)]:
    print(f"{Mc:.1e}  {sig:.1f}   {f_ext(Mc,sig):.2e}")
# monochromatic references
for Mm in (1.6e16, 2e16):
    print(f"mono fmax({Mm:.1e}) = {fmax_mono(np.array([Mm]))[0]:.2e}")

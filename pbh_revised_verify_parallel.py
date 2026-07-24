#!/usr/bin/env python3
"""Independent verification of the parallel draft's quantitative claims."""
import numpy as np
from scipy.optimize import brentq

L0 = 1.0
# six independent FCC bond directions (unit vectors)
dirs = []
for i in range(3):
    for j in range(i + 1, 3):
        for s in (1, -1):
            v = np.zeros(3); v[i] = 1; v[j] = s
            dirs.append(v / np.sqrt(2))
dirs = np.array(dirs)  # 6 pairs
nv = np.sqrt(2) / L0**3   # FCC site density: 1/Vprim, Vprim = L0^3/sqrt2

def N(nhat):
    nhat = nhat / np.linalg.norm(nhat)
    return nv * L0 * np.sum(np.abs(dirs @ nhat))

print("== orientation table (N * L0^2) ==")
for name, n in [("(111)", [1,1,1]), ("(211)", [2,1,1]), ("(311)", [3,1,1]),
                ("(100)", [1,0,0]), ("(110)", [1,1,0])]:
    print(f"  {name}: {N(np.array(n,float)):.4f}")
# sphere average by scan
rng = np.random.default_rng(1)
v = rng.normal(size=(2_000_000, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
scan = nv * L0 * np.mean(np.sum(np.abs(v @ dirs.T), axis=1))
print(f"  scan average (2e6): {scan:.5f}  vs 3*sqrt2 = {3*np.sqrt(2):.5f}")
print(f"  min over scan: {nv*L0*np.min(np.sum(np.abs(v@dirs.T),axis=1)):.4f} (expect 2sqrt3={2*np.sqrt(3):.4f})")
print(f"  max over scan: {nv*L0*np.max(np.sum(np.abs(v@dirs.T),axis=1)):.4f}")

print("== Wulff octahedron vs sphere ==")
# regular octahedron with insphere radius r: a = r*sqrt(6), S = 2*sqrt3*a^2, V = sqrt2/3 a^3
# compare at equal volume with sphere
a = np.sqrt(6.0)  # insphere r=1
S_oct = 2*np.sqrt(3)*a**2; V_oct = np.sqrt(2)/3*a**3
R_eq = (3*V_oct/(4*np.pi))**(1/3.)
S_sph = 4*np.pi*R_eq**2
print(f"  A_oct/A_sph (equal V) = {S_oct/S_sph:.4f}   (claim 1.1826)")
ratio = (2*np.sqrt(3)) * S_oct / (3*np.sqrt(2) * S_sph)
print(f"  N111*A_oct / (<N>*A_sph) = {ratio:.4f}   (claim 0.9656)")

print("== nu* ==")
A1 = np.sqrt(2/3.)
print(f"  nu* = <N>*A1 = {3*np.sqrt(2)*A1:.4f} = 2sqrt3 = {2*np.sqrt(3):.4f}")

print("== cutoffs and slopes ==")
tP, mP, t_univ, G, c = 5.391e-44, 2.176e-5, 4.35e17, 6.674e-11, 2.998e8
RH = lambda M: 2*G*(M*1e-3)/c**2
tau0 = lambda M: 2.17*tP*(M/mP)**2
for name, expo in [("linear", 1), ("areal", 2)]:
    f = lambda lm: np.log(tau0(10**lm)) + (RH(10**lm)/1e-15)**expo - np.log(t_univ)
    lm = brentq(f, 14.0, 18.0)
    M = 10**lm
    # slope d log tau / d log M
    dl = 1e-4
    slope = (np.log(tau0(10**(lm+dl))) + (RH(10**(lm+dl))/1e-15)**expo
             - np.log(tau0(10**(lm-dl))) - (RH(10**(lm-dl))/1e-15)**expo) / (2*dl*np.log(10))
    print(f"  {name}: Mcut = 10^{lm:.2f} g, RH = {RH(M)*1e15:.1f} fm, dlogtau/dlogM = {slope:.0f}")
lm = brentq(lambda lm: np.log(tau0(10**lm)) - np.log(t_univ), 20, 30)
print(f"  unsuppressed M^2 reaches t_univ at 10^{lm:.2f} g")
print(f"  rough interface: tau(1e17 g) = {tau0(1e17):.2f} s, tau(1e22 g) = {tau0(1e22):.2e} s")

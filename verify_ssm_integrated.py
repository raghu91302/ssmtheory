#!/usr/bin/env python3
"""verify_ssm_integrated.py

Numerical verification of the analytic constants quoted in the manuscript
'Emergent Face-Centered Cubic Vacuum from Discrete Entanglement Networks'
(Parts I and II).

This script belongs to that manuscript only. It is distinct from the scripts
supporting the published companion papers and does not supersede them.

Requires numpy only. Run time: ~2 minutes.

Sections:
  1. Lieb-Robinson front on the FCC bond network        (Part II, Sec. lightcone)
  2. Acoustic branch speeds and the trace identity      (Part II, Sec. consequences)
  3. Scalar-dispersion anisotropy order                 (Part I, Sec. isotropy)
  4. Finite-size scaling effective amplitude            (Part I, Sec. scaling)
  5. Eclipsed (AA) stacking under hard-core exclusion   (Part II, Sec. substrate)
  6. Regge deficit and tetrahedral-octahedral closure   (Part II, Sec. fcchcp)
  7. Elastic constants and the isotropy no-go           (Part II, Sec. consequences)
  8. Lindemann constants by Brillouin-zone integration  (Part II, Sec. rigidity)
  9. Rigidity-matrix dissolution (rank criterion)       (Part II, Sec. stability)
 10. FCC vs HCP nearest-neighbor shell degeneracy      (Part II, Sec. fcchcp)
 11. Derived scale relations                            (Part II, Sec. scales)
 12. Construction velocity and growth anisotropy       (Part II, Sec. velocity)
"""
import numpy as np
from itertools import combinations
from math import comb

np.set_printoptions(precision=4, suppress=True)

raw = [(1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
       (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
       (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)]
n = np.array(raw, float) / np.sqrt(2)      # 12 FCC bond vectors, |n| = L = 1


print("== 1. Lieb-Robinson front on the FCC bond network ==")
# Single-excitation hopping between bonds sharing a node is tight binding on the
# line graph of FCC. For a K-regular vertex-transitive graph, A_line = B^T B - 2I
# and B B^T = A_node + K I, so the dispersive line-graph band is the node band
# shifted by K-2: identical group velocities. Node band E(k) = -Gamma sum_j cos(k.n_j).
rng = np.random.default_rng(1)
vmax = 0.0
for _ in range(300000):
    k = rng.uniform(-2 * np.pi, 2 * np.pi, 3)
    vmax = max(vmax, np.linalg.norm((np.sin(n @ k)[:, None] * n).sum(0)))
print("  numerical max |grad E| = %.4f Gamma L/hbar ; analytic 4*sqrt(2) = %.4f"
      % (vmax, 4 * np.sqrt(2)))
print("  attained along [100]. A 1D bond chain gives exactly 2, the chain reference value.")


print("== 2. Acoustic branches and the trace identity ==")
def speeds(khat, eps=1e-4):
    q = eps * np.asarray(khat, float) / np.linalg.norm(khat)
    M = np.zeros((3, 3))
    for v in n:
        M += (1 - np.cos(q @ v)) * np.outer(v, v)
    return np.sqrt(np.linalg.eigvalsh(M)) / eps        # units L sqrt(kappa/m)

for lab, d in [("100", [1, 0, 0]), ("110", [1, 1, 0]), ("111", [1, 1, 1])]:
    s = np.sort(speeds(d))[::-1]
    print("  [%s] v = %s ; sum v^2 = %.4f (= c_s^2 = 2)" % (lab, np.round(s, 4), (s ** 2).sum()))
u = 1 / np.sqrt(2)      # with c_s = c: L sqrt(kappa/m) = c/sqrt(2)
print("  => v_L in [%.3f, %.3f] c ; v_T in [%.3f, %.3f] c"
      % (1 * u, np.sqrt(4 / 3) * u, 0.5 * u, (1 / np.sqrt(2)) * u))
print("  longitudinal anisotropy sqrt(4/3)-1 = %.3f ; transverse sqrt(2)-1 = %.3f"
      % (np.sqrt(4 / 3) - 1, np.sqrt(2) - 1))
print("  kappa from c_s = c with c = 4*sqrt(2) v_lat:  kappa = m c^2/(2 L^2) = 16 m eps^2/hbar^2")


print("== 3. Scalar-dispersion anisotropy order ==")
def w2(k):
    return sum(1 - np.cos(k @ v) for v in n)
for e in [0.2, 0.1, 0.05]:
    r = w2(np.array([1, 1, 1.]) / np.sqrt(3) * e) / w2(np.array([1, 0, 0.]) * e) - 1
    print("  ka=%.2f : rel. anisotropy = %.3e ; /(ka)^2 = %.5f (-1/72 = %.5f)"
          % (e, r, r / e ** 2, -1 / 72))


print("== 4. Part I: bond-set isotropy, centrosymmetry, and bond bookkeeping ==")
# Rank-2 structure tensor of the 12 nearest-neighbour bond directions (Eq. Sexact)
S = sum(np.outer(v, v) for v in n)
print("  S_munu diagonal      =", np.round(np.diag(S), 12))
print("  max |off-diagonal|   = %.2e" % np.abs(S - np.diag(np.diag(S))).max())
print("  S_munu = 4 delta_munu:", np.allclose(S, 4 * np.eye(3), atol=1e-12))
# Rank-3 tensor vanishes by inversion symmetry (Eq. Tzero)
T = np.zeros((3, 3, 3))
for v in n:
    T += v[:, None, None] * v[None, :, None] * v[None, None, :]
print("  max |T_munulambda|   = %.2e  -> centrosymmetric: %s"
      % (np.abs(T).max(), np.abs(T).max() < 1e-12))
# Exact bond bookkeeping B = 3n - 6 - s for a history of s stitches and l lifts
bad = [(st, lf) for st in range(12) for lf in range(12)
       if 3 + 2 * st + 3 * lf != 3 * (3 + st + lf) - 6 - st]
print("  B = 3n-6-s holds for all (s,l) up to 11:", not bad)
print("  => B_max(N) = 3N-6, attained by all-lift histories; deficit m = 3(N-n)+s")


print("== 5. Eclipsed (AA) stacking at close-packed spacing ==")
print("  aligned-node distance sqrt(2/3) L = %.4f L < R_ex = 0.95 L -> hard-core excluded"
      % np.sqrt(2 / 3))


print("== 6. Regge deficit and tetrahedral-octahedral closure ==")
delta = 2 * np.pi - 5 * np.arccos(1 / 3)
print("  five regular tetrahedra about an edge leave")
print("    delta = 2pi - 5 arccos(1/3) = %.4f rad = %.3f deg ; delta/2pi = %.5f"
      % (delta, np.degrees(delta), delta / (2 * np.pi)))
tT = np.arccos(1 / 3); tO = np.arccos(-1 / 3)
print("  tet-oct honeycomb: theta_T = %.6f, theta_O = %.6f, theta_T + theta_O = pi: %s"
      % (tT, tO, abs(tT + tO - np.pi) < 1e-15))
print("    2 theta_T + 2 theta_O - 2pi = %.2e  (edge deficit vanishes identically)"
      % abs(2 * tT + 2 * tO - 2 * np.pi))


print()
print("== 7. NN axially-symmetric force constants: the isotropy no-go ==")
def elastic(alpha, beta):
    # Phi_j = alpha n n^T + beta (1 - n n^T); D(q) = (1/m) sum (1-cos q.r) Phi
    def sp(khat, eps=1e-4):
        q = eps * np.asarray(khat, float) / np.linalg.norm(khat)
        M = np.zeros((3, 3))
        for v in n:
            P = np.outer(v, v)
            M += (1 - np.cos(q @ v)) * (alpha * P + beta * (np.eye(3) - P))
        return np.sort(np.linalg.eigvalsh(M)) / eps ** 2
    s100 = sp([1, 0, 0]); s110 = sp([1, 1, 0])
    C11 = s100[2]; C44 = s100[0]
    tA, tB = s110[0], s110[1]
    half = (tA if abs(tB - C44) < abs(tA - C44) else tB)
    C12 = C11 - 2 * half
    return C11, C12, C44

for r in [0.0, -0.1, -0.2, -1 / 3, -0.5, 0.2]:
    C11, C12, C44 = elastic(1.0, r)
    iso = C11 - C12 - 2 * C44
    A = 2 * C44 / (C11 - C12) if abs(C11 - C12) > 1e-9 else float('nan')
    print("  beta/alpha=%6.3f  C11=%.3f C12=%.3f C44=%.3f  Zener A=%.3f  C11-C12-2C44=%.3f"
          % (r, C11, C12, C44, A, iso))
print("  closed forms: C11 = a+b, C12 = (a-5b)/2, C44 = (a+3b)/2 ; isotropy condition = (b-a)/2")
print("  => isotropy only at b=a (scalar spring), where lambda=-mu and K_bulk=-mu/3<0: unstable")
print("  independently: v_L=v_T => lambda+2mu=mu => lambda=-mu => K_bulk<0 for any nonzero shear")


print()
print("== 8. Lindemann constants by Brillouin-zone integration ==")
a = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float) / np.sqrt(2)
b = 2 * np.pi * np.linalg.inv(a).T
d = n[0]                      # one NN direction; full-BZ sum symmetry-averages
win = (1 - np.sqrt(2 - np.sqrt(2))) ** 2      # (L - R*)^2
for Ng in [16, 24, 32]:
    Xr = Xf = Yf = 0.0
    for i in range(Ng):
        for j in range(Ng):
            for k in range(Ng):
                q = ((np.array([i, j, k]) + 0.5) / Ng) @ b
                M = np.zeros((3, 3))
                for v in n:
                    M += (1 - np.cos(q @ v)) * np.outer(v, v)
                w2v, V = np.linalg.eigh(M)
                w = np.sqrt(np.maximum(w2v, 1e-18))
                oc = 1 - np.cos(q @ d)
                proj = (V.T @ d) ** 2
                Xf += np.sum(oc / w)            # full 3D relative displacement, T=0
                Xr += np.sum(proj * oc / w)     # radial-projected relative, T=0
                Yf += np.sum(oc / w ** 2)       # full relative, classical
    Nq = Ng ** 3
    print("  grid %2d^3: rel-3D X=%.4f -> ZP floor=%.1f | radial X=%.4f -> floor=%.1f"
          " | thermal Y=%.4f -> kT/(kappa L^2) < %.4f"
          % (Ng, Xf / Nq, (Xf / Nq) / win, Xr / Nq, (Xr / Nq) / win, Yf / Nq, win / (2 * Yf / Nq)))
print("  rigidity window (L-R*)^2 = %.4f with R* = sqrt(2-sqrt2) L = %.5f L"
      % (win, np.sqrt(2 - np.sqrt(2))))
print("  quoted in text: zero-point floor 28.6, thermal bound 0.031 (relative-displacement convention)")


print()
print("== 9. Rigidity-matrix dissolution (rank criterion) ==")
q = 1 / (1 + np.e)
def pdiss_rank(dirs, need=3):
    K = len(dirs); tot = 0.0
    for mask in range(2 ** K):
        B = [dirs[i] for i in range(K) if mask >> i & 1]
        r = np.linalg.matrix_rank(np.array(B), tol=1e-9) if B else 0
        if r < need:
            nb = len(B); tot += q ** (K - nb) * (1 - q) ** nb
    return tot
def pdiss_binom(K):
    return sum(comb(K, j) * q ** j * (1 - q) ** (K - j) for j in range(K - 2, K + 1))

tet = np.array([(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)], float) / np.sqrt(3)
hexp = np.array([(np.cos(t), np.sin(t), 0) for t in np.arange(6) * np.pi / 3])
print("  K=12 cuboct : rank criterion P=%.3e | binomial (K-2 rule)=%.3e"
      % (pdiss_rank(n), pdiss_binom(12)))
print("  K=4  tetra  : rank criterion P=%.3e | binomial=%.3e  (coincide exactly)"
      % (pdiss_rank(tet), pdiss_binom(4)))
print("  K=6  sheet  : rank-3 P=%.3f (always rank<=2 out of plane) ; in-plane rank-2 P=%.3e"
      % (pdiss_rank(hexp, 3), pdiss_rank(hexp, 2)))
lc = 1 / (0.05505 * 36)
print("  Lindemann-tied threshold lambda_c/kappa = 1/(0.0551*36) = %.3f" % lc)
print("  K=4 at that threshold requires all four bonds: P = 1-(1-q)^4 = %.3f" % (1 - (1 - q) ** 4))


print()
print("== 10. FCC vs HCP nearest-neighbor shell degeneracy ==")
inp = [(np.cos(t), np.sin(t), 0) for t in np.arange(6) * np.pi / 3]
h = np.sqrt(2 / 3.); r3 = 1 / np.sqrt(3)
up = [(r3 * np.cos(t), r3 * np.sin(t), h) for t in (np.pi / 6 + np.arange(3) * 2 * np.pi / 3)]
dnF = [(r3 * np.cos(t), r3 * np.sin(t), -h) for t in (np.pi / 2 + np.arange(3) * 2 * np.pi / 3)]
dnH = [(r3 * np.cos(t), r3 * np.sin(t), -h) for t in (np.pi / 6 + np.arange(3) * 2 * np.pi / 3)]
def counts(shell):
    S = np.array(shell); E = T = 0
    for i, j in combinations(range(12), 2):
        if abs(np.linalg.norm(S[i] - S[j]) - 1) < 1e-9:
            E += 1
    for i, j, k in combinations(range(12), 3):
        if all(abs(np.linalg.norm(S[x] - S[y]) - 1) < 1e-9 for x, y in [(i, j), (j, k), (i, k)]):
            T += 1
    return E, T
print("  FCC shell: %d unit edges, %d unit triangles | HCP shell: %d unit edges, %d unit triangles"
      % (*counts(inp + up + dnF), *counts(inp + up + dnH)))
print("  => NN pair energy and the three-body triangle term H3 are both exactly degenerate")


print()
print("== 11. Derived scale relations ==")
print("  node inertia: m > 28.6^2/36 = %.1f eps/v_lat^2 ; c^2 = 32 v_lat^2 => m c^2 > %.0f eps"
      % (28.6 ** 2 / 36, 32 * 28.6 ** 2 / 36))
print("  binding floor: kappa L^2 >= 36 eps with kappa = 16 m eps^2/hbar^2"
      " => eps >= 36/16 = %.2f hbar^2/(m L^2)" % (36 / 16))
print("  critical temperature: T_c ~ 0.031 * 33 = %.2f eps/k_B ~ T_0" % (0.031 * 33))
print("  geometric small parameter: delta/2pi = %.4f" % (delta / (2 * np.pi)))
print("  vacuum binding density: -eps * 6*sqrt(2)/L^3 = %.2f eps/L^3" % (-6 * np.sqrt(2)))


print()
print("== 12. Construction velocity and growth anisotropy ==")
q_db = 1 / (1 + np.e)
pref = 1 - 2 * q_db                       # net forward rate factor
P_lift = np.exp(-3)
lat_step = np.sqrt(3) / 2                 # in-plane apex advance, units of L
lift_step = np.sqrt(2 / 3)                # interlayer spacing, units of L
v2d = pref * lat_step
v_stack = pref * P_lift * lift_step
print("  q = %.4f, (1-2q) = %.4f, P_lift = e^-3 = %.5f" % (q_db, pref, P_lift))
print("  v_2D    = %.4f v_lat   (quoted 0.40)" % v2d)
print("  v_stack = %.5f v_lat   (quoted 0.019)" % v_stack)
print("  v_2D / v_sig = %.4f  -> 1/%.1f   (quoted 0.07, one-fourteenth)"
      % (v2d / (4 * np.sqrt(2)), 4 * np.sqrt(2) / v2d))
print("  anisotropy v_2D/v_stack = (1/P_lift)(step ratio) = e^3 x %.3f = %.1f  (quoted ~21)"
      % (lat_step / lift_step, v2d / v_stack))
print("  bare rate ratio 1/P_lift = e^3 = %.2f  (quoted ~20)" % (1 / P_lift))
print("  sheet widening per added layer: %.1f L" % (v2d / v_stack * lift_step))

print("  single-column aspect vs measured (Table sweep, N=1000, 30 seeds):")
geo = lat_step / lift_step
sweep = [(0.01, 0.59), (0.03, 0.76), (0.0498, 0.83), (0.10, 0.84),
         (0.15, 0.90), (0.30, 0.92), (0.50, 0.96), (0.85, 0.99)]
print("     P_lift   ratio   1-column z/xy   measured   factor")
for p, asp in sweep:
    r = geo / p
    print("    %6.3f  %7.1f   %13.3f   %8.2f   %5.1fx" % (p, r, 1 / r, asp, asp * r))
print("  (factor falls monotonically 63 -> 1.2: the branching signature)")

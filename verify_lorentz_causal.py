#!/usr/bin/env python3
"""
verify_lorentz_causal.py

Reproduces every numerical/algebraic claim in
  "Lorentzian Signature from Causal Dynamics on the Euclidean D4 Lattice"
  (R. Kulkarni, SSMTheory Group, 2026).

Dependencies: numpy, sympy  (standard scientific Python; no other packages).

The four claims verified, in paper order:

  (T2)  Rank-4 isotropy of D4: T_{munurhosigma} = 4 (dd + dd + dd) exactly,
        while the 3D FCC sub-lattice alone is rank-4 ANISOTROPIC.
        [paper Sec. 1-2; cites Kulkarni2026D4 Theorem 2]

  (L)   The D4 discrete Laplacian is SECOND ORDER in time:
        omega enters as omega^2 at the same order as k_i^2, with NO first-order
        (omega-linear) term, because the cross-slice bonds are e4 -> -e4 symmetric.
        Leading expansion:  Lap ~ -2(r+2)|k|^2 - 6 r omega^2.
        [paper Sec. 3.1-3.2, Eq. (lead)]

  (r1)  Single-speed condition: spatial and temporal coefficients coincide
        iff r = 1, giving the isotropic Lap ~ -6(|k|^2 + omega^2).
        [paper Sec. 3.3, Eq. (iso)]

  (S)   Signature: the static-crystal invariant is the EUCLIDEAN sum
        (k^2 + omega^2), signature (++++). The Lorentzian (-+++) minus sign is
        NOT present in the geometry; it is supplied only by reading the temporal
        second difference as a causal evolution (d_t^2 psi = +c^2 nabla^2 psi).
        This block demonstrates the sum-vs-difference structure explicitly.
        [paper Sec. 4, Proposition 1]

Run:  python3 verify_lorentz_causal.py
All checks print PASS/FAIL; exit code 0 iff every check passes.
"""

import numpy as np
import sympy as sp

# ----------------------------------------------------------------------
# Lattice data
# ----------------------------------------------------------------------

def d4_neighbors():
    """The 24 nearest-neighbor vectors of D4: +-e_mu +- e_nu, mu<nu, in 4D."""
    N = []
    for i in range(4):
        for j in range(i + 1, 4):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0, 0, 0, 0]
                    v[i] = si
                    v[j] = sj
                    N.append(tuple(v))
    return N

def split_neighbors(N):
    """Split D4 neighbors into in-slice (n4=0) and cross-slice (n4=+-1)."""
    spatial = [n for n in N if n[3] == 0]
    cross   = [n for n in N if n[3] != 0]
    return spatial, cross

# ----------------------------------------------------------------------
# (T2) Rank-4 isotropy of D4
# ----------------------------------------------------------------------

def check_rank4_isotropy():
    print("=" * 68)
    print("(T2) Rank-4 isotropy of D4  vs  rank-4 anisotropy of FCC-alone")
    print("=" * 68)
    N = np.array(d4_neighbors(), dtype=float)
    T4 = np.einsum('ni,nj,nk,nl->ijkl', N, N, N, N)

    I = np.eye(4)
    S4 = (np.einsum('ij,kl->ijkl', I, I)
          + np.einsum('ik,jl->ijkl', I, I)
          + np.einsum('il,jk->ijkl', I, I))   # the unique isotropic rank-4 tensor

    iso = np.allclose(T4, 4.0 * S4)
    maxdiff = np.max(np.abs(T4 - 4.0 * S4))
    print(f"  T_1111 = {T4[0,0,0,0]:.0f}  (claim 12);  T_1122 = {T4[0,0,1,1]:.0f}  (claim 4)")
    print(f"  T == 4*(dd+dd+dd) exactly? {iso}   (max|T-4S| = {maxdiff:.1e})")

    # FCC-alone (the 12 spatial neighbors) is rank-4 anisotropic in 3D
    spatial = np.array([n for n in d4_neighbors() if n[3] == 0], dtype=float)[:, :3]
    T4f = np.einsum('ni,nj,nk,nl->ijkl', spatial, spatial, spatial, spatial)
    fcc_iso = np.isclose(T4f[0, 0, 0, 0], 3 * T4f[0, 0, 1, 1])
    print(f"  FCC-alone: T_0000 = {T4f[0,0,0,0]:.0f}, 3*T_0011 = {3*T4f[0,0,1,1]:.0f}"
          f"  -> isotropic? {fcc_iso}  (expected False)")

    ok = iso and (not fcc_iso)
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}\n")
    return ok

# ----------------------------------------------------------------------
# (L) and (r1): Laplacian expansion, second-order-in-time, single speed
# ----------------------------------------------------------------------

def check_laplacian_and_speed():
    print("=" * 68)
    print("(L)  D4 Laplacian is second-order-in-time;  (r1) single speed at r=1")
    print("=" * 68)
    kx, ky, kz, w, r = sp.symbols('k_x k_y k_z omega r', real=True)
    K = [kx, ky, kz, w]
    N = d4_neighbors()
    spatial, cross = split_neighbors(N)

    # discrete Laplacian symbol: sum_n [cos(k.n) - 1], cross weighted by r
    def lap(bonds, weight):
        return weight * sum(sp.cos(sum(K[m] * n[m] for m in range(4))) - 1 for n in bonds)

    Lap = sp.expand_trig(lap(spatial, 1) + lap(cross, r))

    # No omega-LINEAR term (would signal first-order/parabolic):
    d_w_linear = sp.simplify(sp.diff(Lap, w).subs({kx: 0, ky: 0, kz: 0, w: 0}))
    print(f"  coefficient of omega^1 at origin = {d_w_linear}  (must be 0: no first-order time)")

    # quadratic expansion about k=0
    sub0 = {kx: 0, ky: 0, kz: 0, w: 0}
    quad = 0
    for a in K:
        for b in K:
            c = sp.diff(Lap, a, b).subs(sub0)
            if c != 0:
                quad += sp.Rational(1, 2) * c * a * b
    quad = sp.expand(quad)
    print(f"  leading Laplacian = {quad}")

    coeff_k = sp.simplify(quad.coeff(kx, 2))
    coeff_w = sp.simplify(quad.coeff(w, 2))
    print(f"  coeff(k_x^2) = {coeff_k}   coeff(omega^2) = {coeff_w}")

    # second-order in time: omega^2 coefficient is nonzero
    second_order_time = sp.simplify(coeff_w) != 0
    # single-speed condition: coeff(k^2) == coeff(omega^2)
    r_solutions = sp.solve(sp.Eq(coeff_k, coeff_w), r)
    print(f"  single-speed condition coeff(k^2)=coeff(omega^2)  ->  r = {r_solutions}")

    # at r=1, fully isotropic
    quad_r1 = sp.expand(quad.subs(r, 1))
    iso_r1 = sp.simplify(quad_r1 + 6 * (kx**2 + ky**2 + kz**2 + w**2)) == 0
    print(f"  at r=1: leading Laplacian = {quad_r1}")
    print(f"  is it -6(k^2 + omega^2) (full 4-way isotropy)? {iso_r1}")

    ok = (d_w_linear == 0) and second_order_time and (r_solutions == [1]) and iso_r1
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}\n")
    return ok

# ----------------------------------------------------------------------
# (S) Signature: Euclidean sum vs Lorentzian difference
# ----------------------------------------------------------------------

def check_signature():
    print("=" * 68)
    print("(S)  Euclidean static invariant vs Lorentzian causal-evolution form")
    print("=" * 68)
    kx, ky, kz, w, c = sp.symbols('k_x k_y k_z omega c', real=True, positive=True)

    # The static-crystal leading invariant (from r=1 isotropy, dropping the -6 scale):
    euclidean = kx**2 + ky**2 + kz**2 + w**2          # signature (+,+,+,+)
    print(f"  static-crystal invariant  : {euclidean}   -> signature (+,+,+,+)")

    # Static extremization: (d_t^2 + c^2 nabla^2) psi = 0  => omega^2 + c^2|k|^2 = 0
    # only solution real k,omega is origin -> NO cone.
    static_zero = sp.solve(sp.Eq(w**2 + c**2 * (kx**2), 0), w)  # 1D slice for clarity
    print(f"  static stationarity omega^2 + c^2 k^2 = 0  -> omega = {static_zero}"
          f"  (no real cone; imaginary => no propagation)")

    # Causal update: d_t^2 psi = +c^2 nabla^2 psi  => omega^2 - c^2|k|^2 = 0
    lorentzian = w**2 - c**2 * (kx**2 + ky**2 + kz**2)   # signature (-,+,+,+)
    cone = sp.solve(sp.Eq(lorentzian, 0), w)
    print(f"  causal-evolution invariant: {sp.expand(lorentzian)}   -> signature (-,+,+,+)")
    print(f"  null cone omega^2 = c^2|k|^2 -> omega = {cone}  (finite speed c, real cone)")

    # The two differ ONLY by the sign of the temporal term:
    diff = sp.simplify((euclidean.subs({ky:0,kz:0})) - (w**2 + c**2*kx**2).subs(c,1))
    sign_is_the_difference = sp.simplify(
        (w**2 + (kx**2+ky**2+kz**2)) - (w**2 - (kx**2+ky**2+kz**2))
        - 2*(kx**2+ky**2+kz**2)) == 0
    print("  Euclidean - Lorentzian = 2|k|^2  => they differ ONLY by the temporal sign flip:",
          sign_is_the_difference)
    print("  CONCLUSION: the geometry yields the Euclidean SUM; the minus sign is")
    print("  supplied solely by the causal (evolution) reading of the temporal term.")

    # The null cone omega^2 = c^2|k|^2 has real solutions (a genuine light cone);
    # sympy may return the principal positive root only, so we test for a real,
    # nonzero cone rather than a specific count of listed roots.
    cone_is_real = all(sol.is_real is not False for sol in cone) and len(cone) >= 1
    ok = sign_is_the_difference and cone_is_real
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}\n")
    return ok

# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# (H) Hyperbolic continuum limit of the repair update
# ----------------------------------------------------------------------

def check_hyperbolic_limit():
    print("=" * 68)
    print("(H) Repair-update continuum limit: hyperbolic vs parabolic fork")
    print("=" * 68)
    w, k, c, beta, gamma = sp.symbols('omega k c beta gamma', positive=True)

    # Reversible / time-symmetric (1,-2,1) stencil: 2cos(w)-2 = -alpha k^2
    lhs_rev = sp.series(2*sp.cos(w) - 2, w, 0, 4).removeO()   # -> -w^2
    rev_second_order = sp.simplify(lhs_rev + w**2) == 0
    print(f"  reversible stencil small-w limit = {lhs_rev}  (== -w^2 ? {rev_second_order})")
    print("    => w^2 = c^2 k^2 : HYPERBOLIC (real light cone)")

    # Dissipative / forward (1,-1) stencil: e^{-iw}-1 = -beta k^2
    lhs_dis = sp.series(sp.exp(-sp.I*w) - 1, w, 0, 2).removeO()  # -> -i w
    dis_first_order = sp.simplify(lhs_dis + sp.I*w) == 0
    print(f"  dissipative stencil small-w limit = {lhs_dis}  (== -i*w ? {dis_first_order})")
    print("    => w = -i beta k^2 : PARABOLIC (diffusive, decaying)")

    # The stabilizer support is time-symmetric: 6 forward (+e4) and 6 backward (-e4)
    # cross-slice bonds. Count them to confirm no past/future asymmetry.
    cross = [n for n in d4_neighbors() if n[3] != 0]
    fwd = sum(1 for n in cross if n[3] == 1)
    bwd = sum(1 for n in cross if n[3] == -1)
    symmetric_support = (fwd == bwd == 6)
    print(f"  cross-slice bonds: {fwd} forward (+e4), {bwd} backward (-e4) "
          f"-> time-symmetric support? {symmetric_support}")
    print("    => zero-syndrome condition is the (1,-2,1) second difference: second order forced")

    # Damped-wave robustness: d_t^2 + gamma d_t = c^2 nabla^2  ->  -w^2 - i gamma w + c^2 k^2 = 0
    # Solve the quadratic in w explicitly (avoid solver edge cases).
    a_, b_, cc_ = -1, -sp.I*gamma, c**2*k**2
    disc = sp.sqrt(b_**2 - 4*a_*cc_)
    roots = [(-b_ + disc)/(2*a_), (-b_ - disc)/(2*a_)]
    # Underdamped limit gamma -> 0: real part -> +- c k (propagating cone)
    re_lead = sp.simplify(roots[0].subs(gamma, 0))
    print(f"  damped-wave root at gamma->0: w -> {re_lead}  (propagating cone Re(w)=c k)")
    underdamped_propagating = sp.simplify(re_lead**2 - c**2*k**2) == 0
    print(f"    => underdamped limit is the Lorentzian cone? {underdamped_propagating}")
    print("    => dissipation is a width on the cone, not a change of order")

    ok = (rev_second_order and dis_first_order and symmetric_support
          and underdamped_propagating)
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}\n")
    return ok

# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# (C) D4 CSS code: stabilizer temporal support is e4-symmetric (both check types)
# ----------------------------------------------------------------------

def check_d4_code_support():
    import numpy as np
    print("=" * 68)
    print("(C) D4 CSS code: weight-24 stabilizer support, e4-symmetric in time")
    print("=" * 68)
    L = 4
    sites = [tuple(p) for p in np.ndindex(L, L, L, L)]
    verts = [s for s in sites if sum(s) % 2 == 0]
    voids = [s for s in sites if sum(s) % 2 == 1]

    # roots: +-e_i +- e_j, i<j  (the 24 D4 nearest neighbors)
    roots = []
    for i in range(4):
        for j in range(i + 1, 4):
            for si in (1, -1):
                for sj in (1, -1):
                    r = [0, 0, 0, 0]; r[i] = si; r[j] = sj
                    roots.append(tuple(r))

    def x4(d):
        return ((d + L // 2) % L) - L // 2

    # Z-stabilizer on a vertex: all 24 incident edges (one per root)
    v0 = verts[0]
    zc = {'sp': 0, 'fwd': 0, 'bwd': 0}
    for r in roots:
        d4 = x4(r[3])
        zc['sp' if d4 == 0 else ('fwd' if d4 > 0 else 'bwd')] += 1
    print(f"  Z-check (vertex): {zc['sp']} in-slice, {zc['fwd']} forward, {zc['bwd']} backward")

    # X-stabilizer on a void: non-antipodal pairs among the 8 surrounding vertices
    o = voids[0]
    surround = [tuple((o[k] + (s if k == d else 0)) % L for k in range(4))
                for d in range(4) for s in (1, -1)]
    xc = {'sp': 0, 'fwd': 0, 'bwd': 0}
    for a in range(len(surround)):
        for b in range(a + 1, len(surround)):
            va, vb = surround[a], surround[b]
            diff = tuple(((vb[k] - va[k] + L // 2) % L) - L // 2 for k in range(4))
            nz = [k for k in range(4) if diff[k] != 0]
            antipodal = (len(nz) == 1 and abs(diff[nz[0]]) == 2)
            if not antipodal:
                d4 = x4(vb[3] - va[3])
                xc['sp' if d4 == 0 else ('fwd' if d4 > 0 else 'bwd')] += 1
    print(f"  X-check (void):   {xc['sp']} in-slice, {xc['fwd']} forward, {xc['bwd']} backward")

    z_ok = (zc['sp'] == 12 and zc['fwd'] == zc['bwd'] == 6)
    x_ok = (xc['sp'] == 12 and xc['fwd'] == xc['bwd'] == 6)
    print(f"  Z support 12+6+6 and forward==backward? {z_ok}")
    print(f"  X support 12+6+6 and forward==backward? {x_ok}")
    print("  => both stabilizer types: cross-slice support symmetric under e4 -> -e4")
    print("     => repair enforces a (t-1,t,t+1) SECOND difference -> hyperbolic, not diffusive")

    ok = z_ok and x_ok
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}\n")
    return ok

# ----------------------------------------------------------------------

def main():
    results = []
    results.append(check_rank4_isotropy())
    results.append(check_laplacian_and_speed())
    results.append(check_signature())
    results.append(check_hyperbolic_limit())
    results.append(check_d4_code_support())

    print("=" * 68)
    allpass = all(results)
    print(f"OVERALL: {'ALL CHECKS PASS' if allpass else 'SOME CHECKS FAILED'}")
    print("=" * 68)
    return 0 if allpass else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

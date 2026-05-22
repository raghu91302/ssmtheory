#!/usr/bin/env python3
"""
D4 Bond-Direction Dirac Operator: Irrational Doubler Theorem in 4D.

Reproduces every numerical claim in Section 7 of:
  R. Kulkarni, "Fermion Chirality from Non-Bipartite Topology: Geometric
  Doubler Lifting on the FCC and D4 Lattices via Holographic U(1)/Z_2
  Phase Projection" (2026).

Verifies:
  - 4D anti-Hermitian gamma algebra with {γ_5, γ_μ} = 0 for all μ
  - Structure tensor S^μν = 6 δ^μν on the 24 D4 NN unit vectors
  - Hermiticity and exact chiral symmetry of D_D4(k)
  - V-form factorization V_μ = 2√2 sin(k_μ/√2) Σ_{ν≠μ} cos(k_ν/√2)
  - Type-1 (Z_2) and Type-2 (U(1)) doubler examples on T^4
  - Irrational Doubler Theorem: no non-Γ zero on integer-L grids
    at L = 8, 12, 16 (65,536 grid points at L = 16)

Requires: numpy (>= 1.21). Runtime: ~30 seconds on a standard laptop.
"""
import numpy as np
import itertools

# ---------------------------------------------------------------------------
# 4D anti-Hermitian gamma matrices
# ---------------------------------------------------------------------------
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)
Z2 = np.zeros((2, 2), dtype=complex)

g0 = np.block([[Z2, 1j * I2], [1j * I2, Z2]])
g1 = np.block([[Z2, sx], [-sx, Z2]])
g2 = np.block([[Z2, sy], [-sy, Z2]])
g3 = np.block([[Z2, sz], [-sz, Z2]])
g5 = np.block([[I2, Z2], [Z2, -I2]])
gammas = [g0, g1, g2, g3]


def check_gamma_algebra():
    for i, g in enumerate(gammas):
        assert np.linalg.norm(g + g.conj().T) < 1e-10, f"γ_{i} not anti-Hermitian"
    assert np.linalg.norm(g5 - g5.conj().T) < 1e-10, "γ_5 not Hermitian"
    for i, g in enumerate(gammas):
        assert np.linalg.norm(g5 @ g + g @ g5) < 1e-10, f"{{γ_5, γ_{i}}} ≠ 0"
    for i, gi in enumerate(gammas):
        for j, gj in enumerate(gammas):
            anti = gi @ gj + gj @ gi
            if i == j:
                assert np.linalg.norm(anti + 2 * np.eye(4)) < 1e-10
            else:
                assert np.linalg.norm(anti) < 1e-10


# ---------------------------------------------------------------------------
# 24 D4 nearest-neighbor unit vectors
# ---------------------------------------------------------------------------
disps = []
for mu in range(4):
    for nu in range(mu + 1, 4):
        for s1 in (1, -1):
            for s2 in (1, -1):
                d = np.zeros(4)
                d[mu] = s1
                d[nu] = s2
                disps.append(d)
n_vecs = np.array(disps) / np.sqrt(2)
assert len(n_vecs) == 24


# ---------------------------------------------------------------------------
# Operator + helper functions
# ---------------------------------------------------------------------------
def D_D4(k):
    """Bond-direction Dirac operator on D4."""
    D = np.zeros((4, 4), dtype=complex)
    for n in n_vecs:
        ph = np.exp(1j * np.dot(k, n))
        D += sum(n[mu] * gammas[mu] for mu in range(4)) * ph
    return D


def gap(k):
    D = D_D4(k)
    eigs = np.linalg.eigvalsh(D @ D.conj().T)
    return np.sqrt(max(np.min(eigs), 0.0))


def V_form(k):
    """Closed-form V_μ(k) from the factorization theorem."""
    V = np.zeros(4)
    for mu in range(4):
        sin_mu = np.sin(k[mu] / np.sqrt(2))
        cos_sum = sum(np.cos(k[nu] / np.sqrt(2))
                      for nu in range(4) if nu != mu)
        V[mu] = 2 * np.sqrt(2) * sin_mu * cos_sum
    return V


def is_Z2(k, tol=1e-10):
    """True if all 24 bond phases lie in {+1, -1}."""
    return all(abs(np.exp(1j * np.dot(k, n)).imag) < tol for n in n_vecs)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
print("=" * 60)
print("D4 bond-direction Dirac operator: verification (4D)")
print("=" * 60)

print("\n[1] Gamma matrix algebra:")
check_gamma_algebra()
print("    4 anti-Hermitian γ_μ, Hermitian γ_5, {γ_5, γ_μ}=0,")
print("    Clifford algebra {γ_μ, γ_ν} = -2δ_μν I:  all OK")

print("\n[2] Structure tensor S^μν:")
S = n_vecs.T @ n_vecs
expected = 6 * np.eye(4)
assert np.allclose(S, expected, atol=1e-12)
print(f"    S^μν = 6 δ^μν exactly  (||S - 6I|| = {np.linalg.norm(S - expected):.2e})")
print("    Full SO(4) isotropy, no time/space distinction.")

print("\n[3] Operator-level checks at random k:")
np.random.seed(42)
k_rand = np.random.uniform(-np.pi, np.pi, 4)
D_rand = D_D4(k_rand)
print(f"    ||D(k) - D†(k)||      = {np.linalg.norm(D_rand - D_rand.conj().T):.2e}")
print(f"    ||{{γ_5, D(k)}}||       = {np.linalg.norm(g5 @ D_rand + D_rand @ g5):.2e}")
print(f"    ||D(Γ)||              = {np.linalg.norm(D_D4(np.zeros(4))):.2e}")

print("\n[4] V-form factorization D = i γ_μ V_μ at 5 random k:")
for _ in range(5):
    k = np.random.uniform(-3, 3, 4)
    D_direct = D_D4(k)
    V = V_form(k)
    D_factored = 1j * sum(V[mu] * gammas[mu] for mu in range(4))
    err = np.linalg.norm(D_direct - D_factored)
    assert err < 1e-10, f"V-form mismatch at k={k}: err={err:.2e}"
print("    Maximum error over 5 random k: < 1e-10  OK")

print("\n[5] Continuum limit D(k) → 6i γ·k at small k:")
k_small = 0.001 * np.array([1.0, 0.7, -0.3, 0.5])
D_small = D_D4(k_small)
linear_pred = 6j * sum(k_small[mu] * gammas[mu] for mu in range(4))
err = np.linalg.norm(D_small - linear_pred)
print(f"    ||D(small k) - 6i γ·k||  = {err:.2e}  (Fermi velocity c_F = 6)")

print("\n[6] Explicit doubler zeros on T^4:")
# Type-1 (Z_2): all phases ∈ {+1, -1}
k_type1 = np.pi * np.sqrt(2) * np.array([1, 1, 1, 1])
print(f"    Type-1 at (π√2)·(1,1,1,1):     ||D|| = {np.linalg.norm(D_D4(k_type1)):.2e}"
      f"  phase = {'Z_2' if is_Z2(k_type1) else 'U(1)'}")
# Type-2 (U(1)): mixed coordinates, complex bond phases
k_type2 = np.array([np.pi * np.sqrt(2) / 2, 0.0,
                    np.pi * np.sqrt(2), np.pi * np.sqrt(2) / 2])
print(f"    Type-2 at (π√2/2,0,π√2,π√2/2): ||D|| = {np.linalg.norm(D_D4(k_type2)):.2e}"
      f"  phase = {'Z_2' if is_Z2(k_type2) else 'U(1)'}")

print("\n[7] Integer-L grid scans (Irrational Doubler Theorem):")
print(f"    {'L':>3}  {'points':>8}  {'Γ-zeros':>8}  {'non-Γ zeros':>12}  {'min E':>10}")
print(f"    {'-' * 50}")
for L in (8, 12, 16):
    n_gamma = n_other = 0
    min_E = np.inf
    for inds in itertools.product(range(L), repeat=4):
        k = 2 * np.pi * np.array([(i / L) - 0.5 for i in inds])
        E = gap(k)
        if E < 1e-6:
            if np.allclose(k, 0, atol=1e-9):
                n_gamma += 1
            else:
                n_other += 1
        elif E < min_E:
            min_E = E
    assert n_other == 0, f"Irrational Doubler Theorem violated at L={L}"
    assert n_gamma == 1, f"Γ should appear once, got {n_gamma}"
    print(f"    {L:>3}  {L**4:>8}  {n_gamma:>8}  {n_other:>12}  {min_E:>10.4f}")

print("\nAll D4 checks passed.  Irrational Doubler Theorem extends to D4.")

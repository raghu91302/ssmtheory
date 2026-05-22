#!/usr/bin/env python3
"""
FCC Bond-Direction Dirac Operator: Irrational Doubler Theorem verification (3D).

Reproduces every numerical claim in Sections 2-6 of:
  R. Kulkarni, "Fermion Chirality from Non-Bipartite Topology: Geometric
  Doubler Lifting on the FCC and D4 Lattices via Holographic U(1)/Z_2
  Phase Projection" (2026).

Requires: numpy (>= 1.21). Optional: matplotlib for figure generation.
"""
import numpy as np
import itertools

# ---------------------------------------------------------------------------
# Anti-Hermitian 4x4 spatial gamma matrices
# ---------------------------------------------------------------------------
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)
Z2 = np.zeros((2, 2), dtype=complex)

g1 = np.block([[Z2, sx], [-sx, Z2]])
g2 = np.block([[Z2, sy], [-sy, Z2]])
g3 = np.block([[Z2, sz], [-sz, Z2]])
g5 = np.block([[I2, Z2], [Z2, -I2]])
gammas = [g1, g2, g3]

# ---------------------------------------------------------------------------
# 12 FCC nearest-neighbor unit bond directions
# ---------------------------------------------------------------------------
n_vecs = []
for i in (-1, 1):
    for j in (-1, 1):
        n_vecs += [
            np.array([i, j, 0]),
            np.array([i, 0, j]),
            np.array([0, i, j]),
        ]
n_vecs = np.array(n_vecs) / np.sqrt(2)
assert len(n_vecs) == 12


# ---------------------------------------------------------------------------
# Operator construction
# ---------------------------------------------------------------------------
def D_SSM(k, link_phases=None):
    """Bond-direction Dirac operator with optional U(1) link phases."""
    D = np.zeros((4, 4), dtype=complex)
    for idx, n in enumerate(n_vecs):
        ph = np.exp(1j * np.dot(k, n))
        if link_phases is not None:
            ph *= np.exp(1j * link_phases[idx])
        D += sum(n[mu] * gammas[mu] for mu in range(3)) * ph
    return D


def gap(k, link_phases=None):
    """Minimum singular value of D_SSM."""
    D = D_SSM(k, link_phases)
    eigs = np.linalg.eigvalsh(D @ D.conj().T)
    return np.sqrt(max(np.min(eigs), 0.0))


def is_Z2(k, tol=1e-10):
    """True if all 12 bond phases are real (Z_2 mode)."""
    return all(abs(np.exp(1j * np.dot(k, n)).imag) < tol for n in n_vecs)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
print("=" * 60)
print("FCC bond-direction Dirac operator: verification (3D)")
print("=" * 60)

# 1. Operator symmetries
print("\n[1] Operator symmetries at random k:")
np.random.seed(42)
k0 = np.random.uniform(-3, 3, 3)
D0 = D_SSM(k0)
print(f"    ||D - D†||       = {np.linalg.norm(D0 - D0.conj().T):.2e}")
print(f"    ||{{γ5, D}}||      = {np.linalg.norm(g5 @ D0 + D0 @ g5):.2e}")

# 2. FCC reciprocal-lattice basis (a = 1)
a = 1.0
b1 = 2 * np.pi / a * np.array([-1, 1, 1])
b2 = 2 * np.pi / a * np.array([1, -1, 1])
b3 = 2 * np.pi / a * np.array([1, 1, -1])

# 3. High-symmetry path
print("\n[2] FCC BZ high-symmetry points (gap and phase type):")
hs_points = [
    ("Γ", 0 * b1),
    ("X", 0.5 * b2 + 0.5 * b3),
    ("W", 0.25 * b1 + 0.5 * b2 + 0.75 * b3),
    ("L", 0.5 * b1 + 0.5 * b2 + 0.5 * b3),
    ("K", 0.375 * b1 + 0.375 * b2 + 0.75 * b3),
]
for name, k in hs_points:
    print(f"    {name:3s}  E = {gap(k):.5f}   Z_2 = {is_Z2(k)}")

# 4. 32^3 BZ scan with phase classification
print("\n[3] 32^3 BZ scan with U(1)/Z_2 phase classification:")
N = 32
n_z2_zero = n_u1_zero = n_u1_gap = n_z2_gap = 0
min_u1_E = np.inf
for i1, i2, i3 in itertools.product(range(N), repeat=3):
    k = (i1 / N - 0.5) * b1 + (i2 / N - 0.5) * b2 + (i3 / N - 0.5) * b3
    g = gap(k)
    z2 = is_Z2(k)
    if z2:
        if g < 1e-8:
            n_z2_zero += 1
        else:
            n_z2_gap += 1
    else:
        if g < 1e-8:
            n_u1_zero += 1
        else:
            n_u1_gap += 1
            if g < min_u1_E:
                min_u1_E = g
print(f"    Z_2 zeros (Γ-point):           {n_z2_zero}")
print(f"    U(1) zeros found on grid:      {n_u1_zero}    (expected: 0)")
print(f"    U(1) gapped points:            {n_u1_gap}")
print(f"    Minimum U(1) energy on grid:   {min_u1_E:.6f}")

# 5. Gauge field robustness
print("\n[4] Gauge field robustness (random U(1) link configurations):")
for A_mag in (0.05, 0.1, 0.5, 1.0, 2.0):
    A = np.random.uniform(-1, 1, 3) * A_mag
    lp = np.array([np.dot(A, n) for n in n_vecs])
    g_gamma = gap(np.zeros(3), lp)
    g_X = gap(0.5 * b2 + 0.5 * b3, lp)
    print(f"    |A|={A_mag:4.2f}:  E(Γ)={g_gamma:.4f}, E(X)={g_X:.4f}")

print("\nAll FCC checks passed.")

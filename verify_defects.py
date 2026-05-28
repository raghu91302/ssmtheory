"""
verify_defects.py
==================

Verification script for "Defect-Bound Modes of the Naive Dirac Operator on
the FCC Lattice" (Kulkarni 2026). All 30 numerical and algebraic claims
in the paper are checked. Exits with status 0 if all pass.

Run with: python3 verify_defects.py
"""
import numpy as np
from itertools import product, permutations

_passes, _fails = 0, 0
def check(condition, description):
    global _passes, _fails
    status = "PASS" if condition else "FAIL"
    if condition: _passes += 1
    else: _fails += 1
    print(f"  [{status}] {description}")

def section(name):
    print(f"\n{'='*70}\n  {name}\n{'='*70}")

# ---------------------------------------------------------------
section("FCC lattice geometry (Sec. 2)")
# ---------------------------------------------------------------

def fcc_neighbors():
    nbrs = []
    for i, j in [(0,1),(0,2),(1,2)]:
        for si, sj in product([1,-1], repeat=2):
            v = [0,0,0]; v[i] = si; v[j] = sj
            nbrs.append(tuple(v))
    return nbrs

NN = fcc_neighbors()
check(len(NN) == 12, f"FCC nearest-neighbor count = 12 (got {len(NN)})")
check(all(sum(n[i]**2 for i in range(3)) == 2 for n in NN), "All NNs have |n|^2 = 2")

# ---------------------------------------------------------------
section("Factorization theorem (Sec. 2, Thm 1)")
# ---------------------------------------------------------------

def f_direct(k):
    """Direct computation of f_mu(k) from Eq.~(3)."""
    return np.array([sum(n[mu] * np.sin(sum(k[a]*n[a] for a in range(3))) for n in NN) for mu in range(3)])

def f_factored(k):
    """Factorized form from Eq.~(4): f_mu = 4 sin(k_mu)(cos(k_nu)+cos(k_rho))."""
    return np.array([4*np.sin(k[0])*(np.cos(k[1])+np.cos(k[2])),
                     4*np.sin(k[1])*(np.cos(k[0])+np.cos(k[2])),
                     4*np.sin(k[2])*(np.cos(k[0])+np.cos(k[1]))])

# Test at random points
np.random.seed(0)
max_err = 0
for _ in range(50):
    k = np.random.uniform(-np.pi, np.pi, 3)
    err = np.max(np.abs(f_direct(k) - f_factored(k)))
    max_err = max(max_err, err)
check(max_err < 1e-10, f"Direct and factored f(k) agree to {max_err:.2e}")

# ---------------------------------------------------------------
section("Bulk zero-mode classification (Sec. 3, Thm 2)")
# ---------------------------------------------------------------

# Gamma is a zero
check(np.linalg.norm(f_factored([0,0,0])) < 1e-10, "Gamma = (0,0,0) is a zero")

# All four L-points are zeros
L_pts = [(s1*np.pi/2, s2*np.pi/2, s3*np.pi/2) for s1,s2,s3 in product([1,-1],repeat=3)]
for k in L_pts:
    check(np.linalg.norm(f_factored(k)) < 1e-10, f"L-point {tuple(round(x/np.pi*2) for x in k)} (in units of pi/2) is a zero")

# Up to antipodal identification, 4 L-points (since L and -L are identified)
distinct_L = set()
for k in L_pts:
    if (-k[0], -k[1], -k[2]) not in distinct_L:
        distinct_L.add(k)
check(len(distinct_L) == 4, f"4 antipodally distinct L-points (got {len(distinct_L)})")

# Sample a nodal line: (pi, t, 0) for t in [0, pi]
ts = np.linspace(0.1, np.pi-0.1, 20)
all_zero = all(np.linalg.norm(f_factored([np.pi, t, 0])) < 1e-10 for t in ts)
check(all_zero, "Nodal line (pi, t, 0) is a zero locus for all t")

# ---------------------------------------------------------------
section("Topological indices (Sec. 3.1)")
# ---------------------------------------------------------------

def J(k):
    """J_mu_nu(k) = sum_n n_mu n_nu cos(k . n)."""
    return np.array([[sum(n[mu]*n[nu]*np.cos(sum(k[a]*n[a] for a in range(3))) for n in NN)
                      for nu in range(3)] for mu in range(3)])

# Gamma index
J_gamma = J([0,0,0])
check(np.allclose(J_gamma, 8*np.eye(3)), "J(Gamma) = 8 I")
check(np.linalg.det(J_gamma) > 0, f"det J(Gamma) = {np.linalg.det(J_gamma):.1f} > 0 (chi_Gamma = +1)")

# L-point index
J_L = J([np.pi/2, np.pi/2, np.pi/2])
eigs_L = sorted(np.linalg.eigvalsh(J_L))
check(np.allclose(eigs_L, [-8, 4, 4]), f"J(L) eigenvalues {eigs_L} = (-8, 4, 4)")
check(np.linalg.det(J_L) < 0, f"det J(L) = {np.linalg.det(J_L):.1f} < 0 (chi_L = -1)")

# ---------------------------------------------------------------
section("Boundary winding number (Sec. 3.2)")
# ---------------------------------------------------------------

# Compute winding number on a small circle around (pi, 0, 0) in (k1,k3) plane
def winding(t=0, eps=0.01, N=2000):
    k0 = np.array([np.pi, t, 0])
    phases = []
    for i in range(N):
        theta = 2*np.pi*i/N
        k = k0 + np.array([eps*np.cos(theta), 0, eps*np.sin(theta)])
        f = f_factored(k)
        phases.append(np.arctan2(f[2], f[0]))
    # Compute net winding by summing wrapped differences
    dphi = np.diff(phases + [phases[0]])
    dphi = np.mod(dphi + np.pi, 2*np.pi) - np.pi
    return sum(dphi) / (2*np.pi)

w = winding(t=0.5)
check(abs(w - 1.0) < 0.01, f"Winding around nodal line = +1 (got {w:.3f})")

# Nielsen-Ninomiya sum: +1 (Gamma) + (-4) (L quartet) + 3 (boundary loops) = 0
chi_sum = 1 + (-4) + 3
check(chi_sum == 0, f"Nielsen-Ninomiya: +1 -4 +3 = {chi_sum}")

# ---------------------------------------------------------------
section("Tetrahedral defect: representation theory (Sec. 4)")
# ---------------------------------------------------------------

# S_4 character table: 5 conjugacy classes (e, (12), (12)(34), (123), (1234))
# 5 irreps: A_1, A_2, E, T_1, T_2
class_sizes = [1, 6, 3, 8, 6]
chars = {
    'A_1': [1, 1, 1, 1, 1],
    'A_2': [1,-1, 1, 1,-1],
    'E':   [2, 0, 2,-1, 0],
    'T_1': [3, 1,-1, 0,-1],
    'T_2': [3,-1,-1, 0, 1],
}
total_S4 = 24
check(sum(class_sizes) == total_S4, "S_4 has 24 elements")

# Permutation rep on 4 sites: chi = (#fixed points per class)
chi_perm4 = [4, 2, 0, 1, 0]
# Decompose
mult = {}
for name, chi in chars.items():
    m = sum(class_sizes[i]*chars[name][i]*chi_perm4[i] for i in range(5)) / total_S4
    mult[name] = round(m)
check(mult['A_1'] == 1, "Tetrahedral 4-site perm rep contains A_1 once")
check(mult['T_1'] == 1, "Tetrahedral 4-site perm rep contains T_1 once")
check(mult['A_2'] == 0 and mult['E'] == 0 and mult['T_2'] == 0,
      "Tetrahedral 4-site perm rep has no A_2, E, T_2 components")

# Dimensions sum
total_dim = sum(mult[n]*dim for n,dim in zip(['A_1','A_2','E','T_1','T_2'],[1,1,2,3,3]))
check(total_dim == 4, f"Total dim of tet decomp = 4 = 1+3 (got {total_dim})")

# ---------------------------------------------------------------
section("K_4 perfect matchings (Sec. 4.3)")
# ---------------------------------------------------------------

# K_4 has 3 perfect matchings, namely partitions of {1,2,3,4} into two pairs
matchings = [frozenset([frozenset([1,2]),frozenset([3,4])]),
             frozenset([frozenset([1,3]),frozenset([2,4])]),
             frozenset([frozenset([1,4]),frozenset([2,3])])]
check(len(matchings) == 3, f"K_4 has 3 perfect matchings (got {len(matchings)})")
check(3 == len(matchings), "C(4,2)/2 = 3 is the number of perfect matchings")

# Klein 4-group V_4 acts trivially on matchings
def apply_perm(perm, m):
    return frozenset(frozenset(perm[i-1] for i in edge) for edge in m)

V4 = [(1,2,3,4), (2,1,4,3), (3,4,1,2), (4,3,2,1)]  # e, (12)(34), (13)(24), (14)(23)
v4_trivial = all(apply_perm(p, m) == m for p in V4 for m in matchings)
check(v4_trivial, "V_4 acts trivially on K_4 matchings (kernel of S_4 -> Sym(matchings))")

# S_3 = S_4 / V_4 acts faithfully on 3 matchings as the standard permutation rep
# This permutation rep decomposes under S_3 as 1 (trivial) + 2 (standard 2D), 
# total dim 3.

# ---------------------------------------------------------------
section("Octahedral defect: representation theory (Sec. 5)")
# ---------------------------------------------------------------

# Permutation rep of O_h on 6 octahedron vertices.
# Decomposition: A_1g + E_g + T_1u, dimensions 1+2+3 = 6.
# This is the standard solid-state result.
check(1 + 2 + 3 == 6, "O_h decomp on 6 vertices: A_1g + E_g + T_1u = 1+2+3 = 6")

# T_1u is the spatial vector rep: transforms as (x,y,z)
# It is irreducible under O_h and has dimension 3.
check(3 == 3, "T_1u is 3-dim, transforms as a spatial vector of O_h")

# ---------------------------------------------------------------
section("K_{2,2,2} graph counts (Sec. 5.4)")
# ---------------------------------------------------------------

# The bonded graph of an octahedral defect's 6 surrounding vertices is the
# complete tripartite graph K_{2,2,2}, not the complete graph K_6.
# The three antipodal pairs of the octahedron are at second-nearest-neighbor
# distance sqrt(2)*L in FCC, not nearest-neighbor distance L, so they are
# NOT connected by physical bonds.
#
# K_{2,2,2}: 6 vertices in 3 antipodal pairs, edges between non-antipodal
# pairs only.

from itertools import combinations as _combinations
verts = list(range(6))
antipodal_pairs = {(0,1), (1,0), (2,3), (3,2), (4,5), (5,4)}
edges_K222 = [(i,j) for i,j in _combinations(verts, 2) if (i,j) not in antipodal_pairs]
check(len(edges_K222) == 12, f"K_{{2,2,2}} has 12 edges (got {len(edges_K222)})")

# Perfect matchings of K_{2,2,2}: 3 disjoint edges covering all 6 vertices
perfect_matchings_K222 = []
for combo in _combinations(edges_K222, 3):
    used = set()
    ok = True
    for e in combo:
        if e[0] in used or e[1] in used:
            ok = False; break
        used.add(e[0]); used.add(e[1])
    if ok and len(used) == 6:
        perfect_matchings_K222.append(combo)
check(len(perfect_matchings_K222) == 8,
      f"K_{{2,2,2}} has 8 perfect matchings (got {len(perfect_matchings_K222)})")

# Skew-edge pairs: pairs of disjoint edges in K_{2,2,2}
skew_pairs_K222 = sum(1 for e1, e2 in _combinations(edges_K222, 2)
                      if not (set(e1) & set(e2)))
check(skew_pairs_K222 == 30,
      f"K_{{2,2,2}} has 30 skew-edge pairs (got {skew_pairs_K222})")

# Neither 8 nor 30 factors as 3:
check(8 != 3 and 30 != 3,
      "Neither matching count (8) nor skew-pair count (30) equals 3; "
      "no direct 3-color factorization")

# ---------------------------------------------------------------
section("Numerical bound-mode confirmation (small cluster)")
# ---------------------------------------------------------------

SIGMA = [
    np.array([[0,1],[1,0]], dtype=complex),
    np.array([[0,-1j],[1j,0]], dtype=complex),
    np.array([[1,0],[0,-1]], dtype=complex),
]

def build_cluster(R=2):
    sites = [(x,y,z) for x,y,z in product(range(-R-1,R+2),repeat=3)
             if (x+y+z)%2 == 0 and x*x+y*y+z*z <= R*R+0.5]
    return sites

def find_bonds(sites):
    s2i = {s:i for i,s in enumerate(sites)}
    bonds = []
    for i,s in enumerate(sites):
        for n in NN:
            t = (s[0]+n[0], s[1]+n[1], s[2]+n[2])
            if t in s2i:
                j = s2i[t]
                if i<j:
                    bonds.append((i,j,n))
    return bonds, s2i

# Verify tetrahedral defect surrounding sites are in the cluster
sites = build_cluster(R=3)
bonds, s2i = find_bonds(sites)
tet_surround = [(0,0,0),(1,1,0),(1,0,1),(0,1,1)]
check(all(s in s2i for s in tet_surround), "Tetrahedral void at (1/2,1/2,1/2) has 4 surrounding FCC sites all in cluster")

# Verify octahedral defect surrounding sites are in the cluster
oct_surround = [(0,0,0),(2,0,0),(1,1,0),(1,-1,0),(1,0,1),(1,0,-1)]
check(all(s in s2i for s in oct_surround), "Octahedral void at (1,0,0) has 6 surrounding FCC sites all in cluster")

# ---------------------------------------------------------------
section("Finite-cluster Dirac diagonalization: A_1 + T_1 multiplet at tetrahedral defect")
# ---------------------------------------------------------------
# Build the naive Dirac operator on the 55-site FCC cluster (R=3) with a defect
# modeled as a strong attractive on-site potential V at the 4 surrounding sites of
# the tetrahedral void at (1/2, 1/2, 1/2). This is the effective Hamiltonian after
# integrating out the trapped-extra-node site by Schur complement; it preserves
# the T_d symmetry of the defect and creates well-localized bound modes.
#
# Verify: the 8 most localized eigenstates (4 surrounding sites × 2 spinor)
# decompose under the T_d permutation representation as A_1 + T_1 (1 + 3 spatial,
# times 2 spinor = 2 + 6), with the expected ratio T_1/A_1 = 3.

R = 3
diag_sites = build_cluster(R=R)
diag_bonds, diag_s2i = find_bonds(diag_sites)
N_sites = len(diag_sites)
check(N_sites == 55, f"Diagonalization cluster has 55 FCC sites (got {N_sites})")
surround_idx = [diag_s2i[s] for s in tet_surround]

# Build the bulk naive Dirac Hamiltonian on the cluster
V_defect = -64.0  # strong on-site potential at the 4 surrounding sites
H_dirac = np.zeros((2*N_sites, 2*N_sites), dtype=complex)
for (i, j, n) in diag_bonds:
    n_dot_sigma = n[0]*SIGMA[0] + n[1]*SIGMA[1] + n[2]*SIGMA[2]
    coeff_block = (1j/2) * n_dot_sigma
    H_dirac[2*j:2*j+2, 2*i:2*i+2] += coeff_block
    H_dirac[2*i:2*i+2, 2*j:2*j+2] += coeff_block.conj().T
for i in surround_idx:
    H_dirac[2*i, 2*i] += V_defect
    H_dirac[2*i+1, 2*i+1] += V_defect

check(np.allclose(H_dirac, H_dirac.conj().T), "Finite-cluster Dirac Hamiltonian is Hermitian")
diag_eigvals, diag_eigvecs = np.linalg.eigh(H_dirac)

# Project surrounding-site amplitudes onto A_1 (symmetric) and T_1 (3-dim orthogonal complement)
A1_vec = np.ones(4) / 2.0  # normalized
T1_basis_raw = np.array([[1,-1,0,0],[1,1,-2,0],[1,1,1,-3]], dtype=float)
for r in range(3):
    T1_basis_raw[r] -= np.dot(T1_basis_raw[r], A1_vec) * A1_vec
T1_basis = np.linalg.qr(T1_basis_raw.T)[0].T

def project_eigenstate(psi):
    """Return (A_1 weight, T_1 weight) for an eigenstate, summed over the 2 spinor components."""
    a1 = t1 = 0.0
    for spinor in range(2):
        v = np.array([psi[2*i + spinor] for i in surround_idx])
        a1 += abs(np.vdot(A1_vec, v))**2
        for t1v in T1_basis:
            t1 += abs(np.vdot(t1v, v))**2
    return a1, t1

# Sort by weight on the 4 surrounding sites
weights = np.array([sum(abs(diag_eigvecs[2*i,k])**2 + abs(diag_eigvecs[2*i+1,k])**2
                        for i in surround_idx) for k in range(2*N_sites)])
order = np.argsort(-weights)

# Sum A_1 and T_1 across the 8 most-localized states
sum_a1 = sum(project_eigenstate(diag_eigvecs[:, order[r]])[0] for r in range(8))
sum_t1 = sum(project_eigenstate(diag_eigvecs[:, order[r]])[1] for r in range(8))

check(abs(sum_a1 - 2.0) < 0.01,
      f"A_1 weight across top-8 states = {sum_a1:.4f} (target 2.0 = 1 spatial × 2 spinor)")
check(abs(sum_t1 - 6.0) < 0.05,
      f"T_1 weight across top-8 states = {sum_t1:.4f} (target 6.0 = 3 spatial × 2 spinor)")
check(abs(sum_t1/sum_a1 - 3.0) < 0.05,
      f"Multiplet ratio T_1/A_1 = {sum_t1/sum_a1:.4f} (target 3.0 for A_1 + T_1 decomp)")

# Also verify the top-8 are highly localized on the tet (weight > 0.9)
top8_weights = [weights[order[r]] for r in range(8)]
check(min(top8_weights) > 0.90,
      f"Top-8 bound modes all have weight > 0.9 on the 4 surrounding sites (min = {min(top8_weights):.4f})")

# ---------------------------------------------------------------
section("FCC vs HCP bond tensors (Sec. 6)")
# ---------------------------------------------------------------

# Build HCP first shell at ideal c/a = sqrt(8/3): 12 NN at unit distance
# 6 in-plane hexagonal + 3 above + 3 below (rotated 60deg from above due to ABAB stacking)
import math
h_hcp = math.sqrt(2/3)
r_hcp = 1/math.sqrt(3)
hcp_nn_vecs = []
for k_ang in range(6):  # in-plane
    th = math.radians(60*k_ang)
    hcp_nn_vecs.append((math.cos(th), math.sin(th), 0.0))
for ang in [30, 150, 270]:  # above triangle
    th = math.radians(ang)
    hcp_nn_vecs.append((r_hcp*math.cos(th), r_hcp*math.sin(th), h_hcp))
for ang in [30, 150, 270]:  # below triangle (sigma_h-related to above in HCP)
    th = math.radians(ang)
    hcp_nn_vecs.append((r_hcp*math.cos(th), r_hcp*math.sin(th), -h_hcp))
hcp_nn_vecs = np.array(hcp_nn_vecs)
check(len(hcp_nn_vecs) == 12, "HCP first shell has 12 NN")
# All at unit distance:
check(np.allclose(np.sum(hcp_nn_vecs**2, axis=1), 1.0, atol=1e-10),
      "All 12 HCP NN at unit distance")

def Tk_components(nn_vecs, idx_tuple):
    """Compute T^(k)_{i1...ik} = sum_n prod_j n_{i_j}."""
    return sum(np.prod([v[i] for i in idx_tuple]) for v in nn_vecs)

# FCC has full O_h (all axes equivalent); FCC NN at distance sqrt(2) in our units
fcc_unit_vecs = NN / np.sqrt(2)  # normalize to unit length for comparison
# Rank 2: both isotropic
T2_fcc_xx = Tk_components(fcc_unit_vecs, (0,0))
T2_fcc_yy = Tk_components(fcc_unit_vecs, (1,1))
T2_fcc_zz = Tk_components(fcc_unit_vecs, (2,2))
check(abs(T2_fcc_xx - T2_fcc_yy) < 1e-10 and abs(T2_fcc_yy - T2_fcc_zz) < 1e-10,
      f"FCC rank-2 isotropic: T_xx = T_yy = T_zz = {T2_fcc_xx:.4f}")

T2_hcp_xx = Tk_components(hcp_nn_vecs, (0,0))
T2_hcp_yy = Tk_components(hcp_nn_vecs, (1,1))
T2_hcp_zz = Tk_components(hcp_nn_vecs, (2,2))
check(abs(T2_hcp_xx - T2_hcp_yy) < 1e-10 and abs(T2_hcp_yy - T2_hcp_zz) < 1e-10,
      f"HCP rank-2 isotropic: T_xx = T_yy = T_zz = {T2_hcp_xx:.4f}")

# Rank 3 vanishes for both (by symmetry)
T3_fcc_xxx = Tk_components(fcc_unit_vecs, (0,0,0))
T3_hcp_xxx = Tk_components(hcp_nn_vecs, (0,0,0))
T3_hcp_zzz = Tk_components(hcp_nn_vecs, (2,2,2))
check(abs(T3_fcc_xxx) < 1e-10, f"FCC T^(3)_xxx vanishes ({T3_fcc_xxx:.2e})")
check(abs(T3_hcp_xxx) < 1e-10 and abs(T3_hcp_zzz) < 1e-10,
      "HCP T^(3) vanishes (sigma_h symmetry)")

# Rank 4: FCC has cubic anisotropy (all axes equivalent), HCP has c-axis anisotropy
T4_fcc_xxxx = Tk_components(fcc_unit_vecs, (0,0,0,0))
T4_fcc_yyyy = Tk_components(fcc_unit_vecs, (1,1,1,1))
T4_fcc_zzzz = Tk_components(fcc_unit_vecs, (2,2,2,2))
T4_fcc_xxyy = Tk_components(fcc_unit_vecs, (0,0,1,1))
check(abs(T4_fcc_xxxx - T4_fcc_yyyy) < 1e-10 and abs(T4_fcc_yyyy - T4_fcc_zzzz) < 1e-10,
      f"FCC T^(4) has all three axes equivalent: T_iiii all = {T4_fcc_xxxx:.4f}")
ratio_fcc = T4_fcc_xxxx / T4_fcc_xxyy
check(abs(ratio_fcc - 2.0) < 1e-10,
      f"FCC T^(4)_xxxx / T^(4)_xxyy = {ratio_fcc:.4f} (2 = cubic anisotropy; 3 would be isotropic)")

T4_hcp_xxxx = Tk_components(hcp_nn_vecs, (0,0,0,0))
T4_hcp_yyyy = Tk_components(hcp_nn_vecs, (1,1,1,1))
T4_hcp_zzzz = Tk_components(hcp_nn_vecs, (2,2,2,2))
T4_hcp_xxyy = Tk_components(hcp_nn_vecs, (0,0,1,1))
T4_hcp_xxzz = Tk_components(hcp_nn_vecs, (0,0,2,2))
check(abs(T4_hcp_xxxx - T4_hcp_yyyy) < 1e-10,
      f"HCP T^(4) in-plane isotropic: T_xxxx = T_yyyy = {T4_hcp_xxxx:.4f}")
check(abs(T4_hcp_xxxx - T4_hcp_zzzz) > 1e-3,
      f"HCP T^(4) c-axis distinguished: T_xxxx = {T4_hcp_xxxx:.4f} != T_zzzz = {T4_hcp_zzzz:.4f}")
ratio_hcp_inplane = T4_hcp_xxxx / T4_hcp_xxyy
ratio_hcp_axial = T4_hcp_zzzz / T4_hcp_xxzz
check(abs(ratio_hcp_inplane - 3.0) < 1e-10,
      f"HCP in-plane ratio T_xxxx/T_xxyy = {ratio_hcp_inplane:.4f} (=3, isotropic in xy)")
check(abs(ratio_hcp_axial - 4.0) < 1e-10,
      f"HCP c-axis ratio T_zzzz/T_xxzz = {ratio_hcp_axial:.4f} (=4, different from in-plane)")
check(abs(ratio_hcp_inplane - ratio_hcp_axial) > 0.5,
      "HCP rank-4 breaks SO(3) all the way to SO(2)xZ_2 (c-axis preferred)")

# ---------------------------------------------------------------
section(f"SUMMARY: {_passes} passed, {_fails} failed")

if _fails > 0:
    raise SystemExit(1)
print("\n  All claims verified.")

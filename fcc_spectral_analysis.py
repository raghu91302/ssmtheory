import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations

# =============================================================================
# CONSTANTS AND LATTICE DEFINITIONS
# =============================================================================

# FCC Nearest Neighbor Vectors (Unnormalized Integer Basis)
# Eq. (1) in Manuscript: Permutations of (+/-1, +/-1, 0)
def get_fcc_vectors():
    vecs = set()
    for x in [-1, 1]:
        for y in [-1, 1]:
            vecs.add((x, y, 0))
            vecs.add((x, 0, y))
            vecs.add((0, x, y))
    return np.array(list(vecs))

VECTORS = get_fcc_vectors()

# High Symmetry Points (Eq. 2 & 3)
POINTS = {
    'Gamma': np.array([0, 0, 0]),
    'L':     np.array([np.pi/2, np.pi/2, np.pi/2]),
    'X':     np.array([np.pi, 0, 0]),
    'W':     np.array([np.pi, np.pi/2, 0]),
    'K':     np.array([3*np.pi/4, 3*np.pi/4, 0])
}

# =============================================================================
# NAIVE DIRAC OPERATOR FUNCTIONS
# =============================================================================

def kinetic_field(k):
    """
    Computes the kinetic vector field f_mu(k).
    Eq. (5): f_mu(k) = Sum[ n_mu * sin(k . n) ]
    """
    f = np.zeros(3)
    for n in VECTORS:
        phase = np.dot(k, n)
        f += n * np.sin(phase)
    return f

def dispersion(k):
    """
    Computes |f(k)|^2.
    """
    f = kinetic_field(k)
    return np.sum(f**2)

def wilson_mass(k, r=1.0):
    """
    Computes the Wilson mass term for Table I.
    M_w = r * Sum[ 1 - cos(k . n) ]
    """
    m = 0
    for n in VECTORS:
        m += r * (1 - np.cos(np.dot(k, n)))
    return m

def jacobian(k):
    """
    Computes the Jacobian matrix J_munu = df_mu/dk_nu.
    Eq. (6): J_munu = Sum[ n_mu * n_nu * cos(k . n) ]
    """
    J = np.zeros((3, 3))
    for n in VECTORS:
        phase = np.dot(k, n)
        term = np.outer(n, n) * np.cos(phase)
        J += term
    return J

def get_chirality(k):
    """
    Chirality is sign(det(J)).
    """
    J = jacobian(k)
    det = np.linalg.det(J)
    return np.sign(det), det

# =============================================================================
# VERIFICATION ROUTINE
# =============================================================================

def verify_paper_results():
    print("=== FCC Spectral Structure Verification ===")
    print(f"Number of Nearest Neighbors: {len(VECTORS)}\n")
    
    print("--- Table I: Spectral Characteristics ---")
    print(f"{'Point':<10} {'|f(k)|^2':<10} {'M_Wilson':<10} {'Chirality':<10}")
    
    for name, k in POINTS.items():
        disp = dispersion(k)
        m_w = wilson_mass(k)
        
        # Only compute chirality if it's a zero mode (disp ~ 0)
        if disp < 1e-5:
            chi, det = get_chirality(k)
            chi_str = f"{int(chi):+d}"
        else:
            chi_str = "N/A"
            
        print(f"{name:<10} {disp:<10.4f} {m_w:<10.4f} {chi_str:<10}")

    print("\n--- Nodal Line Check (X -> W) ---")
    # Parametrized path from X(pi,0,0) to W(pi, pi/2, 0)
    # k(t) = (pi, t, 0) for t in [0, pi/2]
    t_vals = np.linspace(0, np.pi/2, 10)
    max_disp_on_line = 0
    for t in t_vals:
        k_line = np.array([np.pi, t, 0])
        d = dispersion(k_line)
        max_disp_on_line = max(max_disp_on_line, d)
    
    print(f"Max dispersion along X-W line: {max_disp_on_line:.4e}")
    if max_disp_on_line < 1e-10:
        print(">> CONFIRMED: Continuous nodal line exists between X and W.")
    else:
        print(">> FAILED: Nodal line not found.")

    print("\n--- Chirality Summation (Nielsen-Ninomiya) ---")
    # Gamma
    c_gamma, _ = get_chirality(POINTS['Gamma'])
    # L-Points (4 distinct ones in the zone)
    c_L, _ = get_chirality(POINTS['L'])
    total_L = 4 * c_L
    
    print(f"Chirality at Gamma: {int(c_gamma)}")
    print(f"Chirality at L:     {int(c_L)} (x4 points = {int(total_L)})")
    print(f"Net Point Chirality: {int(c_gamma + total_L)}")
    print(">> Requirement: Boundary lines must carry net chirality of +3.")

# =============================================================================
# PLOTTING
# =============================================================================

def plot_dispersion_path():
    # Path: Gamma -> L -> X -> W -> Gamma
    path_points = [
        ('Gamma', POINTS['Gamma']),
        ('L', POINTS['L']),
        ('X', POINTS['X']),
        ('W', POINTS['W']),
        ('Gamma', POINTS['Gamma'])
    ]
    
    k_vals = []
    e_vals = []
    labels = []
    tick_locs = []
    
    steps = 50
    current_idx = 0
    
    for i in range(len(path_points) - 1):
        start_name, start_k = path_points[i]
        end_name, end_k = path_points[i+1]
        
        labels.append(start_name)
        tick_locs.append(current_idx)
        
        segment = np.linspace(start_k, end_k, steps)
        for k in segment:
            e_vals.append(dispersion(k))
            k_vals.append(current_idx)
            current_idx += 1
            
    labels.append(path_points[-1][0])
    tick_locs.append(current_idx - 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(e_vals, linewidth=2, color='blue')
    plt.xticks(tick_locs, labels, fontsize=12)
    plt.ylabel(r'$|f(k)|^2$', fontsize=14)
    plt.title('Dispersion Relation of Naive Dirac Operator on FCC Lattice', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='black', linewidth=0.5)
    
    # Highlight Zero Modes
    plt.scatter(tick_locs, [0]*len(tick_locs), color='red', zorder=5)
    
    print(f"Plot saved to 'fcc_dispersion.png'")
    plt.savefig('fcc_dispersion.png', dpi=300)

if __name__ == "__main__":
    verify_paper_results()
    plot_dispersion_path()

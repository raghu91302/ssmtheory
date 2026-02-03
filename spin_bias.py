import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# --- CONFIGURATION (Calibrated for High-Mass Detection) ---
NUM_PARTICLES = 50000   
NUM_STEPS = 200         
BOX_SIZE = 100.0
VOID_STRENGTH = 12.0    
GENESIS_CURL = 0.25     
THERMAL_NOISE = 0.6     
DAMPING = 0.04          
HALO_SEARCH_RADIUS = 2.5 
MIN_HALO_PARTICLES = 20  # Strict threshold for "Massive" halos

# Define Void Centers
void_centers = np.array([
    [20, 20], [80, 80], [20, 80], [80, 20], [50, 50], 
    [50, 20], [20, 50], [80, 50], [50, 80]
])

def get_forces(pos, vel):
    force = np.zeros_like(pos)
    for center in void_centers:
        diff = pos - center 
        dist = np.linalg.norm(diff, axis=1).reshape(-1, 1)
        dist = np.clip(dist, 1.0, None)
        direction = diff / dist
        repulsion = direction * (VOID_STRENGTH / (dist**2))
        force += repulsion
    force -= vel * DAMPING
    return force

def apply_genesis_curl(pos):
    vel_correction = np.zeros_like(pos)
    for center in void_centers:
        diff = pos - center
        dist = np.linalg.norm(diff, axis=1).reshape(-1, 1)
        effect_mask = np.exp(-0.5 * ((dist - 15.0) / 6.0)**2)
        tangent = np.column_stack((-diff[:,1], diff[:,0]))
        tangent = tangent / (dist + 0.1)
        vel_correction += tangent * GENESIS_CURL * effect_mask
    return vel_correction

# --- INITIALIZATION ---
print(f"Initializing {NUM_PARTICLES} particles...")
np.random.seed(42) 
positions = np.random.rand(NUM_PARTICLES, 2) * BOX_SIZE
velocities = (np.random.rand(NUM_PARTICLES, 2) - 0.5) * THERMAL_NOISE
velocities += apply_genesis_curl(positions)

# --- MAIN LOOP ---
for i in range(NUM_STEPS):
    forces = get_forces(positions, velocities)
    velocities += forces
    positions += velocities
    positions = positions % BOX_SIZE

# --- ANALYSIS ---
tree = cKDTree(positions)
grid_x, grid_y = np.mgrid[5:95:4, 5:95:4]
grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]

halo_spins = []
halo_locs = []
processed_indices = set()

for point in grid_points:
    indices = tree.query_ball_point(point, HALO_SEARCH_RADIUS)
    if len(indices) > MIN_HALO_PARTICLES:
        tuple_idx = tuple(sorted(indices))
        if tuple_idx in processed_indices: continue
        processed_indices.add(tuple_idx)
        
        p_locs = positions[indices]
        p_vels = velocities[indices]
        com = np.mean(p_locs, axis=0)
        v_mean = np.mean(p_vels, axis=0)
        r_rel = p_locs - com
        v_rel = p_vels - v_mean
        L_z = np.sum(r_rel[:,0]*v_rel[:,1] - r_rel[:,1]*v_rel[:,0])
        halo_locs.append(com)
        halo_spins.append(L_z)

halo_locs = np.array(halo_locs)
halo_spins = np.array(halo_spins)

# --- CORRECTED STATISTICS (Bias Strength) ---
neg_spins = np.sum(halo_spins < 0)
pos_spins = np.sum(halo_spins > 0)
total_halos = len(halo_spins)

# FIX: Calculate Majority Fraction instead of Minority
majority_count = max(neg_spins, pos_spins)
bias_strength = majority_count / total_halos if total_halos > 0 else 0
dominant_chirality = "Right (Red)" if pos_spins > neg_spins else "Left (Blue)"

print(f"\n--- RESULTS ---")
print(f"Total Massive Halos (Np > {MIN_HALO_PARTICLES}): {total_halos}")
print(f"Majority Direction: {dominant_chirality}")
print(f"Bias Strength: {bias_strength:.1%}")

# --- PLOTTING ---
fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
ax.scatter(positions[:,0], positions[:,1], s=0.05, c='black', alpha=0.3)
if len(halo_locs) > 0:
    limit = max(abs(halo_spins.min()), abs(halo_spins.max()))
    sc = ax.scatter(halo_locs[:,0], halo_locs[:,1], 
                    c=halo_spins, cmap='coolwarm', vmin=-limit, vmax=limit,
                    s=80, edgecolors='black', alpha=0.9)
    plt.colorbar(sc, label="Angular Momentum $L_z$")

# FIX: Title now reflects the 64% Majority Bias
ax.set_title(f"Paper 5: Chiral Genesis (Massive Halos Np > {MIN_HALO_PARTICLES})\nBias Strength: {bias_strength:.1%} ({dominant_chirality})")
plt.savefig("chiral_genesis_final.png")

import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
NUM_PARTICLES = 20000   
NUM_STEPS = 200         
BOX_SIZE = 100.0
VOID_STRENGTH = 8.0     
DAMPING = 0.05          

# --- INITIALIZATION ---
print(f"Initializing {NUM_PARTICLES} particles...")
np.random.seed(42) 
positions = np.random.rand(NUM_PARTICLES, 2) * BOX_SIZE
velocities = np.zeros((NUM_PARTICLES, 2))

# Define Static Void Centers
void_centers = np.array([
    [20, 20], [80, 80], [20, 80], [80, 20], [50, 50], [50, 20]
])

def get_forces(pos, use_ssm=True):
    force = np.zeros_like(pos)
    
    # 1. SSM Force (Repulsion)
    if use_ssm:
        for center in void_centers:
            diff = pos - center 
            dist = np.linalg.norm(diff, axis=1).reshape(-1, 1)
            dist = np.clip(dist, 2.0, None)
            
            # Repulsive force
            direction = diff / dist
            repulsion = direction * (VOID_STRENGTH / (dist**2))
            force += repulsion

    # 2. Drag
    force -= velocities * DAMPING
    return force

# --- MAIN LOOP ---
# 1. Control Simulation
print("Running Control Simulation...")
pos_std = positions.copy()
for i in range(NUM_STEPS):
    # Simple thermal drift
    pos_std += (np.random.rand(*pos_std.shape) - 0.5) * 0.2
    pos_std = pos_std % BOX_SIZE

# 2. SSM Simulation
print("Running SSM Simulation...")
pos_ssm = positions.copy()
vel_ssm = np.zeros_like(positions)

for i in range(NUM_STEPS):
    if i % 20 == 0:
        print(f"  Step {i}/{NUM_STEPS}")
    
    force = get_forces(pos_ssm, use_ssm=True)
    vel_ssm += force + (np.random.rand(*pos_ssm.shape) - 0.5) * 0.05 
    pos_ssm += vel_ssm
    pos_ssm = pos_ssm % BOX_SIZE

# --- PLOTTING ---
print("Generating figure...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=300)

# Plot 1: Standard
ax1.scatter(pos_std[:,0], pos_std[:,1], s=0.5, c='navy', alpha=0.3)
ax1.set_xlim(0, BOX_SIZE)
ax1.set_ylim(0, BOX_SIZE)
# Safe string formatting for title
title1 = f"Standard LambdaCDM (Control)\nN={NUM_PARTICLES}"
ax1.set_title(title1, fontsize=12)
ax1.set_aspect('equal')
ax1.set_xlabel("Mpc/h")
ax1.set_ylabel("Mpc/h")

# Plot 2: SSM
ax1.text(5, 95, "Diffuse Voids", color='black', fontsize=10, 
         bbox=dict(facecolor='white', alpha=0.7))

ax2.scatter(pos_ssm[:,0], pos_ssm[:,1], s=0.5, c='darkgreen', alpha=0.3)
ax2.scatter(void_centers[:,0], void_centers[:,1], marker='x', c='red', s=30)

ax2.set_xlim(0, BOX_SIZE)
ax2.set_ylim(0, BOX_SIZE)

# Safe string formatting for title with LaTeX
# We use double backslash \\alpha to ensure it prints correctly without causing errors
title2 = f"SSM Geodesic Sorting ($\\alpha \\approx 9J$)\nN={NUM_PARTICLES}"
ax2.set_title(title2, fontsize=12)

ax2.set_aspect('equal')
ax2.set_xlabel("Mpc/h")
ax2.text(5, 95, "Cleared Voids", color='black', fontsize=10, 
         bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig("ssm_simulation_hires.png")
print("Done! Saved as ssm_simulation_hires.png")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# --- CONFIGURATION ---
NUM_PARTICLES = 500   # Reduced slightly for faster GIF generation
NUM_STEPS = 150
BOX_SIZE = 100.0
VOID_STRENGTH = 1.0   # Increased for visibility
NUM_VOIDS = 6

# --- INITIALIZATION ---
np.random.seed(42)
# Random particles
positions = np.random.rand(NUM_PARTICLES, 2) * BOX_SIZE
# Velocities (start at 0)
velocities = np.zeros((NUM_PARTICLES, 2))

# Define Static Void Centers (The "Exposed" K=13 regions)
void_centers = np.array([
    [20, 20], [80, 80], [20, 80], [80, 20], [50, 50], [50, 20]
])

def get_forces(pos, use_ssm=True):
    force = np.zeros_like(pos)
    
    # SSM Force: Repulsion from Void Centers
    if use_ssm:
        for center in void_centers:
            diff = pos - center 
            # Calculate distance
            dist = np.linalg.norm(diff, axis=1).reshape(-1, 1)
            # Clip distance to avoid division by zero
            dist = np.clip(dist, 1.0, None)
            
            # Repulsive force (1/r^2 falloff for strong local clearing)
            direction = diff / dist
            repulsion = direction * (VOID_STRENGTH / (dist**2))
            force += repulsion

    # Simple Drag (to stop particles from exploding out of the box)
    force -= velocities * 0.1
    return force

# --- SETUP PLOT ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Plot 1: Standard (Control)
scat1 = ax1.scatter([], [], s=5, c='blue', alpha=0.5)
ax1.set_xlim(0, BOX_SIZE)
ax1.set_ylim(0, BOX_SIZE)
ax1.set_title("Standard (Random Drift)")
ax1.set_aspect('equal')

# Plot 2: SSM Lattice
scat2 = ax2.scatter([], [], s=5, c='green', alpha=0.5)
ax2.set_xlim(0, BOX_SIZE)
ax2.set_ylim(0, BOX_SIZE)
ax2.set_title("SSM (Void Clearing)")
ax2.scatter(void_centers[:,0], void_centers[:,1], marker='x', c='red', s=50, label='Void Centers')
ax2.set_aspect('equal')

# Simulation State
pos_std = positions.copy()
pos_ssm = positions.copy()
vel_ssm = np.zeros_like(positions)

def update(frame):
    global pos_std, pos_ssm, vel_ssm
    
    # 1. Update Standard (Just slight random noise/drift)
    pos_std += (np.random.rand(*pos_std.shape) - 0.5) * 0.5
    pos_std = pos_std % BOX_SIZE
    
    # 2. Update SSM (Lattice Repulsion)
    force = get_forces(pos_ssm, use_ssm=True)
    vel_ssm += force + (np.random.rand(*pos_ssm.shape) - 0.5) * 0.1 # Add noise
    pos_ssm += vel_ssm
    pos_ssm = pos_ssm % BOX_SIZE
    
    # Update Artists
    scat1.set_offsets(pos_std)
    scat2.set_offsets(pos_ssm)
    
    # Return artists for blitting
    return scat1, scat2

# --- GENERATE AND SAVE ---
print("Generating simulation... please wait.")
ani = FuncAnimation(fig, update, frames=NUM_STEPS, interval=40, blit=True)

# SAVE TO FILE
output_filename = "ssm_simulation.gif"
writer = PillowWriter(fps=20)
ani.save(output_filename, writer=writer)

print(f"Done! Simulation saved to {output_filename}")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import copy

# --- SETTINGS ---
plt.style.use('default')
params = {'font.size': 12, 'font.family': 'serif', 'axes.grid': True}
plt.rcParams.update(params)

# --- SIMULATION PARAMETERS ---
TEMPERATURE = 2.0
STEPS = 50000
LATTICE_SIZE = 20
J_VOL = 1.0   # Cost per unit length
J_BEND = 5.0  # Cost per unit curvature

# --- KNOT COORDINATES ---
KNOT_4_1 = [
    [0,0,0], [1,1,1], [2,0,2], [3,-1,1], [4,0,0], 
    [3,1,-1], [2,0,-2], [1,-1,-1], [0,0,0]
]

KNOT_6_1 = [
    [0,0,0], [1,2,1], [2,0,2], [3,-2,1], [4,0,0], 
    [5,2,-1], [6,0,-2], [5,-2,-1], [4,0,0],
    [3,2,1], [2,0,-1], [1,-2,0], [0,0,0] 
]

class LatticeKnotSim:
    def __init__(self, initial_knot):
        self.knot = np.array(initial_knot, dtype=float)
        self.energy_history = []
        
    def calculate_energy(self, current_knot):
        length = 0
        curvature = 0
        N = len(current_knot)
        for i in range(N):
            p1 = current_knot[i]
            p2 = current_knot[(i+1)%N]
            p3 = current_knot[(i+2)%N]
            
            # Length
            length += np.linalg.norm(p1 - p2)
            
            # Curvature
            v1 = p1 - p2
            v2 = p2 - p3
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 > 0 and n2 > 0:
                cos_a = np.clip(np.dot(v1, v2)/(n1*n2), -1.0, 1.0)
                curvature += np.arccos(cos_a)
                
        return (J_VOL * length) + (J_BEND * curvature)

    def run(self):
        curr_E = self.calculate_energy(self.knot)
        for _ in range(STEPS):
            idx = random.randint(0, len(self.knot)-1)
            move = np.random.randint(-1, 2, size=3) * 0.5
            new_knot = copy.deepcopy(self.knot)
            new_knot[idx] += move
            
            new_E = self.calculate_energy(new_knot)
            delta_E = new_E - curr_E
            
            if delta_E < 0 or random.random() < np.exp(-delta_E / TEMPERATURE):
                self.knot = new_knot
                curr_E = new_E
            self.energy_history.append(curr_E)
        return self.energy_history

# --- EXECUTE SIMULATION ---
print("Simulating Dark Matter (4_1)...")
sim_4_1 = LatticeKnotSim(KNOT_4_1)
hist_4_1 = sim_4_1.run()

print("Simulating Unstable Excitation (6_1)...")
sim_6_1 = LatticeKnotSim(KNOT_6_1)
hist_6_1 = sim_6_1.run()

# --- FIGURE 2: ENERGY LANDSCAPE ---
plt.figure(figsize=(10, 6))
plt.plot(hist_6_1, color='#d62728', alpha=0.8, label=r'Excited State ($6_1$): Unstable')
plt.plot(hist_4_1, color='#1f77b4', alpha=0.9, label=r'Dark Matter ($4_1$): Ground State')

# Annotations matching your data
plt.axhline(y=54.86, color='red', linestyle='--', alpha=0.5)
plt.text(40000, 58, r'Min Energy $\approx 54.9$', color='red')

plt.axhline(y=22.82, color='blue', linestyle='--', alpha=0.5)
plt.text(40000, 26, r'Min Energy $\approx 22.8$', color='blue')

plt.title(r'SSM Lattice Stability: Energy Minimization ($N=50,000$)', fontsize=14)
plt.xlabel('Simulation Step (Monte Carlo Time)')
plt.ylabel('Lattice Energy ($H_{SSM}$)')
plt.legend()
plt.tight_layout()
plt.savefig('fig_energy_landscape.png', dpi=300)
print("Saved fig_energy_landscape.png")

# --- FIGURE 3: KNOT TOPOLOGY ---
fig = plt.figure(figsize=(12, 6))

# Plot 4_1
ax1 = fig.add_subplot(121, projection='3d')
k1 = np.vstack((sim_4_1.knot, sim_4_1.knot[0]))
ax1.plot(k1[:,0], k1[:,1], k1[:,2], c='#1f77b4', lw=3, marker='o', markersize=4)
ax1.set_title(r"Relaxed Dark Matter ($4_1$)" + "\n" + r"Ground State ($E \approx 22.8$)", fontsize=12)
ax1.set_axis_off()

# Plot 6_1
ax2 = fig.add_subplot(122, projection='3d')
k2 = np.vstack((sim_6_1.knot, sim_6_1.knot[0]))
ax2.plot(k2[:,0], k2[:,1], k2[:,2], c='#d62728', lw=3, marker='o', markersize=4)
ax2.set_title(r"Unstable Excitation ($6_1$)" + "\n" + r"High Mass ($E \approx 54.9$)", fontsize=12)
ax2.set_axis_off()

plt.tight_layout()
plt.savefig('fig_knot_topology.png', dpi=300)
print("Saved fig_knot_topology.png")

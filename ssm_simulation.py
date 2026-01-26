import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree
import random
import time

# ================= CONFIGURATION =================
N_NODES = 5000            # Size of Universe
UNIT_LENGTH = 1.0         # Stitch Length (Planck Length)
HARD_SHELL = 0.95         # Exclusion Principle (Stiffness)
OUTPUT_IMAGE = "ssm_evolutionary_proof.png"

class SSMVisualSim:
    def __init__(self):
        self.nodes = [] 
        self.G = nx.Graph()
        self.triangles = set()
        self.new_nodes_buffer = [] 
        
    def add_node(self, pos):
        """Adds a node to the graph and spatial index."""
        idx = len(self.nodes)
        pos_arr = np.array(pos)
        self.nodes.append(pos_arr)
        self.new_nodes_buffer.append(pos_arr)
        self.G.add_node(idx, pos=(float(pos[0]), float(pos[1]), float(pos[2])))
        return idx

    def add_edge(self, u, v):
        """Adds a stitch (edge) and detects new triangles."""
        if self.G.has_edge(u, v) or u == v: return
        self.G.add_edge(u, v)
        # Topological check for closed loops (Triangles)
        u_nbrs = set(self.G.neighbors(u))
        v_nbrs = set(self.G.neighbors(v))
        common = u_nbrs.intersection(v_nbrs)
        for w in common:
            self.triangles.add(tuple(sorted((u, v, w))))

    def stabilizer_check(self, candidate, tree):
        """
        The Geometric Stabilizer.
        Enforces the Exclusion Principle: No two nodes can be closer than HARD_SHELL.
        """
        # 1. Check Static Tree (Long-term memory)
        if tree:
            neighbors = tree.query_ball_point(candidate, HARD_SHELL)
            if len(neighbors) > 0: return False
        # 2. Check Dynamic Buffer (Short-term memory for new nodes)
        for pos in self.new_nodes_buffer:
            dist = np.linalg.norm(candidate - pos)
            if dist < HARD_SHELL: return False
        return True

    def run(self):
        print(f"--- INITIATING EVOLUTIONARY RUN (N={N_NODES}) ---")
        start_time = time.time()

        # Genesis: The Primary Triangle (2D Seed)
        n0 = self.add_node([0.0, 0.0, 0.0])
        n1 = self.add_node([UNIT_LENGTH, 0.0, 0.0])
        n2 = self.add_node([0.5, np.sqrt(3)/2, 0.0])
        self.add_edge(n0, n1); self.add_edge(n1, n2); self.add_edge(n2, n0)
        
        tree = cKDTree(self.nodes)
        self.new_nodes_buffer = []
        
        attempts = 0
        current_phase = "Phase I: Surface"

        while len(self.nodes) < N_NODES and attempts < 200000:
            # Rebuild tree periodically for performance optimization
            if len(self.new_nodes_buffer) > 50:
                tree = cKDTree(self.nodes)
                self.new_nodes_buffer = []

            # --- EVOLUTIONARY LOGIC (Timeline) ---
            progress = len(self.nodes) / N_NODES
            
            # Probability of "Lifting" (Creating 3D Volume vs. 2D Sheet)
            if progress < 0.15:
                # PHASE I: The 2D Foundation (Planck Era)
                # Bias heavily towards 2D growth to create the initial "Flat" sheet
                lift_prob = 0.05 
                new_phase = "Phase I: 2D Sheet (Planck)"
            elif progress < 0.60:
                # PHASE II: Inflation (Gappy Foam)
                # Explosive 3D growth. High lift probability to create volume rapidly.
                lift_prob = 0.90 
                new_phase = "Phase II: Inflation (Foam)"
            else:
                # PHASE III: Saturation (The Freeze)
                # Balanced growth allowing the lattice to lock into K=12
                lift_prob = 0.50 
                new_phase = "Phase III: Crystallization (Freeze)"

            if new_phase != current_phase:
                print(f" >> PHASE TRANSITION: {current_phase} -> {new_phase}")
                current_phase = new_phase
            # -------------------------------------

            # Check geometric constraints
            can_lift = (len(self.triangles) > 0)
            is_lift = (random.random() < lift_prob) and can_lift
            
            if is_lift:
                # --- 3D GROWTH (Tetrahedral Stacking) ---
                # This operator creates volume by lifting a triangle into a tetrahedron.
                t = random.sample(sorted(list(self.triangles)), 1)[0]
                u, v, w = t
                p0, p1, p2 = self.nodes[u], self.nodes[v], self.nodes[w]
                
                # Calculate the unique "Lift" position (h = sqrt(2/3))
                centroid = (p0+p1+p2)/3.0
                normal = np.cross(p1-p0, p2-p0)
                norm = np.linalg.norm(normal)
                
                if norm > 0:
                    normal /= norm
                    # Alternate up/down for Phase I to keep it sheet-like if lift happens
                    direction = 1 if random.random() > 0.5 else -1
                    cand = centroid + normal * np.sqrt(2/3)*UNIT_LENGTH * direction
                    
                    if self.stabilizer_check(cand, tree):
                        new_id = self.add_node(cand)
                        self.add_edge(new_id, u); self.add_edge(new_id, v); self.add_edge(new_id, w)
                        attempts = 0
                    else: attempts += 1
            else:
                # --- 2D GROWTH (Lateral Stitching) ---
                # This operator expands the surface area.
                edges = list(self.G.edges())
                if not edges: break
                u, v = edges[random.randint(0, len(edges)-1)]
                p1, p2 = self.nodes[u], self.nodes[v]
                mid = (p1+p2)/2.0
                
                # Find perpendicular direction in the local plane
                # Simple approximation: Cross with Z-axis
                direction = np.cross(p2-p1, [0,0,1])
                norm = np.linalg.norm(direction)
                if norm == 0: direction = np.array([0,1,0])
                else: direction /= norm
                
                cand = mid + direction * np.sqrt(3)/2 * UNIT_LENGTH
                if self.stabilizer_check(cand, tree):
                    new_id = self.add_node(cand)
                    self.add_edge(new_id, u); self.add_edge(new_id, v)
                    attempts = 0
                else: attempts += 1

            if len(self.nodes) % 1000 == 0:
                print(f"  > Nodes: {len(self.nodes)} | Phase: {current_phase}")

        print(f"--- SIMULATION COMPLETE ({time.time()-start_time:.2f}s) ---")

    def render_png(self):
        print(f"\n--- RENDERING PROOF ({OUTPUT_IMAGE}) ---")
        
        degrees = np.array([d for n, d in self.G.degree()])
        coords = np.array(self.nodes)
        
        # We highlight K=12 nodes to prove the Cuboctahedral core
        colors = degrees
        
        fig = plt.figure(figsize=(12, 10), dpi=300)
        ax = fig.add_subplot(111, projection='3d')
        
        # Visualizing the Phases
        # K=12 (Yellow) -> The Solid Core
        # K<12 (Purple/Blue) -> The Gappy Surface/Foam
        sc = ax.scatter(coords[:,0], coords[:,1], coords[:,2], 
                        c=colors, cmap='viridis', vmin=4, vmax=12,
                        s=np.where(degrees >= 12, 40, 5), # Make K=12 huge and visible
                        alpha=0.9, edgecolor='none')
        
        # Wireframe to show the "Mesh"
        edges = list(self.G.edges())
        if len(edges) > 5000:
            import random
            edges = random.sample(edges, 5000)
            
        for u, v in edges:
            p1, p2 = coords[u], coords[v]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                    c='gray', alpha=0.1, linewidth=0.2)

        ax.set_title(f"SSM Evolutionary Proof\nYellow Core = Cuboctahedral Saturation (K=12)", fontsize=14)
        
        cbar = plt.colorbar(sc, shrink=0.6)
        cbar.set_label('Coordination Number (K)')
        cbar.set_ticks([6, 12])
        cbar.set_ticklabels(['Surface/Sheet (K=6)', 'Solid Crystal (K=12)'])
        
        ax.view_init(elev=30, azim=45)
        plt.tight_layout()
        plt.savefig(OUTPUT_IMAGE, dpi=300)
        print(f"Saved visualization to {OUTPUT_IMAGE}")
        
        # --- VERIFICATION STATS ---
        k12_count = np.sum(degrees == 12)
        print(f"\n[GEOMETRIC VERIFICATION]")
        print(f"Total Nodes: {len(self.nodes)}")
        print(f"Max Neighbors Found: {np.max(degrees)}")
        print(f"Nodes locked at K=12: {k12_count}")
        if np.max(degrees) == 12:
            print("SUCCESS: Lattice saturated exactly at K=12 (Cuboctahedral Limit).")
        else:
            print(f"NOTE: Max degree is {np.max(degrees)}. (If >12, Hard Shell is too soft).")

if __name__ == "__main__":
    sim = SSMVisualSim()
    sim.run()
    sim.render_png()

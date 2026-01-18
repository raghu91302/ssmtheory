import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree
import random
import time

# ================= CONFIGURATION =================
N_NODES = 5000            # Size of Universe
UNIT_LENGTH = 1.0         # Stitch Length
HARD_SHELL = 0.95         # Exclusion Principle
OUTPUT_IMAGE = "ssm_saturation_k12.png"

class SSMVisualSim:
    def __init__(self):
        self.nodes = [] 
        self.G = nx.Graph()
        self.triangles = set()
        self.new_nodes_buffer = [] 
        
    def add_node(self, pos):
        idx = len(self.nodes)
        pos_arr = np.array(pos)
        self.nodes.append(pos_arr)
        self.new_nodes_buffer.append(pos_arr)
        self.G.add_node(idx, pos=(float(pos[0]), float(pos[1]), float(pos[2])))
        return idx

    def add_edge(self, u, v):
        if self.G.has_edge(u, v) or u == v: return
        self.G.add_edge(u, v)
        u_nbrs = set(self.G.neighbors(u))
        v_nbrs = set(self.G.neighbors(v))
        common = u_nbrs.intersection(v_nbrs)
        for w in common:
            self.triangles.add(tuple(sorted((u, v, w))))

    def stabilizer_check(self, candidate, tree):
        # 1. Check Static Tree
        if tree:
            neighbors = tree.query_ball_point(candidate, HARD_SHELL)
            if len(neighbors) > 0: return False
        # 2. Check Dynamic Buffer
        for pos in self.new_nodes_buffer:
            dist = np.linalg.norm(candidate - pos)
            if dist < HARD_SHELL: return False
        return True

    def run(self):
        print(f"--- INITIATING SSM VISUALIZATION RUN (N={N_NODES}) ---")
        start_time = time.time()

        # Genesis
        n0 = self.add_node([0.0, 0.0, 0.0])
        n1 = self.add_node([UNIT_LENGTH, 0.0, 0.0])
        n2 = self.add_node([0.5, np.sqrt(3)/2, 0.0])
        self.add_edge(n0, n1); self.add_edge(n1, n2); self.add_edge(n2, n0)
        
        tree = cKDTree(self.nodes)
        self.new_nodes_buffer = []
        
        attempts = 0
        while len(self.nodes) < N_NODES and attempts < 200000:
            if len(self.new_nodes_buffer) > 50:
                tree = cKDTree(self.nodes)
                self.new_nodes_buffer = []

            # Bias towards 3D Lift (60%) to ensure we see a Bulk Core
            is_lift = (random.random() < 0.60) and (len(self.triangles) > 0)
            
            if is_lift:
                t = random.sample(sorted(list(self.triangles)), 1)[0]
                u, v, w = t
                p0, p1, p2 = self.nodes[u], self.nodes[v], self.nodes[w]
                centroid = (p0+p1+p2)/3.0
                normal = np.cross(p1-p0, p2-p0)
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normal /= norm
                    cand = centroid + normal * np.sqrt(2/3)*UNIT_LENGTH
                    if self.stabilizer_check(cand, tree):
                        new_id = self.add_node(cand)
                        self.add_edge(new_id, u); self.add_edge(new_id, v); self.add_edge(new_id, w)
                        attempts = 0
                    else: attempts += 1
            else:
                edges = list(self.G.edges())
                if not edges: break
                u, v = edges[random.randint(0, len(edges)-1)]
                p1, p2 = self.nodes[u], self.nodes[v]
                mid = (p1+p2)/2.0
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
                print(f"  > Generated {len(self.nodes)} nodes...")

        print(f"--- SIMULATION COMPLETE ({time.time()-start_time:.2f}s) ---")

    def render_png(self):
        print(f"\n--- RENDERING HIGH-RES IMAGE ({OUTPUT_IMAGE}) ---")
        
        degrees = np.array([d for n, d in self.G.degree()])
        coords = np.array(self.nodes)
        
        # Color Map: Blue (Low K) -> Yellow/Red (High K=12)
        # We clamp the max color at 12 to make the saturation obvious
        colors = degrees
        
        fig = plt.figure(figsize=(12, 10), dpi=300) # High DPI for paper
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot Scatter
        # Saturated nodes (K>=12) get highlighted larger and brighter
        sc = ax.scatter(coords[:,0], coords[:,1], coords[:,2], 
                        c=colors, cmap='plasma', vmin=4, vmax=12,
                        s=np.where(degrees >= 12, 30, 10), # Size boost for core
                        alpha=0.8, edgecolor='none')
        
        # Add a transparent wireframe (subset of edges) to show "Mesh" structure
        edges = list(self.G.edges())
        if len(edges) > 3000:
            import random
            edges = random.sample(edges, 3000)
            
        for u, v in edges:
            p1, p2 = coords[u], coords[v]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                    c='gray', alpha=0.1, linewidth=0.3)

        # Labels and Style
        ax.set_title(f"SSM Lattice Saturation (N={N_NODES})\nCore Saturation Limit: K=12", fontsize=14)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Colorbar
        cbar = plt.colorbar(sc, shrink=0.6)
        cbar.set_label('Local Coordination Number (K)')
        cbar.set_ticks([4, 6, 8, 10, 12])
        cbar.set_ticklabels(['4 (Surface)', '6', '8', '10', '12 (Saturated)'])
        
        # Orient the view to see depth
        ax.view_init(elev=30, azim=45)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_IMAGE, dpi=300)
        print(f"SUCCESS: Saved {OUTPUT_IMAGE}")
        
        # Print stats for the paper text
        bulk_k = [k for k in degrees if k >= 9]
        print("\n[Paper Data]")
        print(f"Max Degree: {np.max(degrees)}")
        print(f"Bulk Avg: {np.mean(bulk_k):.2f}")

if __name__ == "__main__":
    sim = SSMVisualSim()
    sim.run()
    sim.render_png()

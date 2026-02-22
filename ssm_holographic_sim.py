"""
SSM Holographic Vacuum Simulation
==================================
Constructive verification of K=12 lattice saturation via
2D-dominant holographic growth with proximity bonding.

Usage:
    python ssm_holographic_sim.py [--nodes N] [--lift PROB] [--sweep]

Author: Raghu Kulkarni (raghu@idrive.com)
License: MIT
"""

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from collections import Counter
import random
import argparse
import time

# ============================================================
#  Physical Constants
# ============================================================
UNIT_LENGTH   = 1.0              # Stitch length L (Planck length)
HARD_SHELL    = 0.95             # Exclusion radius (accommodates
                                 #   Regge deficit angle delta~0.128 rad)
BOND_RADIUS   = 1.05             # Proximity bonding threshold
LIFT_HEIGHT   = np.sqrt(2.0/3.0) # Tetrahedral apex: h = sqrt(2/3) * L
LATERAL_HEIGHT = np.sqrt(3.0)/2  # In-plane equilateral: h = sqrt(3)/2 * L

# ============================================================
#  Core Simulation
# ============================================================
class SSMHolographicSim:
    """
    Builds a discrete vacuum via two operators:

      Stitch (2D) : Laterally extends a hexagonal sheet (K=6 in-plane).
      Lift   (3D) : Projects a node from a triangular face at height
                     h = sqrt(2/3) * L, seeding a new adjacent layer.

    Proximity Bonding: After every batch of new nodes, all pairs
    within distance BOND_RADIUS automatically form edges. This is
    the entanglement mechanism that connects adjacent 2D layers,
    producing the 6(in-plane) + 3(above) + 3(below) = K=12
    Cuboctahedral coordination of the FCC lattice.
    """

    def __init__(self, target_nodes=5000, lift_prob=0.05):
        self.target   = target_nodes
        self.lift_prob = lift_prob

        self.nodes    = []            # List of np.array positions
        self.G        = nx.Graph()    # Topological graph
        self.triangles       = set()  # All detected triangles
        self.active_triangles = set() # Triangles available for Lift
        self.active_edges     = set() # Edges available for Stitch
        self.buffer   = []            # New nodes pending tree rebuild

    # --- Node / Edge Management ---

    def _add_node(self, pos):
        idx = len(self.nodes)
        p = np.array(pos, dtype=float)
        self.nodes.append(p)
        self.buffer.append(p)
        self.G.add_node(idx)
        return idx

    def _add_edge(self, u, v):
        if self.G.has_edge(u, v) or u == v:
            return
        self.G.add_edge(u, v)
        self.active_edges.add((min(u,v), max(u,v)))
        # Detect triangles (3-cliques)
        for w in set(self.G.neighbors(u)) & set(self.G.neighbors(v)):
            tri = tuple(sorted((u, v, w)))
            if tri not in self.triangles:
                self.triangles.add(tri)
                self.active_triangles.add(tri)

    # --- Proximity Bonding (Entanglement) ---

    def _proximity_stitch(self, tree):
        """Connect all node pairs within BOND_RADIUS."""
        for u, v in tree.query_pairs(r=BOND_RADIUS):
            if not self.G.has_edge(u, v):
                self._add_edge(u, v)

    # --- Exclusion Check ---

    def _is_valid(self, candidate, tree):
        """Enforce Hard Shell exclusion principle."""
        if tree and tree.query_ball_point(candidate, HARD_SHELL):
            return False
        for p in self.buffer:
            if np.linalg.norm(candidate - p) < HARD_SHELL:
                return False
        return True

    # --- Growth Operators ---

    def _stitch_lateral(self, edge, tree):
        """2D Operator: extend hexagonal sheet in-plane."""
        u, v = edge
        p1, p2 = self.nodes[u], self.nodes[v]
        mid = (p1 + p2) / 2.0

        # Direction: away from attached triangle (in-plane outward)
        attached = [t for t in self.active_triangles
                    if u in t and v in t]
        if attached:
            w = [n for n in attached[0] if n != u and n != v][0]
            direction = mid - self.nodes[w]
        else:
            direction = np.cross(p2 - p1, [0, 0, 1])

        norm = np.linalg.norm(direction)
        if norm > 0:
            direction /= norm
        else:
            direction = np.array([0.0, 1.0, 0.0])

        # Try both planar sides
        c1 = mid + direction * LATERAL_HEIGHT * UNIT_LENGTH
        c2 = mid - direction * LATERAL_HEIGHT * UNIT_LENGTH
        v1, v2 = self._is_valid(c1, tree), self._is_valid(c2, tree)

        if not v1 and not v2:
            self.active_edges.discard((min(u,v), max(u,v)))
            return False

        cand = c1 if v1 else c2
        nid = self._add_node(cand)
        self._add_edge(nid, u)
        self._add_edge(nid, v)
        return True

    def _lift_triangle(self, tri, tree):
        """3D Operator: seed a new layer from a triangular face."""
        u, v, w = tri
        p0, p1, p2 = self.nodes[u], self.nodes[v], self.nodes[w]
        centroid = (p0 + p1 + p2) / 3.0
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm == 0:
            self.active_triangles.discard(tri)
            return False
        normal /= norm

        # Try both directions (up/down)
        signs = [1, -1] if random.random() > 0.5 else [-1, 1]
        for s in signs:
            cand = centroid + normal * LIFT_HEIGHT * UNIT_LENGTH * s
            if self._is_valid(cand, tree):
                nid = self._add_node(cand)
                self._add_edge(nid, u)
                self._add_edge(nid, v)
                self._add_edge(nid, w)
                return True

        self.active_triangles.discard(tri)
        return False

    # --- Main Loop ---

    def run(self, verbose=True):
        """Execute the holographic growth algorithm."""
        t0 = time.time()

        # Genesis: single equilateral triangle
        self._add_node([0.0, 0.0, 0.0])
        self._add_node([UNIT_LENGTH, 0.0, 0.0])
        self._add_node([0.5 * UNIT_LENGTH, LATERAL_HEIGHT * UNIT_LENGTH, 0.0])
        self._add_edge(0, 1)
        self._add_edge(1, 2)
        self._add_edge(2, 0)

        tree = cKDTree(self.nodes)
        self.buffer = []
        attempts = 0

        while len(self.nodes) < self.target and attempts < 500000:
            # Periodic tree rebuild + proximity bonding
            if len(self.buffer) > 40:
                tree = cKDTree(self.nodes)
                self.buffer = []
                self._proximity_stitch(tree)

            can_lift  = len(self.active_triangles) > 0
            can_stitch = len(self.active_edges) > 0

            # Holographic bias: overwhelmingly 2D
            if random.random() < self.lift_prob and can_lift:
                tri = random.choice(sorted(self.active_triangles))
                ok = self._lift_triangle(tri, tree)
            elif can_stitch:
                edge = random.choice(sorted(self.active_edges))
                ok = self._stitch_lateral(edge, tree)
            else:
                ok = False

            attempts = 0 if ok else attempts + 1

            if verbose and len(self.nodes) % 1000 == 0 and ok:
                stats = self.get_stats()
                elapsed = time.time() - t0
                print(f"  N={stats['n']:5d} | K_mean={stats['k_mean']:.2f} "
                      f"| K=12: {stats['pct_k12']:.1f}% "
                      f"| K>=10: {stats['pct_k10']:.1f}% | {elapsed:.1f}s")

        # Final proximity stitch
        tree = cKDTree(self.nodes)
        self._proximity_stitch(tree)

        elapsed = time.time() - t0
        if verbose:
            status = "COMPLETE" if len(self.nodes) >= self.target else "JAMMED"
            print(f"--- {status}: {len(self.nodes)} nodes in {elapsed:.1f}s ---")

        return self

    # --- Analysis ---

    def get_stats(self):
        """Return coordination statistics."""
        degrees = np.array([d for _, d in self.G.degree()])
        dd = Counter(degrees)
        n = len(self.nodes)
        k12 = dd.get(12, 0)
        k10 = sum(dd[k] for k in dd if k >= 10)
        return {
            'n':        n,
            'k_max':    int(degrees.max()) if n > 0 else 0,
            'k_mean':   float(degrees.mean()) if n > 0 else 0,
            'k12':      k12,
            'k10':      k10,
            'pct_k12':  100.0 * k12 / n if n > 0 else 0,
            'pct_k10':  100.0 * k10 / n if n > 0 else 0,
            'deg_dist': dict(sorted(dd.items())),
            'edges':    self.G.number_of_edges(),
            'triangles': len(self.triangles),
        }

    def count_emergent_squares(self):
        """Count square faces that emerged from triangular growth."""
        squares = 0
        checked = set()
        for u in self.G.nodes():
            u_nbrs = list(self.G.neighbors(u))
            for i, a in enumerate(u_nbrs):
                for b in u_nbrs[i+1:]:
                    if self.G.has_edge(a, b):
                        continue
                    key = (min(a,b), max(a,b))
                    if key in checked:
                        continue
                    checked.add(key)
                    for c in set(self.G.neighbors(a)) & set(self.G.neighbors(b)) - {u}:
                        if self.G.has_edge(c, u):
                            continue
                        pu, pa, pb, pc = (self.nodes[u], self.nodes[a],
                                          self.nodes[b], self.nodes[c])
                        d1 = np.linalg.norm(pu - pc)
                        d2 = np.linalg.norm(pa - pb)
                        if 1.2 < d1 < 1.6 and 1.2 < d2 < 1.6:
                            squares += 1
        return squares


# ============================================================
#  CLI Entry Point
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="SSM Holographic Vacuum Simulation")
    parser.add_argument('--nodes', type=int, default=5000,
                        help='Target number of nodes (default: 5000)')
    parser.add_argument('--lift', type=float, default=0.05,
                        help='3D Lift probability (default: 0.05)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--sweep', action='store_true',
                        help='Run lift-probability sweep')
    parser.add_argument('--rex-sweep', action='store_true',
                        help='Run exclusion radius sensitivity sweep')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.sweep:
        # --- Table 1: Lift Probability Sweep ---
        print("=" * 65)
        print("TABLE 1: Lift Probability Sweep (N=3000)")
        print("=" * 65)
        print(f"{'Lift%':>6} | {'K_mean':>7} | {'%K=12':>7} | {'%K>=10':>7}")
        print("-" * 45)
        for lp in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.85]:
            random.seed(args.seed); np.random.seed(args.seed)
            sim = SSMHolographicSim(target_nodes=3000, lift_prob=lp)
            sim.run(verbose=False)
            s = sim.get_stats()
            print(f"{lp*100:5.0f}% | {s['k_mean']:7.2f} | "
                  f"{s['pct_k12']:6.1f}% | {s['pct_k10']:6.1f}%")

    elif args.rex_sweep:
        # --- Exclusion Radius Sensitivity Sweep ---
        print("=" * 65)
        print("EXCLUSION RADIUS SENSITIVITY (N=500, Lift=5%)")
        print("=" * 65)
        print(f"{'Rex':>6} | {'Nodes':>6} | {'K_max':>6} | {'K_mean':>7}")
        print("-" * 40)
        import copy
        for rex in np.arange(0.80, 1.02, 0.02):
            random.seed(args.seed); np.random.seed(args.seed)
            global HARD_SHELL
            old_hs = HARD_SHELL
            HARD_SHELL = rex
            sim = SSMHolographicSim(target_nodes=500, lift_prob=0.05)
            sim.run(verbose=False)
            s = sim.get_stats()
            jammed = "JAMMED" if s['n'] < 450 else ""
            print(f"{rex:5.2f}  | {s['n']:6d} | {s['k_max']:6d} | "
                  f"{s['k_mean']:7.2f}  {jammed}")
            HARD_SHELL = old_hs

    else:
        # --- Standard Run ---
        print(f"SSM Holographic Simulation")
        print(f"  Nodes: {args.nodes}, Lift: {args.lift*100:.0f}%, "
              f"Seed: {args.seed}")
        print(f"  Rex={HARD_SHELL}, Bond={BOND_RADIUS}")
        print()

        sim = SSMHolographicSim(
            target_nodes=args.nodes, lift_prob=args.lift)
        sim.run(verbose=True)
        s = sim.get_stats()

        print(f"\n{'='*50}")
        print(f"FINAL RESULTS")
        print(f"{'='*50}")
        print(f"  Nodes:     {s['n']}")
        print(f"  Edges:     {s['edges']}")
        print(f"  Triangles: {s['triangles']}")
        print(f"  K_max:     {s['k_max']}")
        print(f"  K_mean:    {s['k_mean']:.2f}")
        print(f"  K=12:      {s['k12']} ({s['pct_k12']:.1f}%)")
        print(f"  K>=10:     {s['k10']} ({s['pct_k10']:.1f}%)")
        print(f"\n  Degree Distribution:")
        for k, count in s['deg_dist'].items():
            pct = 100 * count / s['n']
            bar = "#" * max(1, int(pct))
            print(f"    K={k:2d}: {count:5d} ({pct:5.1f}%) {bar}")

        # Emergent square count (slow for large N)
        if s['n'] <= 5000:
            print(f"\n  Counting emergent square faces...")
            nsq = sim.count_emergent_squares()
            print(f"  Emergent squares: {nsq}")
            print(f"  Triangles: {s['triangles']}")
            if nsq > 0:
                print(f"  Tri:Sq ratio = {s['triangles']/nsq:.1f}:1 "
                      f"(cuboctahedron: 1.33:1)")


if __name__ == "__main__":
    main()

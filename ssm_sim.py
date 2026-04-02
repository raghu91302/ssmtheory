import numpy as np, networkx as nx, random, time, argparse
from scipy.spatial import cKDTree
from collections import Counter

UNIT_LENGTH = 1.0
HARD_SHELL = 0.95
BOND_RADIUS = 1.05
LIFT_HEIGHT = np.sqrt(2.0/3.0)   # Tetrahedral apex: h = sqrt(2/3)*L
LATERAL_HEIGHT = np.sqrt(3.0)/2   # Equilateral apex: h = sqrt(3)/2*L

class SSMSim:
    def __init__(self, target_nodes=5000, lift_prob=0.05,
                 hard_shell=HARD_SHELL):
        self.target = target_nodes
        self.lift_prob = lift_prob
        self.hs = hard_shell
        self.nodes = []
        self.G = nx.Graph()
        self.triangles = set()
        self.active_triangles = set()
        self.active_edges = set()
        self.buffer = []

    def _add_node(self, pos):
        idx = len(self.nodes)
        self.nodes.append(np.array(pos, dtype=float))
        self.buffer.append(self.nodes[-1])
        self.G.add_node(idx)
        return idx

    def _add_edge(self, u, v):
        if self.G.has_edge(u, v) or u == v: return
        self.G.add_edge(u, v)
        self.active_edges.add((min(u,v), max(u,v)))
        for w in (set(self.G.neighbors(u))
                  & set(self.G.neighbors(v))):
            tri = tuple(sorted((u, v, w)))
            if tri not in self.triangles:
                self.triangles.add(tri)
                self.active_triangles.add(tri)

    def _proximity_stitch(self, tree):
        for u, v in tree.query_pairs(r=BOND_RADIUS):
            if not self.G.has_edge(u, v):
                self._add_edge(u, v)

    def _is_valid(self, cand, tree):
        if tree and tree.query_ball_point(cand, self.hs):
            return False
        for p in self.buffer:
            if np.linalg.norm(cand - p) < self.hs:
                return False
        return True

    def _stitch_lateral(self, edge, tree):
        u, v = edge
        p1, p2 = self.nodes[u], self.nodes[v]
        mid = (p1 + p2) / 2.0
        att = [t for t in self.active_triangles
               if u in t and v in t]
        if att:
            w = [n for n in att[0]
                 if n != u and n != v][0]
            d = mid - self.nodes[w]
        else:
            d = np.cross(p2 - p1, [0, 0, 1])
        norm = np.linalg.norm(d)
        if norm > 0: d /= norm
        else: d = np.array([0.0, 1.0, 0.0])
        c1 = mid + d * LATERAL_HEIGHT * UNIT_LENGTH
        c2 = mid - d * LATERAL_HEIGHT * UNIT_LENGTH
        v1 = self._is_valid(c1, tree)
        v2 = self._is_valid(c2, tree)
        if not v1 and not v2:
            self.active_edges.discard(
                (min(u,v), max(u,v)))
            return False
        cand = c1 if v1 else c2
        nid = self._add_node(cand)
        self._add_edge(nid, u)
        self._add_edge(nid, v)
        return True

    def _lift_triangle(self, tri, tree):
        u, v, w = tri
        p0 = self.nodes[u]
        p1 = self.nodes[v]
        p2 = self.nodes[w]
        centroid = (p0 + p1 + p2) / 3.0
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm == 0:
            self.active_triangles.discard(tri)
            return False
        normal /= norm
        if random.random() > 0.5:
            signs = [1, -1]
        else:
            signs = [-1, 1]
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

    def run(self, verbose=False):
        self._add_node([0.0, 0.0, 0.0])
        self._add_node([UNIT_LENGTH, 0.0, 0.0])
        self._add_node([0.5*UNIT_LENGTH,
                        LATERAL_HEIGHT*UNIT_LENGTH, 0.0])
        self._add_edge(0, 1)
        self._add_edge(1, 2)
        self._add_edge(2, 0)
        tree = cKDTree(self.nodes)
        self.buffer = []
        attempts = 0
        while (len(self.nodes) < self.target
               and attempts < 500000):
            if len(self.buffer) > 40:
                tree = cKDTree(self.nodes)
                self.buffer = []
                self._proximity_stitch(tree)
            can_lift = len(self.active_triangles) > 0
            can_stitch = len(self.active_edges) > 0
            if (random.random() < self.lift_prob
                    and can_lift):
                tri = random.choice(
                    sorted(self.active_triangles))
                ok = self._lift_triangle(tri, tree)
            elif can_stitch:
                edge = random.choice(
                    sorted(self.active_edges))
                ok = self._stitch_lateral(edge, tree)
            else:
                ok = False
            attempts = 0 if ok else attempts + 1
            if (verbose and ok
                    and len(self.nodes) % 500 == 0):
                s = self.get_stats()
                print(f"  N={s['n']:5d} K12="
                      f"{s['pct_k12']:.1f}%")
        tree = cKDTree(self.nodes)
        self._proximity_stitch(tree)
        return self

    def get_stats(self):
        degrees = np.array(
            [d for _, d in self.G.degree()])
        dd = Counter(degrees)
        n = len(self.nodes)
        k12 = dd.get(12, 0)
        k10 = sum(dd[k] for k in dd if k >= 10)
        return {
            'n': n,
            'k_max': int(degrees.max()) if n else 0,
            'k_mean': float(degrees.mean()) if n else 0,
            'k12': k12,
            'pct_k12': 100.0*k12/n if n else 0,
            'k10': k10,
            'pct_k10': 100.0*k10/n if n else 0,
            'edges': self.G.number_of_edges(),
            'deg_dist': dict(sorted(dd.items())),
        }

    def full_stats(self, tol=0.2):
        """Stats + layers, aspect, Phi, isotropy."""
        s = self.get_stats()
        pos = np.array(self.nodes)
        n = s['n']; z = pos[:, 2]
        # Layers
        visited = np.zeros(n, dtype=bool)
        layers = []; lid = np.full(n, -1, int)
        idx = 0
        for i in np.argsort(z):
            if visited[i]: continue
            mask = ((np.abs(z - z[i]) < tol)
                    & (~visited))
            m = np.where(mask)[0]
            if len(m) >= 3:
                layers.append(m)
                lid[m] = idx; idx += 1
            visited[mask] = True
        big = [(i,m) for i,m in enumerate(layers)
               if len(m) >= 10]
        sigmas = [float(np.std(z[m]))
                  for _, m in big]
        z_means = sorted(
            [float(np.mean(z[m]))
             for _, m in big])
        spacings = (np.diff(z_means)
                    if len(z_means) > 1
                    else np.array([]))
        n_perf = sum(1 for sg in sigmas
                     if sg < 1e-10)
        # IL bonds
        il = np.zeros(n, dtype=int)
        for nd in range(n):
            if lid[nd] < 0: continue
            for nb in self.G.neighbors(nd):
                if (lid[nb] >= 0
                        and lid[nb] != lid[nd]):
                    il[nd] += 1
        in_l = lid >= 0
        ilc = il[in_l]
        # Aspect ratio z/xy
        zr = float(z.max() - z.min())
        xyr = max(
            float(pos[:,0].max()-pos[:,0].min()),
            float(pos[:,1].max()-pos[:,1].min()))
        aspect = zr / max(0.01, xyr)
        # Volumetric yield
        k12f = s['pct_k12'] / 100.0
        phi = k12f * aspect
        # Isotropy eigenvalue ratio
        iso = 0.0
        if n > 10:
            ct = pos - pos.mean(axis=0)
            cov = np.cov(ct.T)
            ev = sorted(np.linalg.eigvalsh(cov))
            iso = ev[0]/ev[2] if ev[2] > 0 else 0
        s.update({
            'n_layers': len(layers),
            'n_big': len(big),
            'n_perf': n_perf,
            'spacings': spacings,
            'il_mean': float(ilc.mean())
                       if len(ilc) else 0,
            'il_frac': float(np.mean(ilc > 0))
                       if len(ilc) else 0,
            'aspect': aspect,
            'phi': phi,
            'iso': iso,
        })
        return s

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--nodes', type=int,
                   default=1000)
    p.add_argument('--lift', type=float,
                   default=0.05)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--analyze', action='store_true')
    args = p.parse_args()

    if args.analyze:
        # === TABLE 2: Lift sweep (N=1000) ===
        print("=" * 65)
        print("TABLE 2: Lift Sweep (N=1000)")
        print("=" * 65)
        print(f"{'Lift':>5} | {'K12%':>7} | "
              f"{'Layers':>6} | {'z/xy':>5} | "
              f"{'Phi':>6} | {'iso':>5}")
        for lp in [0.01, 0.03, 0.05, 0.10,
                    0.15, 0.30, 0.50, 0.85]:
            random.seed(args.seed)
            np.random.seed(args.seed)
            s = SSMSim(1000, lp
                       ).run().full_stats()
            print(f"  {lp*100:4.0f}% | "
                  f"{s['pct_k12']:5.1f} | "
                  f"{s['n_layers']:5d} | "
                  f"{s['aspect']:.2f} | "
                  f"{s['phi']:.3f} | "
                  f"{s['iso']:.3f}")

        # === TABLE 3: Scaling ===
        print("\n" + "=" * 65)
        print("TABLE 3: Finite-Size Scaling")
        print("=" * 65)
        for N in [250, 500, 750, 1000]:
            random.seed(args.seed)
            np.random.seed(args.seed)
            s = SSMSim(N, 0.05
                       ).run().full_stats()
            print(f"  N={s['n']:5d} | "
                  f"K={s['k_mean']:5.2f} | "
                  f"K12={s['pct_k12']:5.1f}% | "
                  f"iso={s['iso']:.3f}")

        # === TABLE 4: Rex Sweep ===
        print("\n" + "=" * 65)
        print("TABLE 4: Rex Sweep (N=500)")
        print("=" * 65)
        for rex in np.arange(0.50, 1.02, 0.02):
            random.seed(args.seed)
            np.random.seed(args.seed)
            s = SSMSim(500, 0.05,
                       hard_shell=rex
                       ).run().get_stats()
            print(f"  Rex={rex:.2f} | "
                  f"N={s['n']:4d} | "
                  f"Kmax={s['k_max']:2d}")

        # === Layer analysis ===
        print("\n" + "=" * 65)
        print(f"LAYER ANALYSIS (N={args.nodes})")
        print("=" * 65)
        random.seed(args.seed)
        np.random.seed(args.seed)
        sim = SSMSim(args.nodes, args.lift)
        sim.run(verbose=True)
        s = sim.full_stats()
        print(f"N={s['n']} K12={s['pct_k12']:.1f}%"
              f" Kmax={s['k_max']}")
        print(f"Layers={s['n_layers']}"
              f" Perfect={s['n_perf']}/{s['n_big']}")
        sp = s['spacings']
        if len(sp):
            print(f"Spacing={np.mean(sp):.4f}"
                  f"+/-{np.std(sp):.4f}"
                  f" (FCC={np.sqrt(2/3):.4f})")
        print(f"IL bonds={s['il_mean']:.1f}"
              f" ({s['il_frac']*100:.0f}% bonded)")
        print(f"Aspect z/xy={s['aspect']:.2f}"
              f" Phi={s['phi']:.3f}"
              f" iso={s['iso']:.3f}")

    else:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"SSM: N={args.nodes}"
              f" lift={args.lift}"
              f" seed={args.seed}")
        sim = SSMSim(args.nodes, args.lift)
        sim.run(verbose=True)
        s = sim.full_stats()
        print(f"\nN={s['n']}"
              f" K={s['k_mean']:.2f}"
              f" K12={s['pct_k12']:.1f}%"
              f" layers={s['n_layers']}"
              f" aspect={s['aspect']:.2f}"
              f" Phi={s['phi']:.3f}"
              f" iso={s['iso']:.3f}")

if __name__ == '__main__':
    main()

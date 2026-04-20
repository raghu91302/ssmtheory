"""
fcc_growth.py  --  Competitive planar/vertical network growth model.

Two operations:
  Stitch: S=2 sphere constraints -> 1D solution manifold (always soluble)
  Lift:   S=3 sphere constraints -> 0D manifold (2 discrete points)
  Optimal lift probability: p = exp(-3) ~ 0.0498

Reproduces all tables and figures in:
  "Emergent FCC Order and Finite-Size Scaling in Competitive
   Planar-Vertical Network Growth", Kulkarni (2026)

Usage:
  python fcc_growth.py               # single run, N=3000, p=e^{-3}
  python fcc_growth.py --analyze     # reproduce all paper tables and sweeps
  python fcc_growth.py --nodes 1000 --lift 0.05 --seed 7

Repository: https://github.com/raghu91302/ssmtheory
"""

import numpy as np
import networkx as nx
import random
import argparse
from scipy.spatial import cKDTree
from collections import Counter

# ── Constants ──────────────────────────────────────────────────────
L          = 1.0               # unit bond length
HARD_SHELL = 0.95              # default exclusion radius
BOND_R     = 1.05              # proximity bonding radius
LIFT_H     = np.sqrt(2.0/3.0)  # FCC stacking height = sqrt(2/3)*L
LAT_H      = np.sqrt(3.0)/2    # equilateral apex height = sqrt(3)/2*L


# ── Core growth class ──────────────────────────────────────────────
class FCCNetworkGrowth:
    """
    Two-operation discrete network growth model producing FCC order.

    Stitch (S=2): lateral planar extension via 2 sphere constraints.
                  Solution manifold: 1D (circle) -- always soluble.
    Lift   (S=3): vertical 3D nucleation via 3 sphere constraints.
                  Solution manifold: 0D (2 discrete points).
                  Geometric suppression: P(success) ~ e^{-S} = e^{-3}.

    Parameters
    ----------
    N          : target number of nodes
    p_lift     : probability of choosing Lift over Stitch at each step
    hard_shell : exclusion radius in units of L (default 0.95)
    """

    def __init__(self, N=3000, p_lift=np.exp(-3), hard_shell=HARD_SHELL):
        self.N   = N
        self.p   = p_lift
        self.hs  = hard_shell
        self.nodes = []
        self.buf   = []          # buffer for kD-tree rebuild
        self.G     = nx.Graph()
        self.tris  = set()       # all triangles ever formed
        self.atri  = set()       # active (growable) triangles
        self.aedg  = set()       # active (growable) edges

    # ── Graph primitives ───────────────────────────────────────────

    def _add_node(self, pos):
        i = len(self.nodes)
        v = np.array(pos, float)
        self.nodes.append(v)
        self.buf.append(v)
        self.G.add_node(i)
        return i

    def _add_edge(self, u, v):
        if self.G.has_edge(u, v) or u == v:
            return
        self.G.add_edge(u, v)
        self.aedg.add((min(u, v), max(u, v)))
        for w in set(self.G.neighbors(u)) & set(self.G.neighbors(v)):
            t = tuple(sorted((u, v, w)))
            if t not in self.tris:
                self.tris.add(t)
                self.atri.add(t)

    def _prox(self, tree):
        """Proximity bonding: connect any two nodes within BOND_R."""
        for u, v in tree.query_pairs(r=BOND_R):
            if not self.G.has_edge(u, v):
                self._add_edge(u, v)

    def _ok(self, c, tree):
        """Hard-shell exclusion check."""
        if tree and tree.query_ball_point(c, self.hs):
            return False
        return all(np.linalg.norm(c - b) >= self.hs for b in self.buf)

    # ── Growth operations ──────────────────────────────────────────

    def _stitch(self, edge, tree):
        """
        Stitch: S=2 constraints, 1D solution manifold.
        Places a new node equidistant from both edge endpoints.
        """
        u, v = edge
        mid  = (self.nodes[u] + self.nodes[v]) / 2.0
        # Direction: away from any existing triangle on this edge
        att = [t for t in self.atri if u in t and v in t]
        if att:
            w = [x for x in att[0] if x not in (u, v)][0]
            d = mid - self.nodes[w]
        else:
            d = np.cross(self.nodes[v] - self.nodes[u], [0, 0, 1])
        n = np.linalg.norm(d)
        d = d / n if n > 0 else np.array([0., 1., 0.])
        for s in [1, -1]:
            c = mid + s * d * LAT_H * L
            if self._ok(c, tree):
                i = self._add_node(c)
                self._add_edge(i, u)
                self._add_edge(i, v)
                return True
        self.aedg.discard((min(u, v), max(u, v)))
        return False

    def _lift(self, tri, tree):
        """
        Lift: S=3 constraints, 0D solution manifold (2 discrete points).
        Geometric suppression factor: P(success) ~ e^{-3}.
        Places a new node above the centroid of the triangle at height sqrt(2/3)*L.
        """
        u, v, w = tri
        cen = (self.nodes[u] + self.nodes[v] + self.nodes[w]) / 3.0
        nm  = np.cross(self.nodes[v] - self.nodes[u],
                       self.nodes[w] - self.nodes[u])
        n = np.linalg.norm(nm)
        if n == 0:
            self.atri.discard(tri)
            return False
        nm /= n
        signs = [1, -1] if random.random() > 0.5 else [-1, 1]
        for s in signs:
            c = cen + s * nm * LIFT_H * L
            if self._ok(c, tree):
                i = self._add_node(c)
                for b in (u, v, w):
                    self._add_edge(i, b)
                return True
        self.atri.discard(tri)
        return False

    # ── Main growth loop ───────────────────────────────────────────

    def run(self, verbose=False):
        """Grow the network to self.N nodes."""
        # Seed: single equilateral triangle
        for pos in [[0., 0., 0.], [L, 0., 0.], [0.5*L, LAT_H*L, 0.]]:
            self._add_node(pos)
        for e in [(0, 1), (1, 2), (2, 0)]:
            self._add_edge(*e)

        tree = cKDTree(self.nodes)
        self.buf = []
        fail = 0

        while len(self.nodes) < self.N and fail < 500000:
            # Rebuild kD-tree and do proximity bonding periodically
            if len(self.buf) > 40:
                tree = cKDTree(self.nodes)
                self.buf = []
                self._prox(tree)

            # Choose operation
            if random.random() < self.p and self.atri:
                ok = self._lift(random.choice(sorted(self.atri)), tree)
            elif self.aedg:
                ok = self._stitch(random.choice(sorted(self.aedg)), tree)
            else:
                ok = False

            fail = 0 if ok else fail + 1

            if verbose and ok and len(self.nodes) % 500 == 0:
                s = self.stats()
                print(f"  N={s['n']:5d}  K12={s['pct_k12']:.1f}%")

        # Final proximity pass
        tree = cKDTree(self.nodes)
        self._prox(tree)
        return self

    # ── Analysis methods ───────────────────────────────────────────

    def stats(self):
        """Return basic coordination statistics."""
        deg = np.array([d for _, d in self.G.degree()])
        dd  = Counter(deg)
        n   = len(self.nodes)
        return dict(
            n        = n,
            k_mean   = float(deg.mean()) if n else 0,
            k_max    = int(deg.max())    if n else 0,
            k12      = dd.get(12, 0),
            pct_k12  = 100. * dd.get(12, 0) / n if n else 0,
        )

    def stats_full(self):
        """Return coordination + aspect ratio + isotropy."""
        pos = np.array(self.nodes)
        deg = np.array([d for _, d in self.G.degree()])
        dd  = Counter(deg)
        n   = len(self.nodes)
        zext  = pos[:, 2].max() - pos[:, 2].min() if n > 1 else 1
        xyext = max(pos[:, 0].max() - pos[:, 0].min(),
                    pos[:, 1].max() - pos[:, 1].min()) if n > 1 else 1
        asp = zext / xyext if xyext > 0 else 0
        C   = np.cov(pos.T) if n > 3 else np.eye(3)
        lam = sorted(np.linalg.eigvalsh(C))
        iso = lam[0] / lam[2] if lam[2] > 0 else 0
        return dict(n=n, pct_k12=100. * dd.get(12, 0) / n if n else 0,
                    asp=asp, iso=iso)

    def layer_analysis(self, tol=0.2):
        """
        Identify planar layers, compute inter-layer spacing and flatness.
        Returns dict with layer statistics; also prints a summary.
        """
        pos  = np.array(self.nodes)
        z    = pos[:, 2]
        n    = len(pos)
        visited = np.zeros(n, bool)
        layers  = []
        lids    = np.full(n, -1)

        for i in np.argsort(z):
            if visited[i]:
                continue
            mask = (np.abs(z - z[i]) < tol) & ~visited
            mem  = np.where(mask)[0]
            if len(mem) >= 3:
                layers.append(mem)
                lids[mem] = len(layers) - 1
            visited[mask] = True

        big     = [(i, m) for i, m in enumerate(layers) if len(m) >= 10]
        sigmas  = [float(np.std(z[m])) for _, m in big]
        z_means = sorted([float(np.mean(z[m])) for _, m in big])
        spacings = np.diff(z_means) if len(z_means) > 1 else np.array([])
        sizes   = [len(m) for _, m in big]

        # Inter-layer bonds
        il = np.zeros(n, int)
        for nd in range(n):
            if lids[nd] < 0:
                continue
            for nb in self.G.neighbors(nd):
                if lids[nb] >= 0 and lids[nb] != lids[nd]:
                    il[nd] += 1
        in_l = lids >= 0
        ilc  = il[in_l]
        n_flat = sum(1 for s in sigmas if s < 1e-10)

        print(f"  Layers total: {len(layers)}  "
              f"substantial (>=10 nodes): {len(big)}")
        print(f"  Exactly flat (sigma<1e-10 L): {n_flat}/{len(big)}")
        if len(spacings):
            print(f"  Spacing: {np.mean(spacings):.4f} +/- "
                  f"{np.std(spacings):.4f} L  "
                  f"(FCC ideal: {np.sqrt(2/3):.4f})")
        if len(ilc):
            print(f"  IL bonds/node: {ilc.mean():.1f}  "
                  f"({np.mean(ilc > 0)*100:.0f}% bonded)")

        return dict(n_layers=len(layers), n_big=len(big),
                    sigmas=sigmas, spacings=spacings.tolist(),
                    layer_sizes=sizes, n_flat=n_flat,
                    il_mean=float(ilc.mean()) if len(ilc) else 0,
                    il_pct=float(np.mean(ilc > 0)*100) if len(ilc) else 0)


# ── Paper sweep functions ──────────────────────────────────────────

def table1_scaling(seeds=30):
    """Table 1: Finite-size scaling  f_K12 = 1 - 6.8/N^{1/3}."""
    print("\n" + "="*60)
    print("TABLE 1: Finite-size scaling (p=e^{-3}, Rex=0.95, 30 seeds)")
    print("="*60)
    print(f"{'N':>6}  {'K_mean':>12}  {'f_K12 (%)':>13}  {'predicted':>10}")
    alpha = 6.8
    for N in [250, 500, 750, 1000]:
        k12s, kmeans = [], []
        for s in range(seeds):
            random.seed(s); np.random.seed(s)
            st = FCCNetworkGrowth(N, np.exp(-3)).run().stats()
            k12s.append(st['pct_k12'])
            kmeans.append(st['k_mean'])
        f  = np.mean(k12s);  ef  = np.std(k12s)
        km = np.mean(kmeans); ekm = np.std(kmeans)
        pred = max(0, 100*(1 - alpha / N**(1/3)))
        print(f"  {N:6d}  {km:.2f}+/-{ekm:.2f}  "
              f"{f:6.1f}+/-{ef:.1f}  {pred:8.1f}%")


def table2_lift_sweep(seeds=30, N=1000):
    """Table 2: Lift probability sweep -- volumetric yield vs p."""
    print("\n" + "="*60)
    print(f"TABLE 2: Lift probability sweep (N={N}, {seeds} seeds)")
    print("="*60)
    print(f"{'p (%)':>7}  {'f_K12':>11}  {'aspect':>8}  "
          f"{'Phi':>8}  {'iso':>5}")
    for p in [0.01, 0.03, np.exp(-3), 0.10, 0.15, 0.30, 0.50, 0.85]:
        k12s, asps, isos = [], [], []
        for s in range(seeds):
            random.seed(s); np.random.seed(s)
            st = FCCNetworkGrowth(N, p).run().stats_full()
            k12s.append(st['pct_k12'])
            asps.append(st['asp'])
            isos.append(st['iso'])
        f   = np.mean(k12s); a = np.mean(asps)
        phi = f / 100 * a
        marker = " <-- e^{-3}" if abs(p - np.exp(-3)) < 0.001 else ""
        print(f"  {p*100:5.1f}%  {f:6.1f}+/-{np.std(k12s):.1f}  "
              f"{a:.3f}+/-{np.std(asps):.3f}  "
              f"{phi:.4f}  {np.mean(isos):.2f}{marker}")


def table3_rex_sweep(seeds=10, N=500):
    """Table 3: Exclusion radius sweep -- phase transition at 1/sqrt(3)."""
    print("\n" + "="*60)
    print(f"TABLE 3: Rex sweep (N={N}, {seeds} seeds)")
    print("="*60)
    print(f"{'Rex/L':>7}  {'K_max':>6}  {'f_K12 (%)':>10}  Outcome")
    for rex in [0.50, 0.55, 0.57, 0.577, 0.58, 0.60, 0.65,
                0.70, 0.80, 0.90, 0.95, 0.99, 1.00, 1.02]:
        kmaxs, k12s = [], []
        for s in range(seeds):
            random.seed(s); np.random.seed(s)
            st = FCCNetworkGrowth(N, np.exp(-3),
                                  hard_shell=rex).run().stats()
            kmaxs.append(st['k_max'])
            k12s.append(st['pct_k12'])
        km = np.mean(kmaxs); f = np.mean(k12s)
        if rex < 0.577:  out = "over-bonded (K>12)"
        elif km < 10:    out = "growth arrested"
        elif km == 12:   out = "K=12 stable"
        else:            out = "partial"
        marker = " <-- 1/sqrt(3)" if abs(rex - 1/np.sqrt(3)) < 0.002 else ""
        print(f"  {rex:.3f}    {km:5.1f}  {f:7.1f}%   {out}{marker}")


def layer_run(N=3000, seed=42):
    """Full layer analysis for Figure 4 data."""
    print("\n" + "="*60)
    print(f"LAYER ANALYSIS (N={N}, seed={seed})")
    print("="*60)
    random.seed(seed); np.random.seed(seed)
    sim = FCCNetworkGrowth(N, np.exp(-3)).run(verbose=True)
    s   = sim.stats()
    print(f"Final: N={s['n']}  K_mean={s['k_mean']:.2f}  "
          f"K12={s['pct_k12']:.1f}%  K_max={s['k_max']}")
    sim.layer_analysis()


# ── Entry point ────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='FCC network growth: competitive planar/vertical model')
    parser.add_argument('--nodes',   type=int,   default=3000)
    parser.add_argument('--lift',    type=float, default=np.exp(-3))
    parser.add_argument('--seed',    type=int,   default=42)
    parser.add_argument('--analyze', action='store_true',
        help='Run all paper sweeps (Tables 1-3 + layer analysis)')
    args = parser.parse_args()

    if args.analyze:
        table1_scaling(seeds=30)
        table2_lift_sweep(seeds=30, N=1000)
        table3_rex_sweep(seeds=10, N=500)
        layer_run(N=3000, seed=42)
    else:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Growing N={args.nodes}, p={args.lift:.4f}, "
              f"seed={args.seed} ...")
        sim = FCCNetworkGrowth(args.nodes, args.lift).run(verbose=True)
        s   = sim.stats()
        print(f"N={s['n']}  K_mean={s['k_mean']:.2f}  "
              f"K12={s['pct_k12']:.1f}%  Kmax={s['k_max']}")
        sim.layer_analysis()

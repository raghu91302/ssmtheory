"""
SSM lattice simulation with bulk-interior diagnostics.

Implements the Selection-Stitch Model (SSM):
  - Stitch operator: 2D lateral hexagonal sheet expansion (K=6 ground state)
  - Lift operator: rare out-of-plane projection at h = sqrt(2/3) L (probability p_lift = e^-3)
  - Proximity bonding: nodes within r_b = 1.05 L automatically bond
  - Hard exclusion: nodes within r_ex = 0.95 L of an existing node are forbidden

Then computes both:
  (i) "all nodes" coordination statistics (current paper's reported numbers)
  (ii) BULK-INTERIOR statistics (nodes >= 2L from any free surface)
       — this is the diagnostic the counterargument requested

Bulk-interior is defined operationally:
  A node v is bulk-interior iff every node within radius 2L of v is itself
  a node of the lattice (i.e., v has no missing neighbors out to second-shell).
  Operationally: v is bulk-interior iff v has at least N_bulk_threshold neighbors
  within 2L. For an ideal FCC bulk, the count within 2L is K + (next shell) ~ 18.
"""

import numpy as np
import time
from scipy.spatial import cKDTree
from collections import Counter

# ---------- SSM parameters ----------
L = 1.0  # unit bond length
H_LIFT = np.sqrt(2.0 / 3.0) * L  # tetrahedral lift altitude
H_LATERAL = np.sqrt(3.0) / 2.0 * L  # equilateral triangle altitude (for stitch)
R_EX = 0.95 * L  # exclusion radius
R_BOND = 1.05 * L  # proximity bond radius
P_LIFT = np.exp(-3.0)  # ~ 0.0498

# Bulk diagnostic: count neighbors within this range — used to mark bulk-interior nodes
R_BULK_INTERIOR = 2.0 * L
# Threshold: a node is "bulk-interior" iff all three FCC coordination shells around it
# are populated. An ideal FCC interior node has 12 + 6 + 24 = 42 neighbors at distances
# L, sqrt(2)*L = 1.414L, sqrt(3)*L = 1.732L — all within 2L. Setting the threshold to
# the full count of 42 is a strict geometric criterion: bulk-interior means "all three
# coordination shells fully populated", not a tunable parameter.
N_BULK_THRESHOLD = 42


def grow_lattice(N_target, seed, p_lift=P_LIFT, verbose=False):
    """
    Grow an SSM lattice to N_target nodes.

    Algorithm:
      1. Seed: place 3 nodes forming an equilateral triangle
      2. While count < N_target:
         - With probability (1 - p_lift): STITCH — pick a random existing edge,
           place a new node at the equilateral apex (in the local 2D plane).
           Reject if exclusion violated; accept if proximity bonds form.
         - With probability p_lift: LIFT — pick a random existing triangle face,
           place a new node above the centroid at height h = sqrt(2/3) L.
           Reject if exclusion violated.
      3. After every placement, form proximity bonds with all existing nodes
         within R_BOND (this links adjacent stacked sheets — the FCC mechanism).

    Returns:
      positions: (N, 3) array
      adjacency: list of sets (adjacency[i] = set of indices bonded to i)
      degrees: (N,) array
    """
    rng = np.random.default_rng(seed)

    # Seed: equilateral triangle in xy-plane centered at origin
    s = L
    positions = [
        np.array([0.0, 0.0, 0.0]),
        np.array([s, 0.0, 0.0]),
        np.array([s / 2.0, s * np.sqrt(3.0) / 2.0, 0.0]),
    ]
    # Initial adjacency: triangle
    adjacency = [{1, 2}, {0, 2}, {0, 1}]

    edges = [(0, 1), (1, 2), (0, 2)]  # for stitch
    triangles = [(0, 1, 2)]  # for lift

    t_start = time.time()
    n_attempt = 0
    n_lift_done = 0
    n_stitch_done = 0
    max_attempts = 200 * N_target  # safety

    while len(positions) < N_target and n_attempt < max_attempts:
        n_attempt += 1

        # Decide operator: lift with probability p_lift, otherwise stitch
        do_lift = rng.random() < p_lift

        candidate = None
        if do_lift and triangles:
            # Pick a random triangle, place new node above centroid
            i, j, k = triangles[rng.integers(len(triangles))]
            p_i, p_j, p_k = positions[i], positions[j], positions[k]
            centroid = (p_i + p_j + p_k) / 3.0
            # Compute the triangle normal
            v1 = p_j - p_i
            v2 = p_k - p_i
            n = np.cross(v1, v2)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n_hat = n / n_norm
            # Two possible apexes: above and below
            sign = 1.0 if rng.random() < 0.5 else -1.0
            candidate = centroid + sign * H_LIFT * n_hat
        elif edges:
            # Stitch: pick a random edge, place at equilateral apex in 2D plane
            # The plane is defined by the edge and a "normal" direction in the local
            # 2D growth plane. We approximate by choosing the apex perpendicular to
            # the edge in the xy-plane (for the dominant 2D growth).
            edge_idx = rng.integers(len(edges))
            i, j = edges[edge_idx]
            p_i, p_j = positions[i], positions[j]
            mid = (p_i + p_j) / 2.0
            edge_vec = p_j - p_i
            edge_len = np.linalg.norm(edge_vec)
            if edge_len < 1e-12:
                continue
            edge_hat = edge_vec / edge_len
            # Find a perpendicular direction in the local growth plane.
            # Look at common neighbors of i and j to determine local plane normal:
            common = adjacency[i] & adjacency[j]
            if common:
                # Plane defined by edge + one common neighbor
                k = next(iter(common))
                p_k = positions[k]
                # Plane normal is edge × (p_k - p_i)
                plane_n = np.cross(edge_vec, p_k - p_i)
                pn_norm = np.linalg.norm(plane_n)
                if pn_norm < 1e-12:
                    continue
                plane_n_hat = plane_n / pn_norm
                # Perpendicular direction in plane: edge_hat × plane_n_hat
                perp = np.cross(plane_n_hat, edge_hat)
                # Choose the side away from k
                p_k_dir = p_k - mid
                p_k_perp = p_k_dir - np.dot(p_k_dir, edge_hat) * edge_hat
                p_k_perp_n = np.linalg.norm(p_k_perp)
                if p_k_perp_n < 1e-12:
                    continue
                if np.dot(perp, p_k_perp / p_k_perp_n) > 0:
                    perp = -perp
            else:
                # Free edge: just pick xy-perpendicular
                if abs(edge_hat[2]) < 0.9:
                    perp = np.cross(edge_hat, np.array([0.0, 0.0, 1.0]))
                else:
                    perp = np.cross(edge_hat, np.array([1.0, 0.0, 0.0]))
                perp = perp / np.linalg.norm(perp)
            candidate = mid + H_LATERAL * perp

        if candidate is None:
            continue

        # Hard-shell exclusion check
        pos_arr = np.array(positions)
        dists = np.linalg.norm(pos_arr - candidate, axis=1)
        if np.any(dists < R_EX):
            continue

        # Accept: add node and form proximity bonds with everything within R_BOND
        new_idx = len(positions)
        positions.append(candidate)
        new_adj = set(np.where(dists < R_BOND)[0].tolist())
        adjacency.append(new_adj)
        for neighbor in new_adj:
            adjacency[neighbor].add(new_idx)
            edges.append((min(neighbor, new_idx), max(neighbor, new_idx)))

        # Update triangle list: any pair of new_adj members that are themselves
        # bonded forms a new triangle with new_idx
        new_adj_list = list(new_adj)
        for a_idx in range(len(new_adj_list)):
            for b_idx in range(a_idx + 1, len(new_adj_list)):
                a, b = new_adj_list[a_idx], new_adj_list[b_idx]
                if b in adjacency[a]:
                    triangles.append(tuple(sorted([new_idx, a, b])))

        if do_lift:
            n_lift_done += 1
        else:
            n_stitch_done += 1

    if verbose:
        elapsed = time.time() - t_start
        print(f"  Grew to N={len(positions)} in {elapsed:.1f}s "
              f"(lifts={n_lift_done}, stitches={n_stitch_done}, "
              f"attempts={n_attempt})")

    pos_array = np.array(positions)
    degrees = np.array([len(adj) for adj in adjacency])
    return pos_array, adjacency, degrees


def compute_bulk_interior_mask(positions, r_bulk=R_BULK_INTERIOR,
                                threshold=N_BULK_THRESHOLD):
    """
    Returns boolean mask: True for bulk-interior nodes.
    A node is bulk-interior iff at least `threshold` other nodes lie within
    radius `r_bulk` of it.
    """
    if len(positions) < threshold + 1:
        return np.zeros(len(positions), dtype=bool)
    tree = cKDTree(positions)
    counts = tree.query_ball_point(positions, r_bulk, return_length=True)
    counts = counts - 1  # subtract self
    return counts >= threshold


def analyze_run(positions, adjacency, degrees):
    """Compute both all-node and bulk-interior coordination statistics."""
    N = len(positions)
    bulk_mask = compute_bulk_interior_mask(positions)
    n_bulk = int(bulk_mask.sum())

    # All-node statistics
    all_mean_K = float(degrees.mean())
    all_K12_frac = float((degrees == 12).mean())
    all_modal_K = int(Counter(degrees.tolist()).most_common(1)[0][0])
    all_std_K = float(degrees.std())

    # Bulk-interior statistics
    if n_bulk > 0:
        bulk_degs = degrees[bulk_mask]
        bulk_mean_K = float(bulk_degs.mean())
        bulk_K12_frac = float((bulk_degs == 12).mean())
        bulk_modal_K = int(Counter(bulk_degs.tolist()).most_common(1)[0][0])
        bulk_std_K = float(bulk_degs.std())
    else:
        bulk_mean_K = bulk_K12_frac = bulk_std_K = float('nan')
        bulk_modal_K = -1

    return dict(N=N, n_bulk=n_bulk,
                all_mean_K=all_mean_K, all_K12_frac=all_K12_frac,
                all_modal_K=all_modal_K, all_std_K=all_std_K,
                bulk_mean_K=bulk_mean_K, bulk_K12_frac=bulk_K12_frac,
                bulk_modal_K=bulk_modal_K, bulk_std_K=bulk_std_K)


def run_size(N, n_seeds=5, verbose=True):
    """Run n_seeds independent simulations at size N. Return aggregated stats."""
    if verbose:
        print(f"\n=== N = {N}, {n_seeds} seeds ===")
    results = []
    for seed in range(n_seeds):
        positions, adjacency, degrees = grow_lattice(N, seed=seed, verbose=verbose)
        stats = analyze_run(positions, adjacency, degrees)
        results.append(stats)
        if verbose:
            print(f"  seed {seed}: bulk_n={stats['n_bulk']:4d}  "
                  f"all_meanK={stats['all_mean_K']:.2f}  "
                  f"all_K12%={100*stats['all_K12_frac']:5.1f}  "
                  f"bulk_meanK={stats['bulk_mean_K']:.2f}  "
                  f"bulk_K12%={100*stats['bulk_K12_frac']:5.1f}  "
                  f"bulk_modal={stats['bulk_modal_K']}  "
                  f"bulk_sigma={stats['bulk_std_K']:.2f}")
    return results


def aggregate(results):
    """Mean ± std across seeds."""
    keys = ['n_bulk', 'all_mean_K', 'all_K12_frac', 'all_std_K',
            'bulk_mean_K', 'bulk_K12_frac', 'bulk_std_K']
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in results if not np.isnan(r[k])])
        if len(vals) > 0:
            out[k + '_mean'] = float(vals.mean())
            out[k + '_std'] = float(vals.std())
        else:
            out[k + '_mean'] = float('nan')
            out[k + '_std'] = float('nan')
    # Modal: take the most-common modal-K across seeds
    bulk_modals = [r['bulk_modal_K'] for r in results if r['bulk_modal_K'] > 0]
    out['bulk_modal_K'] = int(Counter(bulk_modals).most_common(1)[0][0]) if bulk_modals else -1
    return out


if __name__ == '__main__':
    import json
    sizes = [250, 500, 750, 1000]
    n_seeds = 30  # match the 30-seed protocol used for the all-node columns of Table 2
    summary = {}
    for N in sizes:
        results = run_size(N, n_seeds=n_seeds, verbose=True)
        summary[N] = {'aggregate': aggregate(results), 'per_seed': results}
        agg = summary[N]['aggregate']
        print(f"\n  AGGREGATE N={N}:  "
              f"all_K12 = {100*agg['all_K12_frac_mean']:.1f} ± {100*agg['all_K12_frac_std']:.1f}%   "
              f"bulk_K12 = {100*agg['bulk_K12_frac_mean']:.1f} ± {100*agg['bulk_K12_frac_std']:.1f}%   "
              f"bulk_modal = {agg['bulk_modal_K']}   "
              f"bulk_sigma = {agg['bulk_std_K_mean']:.2f} ± {agg['bulk_std_K_std']:.2f}")

    with open('/tmp/bulk_analysis.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nSaved: /tmp/bulk_analysis.json")

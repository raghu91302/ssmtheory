"""Smarter surgery primitive finder for large L.

Strategy: don't exhaustively search all pairs. Instead:
1. For each valid set v (triangle combination commuting with all X-stabs):
   - Compute resulting Op = v @ B (weight in data qubits)
   - Skip if Op = 0 (trivial)
   - Skip if Op is in row span of HZ (stabilizer, not logical)
2. Among non-trivial logicals, sort by weight
3. Take the minimum-weight one

This finds the SAME primitive as find_surgery_primitive but without the O(n^2) pair search.
"""
import sys
import time
import numpy as np

from sheet_code_fcc_lattice import (build_fcc_lattice, vertex_Z_stabilizers,
                                     oct_void_X_stabilizers, fcc_triangles)
from sheet_code_gf2 import gf2_kernel, gf2_rank


def find_surgery_primitive_fast(L, verbose=True):
    """Find a minimum-weight cross-sheet Z-logical via single valid sets.
    
    Returns (triangle_indices, op, B).
    """
    t0 = time.time()
    vertices, vidx, edges, edge_list, edge_to_idx, edge_to_sheet = build_fcc_lattice(L)
    HX, _ = oct_void_X_stabilizers(L)
    HZ, _ = vertex_Z_stabilizers(L)
    triangles, B = fcc_triangles(L)
    
    if verbose:
        print(f"  Setup: {time.time()-t0:.1f}s")
    
    t0 = time.time()
    M_commute = (HX @ B.T) % 2
    valid_sets = gf2_kernel(M_commute)
    if verbose:
        print(f"  gf2_kernel: {time.time()-t0:.1f}s, {valid_sets.shape[0]} valid triangle sets")
    
    # For each valid set, compute Op and its weight, filter non-trivial logicals
    t0 = time.time()
    rank_HZ = gf2_rank(HZ)
    if verbose:
        print(f"  rank(HZ): {time.time()-t0:.1f}s = {rank_HZ}")
    
    t0 = time.time()
    # Compute all Op weights in one batch
    ops = (valid_sets @ B) % 2  # shape (n_valid_sets, n_data)
    weights = ops.sum(axis=1)
    if verbose:
        print(f"  Batch Op computation: {time.time()-t0:.1f}s")
        nontrivial = (weights > 0).sum()
        print(f"  Non-trivial Ops: {nontrivial}")
    
    # Filter out zero-weight (trivial) sets
    nz_mask = weights > 0
    ops_nz = ops[nz_mask]
    weights_nz = weights[nz_mask]
    valid_sets_nz = valid_sets[nz_mask]
    
    # Sort by weight (ascending)
    sort_idx = np.argsort(weights_nz)
    
    # For each candidate from smallest weight up, check if it's a non-stabilizer logical
    t0 = time.time()
    best = None
    for i, idx in enumerate(sort_idx):
        op_candidate = ops_nz[idx]
        # Check if it's a non-trivial logical (not in row span of HZ)
        test = np.vstack([HZ, op_candidate.reshape(1, -1)])
        if gf2_rank(test) > rank_HZ:
            # Non-trivial logical!
            best = (valid_sets_nz[idx], op_candidate)
            if verbose:
                print(f"  Found min-weight Op: weight={int(weights_nz[idx])} after checking {i+1} candidates ({time.time()-t0:.1f}s)")
            break
    
    if best is None:
        raise RuntimeError("No non-trivial logical found")
    
    triangle_set, op = best
    triangle_indices = np.where(triangle_set == 1)[0]
    return triangle_indices, op, B


if __name__ == '__main__':
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    print(f"\n=== Find surgery primitive at L = {L} ===")
    tris, op, B = find_surgery_primitive_fast(L)
    print(f"\n  Triangles in primitive: {len(tris)} (expected ≤ L = {L})")
    print(f"  Op weight: {int(op.sum())} (expected 2L = {2*L})")
    
    # Save for later use
    import json
    cache = {
        'L': L,
        'triangle_indices': [int(t) for t in tris],
        'op': [int(o) for o in op],
    }
    with open(f'/home/claude/surgery_primitive_L{L}.json', 'w') as f:
        json.dump(cache, f)
    print(f"  Saved to /home/claude/surgery_primitive_L{L}.json")

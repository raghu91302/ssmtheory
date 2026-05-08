"""
fcc_code.py: Construction of the [[192, 130, 3]] FCC base code at L = 4
            and the flag stabilizers at tetrahedral voids.

This is the foundational module: it builds the FCC lattice, enumerates
edges and stabilizers, computes the parity-check matrices, and produces
the flag-augmented stabilizer set used elsewhere in the code package.

All other simulation scripts import from this file. The functions here
are deterministic and produce identical output across runs.

Usage:
    from fcc_code import build_base_code, build_flag_stabs, ...
"""

from __future__ import annotations
import numpy as np
from typing import Optional


def build_fcc_lattice(L: int):
    """Enumerate FCC lattice nodes, edges, octahedral voids, and the
    sheet assignment of each edge.

    Returns:
      nodes: dict mapping (x, y, z) -> node_index. Vertices have x+y+z
        even and lie in [0, L)^3 with periodic boundaries.
      edges: dict mapping (v_lo, v_hi) -> edge_index, where each edge
        connects two FCC vertices via a sheet displacement.
      oct_voids: list of (x, y, z) tuples at integer positions with odd
        coordinate sum.
      edge_to_sheet: dict mapping edge_index -> 'xy', 'xz', or 'yz'.
    """
    nodes = {}
    for x in range(L):
        for y in range(L):
            for z in range(L):
                if (x + y + z) % 2 == 0:
                    nodes[(x, y, z)] = len(nodes)

    sheet_disps = {
        'xy': [(1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0)],
        'xz': [(1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1)],
        'yz': [(0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)],
    }
    disp_to_sheet = {d: s for s, ds in sheet_disps.items() for d in ds}

    edges = {}
    edge_to_sheet = {}
    for (x, y, z), v in nodes.items():
        for dx, dy, dz in disp_to_sheet:
            nx, ny, nz = (x + dx) % L, (y + dy) % L, (z + dz) % L
            if (nx, ny, nz) in nodes:
                w = nodes[(nx, ny, nz)]
                key = (min(v, w), max(v, w))
                if key not in edges:
                    edges[key] = len(edges)
                    edge_to_sheet[edges[key]] = disp_to_sheet[(dx, dy, dz)]

    oct_voids = [(x, y, z) for x in range(L) for y in range(L) for z in range(L)
                 if (x + y + z) % 2 == 1]
    return nodes, edges, oct_voids, edge_to_sheet


def build_base_code(L: int):
    """Construct the [[L^3 (or so), k, 3]] base FCC code stabilizers.

    Returns:
      n_data: number of physical data qubits (192 at L = 4).
      Z_stabs: list of weight-12 Z-stabilizer supports (one per FCC vertex).
      X_stabs: list of weight-12 X-stabilizer supports (one per
        octahedral void).
    """
    nodes, edges, oct_voids, _ = build_fcc_lattice(L)

    edge_disps = [
        (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
        (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
        (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
    ]

    # Z-stabilizers: 12 incident edges per FCC vertex
    Z_stabs = []
    for (x, y, z), v in nodes.items():
        stab = []
        for dx, dy, dz in edge_disps:
            nx, ny, nz = (x + dx) % L, (y + dy) % L, (z + dz) % L
            if (nx, ny, nz) in nodes:
                w = nodes[(nx, ny, nz)]
                key = (min(v, w), max(v, w))
                stab.append(edges[key])
        Z_stabs.append(sorted(set(stab)))

    # X-stabilizers: 12 edges of the octahedron surrounding each void
    X_stabs = []
    for (vx, vy, vz) in oct_voids:
        face_neighbors = []
        for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                           (0, 0, 1), (0, 0, -1)]:
            nx, ny, nz = (vx + dx) % L, (vy + dy) % L, (vz + dz) % L
            if (nx, ny, nz) in nodes:
                face_neighbors.append((nx, ny, nz))

        stab = []
        for i in range(len(face_neighbors)):
            for j in range(i + 1, len(face_neighbors)):
                a, b = face_neighbors[i], face_neighbors[j]
                diff = ((b[0] - a[0]) % L, (b[1] - a[1]) % L,
                        (b[2] - a[2]) % L)

                def signed(d):
                    return d - L if d > L // 2 else d

                ds = tuple(signed(d) for d in diff)
                if sum(abs(c) for c in ds) == 2 and 0 in ds:
                    va, vb = nodes[a], nodes[b]
                    key = (min(va, vb), max(va, vb))
                    if key in edges:
                        stab.append(edges[key])

        X_stabs.append(sorted(set(stab)))

    return len(edges), Z_stabs, X_stabs


def build_flag_stabs(L: int, all_corners: bool = True):
    """Construct flag stabilizers at tetrahedral voids.

    Each tetrahedral void is surrounded by 4 FCC vertices forming a
    regular tetrahedron with 6 edges. A flag from a designated corner
    is the Z-parity on the 3 edges from that corner to the other 3.

    Args:
      L: lattice size (must be even).
      all_corners: if True, produce flags from all 4 corners (256 at L=4);
        if False, only from one designated corner (64 at L=4).

    Returns:
      list of flag stabilizer supports, each a list of 3 edge indices.
    """
    nodes, edges, _, _ = build_fcc_lattice(L)
    flags = []
    seen_voids = set()

    for a in range(L):
        for b in range(L):
            for c in range(L):
                if (a + b + c) % 2 == 0:
                    surrounding_raw = [(a, b, c), (a + 1, b + 1, c),
                                       (a + 1, b, c + 1), (a, b + 1, c + 1)]
                else:
                    surrounding_raw = [(a + 1, b, c), (a, b + 1, c),
                                       (a, b, c + 1), (a + 1, b + 1, c + 1)]

                surrounding = [tuple(p % L for p in v) for v in surrounding_raw]
                if not all(v in nodes for v in surrounding):
                    continue

                vset = frozenset(surrounding)
                if vset in seen_voids:
                    continue
                seen_voids.add(vset)

                corners = surrounding if all_corners else [min(surrounding)]
                for corner in corners:
                    others = [v for v in surrounding if v != corner]
                    flag_edges = []
                    for other in others:
                        va, vb = nodes[corner], nodes[other]
                        key = (min(va, vb), max(va, vb))
                        if key in edges:
                            flag_edges.append(edges[key])
                    if len(flag_edges) == 3:
                        flags.append(sorted(flag_edges))
    return flags


# ----- GF(2) linear algebra utilities -----

def to_matrix(stabs, n):
    H = np.zeros((len(stabs), n), dtype=np.int8)
    for i, stab in enumerate(stabs):
        for q in stab:
            H[i, q] = 1
    return H


def gf2_rref(M):
    """Reduced row-echelon form of M over GF(2).

    Returns (RREF_matrix, pivot_columns, rank).
    """
    M = M.copy().astype(np.int8) % 2
    rows, cols = M.shape
    pivots = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        piv = None
        for i in range(r, rows):
            if M[i, c] == 1:
                piv = i
                break
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(rows):
            if i != r and M[i, c] == 1:
                M[i] = (M[i] + M[r]) % 2
        pivots.append(c)
        r += 1
    return M, pivots, r


def gf2_kernel(M):
    """Basis of the kernel of M over GF(2)."""
    rref, pivots, rank = gf2_rref(M)
    rows, cols = M.shape
    free = [c for c in range(cols) if c not in pivots]
    basis = []
    for fc in free:
        v = np.zeros(cols, dtype=np.int8)
        v[fc] = 1
        for i, pc in enumerate(pivots):
            if rref[i, fc] == 1:
                v[pc] = 1
        basis.append(v)
    return np.array(basis) if basis else np.zeros((0, cols), dtype=np.int8)


def find_logical_Z_basis(HX, HZ):
    """Return a (k, n) matrix whose rows form a basis of the logical Z group,
    i.e. ker(HX) / rowspan(HZ)."""
    ker = gf2_kernel(HX)
    HZ_rref, HZ_pivots, HZ_rank = gf2_rref(HZ)
    HZ_basis = HZ_rref[:HZ_rank].astype(np.int8)
    pivots = list(HZ_pivots)

    reduced = []
    for v in ker:
        u = v.copy().astype(np.int8)
        for i, pc in enumerate(pivots):
            if u[pc]:
                u ^= HZ_basis[i]
        reduced.append(u)
    reduced = np.array(reduced, dtype=np.int8)

    rref, _, rank = gf2_rref(reduced)
    return rref[:rank].astype(np.int8)


# ----- Convenience entry point -----

def code_summary(L: int):
    """Print summary of the code parameters at lattice size L."""
    n, Z_stabs, X_stabs = build_base_code(L)
    HX = to_matrix(X_stabs, n)
    HZ = to_matrix(Z_stabs, n)
    css_ok = np.all((HX @ HZ.T) % 2 == 0)
    _, _, rk_z = gf2_rref(HZ)
    _, _, rk_x = gf2_rref(HX)
    k = n - rk_z - rk_x

    flag_stabs_full = build_flag_stabs(L, all_corners=True)
    HZ_aug = np.vstack([HZ, to_matrix(flag_stabs_full, n)])
    _, _, rk_aug = gf2_rref(HZ_aug)
    k_aug = n - rk_aug - rk_x

    print(f"--- FCC code at L = {L} ---")
    print(f"  base:           [[{n}, {k}, ?]]")
    print(f"  CSS valid:      {css_ok}")
    print(f"  rank(HZ) = {rk_z}, rank(HX) = {rk_x}")
    print(f"  Z-stab weights: all = {len(Z_stabs[0])}")
    print(f"  X-stab weights: all = {len(X_stabs[0])}")
    print(f"  rate k/n:       {k}/{n} = {100*k/n:.1f}%")
    print(f"  flag-aug:       [[{n}, {k_aug}, ?]]")
    print(f"  flags added:    {len(flag_stabs_full)}, "
          f"rank gained: {rk_aug - rk_z}")


if __name__ == "__main__":
    code_summary(L=4)


# ============================================================
# Sheet-code construction (one triad sheet of the FCC lattice)
# ============================================================

def build_sheet_code_xy(L: int = 4):
    """Build the [[L^3, 2L, L]] FCC sheet code on the xy-sheet at lattice
    size L (even). Returns (n_data, Z_stabs, X_stabs).
    
    Z-stabilizers: 4 xy-edges incident to each FCC vertex (vertex check).
    X-stabilizers: 4 xy-edges of the octahedron surrounding each octahedral
                   void (octahedral-void check).
    """
    nodes, edges, oct_voids, edge_to_sheet = build_fcc_lattice(L)
    xy_edges = sorted([e for e, sheet in edge_to_sheet.items() if sheet == 'xy'])
    edge_remap = {old_id: new_id for new_id, old_id in enumerate(xy_edges)}
    n_data = len(xy_edges)

    xy_disps = [(1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0)]

    Z_stabs = []
    for (x, y, z), v in nodes.items():
        stab = []
        for dx, dy, dz in xy_disps:
            nx, ny, nz = (x + dx) % L, (y + dy) % L, (z + dz) % L
            if (nx, ny, nz) in nodes:
                w = nodes[(nx, ny, nz)]
                key = (min(v, w), max(v, w))
                if key in edges and edges[key] in edge_remap:
                    stab.append(edge_remap[edges[key]])
        if len(set(stab)) == 4:
            Z_stabs.append(sorted(set(stab)))

    X_stabs = []
    for (vx, vy, vz) in oct_voids:
        face_neighbors = []
        for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                           (0, 0, 1), (0, 0, -1)]:
            nx, ny, nz = (vx + dx) % L, (vy + dy) % L, (vz + dz) % L
            if (nx, ny, nz) in nodes:
                face_neighbors.append((nx, ny, nz))

        stab = []
        for i in range(len(face_neighbors)):
            for j in range(i + 1, len(face_neighbors)):
                a, b = face_neighbors[i], face_neighbors[j]
                diff = ((b[0] - a[0]) % L, (b[1] - a[1]) % L,
                        (b[2] - a[2]) % L)

                def signed(d):
                    return d - L if d > L // 2 else d

                ds = tuple(signed(d) for d in diff)
                if sum(abs(c) for c in ds) == 2 and 0 in ds and ds[2] == 0:
                    va, vb = nodes[a], nodes[b]
                    key = (min(va, vb), max(va, vb))
                    if key in edges and edges[key] in edge_remap:
                        stab.append(edge_remap[edges[key]])
        if len(set(stab)) == 4:
            X_stabs.append(sorted(set(stab)))
    return n_data, Z_stabs, X_stabs


def to_matrix(stabs, n):
    """Convert a list of stabilizer supports to a binary parity-check matrix."""
    H = np.zeros((len(stabs), n), dtype=np.int8)
    for i, stab in enumerate(stabs):
        for q in stab:
            H[i, q] = 1
    return H

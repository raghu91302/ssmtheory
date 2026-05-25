#!/usr/bin/env python3
"""
verify_shared_edge.py
=====================

Computational verification of Section 4 of:

    R. Kulkarni, "Deriving the Dark Matter Annihilation Channel
    from Metric-Wall Confinement on the FCC Vacuum Lattice"
    (Selection-Stitch Model, May 2026).

The script enumerates oct voids on a 4x4x4 FCC supercell and confirms
that every nearest-neighbor pair of oct voids shares exactly an
octahedron edge: two bounding vertices joined by one bond of length L.

Run with:  python3 verify_shared_edge.py
Depends on: numpy  (standard library otherwise).

License: CC-BY-NC-ND 4.0 (matches the rest of the SSMTheory papers).
"""

from __future__ import annotations
import numpy as np
from collections import Counter
from itertools import combinations

# Cubic lattice constant a, FCC nearest-neighbor distance L = a / sqrt(2).
a = 2.0
L = a / np.sqrt(2)
TOL = 1e-4


def build_fcc_atoms(n_cells: int = 4) -> list[np.ndarray]:
    """Build FCC atom positions on an n_cells x n_cells x n_cells supercell.

    FCC = cube corners + three face centers per unit cell.
    """
    atoms = set()
    for ix in range(n_cells):
        for iy in range(n_cells):
            for iz in range(n_cells):
                # cube corner
                atoms.add((ix * a, iy * a, iz * a))
                # face centers on the three coordinate planes
                atoms.add((ix * a + a / 2, iy * a + a / 2, iz * a))
                atoms.add((ix * a + a / 2, iy * a, iz * a + a / 2))
                atoms.add((ix * a, iy * a + a / 2, iz * a + a / 2))
    return [np.array(p) for p in atoms]


def build_oct_centers(n_cells: int = 4) -> list[np.ndarray]:
    """Build candidate oct-void center positions.

    Oct voids of FCC sit at the body center (a/2, a/2, a/2) and at the
    twelve edge midpoints of each cubic unit cell.
    """
    centers = set()
    for ix in range(n_cells):
        for iy in range(n_cells):
            for iz in range(n_cells):
                centers.add((ix * a + a / 2, iy * a + a / 2, iz * a + a / 2))
                centers.add((ix * a + a / 2, iy * a, iz * a))
                centers.add((ix * a, iy * a + a / 2, iz * a))
                centers.add((ix * a, iy * a, iz * a + a / 2))
    return [np.array(p) for p in centers]


def bounding_vertices(
    center: np.ndarray,
    atoms: list[np.ndarray],
    target: float = None,
    tol: float = 1e-6,
) -> frozenset:
    """Return the FCC atoms at distance `target` from `center`.

    For an oct void, target = a / 2 = L / sqrt(2): the six bounding
    vertices sit at this centroid-to-vertex distance.
    """
    if target is None:
        target = a / 2
    hits = []
    for atom in atoms:
        d = np.linalg.norm(atom - center)
        if abs(d - target) < tol:
            hits.append(tuple(np.round(atom, 6)))
    return frozenset(hits)


def collect_void_geometry(
    atoms: list[np.ndarray],
    candidate_centers: list[np.ndarray],
) -> dict:
    """Build a dict mapping oct-void center -> frozenset of 6 bounding vertices.

    Voids whose bounding vertices are not all inside the chunk (i.e., return
    fewer than 6 hits) are dropped. This restricts the analysis to fully-
    interior oct voids of the supercell.
    """
    voids = {}
    for oc in candidate_centers:
        bv = bounding_vertices(oc, atoms)
        if len(bv) == 6:
            voids[tuple(np.round(oc, 6))] = bv
    return voids


def pair_separation_spectrum(
    void_vertices: dict,
    max_distances: int = 4,
) -> dict[float, list[int]]:
    """For each discrete separation between oct-void centers, return the list
    of shared-bounding-vertex counts across all pairs at that separation."""
    centers = list(void_vertices.keys())
    arrs = [np.array(c) for c in centers]
    n = len(arrs)

    # First, gather all unique separations.
    seps = set()
    for i, j in combinations(range(n), 2):
        d = np.linalg.norm(arrs[i] - arrs[j])
        if d > 1e-6:
            seps.add(round(d, 4))
    seps = sorted(seps)[:max_distances]

    # Then bin pairs by separation and record shared-vertex counts.
    out = {}
    for d_bin in seps:
        shared_counts = []
        for i, j in combinations(range(n), 2):
            d = np.linalg.norm(arrs[i] - arrs[j])
            if abs(d - d_bin) < TOL:
                shared = void_vertices[centers[i]] & void_vertices[centers[j]]
                shared_counts.append(len(shared))
        out[d_bin] = shared_counts
    return out


def check_shared_vertex_distances(void_vertices: dict) -> Counter:
    """For nearest-neighbor pairs (separation L) with 2 shared bounding
    vertices: what is the mutual distance between the 2 shared vertices?

    Returns a Counter on the rounded distances. Expected result:
    every pair has its 2 shared vertices at distance L (octahedron edge),
    not L*sqrt(2) (octahedron antipodal axis).
    """
    centers = list(void_vertices.keys())
    arrs = [np.array(c) for c in centers]
    distances = []
    for i, j in combinations(range(len(arrs)), 2):
        d_centers = np.linalg.norm(arrs[i] - arrs[j])
        if abs(d_centers - L) > TOL:
            continue
        shared = void_vertices[centers[i]] & void_vertices[centers[j]]
        if len(shared) != 2:
            continue
        v1, v2 = (np.array(v) for v in shared)
        distances.append(round(float(np.linalg.norm(v1 - v2)), 4))
    return Counter(distances)


def main() -> None:
    print("FCC oct-void shared-edge verification")
    print("=" * 48)
    print(f"Cubic lattice constant a = {a}")
    print(f"NN distance L = a/sqrt(2) = {L:.4f}")
    print()

    atoms = build_fcc_atoms(n_cells=4)
    candidates = build_oct_centers(n_cells=4)
    voids = collect_void_geometry(atoms, candidates)

    print(f"FCC atoms in 4^3 supercell:       {len(atoms)}")
    print(f"Candidate oct-void centers:        {len(candidates)}")
    print(f"Fully-interior oct voids:          {len(voids)}")
    print()

    print(f"{'d / L':>10s}   {'pairs':>6s}   shared-vertex distribution")
    print("-" * 60)
    spectrum = pair_separation_spectrum(voids, max_distances=4)
    for d_abs, counts in spectrum.items():
        d_rel = d_abs / L
        dist = dict(Counter(counts))
        print(f"{d_rel:>10.4f}   {len(counts):>6d}   {dist}")
    print()

    # Now confirm the 2 shared vertices are octahedron-edge-adjacent,
    # not antipodal (which would be at distance L * sqrt(2)).
    edge_check = check_shared_vertex_distances(voids)
    print("For nearest-neighbor oct-void pairs (d = L) with 2 shared")
    print("vertices: the mutual distance between those 2 vertices.")
    print(f"  Distribution: {dict(edge_check)}")
    print(f"  L            = {L:.4f}      (octahedron edge)")
    print(f"  L * sqrt(2)  = {L * np.sqrt(2):.4f}      (octahedron antipodal)")
    print()

    # Compact assertions for automated checks.
    nn_counts = spectrum[round(L, 4)]
    assert Counter(nn_counts) == {2: 450}, \
        f"Expected 450 nearest-neighbor pairs all sharing 2 vertices; got {Counter(nn_counts)}"
    assert set(edge_check.keys()) == {round(L, 4)}, \
        f"Expected all 2 shared vertices at distance L; got {edge_check}"

    print("All assertions passed. The shared edge is verified:")
    print("  450 of 450 nearest-neighbor oct-void pairs share exactly")
    print("  one octahedron edge (2 vertices + 1 bond of length L).")


if __name__ == "__main__":
    main()

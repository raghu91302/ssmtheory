"""
Which cages can host the residual defect after two octahedral-void defects
merge?

Two nearest-neighbor octahedral voids of the FCC lattice share exactly two
bounding vertices, separated by L, that is, one octahedron edge. The union
of their bounding vertices is a ten-vertex interface. Enumerating the cages
contained in that interface returns four: two regular tetrahedra and two
regular octahedra.

This script shows that the four are not equivalent. Writing each cage as

    (shared vertices, vertices exclusive to O1, vertices exclusive to O2)

the two tetrahedra have composition (2, 1, 1) and therefore draw vertices
from both parents, while the two octahedra have composition (2, 4, 0) and
(2, 0, 4): they are the parent voids themselves. A cage that contains no
vertex exclusive to one of the parents is not a product of the merger.

The check runs over every nearest-neighbor pair of octahedral voids in the
supercell, not one representative pair.

Requires numpy only. Runs in about twenty seconds.
"""
from itertools import combinations
from collections import Counter
import numpy as np

# FCC nodes are the integer points with x + y + z even, so the
# nearest-neighbor distance is L = sqrt(2). Octahedral voids sit at the
# odd-parity integer points.
L_SQ = 2
AXES = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def dist_sq(a, b):
    return sum((a[i] - b[i]) ** 2 for i in range(3))


def bounding_cage(centre):
    """The six FCC vertices bounding an octahedral void."""
    return [tuple(np.array(centre) + np.array(d)) for d in AXES]


def tetrahedral_cages(vertices):
    """Four mutually nearest-neighbor vertices."""
    return [g for g in combinations(vertices, 4)
            if all(dist_sq(a, b) == L_SQ for a, b in combinations(g, 2))]


def octahedral_cages(vertices):
    """Six vertices carrying twelve bonds, every vertex of degree four."""
    out = []
    for g in combinations(vertices, 6):
        edges = [(a, b) for a, b in combinations(g, 2)
                 if dist_sq(a, b) == L_SQ]
        if len(edges) != 12:
            continue
        deg = Counter(v for e in edges for v in e)
        if set(deg.values()) == {4}:
            out.append(set(g))
    return out


def is_fcc_tetrahedral_site(centre):
    """Genuine FCC tetrahedral voids sit at the quarter-cell positions."""
    return all(abs(abs(c) % 1 - 0.5) < 1e-9 for c in centre)


def main(R=3):
    centres = [(x, y, z)
               for x in range(-R, R + 1)
               for y in range(-R, R + 1)
               for z in range(-R, R + 1)
               if (x + y + z) % 2 == 1]
    pairs = [(a, b) for a, b in combinations(centres, 2)
             if dist_sq(a, b) == L_SQ and max(abs(v) for v in a + b) <= R - 1]

    tet_comp, oct_comp = Counter(), Counter()
    n_tet, n_oct_new, n_sites = Counter(), Counter(), 0

    for O1, O2 in pairs:
        C1, C2 = bounding_cage(O1), bounding_cage(O2)
        shared = set(C1) & set(C2)
        assert len(shared) == 2, "nearest-neighbor voids should share two vertices"
        assert dist_sq(*list(shared)) == L_SQ, "the shared pair should be an edge"

        only1, only2 = set(C1) - shared, set(C2) - shared
        interface = sorted(set(C1) | set(C2))
        assert len(interface) == 10

        def composition(g):
            g = set(g)
            return (len(g & shared), len(g & only1), len(g & only2))

        tets = tetrahedral_cages(interface)
        n_tet[len(tets)] += 1
        for g in tets:
            tet_comp[composition(g)] += 1
            if is_fcc_tetrahedral_site(np.mean(np.array(g), axis=0)):
                n_sites += 1

        octs = octahedral_cages(interface)
        for g in octs:
            oct_comp[composition(g)] += 1
        n_oct_new[sum(1 for g in octs
                      if g != set(C1) and g != set(C2))] += 1

    total_tets = sum(tet_comp.values())
    print(f"nearest-neighbor octahedral-void pairs examined: {len(pairs)}")
    print(f"\ntetrahedral cages per interface : {dict(n_tet)}")
    print(f"octahedral cages per interface that are NOT a parent void:"
          f" {dict(n_oct_new)}")

    print("\ncage composition (shared, exclusive to O1, exclusive to O2)")
    for comp, n in sorted(tet_comp.items()):
        print(f"    tetrahedra {comp}: {n:5d}   draws from both parents: "
              f"{comp[1] > 0 and comp[2] > 0}")
    for comp, n in sorted(oct_comp.items()):
        print(f"    octahedra  {comp}: {n:5d}   draws from both parents: "
              f"{comp[1] > 0 and comp[2] > 0}")

    print(f"\ntetrahedral cage centres at genuine FCC tetrahedral-void sites:"
          f" {n_sites} of {total_tets}")

    assert set(n_tet) == {2}
    assert set(n_oct_new) == {0}
    assert set(tet_comp) == {(2, 1, 1)}
    assert set(oct_comp) == {(2, 4, 0), (2, 0, 4)}
    assert n_sites == total_tets

    print("\nEvery tetrahedral cage draws a vertex from each parent, so it exists")
    print("only because the two defects are in contact. Every octahedral cage in")
    print("the interface is one of the parent voids, containing no vertex")
    print("exclusive to the other parent. The residual is therefore tetrahedral,")
    print("with no appeal to the measured line energy.")



# ----------------------------------------------------------------------
# Two further properties of the tetrahedral residual site.
# ----------------------------------------------------------------------
def site_symmetry(vertices):
    """Orthogonal maps fixing the void centre and permuting the vertices."""
    from itertools import permutations
    V = np.array(vertices, dtype=float)
    V = V - V.mean(0)
    ops = []
    for perm in permutations(range(len(V))):
        M, *_ = np.linalg.lstsq(V, V[list(perm)], rcond=None)
        if np.allclose(V @ M, V[list(perm)], atol=1e-9) \
           and np.allclose(M.T @ M, np.eye(3), atol=1e-9):
            ops.append(perm)
    orbits = {frozenset(p[v] for p in ops) for v in range(len(V))}
    return len(ops), sorted(len(o) for o in orbits)


def hessian_eigenvalues(vertices):
    """Curvature at the equidistant centre, bonds as unit-length springs."""
    V = np.array(vertices, dtype=float)
    c = V.mean(0)
    r0 = np.linalg.norm(V - c, axis=1).mean()

    def energy(x):
        d = np.linalg.norm(V - x, axis=1)
        return 0.5 * np.sum((d - r0) ** 2)

    h, H = 1e-5, np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            ei, ej = np.zeros(3), np.zeros(3)
            ei[i] = h; ej[j] = h
            H[i, j] = (energy(c + ei + ej) - energy(c + ei - ej)
                       - energy(c - ei + ej) + energy(c - ei - ej)) / (4 * h * h)
    return np.sort(np.linalg.eigvalsh(H))


def residual_site_report():
    tet = [(0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
    n_ops, orbits = site_symmetry(tet)
    eig = hessian_eigenvalues(tet)
    print("\ntetrahedral residual site")
    print(f"    site-symmetry operations: {n_ops}   vertex orbits: {orbits}")
    print(f"    Hessian eigenvalues at the symmetric centre:"
          f" {', '.join(f'{v:+.3f}' for v in eig)}")
    assert n_ops == 24 and orbits == [4]
    assert np.all(eig > 0) and np.allclose(eig, 4/3, atol=1e-3)
    print("    all four vertices lie in one orbit, so no vertex is")
    print("    distinguished and an anchor-free occupant is admissible;")
    print("    the Hessian is positive definite and isotropic, so there is")
    print("    no soft mode toward anchor selection or face escape.")



# ----------------------------------------------------------------------
# Exhaustive cage enumeration and face adjacency.
# ----------------------------------------------------------------------
WALL = 1 / np.sqrt(3)            # metric wall, in units of L
L_BOND = np.sqrt(L_SQ)           # bond length in these coordinates


def equidistant_centre(vertices):
    """The unique point equidistant from all vertices, or None."""
    V = np.array(vertices, dtype=float)
    A = 2 * (V[1:] - V[0])
    b = (V[1:] ** 2).sum(1) - (V[0] ** 2).sum()
    if np.linalg.matrix_rank(A, tol=1e-9) < 3:
        return None
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = np.linalg.norm(V - x, axis=1)
    return (x, r.mean()) if r.std() < 1e-9 else None


def is_connected(subset, adjacency):
    subset = list(subset)
    seen, stack = {subset[0]}, [subset[0]]
    while stack:
        v = stack.pop()
        for w in subset:
            if w not in seen and w in adjacency[v]:
                seen.add(w)
                stack.append(w)
    return len(seen) == len(subset)


def stable_cages(interface):
    """Every connected subset that survives the four stability constraints.

    A trapped node is stable only if the subset (a) admits a unique
    equidistant centre, (b) has radius at or above the metric wall
    L/sqrt(3), (c) is strain-balanced, so the unit bond vectors sum to
    zero, and (d) is non-coplanar, since a coplanar set lets the node
    slide normal to the plane and is not a trap.
    """
    adj = {v: {w for w in interface if dist_sq(v, w) == L_SQ}
           for v in interface}
    n_subsets = n_connected = 0
    survivors = []
    for k in range(3, len(interface) + 1):
        for sub in combinations(interface, k):
            n_subsets += 1
            if not is_connected(sub, adj):
                continue
            n_connected += 1
            found = equidistant_centre(sub)
            if found is None:
                continue
            centre, radius = found
            if radius / L_BOND < WALL - 1e-12:
                continue
            V = np.array(sub, dtype=float)
            u = (V - centre) / np.linalg.norm(V - centre, axis=1)[:, None]
            if np.linalg.norm(u.sum(0)) > 1e-9:
                continue
            if np.linalg.matrix_rank(V - V.mean(0), tol=1e-9) < 3:
                continue
            survivors.append((len(sub), round(radius / L_BOND, 4),
                              tuple(sorted(sub))))
    return n_subsets, n_connected, survivors


def triangular_faces(cage_vertices):
    """Three mutually bonded vertices."""
    return [f for f in combinations(sorted(cage_vertices), 3)
            if all(dist_sq(a, b) == L_SQ for a, b in combinations(f, 2))]


def exhaustive_report(R=2):
    """Enumerate every connected subset, and check face adjacency."""
    centres = [(x, y, z)
               for x in range(-R, R + 1)
               for y in range(-R, R + 1)
               for z in range(-R, R + 1)
               if (x + y + z) % 2 == 1]
    pairs = [(a, b) for a, b in combinations(centres, 2)
             if dist_sq(a, b) == L_SQ and max(abs(v) for v in a + b) <= R - 1]

    counts, kinds, faces, edge_in_face = Counter(), Counter(), Counter(), Counter()
    for O1, O2 in pairs:
        C1, C2 = bounding_cage(O1), bounding_cage(O2)
        shared = set(C1) & set(C2)
        only1, only2 = set(C1) - shared, set(C2) - shared
        interface = sorted(set(C1) | set(C2))

        n_sub, n_conn, surv = stable_cages(interface)
        counts[(n_sub, n_conn, len(surv))] += 1
        for k, r, vs in surv:
            comp = (len(set(vs) & shared), len(set(vs) & only1),
                    len(set(vs) & only2))
            kinds[(k, r, comp)] += 1

        f1, f2 = set(triangular_faces(C1)), set(triangular_faces(C2))
        for k, r, vs in surv:
            if k != 4:
                continue
            tf = set(triangular_faces(vs))
            faces[(len(tf & f1), len(tf & f2))] += 1
            for f in (tf & f1) | (tf & f2):
                edge_in_face[shared <= set(f)] += 1

    print(f"\nexhaustive enumeration over {len(pairs)} interfaces")
    print(f"    (subsets, connected, survivors) per interface: {dict(counts)}")
    print("    survivors (vertices, radius/L, composition):")
    for key, n in sorted(kinds.items()):
        print(f"        {key[0]} vertices, r={key[1]} L, composition {key[2]}"
              f"   x{n}")
    print(f"    triangular faces each tetrahedron shares with (O1, O2):"
          f" {dict(faces)}")
    print(f"    shared faces containing both merger-edge vertices:"
          f" {dict(edge_in_face)}")

    assert set(counts) == {(968, 794, 4)}
    assert set(faces) == {(1, 1)}
    assert set(edge_in_face) == {True}
    sizes = {k[0] for k in kinds}
    assert sizes == {4, 6}
    print("\n    Only the regular tetrahedron (r = 0.6124 L) and the regular")
    print("    octahedron (r = 0.7071 L) survive the four stability")
    print("    constraints, over the full connected-subset enumeration. Each")
    print("    tetrahedron shares one full triangular face with each parent")
    print("    octahedron, and every such face contains both merger-edge")
    print("    vertices, so the residual reaches its site across a 2D face")
    print("    without crossing 3D bulk.")


if __name__ == "__main__":
    main()
    exhaustive_report()
    residual_site_report()

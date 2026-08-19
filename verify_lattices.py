"""
Lattice-robustness check for the octahedral-void construction.

Applies the same construction to three candidate vacuum lattices: BCC,
ideal HCP, and FCC. All three are generated at nearest-neighbor distance
L = 1. For each lattice the script reports:

  * whether a tetrahedral cage (a K_4 clique of mutually nearest-neighbor
    nodes) exists at all;
  * whether an octahedral cage (a K_{2,2,2} cage) exists;
  * the first-shell overlap K_pairwise on each cage type;
  * the first-shell triple overlap K_triple on the octahedral cage;
  * the resulting counts C_p and C_DM.

Requires numpy only. Runs in about ten seconds.
"""
from itertools import combinations
from collections import Counter
import numpy as np

TOL = 1e-6


# ----------------------------------------------------------------------
# Lattice generators. Each returns node coordinates at NN distance L = 1.
# ----------------------------------------------------------------------
def build_fcc(R=4):
    basis = [(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0)]
    P = [(i + b[0], j + b[1], k + b[2])
         for i in range(-R, R + 1)
         for j in range(-R, R + 1)
         for k in range(-R, R + 1)
         for b in basis]
    return np.array(P) * np.sqrt(2.0)


def build_bcc(R=4):
    a = 2.0 / np.sqrt(3.0)              # NN distance a*sqrt(3)/2 = 1
    P = []
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            for k in range(-R, R + 1):
                P.append((i, j, k))
                P.append((i + .5, j + .5, k + .5))
    return np.array(P) * a


def build_hcp(R=4):
    c = np.sqrt(8.0 / 3.0)              # ideal axial ratio
    a1 = np.array([1.0, 0.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3.0) / 2.0, 0.0])
    off = np.array([0.5, np.sqrt(3.0) / 6.0, 0.0])
    P = []
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            for k in range(-R, R + 1):
                b = i * a1 + j * a2 + np.array([0.0, 0.0, k * c])
                P.append(b)
                P.append(b + off + np.array([0.0, 0.0, c / 2.0]))
    return np.array(P)


# ----------------------------------------------------------------------
# Neighbor structure
# ----------------------------------------------------------------------
def neighbor_sets(P, nn):
    """First shell of every node, as frozensets of node indices."""
    shells = []
    for i in range(len(P)):
        d = np.linalg.norm(P - P[i], axis=1)
        idx = np.where((d < nn * 1.02) & (d > TOL))[0]
        shells.append(frozenset(int(x) for x in idx))
    return shells


def nn_distance(P, center_idx):
    d = np.linalg.norm(P - P[center_idx], axis=1)
    return np.sort(d)[1]


# ----------------------------------------------------------------------
# Cage finders
# ----------------------------------------------------------------------
def find_tet_cages(nbr, deep):
    """K_4 cliques: four mutually nearest-neighbor nodes."""
    cages = set()
    for v in deep:
        for trio in combinations(sorted(nbr[v]), 3):
            g = (v,) + trio
            if all(b in nbr[a] for a, b in combinations(g, 2)):
                cages.add(tuple(sorted(g)))
    return sorted(cages)


def find_oct_cages(P, nbr, deep, nn):
    """K_{2,2,2} cages: six nodes, twelve bonds, every degree 4."""
    cages = set()
    for v in deep:
        d = np.linalg.norm(P - P[v], axis=1)
        second = np.where(np.abs(d - nn * np.sqrt(2.0)) < nn * 0.02)[0]
        for w in second:
            w = int(w)
            common = sorted(nbr[v] & nbr[w])
            if len(common) != 4:
                continue
            g = tuple(sorted((v, w) + tuple(common)))
            E = [(a, b) for a, b in combinations(g, 2) if b in nbr[a]]
            if len(E) != 12:
                continue
            deg = Counter(x for e in E for x in e)
            if set(deg.values()) == {4}:
                cages.add(g)
    return sorted(cages)


# ----------------------------------------------------------------------
# Overlap statistics
# ----------------------------------------------------------------------
def cage_overlaps(g, nbr):
    """Return (|N(e)| set, K_pairwise counter, K_triple counter) for one cage."""
    E = [(a, b) for a, b in combinations(g, 2) if b in nbr[a]]
    N = {e: nbr[e[0]] | nbr[e[1]] for e in E}
    sizes = {len(N[e]) for e in E}

    skew = [(x, y) for x, y in combinations(E, 2) if not set(x) & set(y)]
    pair = Counter(len(N[x] & N[y]) for x, y in skew)

    match = [t for t in combinations(E, 3)
             if len({v for e in t for v in e}) == 6]
    trip = Counter(len(N[t[0]] & N[t[1]] & N[t[2]]) for t in match)
    return sizes, pair, trip


def survey(P, nbr, cages, dist, rad, label):
    """Aggregate overlap statistics over all fully interior cages."""
    if not cages:
        print(f"    {label}: no such cage exists in this lattice")
        return None, None, 0

    tot_pair, tot_trip, used, radii, sizes = Counter(), Counter(), 0, [], set()
    for g in cages:
        if any(dist[i] > rad - 1.2 for i in g):
            continue
        used += 1
        X = P[list(g)]
        r = np.linalg.norm(X - X.mean(0), axis=1)
        assert r.std() < 1e-9, "cage is not equidistant"
        radii.append(r.mean())
        s, pair, trip = cage_overlaps(g, nbr)
        sizes |= s
        tot_pair += pair
        tot_trip += trip

    per_cage_pair = {k: v // used for k, v in tot_pair.items()}
    per_cage_trip = {k: v // used for k, v in tot_trip.items()}
    print(f"    {label}: {used} interior cages, "
          f"circumradius {np.mean(radii):.4f} L, |N(e)| = {sorted(sizes)}")
    print(f"        K_pairwise per cage: {per_cage_pair}"
          f"   {'uniform' if len(tot_pair) == 1 else 'NOT UNIFORM'}")
    if tot_trip:
        print(f"        K_triple   per cage: {per_cage_trip}"
              f"   {'uniform' if len(tot_trip) == 1 else 'NOT UNIFORM'}")
    return tot_pair, tot_trip, used


def mean_of(counter):
    return sum(k * v for k, v in counter.items()) / sum(counter.values())


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def analyse(name, P, rad=2.6):
    center = int(np.argmin(np.linalg.norm(P - P.mean(0), axis=1)))
    nn = nn_distance(P, center)
    P = P / nn                                   # normalize to L = 1
    nn = 1.0

    dist = np.linalg.norm(P - P.mean(0), axis=1)
    keep = np.where(dist < rad + 2.5)[0]         # trim for speed
    P = P[keep]
    dist = np.linalg.norm(P - P.mean(0), axis=1)
    nbr = neighbor_sets(P, nn)
    deep = [i for i in range(len(P)) if dist[i] < rad]

    modal_K = Counter(len(nbr[i]) for i in deep).most_common(1)[0][0]
    print(f"\n{name}: interior coordination K = {modal_K}")

    tets = find_tet_cages(nbr, deep)
    octs = find_oct_cages(P, nbr, deep, nn)
    tp, _, _ = survey(P, nbr, tets, dist, rad, "tetrahedral cage (K_4)")
    op, ot, _ = survey(P, nbr, octs, dist, rad, "octahedral cage (K_{2,2,2})")

    if tp is None:
        print("    => construction undefined: no tetrahedral cage to count")
        return
    kp = mean_of(tp)
    C_p = 13 * 144 - 3 * kp
    print(f"    => C_p = 13*144 - 3*{kp:g} = {C_p:g}")
    if op is not None:
        kpo, kto = mean_of(op), mean_of(ot)
        C_DM = 25 * 144 - 30 * kpo + 8 * kto
        print(f"    => C_DM = 25*144 - 30*{kpo:g} + 8*{kto:g} = {C_DM:g}")
        print(f"    => m_DM = {C_DM / C_p * 938.272 / 1000:.4f} GeV")


if __name__ == "__main__":
    print("Lattice robustness check, all lattices at NN distance L = 1")
    analyse("BCC", build_bcc())
    analyse("HCP (ideal c/a = sqrt(8/3))", build_hcp())
    analyse("FCC", build_fcc())
    print("\nObserved proton-to-electron mass ratio: 1836.153")

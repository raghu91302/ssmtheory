"""
Verification script for the Embedding Uniformity Lemma (Section 4.4 of
"Dark Matter as a Trapped K=6 Remnant in the Octahedral Voids of the
FCC Vacuum Lattice"). Standard library + numpy only.

Establishes:
  (i)  The 8 perfect matchings of K_{2,2,2} form a single O_h-orbit, so
       K_triple = 8 is O_h-forced.
  (ii) The 30 skew pairs split into two O_h-orbits (sizes 24 and 6); both
       orbits independently yield K_pairwise = 10, a structural rigidity
       stronger than O_h-symmetry alone forces.
  (a)-(d) Body-diagonal labeling: each matching is labeled by a unique
       body diagonal of the surrounding cube; matchings pair under
       inversion through the void center; the three edge midpoints of
       each matching lie in the plane perpendicular to its body diagonal.
"""
from itertools import combinations, permutations
import numpy as np

# --- Octahedral void embedding (Section 4.3 of the main text) ---
VOID_CENTER = np.array([1.0, 0.0, 0.0])
VERTICES = {
    'A': np.array([2.0, 0.0, 0.0]),    # +x
    'B': np.array([0.0, 0.0, 0.0]),    # -x
    'C': np.array([1.0, 1.0, 0.0]),    # +y
    'D': np.array([1.0,-1.0, 0.0]),    # -y
    'E': np.array([1.0, 0.0, 1.0]),    # +z
    'F': np.array([1.0, 0.0,-1.0]),    # -z
}
LABELS = list(VERTICES.keys())

NN = [( 1, 1, 0), ( 1,-1, 0), (-1, 1, 0), (-1,-1, 0),
      ( 1, 0, 1), ( 1, 0,-1), (-1, 0, 1), (-1, 0,-1),
      ( 0, 1, 1), ( 0, 1,-1), ( 0,-1, 1), ( 0,-1,-1)]

def first_shell(v):
    x, y, z = v
    return frozenset((x+dx, y+dy, z+dz) for dx, dy, dz in NN)

def dist_sq(a, b):
    return int(np.sum((a - b) ** 2))

edges = [(u, v) for u, v in combinations(LABELS, 2)
         if dist_sq(VERTICES[u], VERTICES[v]) == 2]
edge_N = {e: first_shell(tuple(VERTICES[e[0]].astype(int)))
            | first_shell(tuple(VERTICES[e[1]].astype(int)))
          for e in edges}

def canonical(u, v):
    return (u, v) if (u, v) in edges else (v, u)

# --- O_h: 48 signed permutations acting (relative to void center) ---
def Oh_elements():
    elems = []
    for perm in permutations(range(3)):
        for sx, sy, sz in [(a, b, c) for a in (1,-1) for b in (1,-1) for c in (1,-1)]:
            M = np.zeros((3, 3), dtype=int)
            for i, j in enumerate(perm):
                M[i, j] = (sx, sy, sz)[i]
            elems.append(M)
    return elems
Oh = Oh_elements()
assert len(Oh) == 48

coord_to_label = {tuple(VERTICES[L].astype(int)): L for L in LABELS}
def act_label(g, L):
    rel = VERTICES[L] - VOID_CENTER
    gv = g @ rel + VOID_CENTER
    return coord_to_label[tuple(gv.astype(int))]
def act_edge(g, e):
    return canonical(act_label(g, e[0]), act_label(g, e[1]))

# === Part (i): perfect matchings form a single O_h-orbit ===
matchings = [tuple(sorted(t)) for t in combinations(edges, 3)
             if len({v for e in t for v in e}) == 6]
assert len(matchings) == 8

def act_matching(g, m):
    return tuple(sorted(act_edge(g, e) for e in m))

unseen = set(matchings)
m_orbits = []
while unseen:
    m0 = next(iter(unseen))
    orb = {act_matching(g, m0) for g in Oh}
    m_orbits.append(orb); unseen -= orb
assert [len(o) for o in m_orbits] == [8], "matchings should form one orbit of size 8"

# K_triple = 8 by direct evaluation on one representative
m_rep = next(iter(m_orbits[0]))
K_triple = len(edge_N[m_rep[0]] & edge_N[m_rep[1]] & edge_N[m_rep[2]])
assert K_triple == 8

# === Part (ii): skew pairs split into two O_h-orbits of sizes 24 and 6 ===
skew_pairs = [tuple(sorted([e1, e2]))
              for e1, e2 in combinations(edges, 2)
              if not (set(e1) & set(e2))]
assert len(skew_pairs) == 30

def act_skew(g, sp):
    return tuple(sorted([act_edge(g, sp[0]), act_edge(g, sp[1])]))

unseen = set(skew_pairs)
sp_orbits = []
while unseen:
    sp0 = next(iter(unseen))
    orb = {act_skew(g, sp0) for g in Oh}
    sp_orbits.append(orb); unseen -= orb
assert sorted(len(o) for o in sp_orbits) == [6, 24], \
    "skew pairs should split into orbits of sizes 6 and 24"

# K_pairwise = 10 in both orbits
K_pair_per_orbit = []
for orb in sp_orbits:
    sp = next(iter(orb))
    K_pair_per_orbit.append(len(edge_N[sp[0]] & edge_N[sp[1]]))
assert set(K_pair_per_orbit) == {10}, \
    "Both orbits must independently give K_pairwise = 10"

# === Body-diagonal labeling ===
CUBE_CORNERS = [tuple((VOID_CENTER + np.array([sx, sy, sz])).astype(int))
                for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)]
body_diagonals = []
for c1, c2 in combinations(CUBE_CORNERS, 2):
    d = tuple(c1[i] + c2[i] - 2*int(VOID_CENTER[i]) for i in range(3))
    if d == (0, 0, 0):
        body_diagonals.append(frozenset({c1, c2}))
assert len(body_diagonals) == 4

bounding_set = {tuple(VERTICES[L].astype(int)) for L in LABELS}

# (a),(b): each matching is labeled by a unique body diagonal,
#          and each body diagonal labels exactly 2 matchings
label_of = {}
for m in matchings:
    inter = edge_N[m[0]] & edge_N[m[1]] & edge_N[m[2]]
    other = frozenset(inter - bounding_set)
    assert len(other) == 2
    assert other in body_diagonals
    label_of[m] = body_diagonals.index(other)

from collections import Counter
diag_counts = Counter(label_of.values())
assert all(c == 2 for c in diag_counts.values()) and len(diag_counts) == 4

# (c): paired matchings are related by inversion through void center
def invert_label(L):
    inv = 2*VOID_CENTER - VERTICES[L]
    return coord_to_label[tuple(inv.astype(int))]
def invert_matching(m):
    return tuple(sorted(canonical(invert_label(e[0]), invert_label(e[1])) for e in m))

for diag_idx in range(4):
    paired = [m for m in matchings if label_of[m] == diag_idx]
    assert len(paired) == 2
    m1, m2 = paired
    assert invert_matching(m1) == m2

# (d): the three edge midpoints of any matching lie in the plane through
#      the void center perpendicular to that matching's body diagonal
for m in matchings:
    diag = list(body_diagonals[label_of[m]])
    diag_vec = np.array(diag[1], dtype=float) - np.array(diag[0], dtype=float)
    diag_unit = diag_vec / np.linalg.norm(diag_vec)
    for e in m:
        midpoint = (VERTICES[e[0]] + VERTICES[e[1]]) / 2.0
        proj = np.dot(midpoint - VOID_CENTER, diag_unit)
        assert abs(proj) < 1e-9, "midpoints should lie in the perpendicular plane"

# === Report ===
print("Embedding Uniformity Lemma -- verification report")
print("-" * 56)
print(f"|O_h|                                        = {len(Oh)}")
print(f"Perfect matchings of K_2,2,2                 = {len(matchings)}")
print(f"O_h orbits on matchings (sizes)              = "
      f"{sorted(len(o) for o in m_orbits)}")
print(f"K_triple (any matching)                      = {K_triple}")
print(f"K_triple uniformity                          = O_h-forced (single orbit)")
print()
print(f"Skew pairs of K_2,2,2                        = {len(skew_pairs)}")
print(f"O_h orbits on skew pairs (sizes)             = "
      f"{sorted(len(o) for o in sp_orbits)}")
print(f"K_pairwise per orbit                         = {K_pair_per_orbit}")
print(f"K_pairwise uniformity                        = "
      f"stronger than O_h alone forces")
print()
print(f"Body diagonals of circumscribing cube         = {len(body_diagonals)}")
print(f"Matching-to-diagonal map: 4 pairs            = "
      f"{dict(Counter(diag_counts.values()))}")
print(f"Within-pair relation                         = "
      f"inversion through void center (-I in O_h)")
print(f"Edge midpoints in perpendicular plane        = verified for all 8 matchings")
print()
print("All lemma claims verified.")

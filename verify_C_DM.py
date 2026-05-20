"""
Reference implementation for the K_{2,2,2} forward derivation of
C_DM = 3364 (Eq. 6 of "Dark Matter as a Trapped K=6 Remnant
in the Octahedral Voids of the FCC Vacuum Lattice"). Standard library only.
"""
from itertools import combinations
from collections import Counter

# 12 FCC nearest-neighbor displacement vectors (Eq. 7 of the main text).
NN = [( 1, 1, 0), ( 1,-1, 0), (-1, 1, 0), (-1,-1, 0),
      ( 1, 0, 1), ( 1, 0,-1), (-1, 0, 1), (-1, 0,-1),
      ( 0, 1, 1), ( 0, 1,-1), ( 0,-1, 1), ( 0,-1,-1)]

def first_shell(v):
    """Return the 12 FCC nearest neighbors of vertex v as a frozenset."""
    x, y, z = v
    return frozenset((x+dx, y+dy, z+dz) for dx, dy, dz in NN)

# Octahedral void at (1,0,0); 6 bounding vertices A..F (Table A.1).
VERTICES = {'A': (2, 0, 0), 'B': (0, 0, 0),
            'C': (1, 1, 0), 'D': (1,-1, 0),
            'E': (1, 0, 1), 'F': (1, 0,-1)}

def dist_sq(a, b):
    return sum((a[i]-b[i])**2 for i in range(3))

# Edges of K_{2,2,2}: vertex pairs at NN distance (squared distance = 2).
labels = list(VERTICES.keys())
edges = [(u, v) for u, v in combinations(labels, 2)
         if dist_sq(VERTICES[u], VERTICES[v]) == 2]
assert len(edges) == 12, "K_{2,2,2} should have 12 edges"

# Pre-compute N(e) = N_1(v_a) U N_1(v_b) for each edge.
N = {e: first_shell(VERTICES[e[0]]) | first_shell(VERTICES[e[1]])
     for e in edges}
assert all(len(N[e]) == 20 for e in edges), "|N(e)| should be 20"

# --- Skew pairs and pairwise intersections ---
def share_vertex(e1, e2):
    return bool(set(e1) & set(e2))

skew_pairs = [(e1, e2) for e1, e2 in combinations(edges, 2)
              if not share_vertex(e1, e2)]
assert len(skew_pairs) == 30, "K_{2,2,2} should have 30 skew pairs"

pair_sizes = [len(N[e1] & N[e2]) for e1, e2 in skew_pairs]
assert set(pair_sizes) == {10}, "all pair intersections should equal 10"
K_pairwise = 10

# --- Perfect matchings and triple intersections ---
matchings = []
for t in combinations(edges, 3):
    vs = set()
    for e in t:
        vs.update(e)
    if len(vs) == 6:           # 4-matching needs 8 vertices; impossible
        matchings.append(t)
assert len(matchings) == 8, "K_{2,2,2} should have 8 perfect matchings"

bounding = set(VERTICES.values())
triple_sizes = []
for m in matchings:
    inter = N[m[0]] & N[m[1]] & N[m[2]]
    n_bd = len(inter & bounding)
    n_other = len(inter - bounding)
    triple_sizes.append(n_bd + n_other)
    assert n_bd == 6 and n_other == 2, "6 + 2 decomposition expected"
assert set(triple_sizes) == {8}, "all triple intersections should equal 8"
K_triple = 8

# --- Assemble C_DM and forward mass prediction ---
N_O          = 25            # 6 bounding * 4 internal bonds + 1 trapped node
K_squared    = 144           # K^2 per-node disruption count, K = 12
c_skew       = len(skew_pairs)    # 30
c_triple     = len(matchings)     # 8
C_DM = N_O * K_squared - c_skew * K_pairwise + c_triple * K_triple
assert C_DM == 3364

C_p          = 1836
m_p_MeV      = 938.272       # CODATA 2022 proton mass
m_DM_MeV     = (C_DM / C_p) * m_p_MeV

print(f"c_skew^(O)      = {c_skew}")        # 30
print(f"c_triple^(O)    = {c_triple}")      # 8
print(f"K_pairwise^(O)  = {K_pairwise}")    # 10
print(f"K_triple^(O)    = {K_triple}")      # 8
print(f"C_DM            = {C_DM}")          # 3364
print(f"m_DM            = {m_DM_MeV:.1f} MeV "
      f"= {m_DM_MeV/1000:.4f} GeV")          # 1719.1 MeV = 1.7191 GeV

"""
Verify the bulk redundancy claim:
- Each bulk edge has 30 informative stabilizers in its first-neighbour shell
- Decomposition: 20 vertex-based (Z) + 10 void-based (X)
- Compare to 4 primary stabilizers per edge

In the L=4 torus, all edges are bulk-equivalent (periodic boundary).
This gives us the 'pure bulk' baseline.

Then for the bulk-vs-boundary question we need to construct an OPEN
boundary lattice (which is what ssm_sim.py actually produces).
"""
import numpy as np

data = np.load('/home/claude/lorentz_paper/code_L4.npz', allow_pickle=True)
H_X = data['H_X']
H_Z = data['H_Z']
edges = data['edges']

n_v = H_Z.shape[0]
n_o = H_X.shape[0]
n_e = len(edges)

# Build adjacency: for each vertex, which vertices are neighbors?
neighbors = [set() for _ in range(n_v)]
for e_idx, (u, v) in enumerate(edges):
    neighbors[u].add(v)
    neighbors[v].add(u)

print("=== Per-edge stabilizer coverage ===\n")

# For each edge, count primary and informative stabilizers
primary_Z_per_edge = []   # vertex Z-stabilizers whose support contains this edge
primary_X_per_edge = []   # void X-stabilizers whose support contains this edge
informative_Z_per_edge = []  # Z-stabilizers whose support touches any edge sharing a vertex with this edge
informative_X_per_edge = []  # X-stabilizers whose support touches u or v

# Build vertex -> set of edges incident
vertex_edges = [set() for _ in range(n_v)]
for e_idx, (u, v) in enumerate(edges):
    vertex_edges[u].add(e_idx)
    vertex_edges[v].add(e_idx)

# Build void -> set of edges in its X-stabilizer
void_edges = [set(np.where(H_X[o])[0]) for o in range(n_o)]
# Build vertex -> set of voids that contain it as surrounding vertex
# Equivalent: void o's surrounding vertex set
# Better: for each edge in void o's stab, the two endpoints are surrounding vertices of o
vertex_voids = [set() for _ in range(n_v)]
for o in range(n_o):
    for e_idx in void_edges[o]:
        u, v = edges[e_idx]
        vertex_voids[u].add(o)
        vertex_voids[v].add(o)

for e_idx, (u, v) in enumerate(edges):
    # Primary Z: vertices that this edge is incident to (i.e. u and v themselves)
    primary_Z = {u, v}
    # Primary X: voids whose stab contains this edge
    primary_X = set(np.where(H_X[:, e_idx])[0])

    # Informative Z: vertices whose stab support touches a vertex of this edge
    # i.e. vertices w such that there is an edge from w to {u,v}, OR w in {u,v}
    # Equivalently: w in {u, v} ∪ N(u) ∪ N(v)
    informative_Z = {u, v} | neighbors[u] | neighbors[v]
    # Informative X: voids surrounding u or v (i.e. voids whose stab contains an edge touching u or v)
    informative_X = vertex_voids[u] | vertex_voids[v]

    primary_Z_per_edge.append(len(primary_Z))
    primary_X_per_edge.append(len(primary_X))
    informative_Z_per_edge.append(len(informative_Z))
    informative_X_per_edge.append(len(informative_X))

primary_Z_per_edge = np.array(primary_Z_per_edge)
primary_X_per_edge = np.array(primary_X_per_edge)
informative_Z_per_edge = np.array(informative_Z_per_edge)
informative_X_per_edge = np.array(informative_X_per_edge)

print(f"Primary Z per edge:        min={primary_Z_per_edge.min()}, max={primary_Z_per_edge.max()}, mean={primary_Z_per_edge.mean():.2f}  (predicted: 2)")
print(f"Primary X per edge:        min={primary_X_per_edge.min()}, max={primary_X_per_edge.max()}, mean={primary_X_per_edge.mean():.2f}  (predicted: 2)")
print(f"Total primary per edge:    {(primary_Z_per_edge + primary_X_per_edge).mean():.2f}  (predicted: 4)")
print()
print(f"Informative Z per edge:    min={informative_Z_per_edge.min()}, max={informative_Z_per_edge.max()}, mean={informative_Z_per_edge.mean():.2f}  (predicted: 20)")
print(f"Informative X per edge:    min={informative_X_per_edge.min()}, max={informative_X_per_edge.max()}, mean={informative_X_per_edge.mean():.2f}  (predicted: 10)")
print(f"Total informative per edge: {(informative_Z_per_edge + informative_X_per_edge).mean():.2f}  (predicted: 30)")
print()

R_bulk = (informative_Z_per_edge + informative_X_per_edge).mean() / (primary_Z_per_edge + primary_X_per_edge).mean()
print(f"Bulk redundancy ratio R_bulk = {R_bulk:.3f}  (predicted: 7.5)")
print()

# Common neighbours per edge: how many triangles per edge?
print("=== Triangles per edge (cross-check on FCC fact: 4 per edge) ===")
common_neighbours_per_edge = []
for e_idx, (u, v) in enumerate(edges):
    common = neighbors[u] & neighbors[v]
    common_neighbours_per_edge.append(len(common))
common_neighbours_per_edge = np.array(common_neighbours_per_edge)
print(f"Common neighbours per edge: min={common_neighbours_per_edge.min()}, max={common_neighbours_per_edge.max()}, mean={common_neighbours_per_edge.mean():.2f}  (FCC: 4)")
print()

# Voids per vertex
print("=== Voids per vertex (cross-check on FCC fact: 6 per vertex) ===")
voids_per_vertex = np.array([len(vertex_voids[v]) for v in range(n_v)])
print(f"Voids per vertex: min={voids_per_vertex.min()}, max={voids_per_vertex.max()}, mean={voids_per_vertex.mean():.2f}  (predicted: 6)")

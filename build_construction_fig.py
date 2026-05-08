"""
Build a 3-panel construction figure for the manuscript:
(a) Cuboctahedral K=12 neighborhood around an FCC vertex
(b) Triad-sheet decomposition (xy, xz, yz with distinct colors)
(c) Single sheet showing Z-ancilla (vertex) and X-ancilla (octahedral void)
    with weight-4 stabilizer supports highlighted
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import os

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

OUT = "fig1_construction"


# ------------------------------------------------------------------
# Define the 12 FCC nearest-neighbor vectors, partitioned by sheet
# ------------------------------------------------------------------
sheet_xy = [(1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0)]
sheet_xz = [(1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1)]
sheet_yz = [(0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)]

C_XY = "#c0392b"  # red
C_XZ = "#27ae60"  # green
C_YZ = "#2874a6"  # blue
C_GRAY = "#7a7a7a"

fig = plt.figure(figsize=(13.5, 4.6))


# ------------------------------------------------------------------
# Panel (a): full K=12 cuboctahedral neighborhood
# ------------------------------------------------------------------
ax1 = fig.add_subplot(131, projection='3d')

# Plot all 12 edges from origin
for v in sheet_xy + sheet_xz + sheet_yz:
    ax1.plot([0, v[0]], [0, v[1]], [0, v[2]], color=C_GRAY, lw=1.6, alpha=0.9)

# Vertices: center + 12 neighbors
ax1.scatter([0], [0], [0], s=140, c='black', zorder=10, edgecolors='black')
for v in sheet_xy + sheet_xz + sheet_yz:
    ax1.scatter(v[0], v[1], v[2], s=70, c='white',
                edgecolors='black', lw=1.0, zorder=8)

# Cuboctahedron faces (square + triangle) - draw as transparent surface
# The 12 vertices are at all permutations of (±1, ±1, 0) - this is a cuboctahedron
all12 = sheet_xy + sheet_xz + sheet_yz
# Square faces: at each cube face, four vertices
sq_faces = [
    [(1,1,0),(1,0,1),(1,-1,0),(1,0,-1)],   # x=+1 plane
    [(-1,1,0),(-1,0,1),(-1,-1,0),(-1,0,-1)], # x=-1 plane
    [(1,1,0),(0,1,1),(-1,1,0),(0,1,-1)],    # y=+1
    [(1,-1,0),(0,-1,1),(-1,-1,0),(0,-1,-1)],# y=-1
    [(1,0,1),(0,1,1),(-1,0,1),(0,-1,1)],    # z=+1
    [(1,0,-1),(0,1,-1),(-1,0,-1),(0,-1,-1)],# z=-1
]
for f in sq_faces:
    poly = Poly3DCollection([f], alpha=0.05, facecolor='steelblue',
                             edgecolor='gray', linewidths=0.3)
    ax1.add_collection3d(poly)

ax1.set_title("(a)  FCC lattice: K = 12 cuboctahedral neighborhood", pad=8)
ax1.set_xlim(-1.4, 1.4); ax1.set_ylim(-1.4, 1.4); ax1.set_zlim(-1.4, 1.4)
ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_zticks([])
ax1.set_box_aspect([1,1,1])
ax1.view_init(elev=18, azim=30)


# ------------------------------------------------------------------
# Panel (b): triad sheet decomposition
# ------------------------------------------------------------------
ax2 = fig.add_subplot(132, projection='3d')
for v in sheet_xy:
    ax2.plot([0, v[0]], [0, v[1]], [0, v[2]], color=C_XY, lw=2.2)
for v in sheet_xz:
    ax2.plot([0, v[0]], [0, v[1]], [0, v[2]], color=C_XZ, lw=2.2)
for v in sheet_yz:
    ax2.plot([0, v[0]], [0, v[1]], [0, v[2]], color=C_YZ, lw=2.2)
ax2.scatter([0], [0], [0], s=140, c='black', zorder=10, edgecolors='black')
for v in all12:
    ax2.scatter(v[0], v[1], v[2], s=60, c='white',
                edgecolors='black', lw=1.0, zorder=8)

# Add a small legend
ax2.plot([], [], color=C_XY, lw=2.5,
         label=r"$S_{xy}$:  $(\pm 1, \pm 1, 0)$")
ax2.plot([], [], color=C_XZ, lw=2.5,
         label=r"$S_{xz}$:  $(\pm 1, 0, \pm 1)$")
ax2.plot([], [], color=C_YZ, lw=2.5,
         label=r"$S_{yz}$:  $(0, \pm 1, \pm 1)$")
ax2.legend(loc='upper left', bbox_to_anchor=(-0.02, 1.0),
           fontsize=8.5, framealpha=0.92, handlelength=1.3)

ax2.set_title("(b)  Three triad sheets, 4 edges each", pad=8)
ax2.set_xlim(-1.4, 1.4); ax2.set_ylim(-1.4, 1.4); ax2.set_zlim(-1.4, 1.4)
ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_zticks([])
ax2.set_box_aspect([1,1,1])
ax2.view_init(elev=18, azim=30)


# ------------------------------------------------------------------
# Panel (c): one sheet showing data, Z-ancilla, X-ancilla
# ------------------------------------------------------------------
ax3 = fig.add_subplot(133, projection='3d')

# Show xy sheet edges - draw 2 unit cells worth so the octahedral void is visible
# The xy sheet at z=0 layer of the FCC lattice has vertices at (i+j even, i, j, 0)
# Actually let me just show one Z-ancilla vertex with its 4 xy edges
# and one nearby octahedral void with its 4 xy stabilizer edges

# Z-ancilla at origin, 4 xy-data qubits at midpoints of the 4 xy edges
z_anc_pos = (0, 0, 0)
xy_endpoints = sheet_xy
xy_midpoints = [(v[0]/2, v[1]/2, 0) for v in xy_endpoints]

for ep, mp in zip(xy_endpoints, xy_midpoints):
    ax3.plot([0, ep[0]], [0, ep[1]], [0, 0], color=C_XY, lw=2.0, alpha=0.85)
    # data qubit at midpoint
    ax3.scatter(mp[0], mp[1], mp[2], s=85, c='#f5d76e',
                edgecolors='black', lw=1.0, zorder=10, marker='s')

# Z-ancilla at center
ax3.scatter([0], [0], [0], s=180, c='#e74c3c', zorder=11,
             edgecolors='black', lw=1.2, marker='o')

# X-ancilla at an octahedral void - position it 1 unit away in z direction
# The octahedral void is at a body-center-like position
# Actually for FCC, oct voids are at face centers of the conventional cube
# Let's put X-ancilla at (1, 0, 0) (a face center between two FCC vertices)
# But this needs to be edge-disjoint from our data qubits

# Use offset visualization: show the octahedron around the void
# Oct void at (0, 0, 1) - 6 surrounding FCC vertices form an octahedron
# Wait, this isn't quite right. Let me think.
# In FCC, oct voids exist at half-integer positions.
# For visualization just put X-ancilla at a position that shows the concept

# Place X-ancilla showing 4 xy-edges of an octahedron
# The octahedron at center (1, 1, 0) has vertices at:
# (1+1, 1, 0) (1-1, 1, 0) (1, 1+1, 0) (1, 1-1, 0) (1, 1, 1) (1, 1, -1)
# of which (2,1,0), (0,1,0), (1,2,0), (1,0,0) are 4 in z=0 plane
# Edges in xy plane: (2,1,0)-(1,2,0), (1,2,0)-(0,1,0), (0,1,0)-(1,0,0), (1,0,0)-(2,1,0)
# These are exactly 4 xy-sheet edges! Each has displacement (±1, ±1, 0) ✓

x_anc_pos = (1.5, 1.5, 0)  # offset for visualization
square_verts = [(2.5, 1.5, 0), (1.5, 2.5, 0), (0.5, 1.5, 0), (1.5, 0.5, 0)]
square_edges = [(0,1), (1,2), (2,3), (3,0)]

# Draw the 4 xy data qubits of this octahedron as connecting edges
for i, j in square_edges:
    a = square_verts[i]
    b = square_verts[j]
    ax3.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
             color=C_XY, lw=2.0, alpha=0.85)
    # midpoint = data qubit
    mid = ((a[0]+b[0])/2, (a[1]+b[1])/2, 0)
    ax3.scatter(mid[0], mid[1], mid[2], s=85, c='#f5d76e',
                edgecolors='black', lw=1.0, zorder=10, marker='s')

# Vertices of the surrounding octahedron (FCC vertex sites)
for v in square_verts:
    ax3.scatter(v[0], v[1], v[2], s=40, c='lightgray',
                edgecolors='gray', lw=0.7, zorder=8)

# X-ancilla at oct-void center
ax3.scatter([x_anc_pos[0]], [x_anc_pos[1]], [x_anc_pos[2]],
             s=180, c='#3498db', zorder=11,
             edgecolors='black', lw=1.2, marker='D')

# Legend entries
ax3.scatter([], [], c='#e74c3c', s=120, marker='o',
             edgecolors='black', lw=1.0, label='Z-ancilla (FCC vertex)')
ax3.scatter([], [], c='#3498db', s=120, marker='D',
             edgecolors='black', lw=1.0, label='X-ancilla (oct. void)')
ax3.scatter([], [], c='#f5d76e', s=80, marker='s',
             edgecolors='black', lw=1.0, label='Data qubit (sheet edge)')
ax3.legend(loc='upper left', fontsize=8.5, framealpha=0.92,
           bbox_to_anchor=(-0.02, 1.0), handlelength=1.0)

ax3.set_title(r"(c)  Single sheet ($S_{xy}$): weight-4 stabilizers", pad=8)
ax3.set_xlim(-1.4, 2.9); ax3.set_ylim(-1.4, 2.9); ax3.set_zlim(-1.0, 1.0)
ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([])
ax3.set_box_aspect([1.5, 1.5, 0.7])
ax3.view_init(elev=30, azim=-55)


fig.tight_layout()
fig.savefig(OUT + ".png", dpi=200, bbox_inches='tight')
fig.savefig(OUT + ".pdf", bbox_inches='tight')
print(f"Saved {OUT}.png and .pdf")
plt.close(fig)

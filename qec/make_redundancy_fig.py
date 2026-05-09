"""
Publication-quality 3D figure: bulk redundancy of FCC quantum code.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
import numpy as np
import itertools


def fcc_neighbors():
    nbrs = []
    for i, j in itertools.combinations(range(3), 2):
        for si in [+1, -1]:
            for sj in [+1, -1]:
                vec = [0, 0, 0]
                vec[i] = si
                vec[j] = sj
                nbrs.append(np.array(vec, dtype=float))
    return nbrs


def find_common_neighbors(u, v):
    nbrs = fcc_neighbors()
    N_u = [u + n for n in nbrs]
    N_v = [v + n for n in nbrs]
    common = []
    for w_u in N_u:
        for w_v in N_v:
            if np.allclose(w_u, w_v) and not np.allclose(w_u, u) and not np.allclose(w_u, v):
                common.append(w_u)
                break
    return common


def voids_at_vertex(u):
    voids = []
    basis = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]
    for d in range(3):
        for s in [+1, -1]:
            voids.append(u + s * basis[d])
    return voids


def voids_containing_edge(u, v):
    diff = v - u
    nz = np.nonzero(diff)[0]
    if len(nz) != 2:
        return []
    a, b = nz[0], nz[1]
    s_a, s_b = diff[a], diff[b]
    basis = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]
    return [u + s_a * basis[a], u + s_b * basis[b]]


def draw_cuboctahedron_hull(ax, center, color='#1f77b4', alpha=0.10):
    nbrs = fcc_neighbors()
    pts = np.array([center + n for n in nbrs])
    try:
        hull = ConvexHull(pts)
        polys = [pts[simplex] for simplex in hull.simplices]
        coll = Poly3DCollection(polys, alpha=alpha, facecolor=color,
                                edgecolor=color, linewidth=0.5)
        ax.add_collection3d(coll)
    except Exception as e:
        print(f"Hull failed: {e}")


def draw_octahedron_hull(ax, center, color='#9467bd', alpha=0.20):
    pts = np.array([center + v for v in
                    [np.array([1,0,0]), np.array([-1,0,0]),
                     np.array([0,1,0]), np.array([0,-1,0]),
                     np.array([0,0,1]), np.array([0,0,-1])]])
    try:
        hull = ConvexHull(pts)
        polys = [pts[simplex] for simplex in hull.simplices]
        coll = Poly3DCollection(polys, alpha=alpha, facecolor=color,
                                edgecolor=color, linewidth=0.7)
        ax.add_collection3d(coll)
    except Exception:
        pass


def setup_3d_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.grid(False)
    ax.xaxis.pane.set_alpha(0.03)
    ax.yaxis.pane.set_alpha(0.03)
    ax.zaxis.pane.set_alpha(0.03)


# ---- Setup geometry ----
u = np.array([0., 0., 0.])
v = np.array([1., 1., 0.])
nbrs = fcc_neighbors()
N_u = [u + n for n in nbrs]
N_v = [v + n for n in nbrs]
common = find_common_neighbors(u, v)
voids_u = voids_at_vertex(u)
voids_v = voids_at_vertex(v)
shared_voids = voids_containing_edge(u, v)


# ---- Build figure ----
fig = plt.figure(figsize=(16, 6.5))

# === Panel A ===
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.scatter(*u, color='#1f77b4', s=380, edgecolor='black', linewidth=1.5, zorder=10)
ax1.text(u[0]+0.05, u[1]+0.05, u[2]+0.18, r'$u$', fontsize=20, fontweight='bold', zorder=11)
draw_cuboctahedron_hull(ax1, u, color='#1f77b4', alpha=0.18)
for w in N_u:
    ax1.scatter(*w, color='#aec7e8', s=140, edgecolor='black', linewidth=0.8, zorder=8)
    ax1.plot([u[0], w[0]], [u[1], w[1]], [u[2], w[2]],
             color='#1f77b4', alpha=0.55, linewidth=1.0, zorder=2)
ax1.set_title("(a) Z-stabilizer of vertex $u$\nacts on its 12 cuboctahedral neighbors",
              fontsize=11, pad=8)
ax1.view_init(elev=20, azim=30)
setup_3d_axes(ax1)
ax1.set_box_aspect((1, 1, 1))

# === Panel B ===
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
draw_cuboctahedron_hull(ax2, u, color='#1f77b4', alpha=0.13)
draw_cuboctahedron_hull(ax2, v, color='#ff7f0e', alpha=0.13)
ax2.scatter(*u, color='#1f77b4', s=380, edgecolor='black', linewidth=1.5, zorder=10)
ax2.scatter(*v, color='#ff7f0e', s=380, edgecolor='black', linewidth=1.5, zorder=10)
ax2.text(u[0]-0.30, u[1]-0.15, u[2]-0.40, r'$u$', fontsize=20, fontweight='bold', zorder=11)
ax2.text(v[0]+0.15, v[1]+0.15, v[2]+0.25, r'$v$', fontsize=20, fontweight='bold', zorder=11)

for w in common:
    ax2.scatter(*w, color='#d62728', s=240, edgecolor='black', linewidth=1.2, zorder=9)
    tri = np.array([u, v, w, u])
    ax2.plot(tri[:, 0], tri[:, 1], tri[:, 2],
             color='#d62728', linewidth=1.8, alpha=0.75, zorder=5)

for w in N_u:
    skip = np.allclose(w, v) or any(np.allclose(w, c) for c in common)
    if not skip:
        ax2.scatter(*w, color='#aec7e8', s=110, edgecolor='gray', linewidth=0.6, zorder=7, alpha=0.85)
for w in N_v:
    skip = np.allclose(w, u) or any(np.allclose(w, c) for c in common)
    if not skip:
        ax2.scatter(*w, color='#ffbb78', s=110, edgecolor='gray', linewidth=0.6, zorder=7, alpha=0.85)

ax2.plot([u[0], v[0]], [u[1], v[1]], [u[2], v[2]],
         color='red', linewidth=5.5, zorder=12, alpha=0.95)

ax2.set_title("(b) Edge $(u,v)$ shared by both cuboctahedra\n"
              r"$Z$-informative count $= 2 + 11 + 11 - 4 = 20$",
              fontsize=11, pad=8)
ax2.view_init(elev=18, azim=35)
setup_3d_axes(ax2)
ax2.set_box_aspect((1, 1, 1))

# === Panel C ===
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.scatter(*u, color='#1f77b4', s=380, edgecolor='black', linewidth=1.5, zorder=10)
ax3.scatter(*v, color='#ff7f0e', s=380, edgecolor='black', linewidth=1.5, zorder=10)
ax3.text(u[0]-0.30, u[1]-0.15, u[2]-0.40, r'$u$', fontsize=20, fontweight='bold', zorder=11)
ax3.text(v[0]+0.15, v[1]+0.15, v[2]+0.25, r'$v$', fontsize=20, fontweight='bold', zorder=11)

for o in voids_u:
    is_shared = any(np.allclose(o, sv) for sv in shared_voids)
    if is_shared:
        draw_octahedron_hull(ax3, o, color='#9467bd', alpha=0.40)
    else:
        draw_octahedron_hull(ax3, o, color='#9edae5', alpha=0.18)

for o in voids_v:
    is_shared = any(np.allclose(o, sv) for sv in shared_voids)
    if is_shared:
        continue
    draw_octahedron_hull(ax3, o, color='#dbdb8d', alpha=0.18)

for o in voids_u:
    is_shared = any(np.allclose(o, sv) for sv in shared_voids)
    if is_shared:
        ax3.scatter(*o, color='#9467bd', s=320, marker='s', edgecolor='black', linewidth=1.5, zorder=9)
    else:
        ax3.scatter(*o, color='#9edae5', s=130, marker='s', edgecolor='gray', linewidth=0.6, zorder=8)
for o in voids_v:
    is_shared = any(np.allclose(o, sv) for sv in shared_voids)
    if is_shared:
        continue
    ax3.scatter(*o, color='#dbdb8d', s=130, marker='s', edgecolor='gray', linewidth=0.6, zorder=8)

ax3.plot([u[0], v[0]], [u[1], v[1]], [u[2], v[2]],
         color='red', linewidth=5.5, zorder=12, alpha=0.95)

ax3.set_title("(c) Voids touching $u$ or $v$ give X-stabilizer cover\n"
              r"$X$-informative count $= 6 + 6 - 2 = 10$",
              fontsize=11, pad=8)
ax3.view_init(elev=15, azim=10)
setup_3d_axes(ax3)
ax3.set_box_aspect((1, 1, 1))


# Common legend below
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=12,
           markeredgecolor='black', label=r'endpoint $u$'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', markersize=12,
           markeredgecolor='black', label=r'endpoint $v$'),
    Line2D([0], [0], color='red', linewidth=4, label=r'shared edge $(u,v)$'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=11,
           markeredgecolor='black', label=r'common neighbors (4)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#aec7e8', markersize=9,
           markeredgecolor='gray', label=r"$u$'s 11 other neighbors"),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffbb78', markersize=9,
           markeredgecolor='gray', label=r"$v$'s 11 other neighbors"),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#9467bd', markersize=11,
           markeredgecolor='black', label=r'2 shared voids'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#9edae5', markersize=9,
           markeredgecolor='gray', label=r"$u$'s 4 other voids"),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#dbdb8d', markersize=9,
           markeredgecolor='gray', label=r"$v$'s 4 other voids"),
]
fig.legend(handles=legend_elems, loc='lower center', ncol=5, fontsize=9.5,
           frameon=True, bbox_to_anchor=(0.5, -0.01))

plt.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.13, wspace=0.04)
plt.savefig('/home/claude/lorentz_paper/fig_bulk_redundancy.pdf', bbox_inches='tight', dpi=200)
plt.savefig('/home/claude/lorentz_paper/fig_bulk_redundancy.png', bbox_inches='tight', dpi=200)
print("Saved fig_bulk_redundancy.pdf")

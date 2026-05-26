"""Generate the four figures for the paper:

  fig1_triangle.pdf   - FCC triangle structure (3 sheets)
  fig2_primitive.pdf  - L=4 surgery primitive (4 triangles + ancillas)
  fig3_threshold.pdf  - log-log threshold plot
  fig4_comparison.pdf - logical qubits at fixed distance, code comparison

All figures are saved as PDF (vector graphics) in the current directory.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import warnings
warnings.filterwarnings('ignore')


SHEET_COLORS = {'xy': '#1f77b4', 'xz': '#ff7f0e', 'yz': '#2ca02c'}


# ---------- Figure 1: FCC triangle in 3D ----------

def fig_triangle():
    fig = plt.figure(figsize=(6.5, 4.8))
    ax = fig.add_subplot(111, projection='3d')

    # Three vertices forming an FCC triangle
    v0 = np.array([0, 0, 0])
    v1 = np.array([0, 1, 1])
    v2 = np.array([1, 0, 1])

    # Edges, labeled by sheet
    edges = [
        (v0, v1, 'yz', '$(0,1,1) \\in S_{yz}$'),
        (v0, v2, 'xz', '$(1,0,1) \\in S_{xz}$'),
        (v1, v2, 'xy', '$(1,\\!-\\!1,0) \\in S_{xy}$'),
    ]
    for a, b, sheet, _ in edges:
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                color=SHEET_COLORS[sheet], linewidth=3.5, zorder=3)

    # Vertex markers
    for v, label in [(v0, '$v_a$'), (v1, '$v_b$'), (v2, '$v_c$')]:
        ax.scatter(*v, s=130, color='black', zorder=5)
        ax.text(v[0]+0.06, v[1]+0.06, v[2]+0.08, label,
                fontsize=12, ha='left')

    # Triangle face
    pts = np.array([v0, v1, v2, v0])
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], '-',
            color='gray', linewidth=0.5, alpha=0.3, zorder=1)

    # Legend
    patches = [mpatches.Patch(color=SHEET_COLORS[s], label=f'sheet $S_{{{s}}}$')
               for s in ['xy', 'xz', 'yz']]
    ax.legend(handles=patches, loc='upper left', fontsize=10,
              bbox_to_anchor=(0.0, 1.0))

    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_zticks([0, 1])
    ax.view_init(elev=18, azim=-55)
    ax.set_title('Every FCC triangle has one edge per triad sheet',
                 fontsize=11, pad=10)
    plt.tight_layout()
    plt.savefig('fig1_triangle.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("  fig1_triangle.pdf")


# ---------- Figure 2: surgery primitive at L=4 ----------

def fig_primitive():
    fig = plt.figure(figsize=(7.5, 5.0))
    ax = fig.add_subplot(111, projection='3d')

    # 4-triangle surgery primitive at L=4
    # Vertices used: 0, 1, 2, 3, 8, 9 at coordinates from our computation
    V = {
        0: (0, 0, 0),
        1: (0, 0, 2),
        2: (0, 1, 1),
        3: (0, 1, 3),
        8: (1, 0, 1),
        9: (1, 0, 3),
    }

    triangles = [
        ('T0',  [0, 2, 8]),
        ('T4',  [0, 3, 9]),
        ('T24', [1, 2, 8]),
        ('T28', [1, 3, 9]),
    ]

    # Edges with sheet labels
    edges_in_triangles = [
        ([0, 2], 'yz'), ([0, 8], 'xz'), ([2, 8], 'xy'),  # T0
        ([0, 3], 'yz'), ([0, 9], 'xz'), ([3, 9], 'xy'),  # T4
        ([1, 2], 'yz'), ([1, 8], 'xz'),                  # T24 (xy shared)
        ([1, 3], 'yz'), ([1, 9], 'xz'),                  # T28 (xy shared)
    ]

    drawn_edges = set()
    for [u, v], sheet in edges_in_triangles:
        key = tuple(sorted([u, v]))
        if key in drawn_edges:
            continue
        drawn_edges.add(key)
        a, b = np.array(V[u]), np.array(V[v])
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                color=SHEET_COLORS[sheet], linewidth=2.5, zorder=2)

    # Vertices
    for v_id, pos in V.items():
        ax.scatter(*pos, s=80, color='black', zorder=5)
        ax.text(pos[0]+0.05, pos[1]+0.05, pos[2]+0.1, f'$v_{{{v_id}}}$',
                fontsize=9)

    # Triangle ancilla positions (centroids)
    for name, verts in triangles:
        c = np.mean([V[v] for v in verts], axis=0)
        ax.scatter(*c, s=140, marker='*', color='red',
                   zorder=6, edgecolor='black', linewidths=0.8)
        ax.text(c[0]+0.05, c[1]+0.05, c[2]+0.1, name,
                fontsize=9, color='darkred')

    patches = [mpatches.Patch(color=SHEET_COLORS[s], label=f'$S_{{{s}}}$')
               for s in ['xy', 'xz', 'yz']]
    patches.append(plt.Line2D([0], [0], marker='*', color='w',
                              markerfacecolor='red', markeredgecolor='black',
                              markersize=10, linestyle='', label='triangle ancilla'))
    ax.legend(handles=patches, loc='upper left', fontsize=9,
              bbox_to_anchor=(0.0, 1.0))

    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.view_init(elev=15, azim=-55)
    ax.set_title('Surgery primitive at $L=4$: 4 triangles spanning two sheets\n'
                 '(weight-8 joint $Z_A\\otimes Z_B$ measurement)',
                 fontsize=10, pad=8)
    plt.tight_layout()
    plt.savefig('fig2_primitive.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("  fig2_primitive.pdf")


# ---------- Figure 3: threshold log-log plot ----------

def fig_threshold():
    # Data from gate4_levelC simulation (n_shots=30000, MWPM via PyMatching)
    p_values = np.array([0.001, 0.002, 0.003, 0.005, 0.008])
    distances = [3, 5, 7, 9]

    # Standard memory (d rounds)
    std_rates = {
        3: [3.7e-4, 1.3e-3, 4.03e-3, 1.123e-2, 2.543e-2],
        5: [7e-5,   6.7e-4, 1.63e-3, 6.23e-3,  2.537e-2],
        7: [1e-5,   1.0e-4, 7.7e-4,  3.47e-3,  2.42e-2],
        9: [1e-5,   1e-5,   3.0e-4,  2.50e-3,  2.107e-2],
    }
    # Surgery-extended (3d rounds)
    surg_rates = {
        3: [1.4e-3, 4.13e-3, 1.103e-2, 2.833e-2, 6.25e-2],
        5: [2.0e-4, 1.9e-3,  5.37e-3,  2.003e-2, 7.683e-2],
        7: [1e-5,   3.3e-4,  1.57e-3,  1.303e-2, 7.377e-2],
        9: [1e-5,   7e-5,    3.3e-4,   7.37e-3,  7.127e-2],
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(distances)))

    for ax, rates, title in [
        (ax1, std_rates, 'Standard memory ($d$ rounds)'),
        (ax2, surg_rates, 'Surgery-extended ($3d$ rounds)')]:
        for i, d in enumerate(distances):
            ax.loglog(p_values * 100, rates[d], 'o-',
                      color=colors[i], label=f'$d = {d}$',
                      markersize=7, linewidth=1.6)
        ax.set_xlabel('Physical error rate $p$ (%)', fontsize=11)
        ax.set_ylabel('Logical error rate $p_L$', fontsize=11)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(loc='lower right', fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(8e-6, 0.3)

    fig.suptitle('Threshold simulation: surface code memory experiments under '
                 'circuit-level depolarizing noise',
                 fontsize=12, y=1.03)
    plt.tight_layout()
    plt.savefig('fig3_threshold.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("  fig3_threshold.pdf")


# ---------- Figure 4: code comparison ----------

def fig_comparison():
    """Compare codes on the most important axes: connectivity, cross-block gates,
    and logical qubits per total qubit at a fixed distance d=8."""
    codes = [
        ('Surface (rotated)',    1, 'd=8',  '4',  'Within patch'),
        ('2D toric',              2, 'd=8',  '4+wraps', 'Within patch'),
        ('3D toric',              3, 'L=8',  '6',  'Limited'),
        ('Gross BB',             12, 'd=12', '6+L', 'Open problem'),
        ('Sheet code, 1 sheet (planar)',   8, 'L=8',  '4',  'Yes (triangles)'),
        ('Sheet code, 1 sheet (toric)',   16, 'L=8',  '4+wraps', 'Yes (triangles)'),
        ('Sheet code, 3 sheets (toric)',  48, 'L=8',  '4+wraps', 'Yes (triangles)'),
    ]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    names = [c[0] for c in codes]
    k_vals = [c[1] for c in codes]

    bar_colors = ['#888888', '#888888', '#888888', '#cc6677',
                  '#117733', '#117733', '#117733']
    bars = ax.barh(range(len(codes)), k_vals, color=bar_colors,
                   edgecolor='black', linewidth=0.5)

    # Annotations: K and cross-block gate availability
    for i, (name, k, d, K, xblock) in enumerate(codes):
        ax.text(k + max(k_vals)*0.012, i, 
                f'  $K{{=}}{K}$  |  cross-block: {xblock}',
                va='center', fontsize=9)

    ax.set_yticks(range(len(codes)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('Logical qubits $k$', fontsize=11)
    ax.set_xlim(0, max(k_vals) * 1.85)
    ax.set_title('Code comparison at distance $\\approx 8{-}12$',
                 fontsize=11)
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)

    # Highlight sheet code rows
    for i in range(len(codes)):
        if 'Sheet' in names[i]:
            ax.get_yticklabels()[i].set_fontweight('bold')

    plt.tight_layout()
    plt.savefig('fig4_comparison.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("  fig4_comparison.pdf")


if __name__ == '__main__':
    print("Generating figures...")
    fig_triangle()
    fig_primitive()
    fig_threshold()
    fig_comparison()
    print("\nAll figures generated.")

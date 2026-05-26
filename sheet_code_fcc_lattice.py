"""FCC sheet code: lattice construction and stabilizers.

This module builds the FCC lattice with all 3 triad sheets, enumerates edges
and triangles, and constructs the per-sheet stabilizer matrices.
"""
import numpy as np
from itertools import product


# Nearest-neighbor displacement vectors of the FCC lattice, grouped by triad sheet.
NN_DISPLACEMENTS = {
    'xy': [(1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0)],
    'xz': [(1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1)],
    'yz': [(0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)],
}

SHEETS = ['xy', 'xz', 'yz']


def build_fcc_lattice(L):
    """Build the FCC lattice on an L x L x L torus.

    Args:
        L: linear size (even L gives the toric variant of the sheet code)

    Returns:
        vertices: list of (x, y, z) vertex positions (x+y+z even)
        vidx: dict mapping position -> vertex index
        edges: dict {sheet: list of (i, j) edge endpoint pairs}
        edge_list: flat list of all edges, ordered by sheet
        edge_to_idx: dict mapping (i, j) -> flat edge index
        edge_to_sheet: list, same length as edge_list, of sheet labels
    """
    vertices = []
    vidx = {}
    for x, y, z in product(range(L), repeat=3):
        if (x + y + z) % 2 == 0:
            vidx[(x, y, z)] = len(vertices)
            vertices.append((x, y, z))

    edges = {s: set() for s in SHEETS}
    for vp in vertices:
        i = vidx[vp]
        x, y, z = vp
        for sheet, dirs in NN_DISPLACEMENTS.items():
            for dx, dy, dz in dirs:
                wp = ((x + dx) % L, (y + dy) % L, (z + dz) % L)
                j = vidx[wp]
                edges[sheet].add((min(i, j), max(i, j)))
    edges = {k: sorted(v) for k, v in edges.items()}

    edge_list = []
    edge_to_sheet = []
    edge_to_idx = {}
    for sheet in SHEETS:
        for e in edges[sheet]:
            edge_to_idx[e] = len(edge_list)
            edge_list.append(e)
            edge_to_sheet.append(sheet)

    return vertices, vidx, edges, edge_list, edge_to_idx, edge_to_sheet


def vertex_Z_stabilizers(L):
    """For each (vertex, sheet) pair, build the weight-4 vertex Z-stabilizer
    matrix (rows) over all data qubits (columns)."""
    vertices, vidx, edges, edge_list, edge_to_idx, _ = build_fcc_lattice(L)
    n_edges = len(edge_list)
    rows = []
    meta = []
    for vp in vertices:
        i = vidx[vp]
        x, y, z = vp
        for sheet, dirs in NN_DISPLACEMENTS.items():
            row = np.zeros(n_edges, dtype=np.int8)
            for dx, dy, dz in dirs:
                wp = ((x + dx) % L, (y + dy) % L, (z + dz) % L)
                j = vidx[wp]
                e = (min(i, j), max(i, j))
                row[edge_to_idx[e]] = 1
            rows.append(row)
            meta.append((sheet, vp))
    return np.array(rows, dtype=np.int8), meta


def oct_void_X_stabilizers(L):
    """For each (oct void, sheet) pair, build the weight-4 octahedral X-stabilizer
    matrix. Octahedral void centers are at FCC sites with x+y+z odd; each oct
    void has 6 surrounding vertices, and the 4 sheet edges in the void connect
    pairs whose difference lies in that sheet."""
    vertices, vidx, edges, edge_list, edge_to_idx, _ = build_fcc_lattice(L)
    n_edges = len(edge_to_idx)
    oct_centers = [(x, y, z) for x, y, z in product(range(L), repeat=3)
                   if (x + y + z) % 2 == 1]

    rows = []
    meta = []
    for c in oct_centers:
        cx, cy, cz = c
        for sheet in SHEETS:
            if sheet == 'xy':
                pairs = [((cx+1, cy, cz), (cx, cy+1, cz)),
                         ((cx+1, cy, cz), (cx, cy-1, cz)),
                         ((cx-1, cy, cz), (cx, cy+1, cz)),
                         ((cx-1, cy, cz), (cx, cy-1, cz))]
            elif sheet == 'xz':
                pairs = [((cx+1, cy, cz), (cx, cy, cz+1)),
                         ((cx+1, cy, cz), (cx, cy, cz-1)),
                         ((cx-1, cy, cz), (cx, cy, cz+1)),
                         ((cx-1, cy, cz), (cx, cy, cz-1))]
            else:  # 'yz'
                pairs = [((cx, cy+1, cz), (cx, cy, cz+1)),
                         ((cx, cy+1, cz), (cx, cy, cz-1)),
                         ((cx, cy-1, cz), (cx, cy, cz+1)),
                         ((cx, cy-1, cz), (cx, cy, cz-1))]
            row = np.zeros(n_edges, dtype=np.int8)
            for ap, bp in pairs:
                ap = (ap[0] % L, ap[1] % L, ap[2] % L)
                bp = (bp[0] % L, bp[1] % L, bp[2] % L)
                ai, bi = vidx[ap], vidx[bp]
                e = (min(ai, bi), max(ai, bi))
                if e in edge_to_idx:
                    row[edge_to_idx[e]] = 1
            rows.append(row)
            meta.append((sheet, c))

    return np.array(rows, dtype=np.int8), meta


def fcc_triangles(L):
    """Enumerate all FCC triangles (3-cycles in the FCC graph)."""
    from itertools import combinations
    vertices, vidx, _, edge_list, edge_to_idx, _ = build_fcc_lattice(L)
    nn_all = []
    for dirs in NN_DISPLACEMENTS.values():
        nn_all.extend(dirs)
    nn_set = set(nn_all)

    triangles = set()
    for vp in vertices:
        i = vidx[vp]
        x, y, z = vp
        nbrs = []
        for dx, dy, dz in nn_all:
            wp = ((x + dx) % L, (y + dy) % L, (z + dz) % L)
            nbrs.append((vidx[wp], wp))
        for a, b in combinations(range(len(nbrs)), 2):
            ja, ap = nbrs[a]
            jb, bp = nbrs[b]
            diff = ((bp[0] - ap[0]) % L,
                    (bp[1] - ap[1]) % L,
                    (bp[2] - ap[2]) % L)
            diff_signed = tuple((d if d <= L // 2 else d - L) for d in diff)
            if diff_signed in nn_set or tuple(-x for x in diff_signed) in nn_set:
                triangles.add(tuple(sorted([i, ja, jb])))

    triangle_list = sorted(triangles)
    n_edges = len(edge_list)
    B = np.zeros((len(triangle_list), n_edges), dtype=np.int8)
    for t_idx, tri in enumerate(triangle_list):
        v1, v2, v3 = tri
        for ii, jj in [(v1, v2), (v1, v3), (v2, v3)]:
            e = (min(ii, jj), max(ii, jj))
            if e in edge_to_idx:
                B[t_idx, edge_to_idx[e]] = 1
    return triangle_list, B

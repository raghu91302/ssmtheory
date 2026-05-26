"""Cross-sheet triangle surgery on the FCC sheet code."""
import numpy as np
from sheet_code_fcc_lattice import (build_fcc_lattice, vertex_Z_stabilizers,
                          oct_void_X_stabilizers, fcc_triangles, SHEETS)
from sheet_code_gf2 import gf2_rank, gf2_kernel, gf2_rref, gf2_row_span


def per_sheet_Z_logicals(L, target_sheet):
    """Compute Z-logical operators of one sheet."""
    _, _, edges_by_sheet, _, _, edge_to_sheet = build_fcc_lattice(L)
    sheet_arr = np.array(
        [{'xy': 0, 'xz': 1, 'yz': 2}[s] for s in edge_to_sheet],
        dtype=np.int8)
    sheet_idx = {'xy': 0, 'xz': 1, 'yz': 2}[target_sheet]
    sheet_cols = np.where(sheet_arr == sheet_idx)[0]

    HX_full, _ = oct_void_X_stabilizers(L)
    HZ_full, _ = vertex_Z_stabilizers(L)

    sheet_X = []
    sheet_Z = []
    for row in HX_full:
        nonzero = np.where(row == 1)[0]
        if len(nonzero) > 0 and np.all(sheet_arr[nonzero] == sheet_idx):
            sheet_X.append(row[sheet_cols])
    for row in HZ_full:
        nonzero = np.where(row == 1)[0]
        if len(nonzero) > 0 and np.all(sheet_arr[nonzero] == sheet_idx):
            sheet_Z.append(row[sheet_cols])
    HX_sheet = np.array(sheet_X, dtype=np.int8)
    HZ_sheet = np.array(sheet_Z, dtype=np.int8)

    ker = gf2_kernel(HX_sheet)
    basis = []
    current = HZ_sheet.copy()
    current_rank = gf2_rank(current)
    for v in ker:
        test = np.vstack([current, v.reshape(1, -1)])
        if gf2_rank(test) > current_rank:
            current = test
            current_rank += 1
            basis.append(v)
    return np.array(basis, dtype=np.int8), sheet_cols


def triangle_reachable_logicals(L):
    """Compute the space of cross-sheet Z-logicals reachable via triangle products."""
    HX_full, _ = oct_void_X_stabilizers(L)
    HZ_full, _ = vertex_Z_stabilizers(L)
    _, B = fcc_triangles(L)

    M_commute = (HX_full @ B.T) % 2
    valid_sets = gf2_kernel(M_commute)

    rank_HZ = gf2_rank(HZ_full)
    basis = HZ_full.copy()
    basis_rank = rank_HZ
    triangle_logicals = []
    triangle_sets = []
    for v in valid_sets:
        candidate = (v @ B) % 2
        test = np.vstack([basis, candidate.reshape(1, -1)])
        if gf2_rank(test) > basis_rank:
            basis = test
            basis_rank += 1
            triangle_logicals.append(candidate)
            triangle_sets.append(v)
    return (np.array(triangle_logicals, dtype=np.int8),
            np.array(triangle_sets, dtype=np.int8))


def decompose_by_sheet(operator, L):
    """Decompose an operator's support by sheet and project onto per-sheet
    Z-logical bases."""
    _, _, _, _, _, edge_to_sheet = build_fcc_lattice(L)
    sheet_arr = np.array(
        [{'xy': 0, 'xz': 1, 'yz': 2}[s] for s in edge_to_sheet],
        dtype=np.int8)
    HZ_full, _ = vertex_Z_stabilizers(L)
    sheet_Z_logs = {}
    sheet_stabs = {}
    for sheet in SHEETS:
        logs, _ = per_sheet_Z_logicals(L, sheet)
        sheet_Z_logs[sheet] = logs
        sheet_idx = {'xy': 0, 'xz': 1, 'yz': 2}[sheet]
        sheet_cols = np.where(sheet_arr == sheet_idx)[0]
        sheet_stabs[sheet] = np.array(
            [row[sheet_cols] for row in HZ_full
             if np.all(sheet_arr[np.where(row == 1)[0]] == sheet_idx)],
            dtype=np.int8)

    decomposition = {}
    for sheet in SHEETS:
        sheet_idx = {'xy': 0, 'xz': 1, 'yz': 2}[sheet]
        sheet_cols = np.where(sheet_arr == sheet_idx)[0]
        op_on_sheet = operator[sheet_cols]
        if np.all(op_on_sheet == 0):
            decomposition[sheet] = None
            continue
        stabs = sheet_stabs[sheet]
        logs = sheet_Z_logs[sheet]
        full_basis = np.vstack([stabs, logs])
        rows, cols = full_basis.T.shape
        aug = np.hstack([full_basis.T, op_on_sheet.reshape(-1, 1)])
        aug_rref, pivots, rank = gf2_rref(aug)
        if cols in pivots:
            decomposition[sheet] = ('inconsistent', None)
            continue
        x = np.zeros(cols, dtype=np.int8)
        for i, p in enumerate(pivots):
            x[p] = aug_rref[i, cols]
        decomposition[sheet] = x[stabs.shape[0]:]
    return decomposition


def verify_merge_distance(L):
    """Verify that a minimum-weight cross-sheet surgery preserves code distance."""
    HX_full, _ = oct_void_X_stabilizers(L)
    HZ_full, _ = vertex_Z_stabilizers(L)

    triangle_logicals, _ = triangle_reachable_logicals(L)
    if triangle_logicals.shape[0] == 0:
        return {'ok': False, 'reason': 'No triangle-reachable logicals found'}

    weights = triangle_logicals.sum(axis=1)
    Op = triangle_logicals[int(np.argmin(weights))]
    op_weight = int(weights.min())

    syndrome = (HX_full @ Op) % 2
    if np.any(syndrome != 0):
        return {'ok': False, 'reason': 'Op anticommutes with X-stabilizers'}

    HZ_merged = np.vstack([HZ_full, Op.reshape(1, -1)])
    rank_HZ_merged = gf2_rank(HZ_merged)
    rank_HX = gf2_rank(HX_full)
    n_edges = HX_full.shape[1]
    k_post = n_edges - rank_HZ_merged - rank_HX
    k_pre = n_edges - gf2_rank(HZ_full) - rank_HX

    all_logicals = []
    for sheet in SHEETS:
        logs, sheet_cols = per_sheet_Z_logicals(L, sheet)
        for log in logs:
            embedded = np.zeros(n_edges, dtype=np.int8)
            for i, col in enumerate(sheet_cols):
                if log[i] == 1:
                    embedded[col] = 1
            all_logicals.append(embedded)
    all_logicals = np.array(all_logicals, dtype=np.int8)

    min_class_weight = min(
        min(int(np.sum(g)), int(np.sum((g + Op) % 2)))
        for g in all_logicals)

    return {
        'ok': k_post == k_pre - 1 and min_class_weight >= L,
        'k_pre': k_pre,
        'k_post': k_post,
        'op_weight': op_weight,
        'min_class_weight': min_class_weight,
        'L': L,
    }

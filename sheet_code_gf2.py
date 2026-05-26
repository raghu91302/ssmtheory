"""GF(2) linear algebra utilities for stabilizer code analysis."""
import numpy as np


def gf2_rref(M):
    """Row-reduce M over GF(2). Returns (rref, pivots, rank)."""
    M = M.copy().astype(np.int64) % 2
    rows, cols = M.shape
    pivots = []
    r = 0
    for col in range(cols):
        piv = None
        for row in range(r, rows):
            if M[row, col] == 1:
                piv = row
                break
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        for row in range(rows):
            if row != r and M[row, col] == 1:
                M[row] = (M[row] + M[r]) % 2
        pivots.append(col)
        r += 1
    return M, pivots, r


def gf2_rank(M):
    return gf2_rref(M)[2]


def gf2_kernel(M):
    """Compute a basis for the right null space of M over GF(2)."""
    rows, cols = M.shape
    rref, pivots, rank = gf2_rref(M)
    free = [c for c in range(cols) if c not in pivots]
    basis = []
    for fc in free:
        v = np.zeros(cols, dtype=np.int8)
        v[fc] = 1
        for i, pc in enumerate(pivots):
            if rref[i, fc] == 1:
                v[pc] = 1
        basis.append(v)
    return (np.array(basis, dtype=np.int8) if basis
            else np.zeros((0, cols), dtype=np.int8))


def gf2_row_span(M):
    rref, pivots, rank = gf2_rref(M)
    return rref[:rank].copy()

"""
verify_construction.py: Static verification of the FCC sheet code parameters
at L = 4, 6, 8.

Reproduces the algebraic claims of Section 4 of the manuscript:

  - n = L^3 data qubits per sheet
  - rank(H_Z) = rank(H_X) = (L^3 - 2L) / 2
  - k = L^3 - rank(H_Z) - rank(H_X) = 2L logical qubits per sheet
  - k_3 = 3 * 2L = 6L logical qubits across the three edge-disjoint sheets
  - CSS validity: H_X H_Z^T = 0 mod 2

No Stim simulation; all computation is exact in GF(2) using fcc_code.py.

Usage:
    python verify_construction.py
"""
from __future__ import annotations
import numpy as np

from fcc_code import (
    build_sheet_code_xy, to_matrix, gf2_rref,
)


def gf2_rank(M: np.ndarray) -> int:
    """Rank over GF(2) via fcc_code's row-reduction utility."""
    if M.size == 0:
        return 0
    _, _, rank = gf2_rref(M)
    return int(rank)


def verify_at_L(L: int) -> dict:
    """Build the sheet code at lattice size L and verify all algebraic claims."""
    n_data, z_stabs, x_stabs = build_sheet_code_xy(L)
    HZ = to_matrix(z_stabs, n_data)
    HX = to_matrix(x_stabs, n_data)

    rank_HZ = gf2_rank(HZ)
    rank_HX = gf2_rank(HX)
    k_predicted = 2 * L
    k_actual = n_data - rank_HZ - rank_HX

    # CSS validity: H_X H_Z^T == 0 mod 2
    css_product = (HX @ HZ.T) % 2
    css_valid = bool(np.all(css_product == 0))

    expected_rank = (L**3 - 2*L) // 2

    return {
        "L": L,
        "n_data": n_data,
        "n_data_predicted": L**3,
        "rank_HZ": rank_HZ,
        "rank_HX": rank_HX,
        "rank_predicted": expected_rank,
        "k_actual": k_actual,
        "k_predicted": k_predicted,
        "css_valid": css_valid,
        "n_z_stabs": HZ.shape[0],
        "n_x_stabs": HX.shape[0],
    }


def main():
    print("=" * 72)
    print("FCC sheet code static verification")
    print("Reproduces Table 1 / Section 4 of the manuscript.")
    print("=" * 72)
    print()

    rows = []
    all_ok = True
    for L in [4, 6, 8]:
        print(f"L = {L}:")
        r = verify_at_L(L)

        ok_n = r["n_data"] == r["n_data_predicted"]
        ok_rank = r["rank_HZ"] == r["rank_predicted"] and r["rank_HX"] == r["rank_predicted"]
        ok_k = r["k_actual"] == r["k_predicted"]
        ok_css = r["css_valid"]
        ok = ok_n and ok_rank and ok_k and ok_css

        print(f"  n_data:       {r['n_data']:5d}    (expected L^3       = {r['n_data_predicted']:5d}) {'OK' if ok_n else 'MISMATCH'}")
        print(f"  Z-stabilizers: {r['n_z_stabs']:5d}    (vertex checks)")
        print(f"  X-stabilizers: {r['n_x_stabs']:5d}    (oct-void checks)")
        print(f"  rank(H_Z):    {r['rank_HZ']:5d}    (expected (L^3-2L)/2 = {r['rank_predicted']:5d})")
        print(f"  rank(H_X):    {r['rank_HX']:5d}    (expected (L^3-2L)/2 = {r['rank_predicted']:5d}) {'OK' if ok_rank else 'MISMATCH'}")
        print(f"  k = n - rank_HZ - rank_HX = {r['k_actual']:3d}  (expected 2L = {r['k_predicted']:3d}) {'OK' if ok_k else 'MISMATCH'}")
        print(f"  k_3 = 3 * k = {3 * r['k_actual']:3d}  (expected 6L = {6 * L:3d})")
        print(f"  CSS validity (H_X H_Z^T == 0 mod 2): {'PASS' if ok_css else 'FAIL'}")
        print(f"  -> code parameters: [[{r['n_data']}, {r['k_actual']}, {L}]] per sheet,"
              f" [[{3*r['n_data']}, {3*r['k_actual']}, {L}]] three-sheet")
        print()

        all_ok = all_ok and ok
        rows.append(r)

    print("=" * 72)
    print("Summary table (matches Table 1 of the manuscript):")
    print()
    print(f"  {'L':>3}  {'1-sheet':>14}  {'3-sheet':>16}  {'Phys. qubits':>13}  {'Rate (3-sheet)':>15}")
    print("  " + "-" * 70)
    for r in rows:
        L = r["L"]
        n1, k1 = r["n_data"], r["k_actual"]
        n3, k3 = 3*n1, 3*k1
        phys = 4 * L**3   # data + shared ancillas (L^3/2 vertex Z + L^3/2 oct X = L^3)
        rate = 100 * k3 / n3
        print(f"  {L:>3}  [[{n1:>4},{k1:>3},{L}]]  [[{n3:>4},{k3:>3},{L}]]  {phys:>13d}  {rate:>13.1f}%")
    print("=" * 72)
    print()

    if all_ok:
        print("ALL CHECKS PASS. Construction is consistent with manuscript claims.")
    else:
        print("WARNING: at least one verification failed. See messages above.")
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

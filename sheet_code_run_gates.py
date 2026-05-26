"""Reproduce Section 4: triangle algebra and cross-sheet logicals."""
import numpy as np
from sheet_code_fcc_lattice import build_fcc_lattice
from sheet_code_surgery import (triangle_reachable_logicals, decompose_by_sheet,
                      per_sheet_Z_logicals)
from sheet_code_gf2 import gf2_rank


def main():
    for L in [4, 6]:
        print(f"\n{'='*70}")
        print(f"  Triangle algebra at L = {L}")
        print(f"{'='*70}\n")

        logicals, sets = triangle_reachable_logicals(L)
        print(f"  Triangle-reachable cross-sheet Z-logicals: {logicals.shape[0]}")
        print(f"  Expected (6L - 3): {6*L - 3}")

        _, _, _, _, _, edge_to_sheet = build_fcc_lattice(L)
        sheet_arr = np.array(
            [{'xy': 0, 'xz': 1, 'yz': 2}[s] for s in edge_to_sheet])

        n_2sheet = 0
        for op in logicals:
            nonzero_sheets = set()
            for sheet, idx in [('xy', 0), ('xz', 1), ('yz', 2)]:
                if np.any(op[sheet_arr == idx] == 1):
                    nonzero_sheets.add(sheet)
            if len(nonzero_sheets) == 2:
                n_2sheet += 1
        print(f"  2-sheet operators: {n_2sheet}/{logicals.shape[0]}")

        print(f"\n  Per-sheet coverage:")
        for target in ['xy', 'xz', 'yz']:
            decomps_by_partner = {'xy': [], 'xz': [], 'yz': []}
            for op in logicals:
                d = decompose_by_sheet(op, L)
                nonzero = [s for s in d if d[s] is not None
                           and not np.all(d[s] == 0)]
                if target in nonzero:
                    for partner in nonzero:
                        if partner != target:
                            decomps_by_partner[partner].append(d[target])
            all_combined = []
            for p in ['xy', 'xz', 'yz']:
                if p != target and decomps_by_partner[p]:
                    all_combined.extend(decomps_by_partner[p])
            union_rank = (gf2_rank(np.array(all_combined, dtype=np.int8))
                          if all_combined else 0)
            print(f"    sheet {target}: union coverage = {union_rank}/{2*L}")


if __name__ == '__main__':
    main()

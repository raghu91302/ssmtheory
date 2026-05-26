"""Verification tests for the sheet code construction."""
import numpy as np
from sheet_code_fcc_lattice import (build_fcc_lattice, vertex_Z_stabilizers,
                          oct_void_X_stabilizers, fcc_triangles)
from sheet_code_gf2 import gf2_rank
from sheet_code_surgery import triangle_reachable_logicals, verify_merge_distance


def test_lattice_size():
    L = 4
    _, _, edges, edge_list, _, _ = build_fcc_lattice(L)
    assert len(edge_list) == 3 * L**3
    print(f"  ok Edge count at L={L}: 3 L^3 = {3*L**3}")


def test_css_condition():
    L = 4
    HX, _ = oct_void_X_stabilizers(L)
    HZ, _ = vertex_Z_stabilizers(L)
    product = (HX @ HZ.T) % 2
    assert np.all(product == 0)
    print(f"  ok CSS condition at L={L}: H_X H_Z^T = 0")


def test_code_parameters():
    for L in [4, 6]:
        HX, _ = oct_void_X_stabilizers(L)
        HZ, _ = vertex_Z_stabilizers(L)
        n = HX.shape[1]
        k = n - gf2_rank(HX) - gf2_rank(HZ)
        assert k == 6 * L
        print(f"  ok L={L}: n={n}, k={k} (= 6L across 3 sheets)")


def test_triangle_count():
    L = 4
    tris, B = fcc_triangles(L)
    assert len(tris) == 4 * L**3
    assert np.all(B.sum(axis=1) == 3)
    print(f"  ok Triangle count at L={L}: 4 L^3 = {4*L**3}")


def test_triangle_reachable():
    for L in [4, 6]:
        logicals, _ = triangle_reachable_logicals(L)
        assert logicals.shape[0] == 6*L - 3
        print(f"  ok L={L}: triangle-reachable Z-logicals = {6*L-3}")


def test_distance_preservation():
    for L in [4, 6]:
        result = verify_merge_distance(L)
        assert result['ok']
        assert result['k_post'] == result['k_pre'] - 1
        assert result['min_class_weight'] >= L
        print(f"  ok L={L}: merge preserves distance d={L} "
              f"(k: {result['k_pre']} -> {result['k_post']})")


if __name__ == '__main__':
    print("Running verification tests...\n")
    test_lattice_size()
    test_css_condition()
    test_code_parameters()
    test_triangle_count()
    test_triangle_reachable()
    test_distance_preservation()
    print("\nAll tests passed.")

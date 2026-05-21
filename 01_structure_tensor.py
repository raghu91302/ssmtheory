#!/usr/bin/env python3
"""
Verify the structure-tensor identities and FCC sub-lattice emergence
claimed in Section 2 of Part II.

Outputs:
  - 24 nearest-neighbor displacement vectors of D4
  - Structure tensor S^{mu nu} = sum_j n_j^mu n_j^nu, claim: 12 * I_4
  - Odd-rank tensor T^{mu nu lambda}, claim: identically zero
  - Bond partition after picking coord 0 as time: 12 + 12 spatial/time-mixed
  - Spatial 12 should equal the FCC nearest-neighbor displacements
"""
import numpy as np


def gen_d4_nn():
    """Return the 24 D4 nearest-neighbor displacement vectors as a 24x4 array."""
    nn = []
    for i in range(4):
        for j in range(i + 1, 4):
            for si in (-1, 1):
                for sj in (-1, 1):
                    d = [0, 0, 0, 0]
                    d[i] = si
                    d[j] = sj
                    nn.append(d)
    return np.array(nn)


def gen_fcc_nn():
    """Return the 12 FCC nearest-neighbor displacement vectors as a 12x3 array."""
    nn = []
    for i in range(3):
        for j in range(i + 1, 3):
            for si in (-1, 1):
                for sj in (-1, 1):
                    d = [0, 0, 0]
                    d[i] = si
                    d[j] = sj
                    nn.append(d)
    return np.array(nn)


def main():
    NN = gen_d4_nn()
    print(f"D4 NN count: {len(NN)}   [expected 24]")
    assert len(NN) == 24

    # Distances should all be sqrt(2)
    norms = np.linalg.norm(NN, axis=1)
    assert np.allclose(norms, np.sqrt(2)), "All NN should be at distance sqrt(2)"
    print(f"All NN at distance sqrt(2): True")

    # Structure tensor
    S = NN.T @ NN
    print(f"\nStructure tensor S^(mu nu):\n{S}")
    target = 12 * np.eye(4, dtype=int)
    print(f"Equals 12 I_4 exactly: {np.array_equal(S, target)}")
    assert np.array_equal(S, target)

    # Odd-rank tensor (centrosymmetry)
    T = np.einsum('ji,jk,jl->ikl', NN, NN, NN)
    print(f"\nT^(mu nu lambda) max abs: {np.abs(T).max()}")
    assert np.abs(T).max() == 0
    print(f"T identically zero: True")

    # Bond partition with coord 0 = time
    is_spatial = (NN[:, 0] == 0)
    spatial = NN[is_spatial]
    time_mixed = NN[~is_spatial]
    print(f"\nWith coord 0 = time:")
    print(f"  Spatial bonds (zero time component): {len(spatial)}   [expected 12]")
    print(f"  Time-mixed bonds                   : {len(time_mixed)}   [expected 12]")
    assert len(spatial) == 12 and len(time_mixed) == 12

    # Spatial 12 should match FCC
    fcc = gen_fcc_nn()
    spatial_set = {tuple(v) for v in spatial[:, 1:].tolist()}
    fcc_set = {tuple(v) for v in fcc.tolist()}
    match = (spatial_set == fcc_set)
    print(f"  Spatial 12 equal FCC NN: {match}")
    assert match

    print("\nAll structure-tensor checks passed.")


if __name__ == '__main__':
    main()

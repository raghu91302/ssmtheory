"""Custom Stim circuit (v2): separate Z and X stab measurement phases.

Fixes the parallel-CNOT scheduling conflict from v1 by measuring all Z-stabs first,
then all X-stabs, in separate time blocks within each syndrome round.
"""
import sys
import numpy as np
import stim

from sheet_code_fcc_lattice import (build_fcc_lattice, vertex_Z_stabilizers,
                                     oct_void_X_stabilizers)


def build_single_sheet_circuit(L, sheet, d_rounds, p, basis='Z'):
    """Memory experiment circuit for one sheet, with corrected scheduling."""
    sheet_idx = {'xy': 0, 'xz': 1, 'yz': 2}[sheet]
    vertices, vidx, edges, edge_list, edge_to_idx, edge_to_sheet = build_fcc_lattice(L)
    sheet_arr = np.array([{'xy': 0, 'xz': 1, 'yz': 2}[s] for s in edge_to_sheet])
    sheet_cols = np.where(sheet_arr == sheet_idx)[0]
    n_data = len(sheet_cols)
    sheet_edge_global_to_local = {c: i for i, c in enumerate(sheet_cols)}
    
    HZ_full, Z_meta = vertex_Z_stabilizers(L)
    HX_full, X_meta = oct_void_X_stabilizers(L)
    
    sheet_Z_stabs = []
    for row, (s_label, pos) in zip(HZ_full, Z_meta):
        if s_label != sheet:
            continue
        nonzero = np.where(row == 1)[0]
        if len(nonzero) == 0:
            continue
        local = [sheet_edge_global_to_local[c] for c in nonzero]
        sheet_Z_stabs.append((local, pos))
    
    sheet_X_stabs = []
    for row, (s_label, pos) in zip(HX_full, X_meta):
        if s_label != sheet:
            continue
        nonzero = np.where(row == 1)[0]
        if len(nonzero) == 0:
            continue
        local = [sheet_edge_global_to_local[c] for c in nonzero]
        sheet_X_stabs.append((local, pos))
    
    n_Z = len(sheet_Z_stabs)
    n_X = len(sheet_X_stabs)
    Z_anc_offset = n_data
    X_anc_offset = n_data + n_Z
    
    c = stim.Circuit()
    
    all_data = list(range(n_data))
    all_Z_anc = list(range(Z_anc_offset, Z_anc_offset + n_Z))
    all_X_anc = list(range(X_anc_offset, X_anc_offset + n_X))
    
    if basis == 'Z':
        c.append('R', all_data + all_Z_anc + all_X_anc)
    else:
        c.append('RX', all_data)
        c.append('R', all_Z_anc + all_X_anc)
    if p > 0:
        c.append('DEPOLARIZE1', all_data + all_Z_anc + all_X_anc, p)
    
    for round_idx in range(d_rounds):
        if round_idx > 0:
            c.append('R', all_Z_anc + all_X_anc)
            if p > 0:
                c.append('DEPOLARIZE1', all_Z_anc + all_X_anc, p)
        
        # PHASE 1: Z-stab measurement (CX data→Z_anc, no H involved)
        for slot in range(4):  # weight-4 stabs need 4 slots
            cnots = []
            for stab_idx, (data_qs, _) in enumerate(sheet_Z_stabs):
                if slot < len(data_qs):
                    cnots.extend([data_qs[slot], Z_anc_offset + stab_idx])
            if cnots:
                c.append('CX', cnots)
                if p > 0:
                    c.append('DEPOLARIZE2', cnots, p)
                c.append('TICK')
        
        # Measure Z-ancs
        if p > 0:
            c.append('X_ERROR', all_Z_anc, p)
        c.append('M', all_Z_anc)
        
        # PHASE 2: X-stab measurement
        # Initialize X-ancs (already in |0> after R or freshly reset above)
        c.append('H', all_X_anc)
        if p > 0:
            c.append('DEPOLARIZE1', all_X_anc, p)
        c.append('TICK')
        
        for slot in range(4):
            cnots = []
            for stab_idx, (data_qs, _) in enumerate(sheet_X_stabs):
                if slot < len(data_qs):
                    cnots.extend([X_anc_offset + stab_idx, data_qs[slot]])
            if cnots:
                c.append('CX', cnots)
                if p > 0:
                    c.append('DEPOLARIZE2', cnots, p)
                c.append('TICK')
        
        c.append('H', all_X_anc)
        if p > 0:
            c.append('DEPOLARIZE1', all_X_anc, p)
        c.append('TICK')
        
        if p > 0:
            c.append('X_ERROR', all_X_anc, p)
        c.append('M', all_X_anc)
        
        # Detector annotations
        # Within one round, we measured: Z_ancs (n_Z), then X_ancs (n_X)
        # Most recent measurements: X_anc[n_X-1] at -1, ..., X_anc[0] at -n_X
        # Then Z_anc[n_Z-1] at -(n_X+1), ..., Z_anc[0] at -(n_X+n_Z)
        # Each round contributes n_Z+n_X measurements
        
        if round_idx > 0:
            # Z detectors: compare to previous round's Z measurements
            # Current Z_anc[i] at rec[-(n_X+n_Z) + i]
            # Previous Z_anc[i] at rec[-(n_X+n_Z) - (n_X+n_Z) + i] = rec[-2(n_X+n_Z) + i]
            for i in range(n_Z):
                c.append('DETECTOR', [
                    stim.target_rec(-(n_X + n_Z) + i),
                    stim.target_rec(-2 * (n_X + n_Z) + i)
                ])
            # X detectors: compare current X_anc[i] to previous X_anc[i]
            # Current X_anc[i] at rec[-n_X + i]
            # Previous X_anc[i] at rec[-(n_X+n_Z) - n_X + i] = rec[-(2n_X+n_Z) + i]
            for i in range(n_X):
                c.append('DETECTOR', [
                    stim.target_rec(-n_X + i),
                    stim.target_rec(-(2 * n_X + n_Z) + i)
                ])
        else:
            # Round 0: only Z detectors are deterministic for Z-basis init
            if basis == 'Z':
                for i in range(n_Z):
                    c.append('DETECTOR', [stim.target_rec(-(n_X + n_Z) + i)])
            else:
                for i in range(n_X):
                    c.append('DETECTOR', [stim.target_rec(-n_X + i)])
    
    # Final destructive measurement
    if basis == 'Z':
        if p > 0:
            c.append('X_ERROR', all_data, p)
        c.append('M', all_data)
        # Final detectors: compute Z-stab parity from data, compare to last Z-anc measurement
        for stab_idx, (data_qs, _) in enumerate(sheet_Z_stabs):
            data_recs = [stim.target_rec(-n_data + dq) for dq in data_qs]
            # Last Z-anc[stab_idx] is at rec[-n_data - (n_X+n_Z) + stab_idx]
            # (after this round's M of Z and X ancs, then M of data)
            anc_rec = stim.target_rec(-n_data - (n_X + n_Z) + stab_idx)
            c.append('DETECTOR', data_recs + [anc_rec])
    else:
        if p > 0:
            c.append('Z_ERROR', all_data, p)
        c.append('MX', all_data)
        for stab_idx, (data_qs, _) in enumerate(sheet_X_stabs):
            data_recs = [stim.target_rec(-n_data + dq) for dq in data_qs]
            # Last X-anc[stab_idx] is at rec[-n_data - n_X + stab_idx]
            anc_rec = stim.target_rec(-n_data - n_X + stab_idx)
            c.append('DETECTOR', data_recs + [anc_rec])
    
    # Logical observable: non-contractible cycle of length L in layer z=0
    if basis == 'Z' and sheet == 'xy':
        logical_edges = []
        z0 = 0
        # In rotated lattice for S_xy at z=0: vertices satisfy x+y even.
        # Find a horizontal cycle.
        for x_step in range(L):
            v1 = (x_step % L, (x_step % 2), z0)
            v2 = ((x_step + 1) % L, ((x_step + 1) % 2), z0)
            if (sum(v1) % 2 != 0) or (sum(v2) % 2 != 0):
                continue
            if v1 in vidx and v2 in vidx:
                i1, i2 = vidx[v1], vidx[v2]
                e = (min(i1, i2), max(i1, i2))
                if e in edge_to_idx:
                    global_col = edge_to_idx[e]
                    if global_col in sheet_edge_global_to_local:
                        logical_edges.append(sheet_edge_global_to_local[global_col])
        if len(logical_edges) == L:
            obs_recs = [stim.target_rec(-n_data + le) for le in logical_edges]
            c.append('OBSERVABLE_INCLUDE', obs_recs, 0)
    
    return c, n_data, n_Z, n_X


def test_v2():
    L = 4
    d_rounds = 4
    p = 0.001
    print(f"Building L={L}, d={d_rounds}, p={p} single-sheet circuit (v2)...")
    c, n_data, n_Z, n_X = build_single_sheet_circuit(L, 'xy', d_rounds, p, 'Z')
    
    print(f"  total qubits: {c.num_qubits}, measurements: {c.num_measurements}, "
          f"detectors: {c.num_detectors}, observables: {c.num_observables}")
    
    # Try to construct DEM
    try:
        dem = c.detector_error_model(decompose_errors=True)
        n_err = sum(1 for inst in dem if inst.type == 'error')
        print(f"  DEM constructed: {n_err} error mechanisms")
        return c, dem
    except Exception as e:
        msg = str(e)
        # Print just first line of error
        print(f"  DEM error: {msg.splitlines()[0]}")
        return c, None


if __name__ == '__main__':
    c, dem = test_v2()

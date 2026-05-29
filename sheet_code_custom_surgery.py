"""Surgery circuit v3: handle broken X-stabilizers during merge.

During merge rounds, the L individual triangle Z-measurements anticommute with
6L per-sheet X-stabilizers (Lemma 2 in the paper). Those X-stab detectors must
be SKIPPED during merge rounds because their values are randomized by the
triangle measurements.

After merge: re-initialize the broken X-stabs (their post-merge values follow
from the per-sheet codes' Z-eigenvalue, which we know from the data).

For a Z-memory experiment specifically, the logical observable is Z⊗Z on the
data, so:
- Z-stab detectors: always active (Z-stabs commute with triangle Z-measurements)
- X-stab detectors: skip the 12 "broken" X-stabs during merge rounds at L=4
- After merge: skip the broken X-stab DETECTORS for one round (they're random
  during the first post-merge round) then resume normally
"""
import sys
import numpy as np
import stim
import pymatching


from sheet_code_fcc_lattice import (build_fcc_lattice, vertex_Z_stabilizers,
                                     oct_void_X_stabilizers, fcc_triangles)
from sheet_code_gf2 import gf2_kernel, gf2_rank
from sheet_code_custom_surgery import find_surgery_primitive


def build_surgery_circuit_v3(L, d_each, p):
    """Surgery circuit with broken-X-stab handling."""
    vertices, vidx, edges, edge_list, edge_to_idx, edge_to_sheet = build_fcc_lattice(L)
    sheet_arr = np.array([{'xy': 0, 'xz': 1, 'yz': 2}[s] for s in edge_to_sheet])
    
    xy_cols = np.where(sheet_arr == 0)[0]
    xz_cols = np.where(sheet_arr == 1)[0]
    yz_cols = np.where(sheet_arr == 2)[0]
    n_xy, n_xz, n_yz = len(xy_cols), len(xz_cols), len(yz_cols)
    n_data = n_xz + n_yz + n_xy
    
    global_to_local = {}
    for i, c in enumerate(xz_cols): global_to_local[c] = i
    for i, c in enumerate(yz_cols): global_to_local[c] = n_xz + i
    for i, c in enumerate(xy_cols): global_to_local[c] = n_xz + n_yz + i
    
    HZ_full, Z_meta = vertex_Z_stabilizers(L)
    HX_full, X_meta = oct_void_X_stabilizers(L)
    
    sheet_Z_stabs = {'xz': [], 'yz': []}
    sheet_X_stabs = {'xz': [], 'yz': []}
    sheet_X_stabs_global_rows = {'xz': [], 'yz': []}  # keep global row index for break check
    
    for row_idx, (row, (s_label, pos)) in enumerate(zip(HZ_full, Z_meta)):
        if s_label not in sheet_Z_stabs: continue
        nonzero = np.where(row == 1)[0]
        if len(nonzero) == 0: continue
        local = [global_to_local[c] for c in nonzero]
        sheet_Z_stabs[s_label].append((local, pos))
    for row_idx, (row, (s_label, pos)) in enumerate(zip(HX_full, X_meta)):
        if s_label not in sheet_X_stabs: continue
        nonzero = np.where(row == 1)[0]
        if len(nonzero) == 0: continue
        local = [global_to_local[c] for c in nonzero]
        sheet_X_stabs[s_label].append((local, pos))
        sheet_X_stabs_global_rows[s_label].append(row_idx)
    
    n_Zxz = len(sheet_Z_stabs['xz']); n_Zyz = len(sheet_Z_stabs['yz'])
    n_Xxz = len(sheet_X_stabs['xz']); n_Xyz = len(sheet_X_stabs['yz'])
    
    # Find triangle primitive and determine which X-stabs are broken
    triangles_in_primitive, op_global, B = find_surgery_primitive(L)
    n_tri = len(triangles_in_primitive)
    triangle_data_qubits = []
    for ti in triangles_in_primitive:
        edges_in_T = np.where(B[ti] == 1)[0]
        data_qs = [global_to_local[c] for c in edges_in_T]
        triangle_data_qubits.append(data_qs)
    
    # Compute which X-stabs are broken by ANY triangle in the primitive
    broken_X_global = set()
    for ti in triangles_in_primitive:
        tri_Z_op = B[ti]
        overlap = (HX_full @ tri_Z_op) % 2
        for x_idx in np.where(overlap == 1)[0]:
            broken_X_global.add(x_idx)
    
    # Map broken X-stabs to indices within sheet_X_stabs['xz'] and ['yz']
    broken_X_local = {'xz': set(), 'yz': set()}
    for sheet_name in ['xz', 'yz']:
        for local_idx, global_row in enumerate(sheet_X_stabs_global_rows[sheet_name]):
            if global_row in broken_X_global:
                broken_X_local[sheet_name].add(local_idx)
    
    print(f"  Triangle primitive: {n_tri} triangles")
    print(f"  Broken X-stabs: {len(broken_X_local['xz'])} in xz, {len(broken_X_local['yz'])} in yz (total {len(broken_X_global)})")
    
    # Ancilla allocation
    Zxz_off = n_data
    Xxz_off = Zxz_off + n_Zxz
    Zyz_off = Xxz_off + n_Xxz
    Xyz_off = Zyz_off + n_Zyz
    Tri_off = Xyz_off + n_Xyz
    total_qubits = Tri_off + n_tri
    
    c = stim.Circuit()
    all_data = list(range(n_data))
    Zxz_ancs = list(range(Zxz_off, Zxz_off + n_Zxz))
    Xxz_ancs = list(range(Xxz_off, Xxz_off + n_Xxz))
    Zyz_ancs = list(range(Zyz_off, Zyz_off + n_Zyz))
    Xyz_ancs = list(range(Xyz_off, Xyz_off + n_Xyz))
    Tri_ancs = list(range(Tri_off, Tri_off + n_tri))
    all_per_sheet_ancs = Zxz_ancs + Xxz_ancs + Zyz_ancs + Xyz_ancs
    
    c.append('R', all_data + all_per_sheet_ancs + Tri_ancs)
    if p > 0:
        c.append('DEPOLARIZE1', all_data + all_per_sheet_ancs + Tri_ancs, p)
    
    # Track measurement order so we can refer to specific measurements later
    # Each round produces measurements in a fixed pattern:
    # phase 1: Zxz, Zyz, [Tri if merge]
    # phase 2: Xxz, Xyz
    # We track cumulative measurement counts to compute rec indices.
    
    measurement_records = []  # list of dicts {('Z', sheet, idx): m_idx, ('X', sheet, idx): m_idx, ('T', tri_idx): m_idx}
    total_measurements = 0
    
    def syndrome_round(round_idx, with_triangles):
        nonlocal total_measurements
        if round_idx > 0:
            c.append('R', all_per_sheet_ancs)
            if p > 0:
                c.append('DEPOLARIZE1', all_per_sheet_ancs, p)
            if with_triangles:
                c.append('R', Tri_ancs)
                if p > 0:
                    c.append('DEPOLARIZE1', Tri_ancs, p)
        
        # Phase 1: Z-stab CNOTs and triangle CNOTs (if merge)
        for slot in range(4):
            cnots = []
            for stab_list, anc_offset in [(sheet_Z_stabs['xz'], Zxz_off),
                                            (sheet_Z_stabs['yz'], Zyz_off)]:
                for stab_idx, (data_qs, _) in enumerate(stab_list):
                    if slot < len(data_qs):
                        cnots.extend([data_qs[slot], anc_offset + stab_idx])
            if with_triangles:
                for tri_idx, data_qs in enumerate(triangle_data_qubits):
                    if slot < len(data_qs):
                        cnots.extend([data_qs[slot], Tri_off + tri_idx])
            if cnots:
                c.append('CX', cnots)
                if p > 0:
                    c.append('DEPOLARIZE2', cnots, p)
                c.append('TICK')
        
        z_ancs_to_measure = Zxz_ancs + Zyz_ancs
        if with_triangles:
            z_ancs_to_measure = Zxz_ancs + Zyz_ancs + Tri_ancs
        if p > 0:
            c.append('X_ERROR', z_ancs_to_measure, p)
        c.append('M', z_ancs_to_measure)
        
        # Record measurement indices
        record = {}
        for i in range(n_Zxz):
            record[('Z', 'xz', i)] = total_measurements + i
        for i in range(n_Zyz):
            record[('Z', 'yz', i)] = total_measurements + n_Zxz + i
        if with_triangles:
            for i in range(n_tri):
                record[('T', i)] = total_measurements + n_Zxz + n_Zyz + i
        total_measurements += len(z_ancs_to_measure)
        
        # Phase 2: X-stab measurements
        c.append('H', Xxz_ancs + Xyz_ancs)
        if p > 0:
            c.append('DEPOLARIZE1', Xxz_ancs + Xyz_ancs, p)
        c.append('TICK')
        
        for slot in range(4):
            cnots = []
            for stab_list, anc_offset in [(sheet_X_stabs['xz'], Xxz_off),
                                            (sheet_X_stabs['yz'], Xyz_off)]:
                for stab_idx, (data_qs, _) in enumerate(stab_list):
                    if slot < len(data_qs):
                        cnots.extend([anc_offset + stab_idx, data_qs[slot]])
            if cnots:
                c.append('CX', cnots)
                if p > 0:
                    c.append('DEPOLARIZE2', cnots, p)
                c.append('TICK')
        
        c.append('H', Xxz_ancs + Xyz_ancs)
        if p > 0:
            c.append('DEPOLARIZE1', Xxz_ancs + Xyz_ancs, p)
        c.append('TICK')
        
        if p > 0:
            c.append('X_ERROR', Xxz_ancs + Xyz_ancs, p)
        c.append('M', Xxz_ancs + Xyz_ancs)
        
        for i in range(n_Xxz):
            record[('X', 'xz', i)] = total_measurements + i
        for i in range(n_Xyz):
            record[('X', 'yz', i)] = total_measurements + n_Xxz + i
        total_measurements += n_Xxz + n_Xyz
        
        measurement_records.append(record)
    
    def add_detectors_for_round(round_idx, prev_with_tri, curr_with_tri,
                                  prev_round_broken_X=False, curr_round_broken_X=False):
        """Add detectors for this round, handling broken X-stabs.
        
        prev_round_broken_X: True if the previous round was a merge round (and thus X-stab values may have changed due to triangle measurements)
        curr_round_broken_X: True if current round is a merge round
        """
        curr_records = measurement_records[round_idx]
        prev_records = measurement_records[round_idx - 1] if round_idx > 0 else None
        
        # Z-stab detectors (always active, both sheets)
        for sheet_name, n_Z in [('xz', n_Zxz), ('yz', n_Zyz)]:
            for i in range(n_Z):
                curr_m = curr_records[('Z', sheet_name, i)]
                if round_idx == 0:
                    # Deterministic in Z-basis
                    rec_idx = curr_m - total_measurements
                    c.append('DETECTOR', [stim.target_rec(rec_idx)])
                else:
                    prev_m = prev_records[('Z', sheet_name, i)]
                    c.append('DETECTOR', [stim.target_rec(curr_m - total_measurements),
                                           stim.target_rec(prev_m - total_measurements)])
        
        # X-stab detectors: skip broken ones if any round in between disrupted them
        if round_idx > 0:
            for sheet_name, n_X in [('xz', n_Xxz), ('yz', n_Xyz)]:
                for i in range(n_X):
                    # Skip if this X-stab is broken AND either prev or curr round has triangles
                    is_broken = i in broken_X_local[sheet_name]
                    skip = is_broken and (prev_with_tri or curr_with_tri)
                    if skip:
                        continue
                    curr_m = curr_records[('X', sheet_name, i)]
                    prev_m = prev_records[('X', sheet_name, i)]
                    c.append('DETECTOR', [stim.target_rec(curr_m - total_measurements),
                                           stim.target_rec(prev_m - total_measurements)])
    
    # PRE-MERGE phase
    for r in range(d_each):
        syndrome_round(r, with_triangles=False)
        add_detectors_for_round(r, prev_with_tri=False, curr_with_tri=False)
    
    # MERGE phase
    for r in range(d_each):
        round_idx = d_each + r
        syndrome_round(round_idx, with_triangles=True)
        prev_with_tri = (r > 0)  # only true if previous round was also merge
        add_detectors_for_round(round_idx, prev_with_tri=prev_with_tri, curr_with_tri=True)
    
    # POST-MERGE phase
    for r in range(d_each):
        round_idx = 2 * d_each + r
        syndrome_round(round_idx, with_triangles=False)
        prev_with_tri = (r == 0)
        add_detectors_for_round(round_idx, prev_with_tri=prev_with_tri, curr_with_tri=False)
    
    # Final destructive Z measurement of data
    if p > 0:
        c.append('X_ERROR', all_data, p)
    c.append('M', all_data)
    data_m_start = total_measurements
    total_measurements += n_data
    
    # Final Z-stab detectors from data
    last_record = measurement_records[-1]
    for stab_idx, (data_qs, _) in enumerate(sheet_Z_stabs['xz']):
        data_recs = [stim.target_rec(data_m_start + dq - total_measurements) for dq in data_qs]
        anc_rec_m = last_record[('Z', 'xz', stab_idx)]
        anc_rec = stim.target_rec(anc_rec_m - total_measurements)
        c.append('DETECTOR', data_recs + [anc_rec])
    for stab_idx, (data_qs, _) in enumerate(sheet_Z_stabs['yz']):
        data_recs = [stim.target_rec(data_m_start + dq - total_measurements) for dq in data_qs]
        anc_rec_m = last_record[('Z', 'yz', stab_idx)]
        anc_rec = stim.target_rec(anc_rec_m - total_measurements)
        c.append('DETECTOR', data_recs + [anc_rec])
    
    # Observable: Z_A ⊗ Z_B from data measurements
    op_data_qs = [global_to_local[c_idx] for c_idx in np.where(op_global == 1)[0]]
    obs_recs = [stim.target_rec(data_m_start + dq - total_measurements) for dq in op_data_qs]
    c.append('OBSERVABLE_INCLUDE', obs_recs, 0)
    
    return c, total_qubits, n_tri


def test():
    L = 4
    d = 3
    p = 0.001
    
    print(f"=== Surgery circuit v3 ===")
    print(f"  L={L}, d_each={d}, p={p}")
    c, nq, n_tri = build_surgery_circuit_v3(L, d, p)
    print(f"  Qubits: {nq}, Measurements: {c.num_measurements}, Detectors: {c.num_detectors}")
    
    try:
        dem = c.detector_error_model(decompose_errors=True)
        n_err = sum(1 for inst in dem if inst.type == 'error')
        print(f"  DEM OK: {n_err} error mechanisms")
        
        # Sample and decode
        sampler = c.compile_detector_sampler()
        events, flips = sampler.sample(shots=2000, separate_observables=True)
        matcher = pymatching.Matching.from_detector_error_model(dem)
        pred = matcher.decode_batch(events)
        n_err_log = int(np.sum(pred != flips))
        print(f"  Logical error rate: {n_err_log/2000:.4f}")
        return c, dem
    except Exception as e:
        msg = str(e)
        print(f"  Error: {msg.splitlines()[0]}")
        return None, None


if __name__ == '__main__':
    test()

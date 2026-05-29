"""Run L=8 surgery using cached primitive.

The original build_surgery_circuit_v3 calls find_surgery_primitive every time,
which is O(n^2) and times out at L=8. This module loads the cached primitive
from JSON instead.
"""
import os
import sys
import json
import time
import numpy as np
import stim
import pymatching



from sheet_code_fcc_lattice import (build_fcc_lattice, vertex_Z_stabilizers,
                                     oct_void_X_stabilizers, fcc_triangles)


def get_or_build_primitive(L):
    """Load surgery primitive from cache, or compute and cache it."""
    cache_path = f'/tmp/surgery_primitive_L{L}.json'
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
        triangle_indices = np.array(cache['triangle_indices'])
        op = np.array(cache['op'], dtype=np.int8)
        return triangle_indices, op
    
    # Compute fresh
    print(f"  No cache for L={L}, computing primitive...")
    from sheet_code_find_primitive_fast import find_surgery_primitive_fast
    tris, op, B = find_surgery_primitive_fast(L, verbose=False)
    cache = {
        'L': L,
        'triangle_indices': [int(t) for t in tris],
        'op': [int(o) for o in op],
    }
    with open(cache_path, 'w') as f:
        json.dump(cache, f)
    return tris, op


def build_surgery_circuit_cached(L, d_each, p):
    """Surgery circuit using cached primitive — avoids slow O(n^2) search at large L."""
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
    sheet_X_stabs_global_rows = {'xz': [], 'yz': []}
    
    for row_idx, (row, (s_label, pos)) in enumerate(zip(HZ_full, Z_meta)):
        if s_label not in sheet_Z_stabs:
            continue
        nonzero = np.where(row == 1)[0]
        if len(nonzero) == 0:
            continue
        local = [global_to_local[c] for c in nonzero]
        sheet_Z_stabs[s_label].append((local, pos))
    for row_idx, (row, (s_label, pos)) in enumerate(zip(HX_full, X_meta)):
        if s_label not in sheet_X_stabs:
            continue
        nonzero = np.where(row == 1)[0]
        if len(nonzero) == 0:
            continue
        local = [global_to_local[c] for c in nonzero]
        sheet_X_stabs[s_label].append((local, pos))
        sheet_X_stabs_global_rows[s_label].append(row_idx)
    
    n_Zxz = len(sheet_Z_stabs['xz'])
    n_Zyz = len(sheet_Z_stabs['yz'])
    n_Xxz = len(sheet_X_stabs['xz'])
    n_Xyz = len(sheet_X_stabs['yz'])
    
    triangle_indices, op_global = get_or_build_primitive(L)
    n_tri = len(triangle_indices)
    
    # Get B (triangle data qubit list) without computing find_surgery_primitive
    triangles, B = fcc_triangles(L)
    triangle_data_qubits = []
    for ti in triangle_indices:
        edges_in_T = np.where(B[ti] == 1)[0]
        data_qs = [global_to_local[c] for c in edges_in_T]
        triangle_data_qubits.append(data_qs)
    
    # Broken X-stabs
    broken_X_global = set()
    for ti in triangle_indices:
        tri_Z_op = B[ti]
        overlap = (HX_full @ tri_Z_op) % 2
        for x_idx in np.where(overlap == 1)[0]:
            broken_X_global.add(x_idx)
    
    broken_X_local = {'xz': set(), 'yz': set()}
    for sheet_name in ['xz', 'yz']:
        for local_idx, global_row in enumerate(sheet_X_stabs_global_rows[sheet_name]):
            if global_row in broken_X_global:
                broken_X_local[sheet_name].add(local_idx)
    
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
    
    measurement_records = []
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
        
        record = {}
        for i in range(n_Zxz):
            record[('Z', 'xz', i)] = total_measurements + i
        for i in range(n_Zyz):
            record[('Z', 'yz', i)] = total_measurements + n_Zxz + i
        if with_triangles:
            for i in range(n_tri):
                record[('T', i)] = total_measurements + n_Zxz + n_Zyz + i
        total_measurements += len(z_ancs_to_measure)
        
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
    
    def add_detectors_for_round(round_idx, prev_with_tri, curr_with_tri):
        curr_records = measurement_records[round_idx]
        prev_records = measurement_records[round_idx - 1] if round_idx > 0 else None
        
        for sheet_name, n_Z in [('xz', n_Zxz), ('yz', n_Zyz)]:
            for i in range(n_Z):
                curr_m = curr_records[('Z', sheet_name, i)]
                if round_idx == 0:
                    rec_idx = curr_m - total_measurements
                    c.append('DETECTOR', [stim.target_rec(rec_idx)])
                else:
                    prev_m = prev_records[('Z', sheet_name, i)]
                    c.append('DETECTOR', [stim.target_rec(curr_m - total_measurements),
                                           stim.target_rec(prev_m - total_measurements)])
        
        if round_idx > 0:
            for sheet_name, n_X in [('xz', n_Xxz), ('yz', n_Xyz)]:
                for i in range(n_X):
                    is_broken = i in broken_X_local[sheet_name]
                    skip = is_broken and (prev_with_tri or curr_with_tri)
                    if skip:
                        continue
                    curr_m = curr_records[('X', sheet_name, i)]
                    prev_m = prev_records[('X', sheet_name, i)]
                    c.append('DETECTOR', [stim.target_rec(curr_m - total_measurements),
                                           stim.target_rec(prev_m - total_measurements)])
    
    for r in range(d_each):
        syndrome_round(r, with_triangles=False)
        add_detectors_for_round(r, prev_with_tri=False, curr_with_tri=False)
    
    for r in range(d_each):
        round_idx = d_each + r
        syndrome_round(round_idx, with_triangles=True)
        prev_with_tri = (r > 0)
        add_detectors_for_round(round_idx, prev_with_tri=prev_with_tri, curr_with_tri=True)
    
    for r in range(d_each):
        round_idx = 2 * d_each + r
        syndrome_round(round_idx, with_triangles=False)
        prev_with_tri = (r == 0)
        add_detectors_for_round(round_idx, prev_with_tri=prev_with_tri, curr_with_tri=False)
    
    if p > 0:
        c.append('X_ERROR', all_data, p)
    c.append('M', all_data)
    data_m_start = total_measurements
    total_measurements += n_data
    
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
    
    op_data_qs = [global_to_local[c_idx] for c_idx in np.where(op_global == 1)[0]]
    obs_recs = [stim.target_rec(data_m_start + dq - total_measurements) for dq in op_data_qs]
    c.append('OBSERVABLE_INCLUDE', obs_recs, 0)
    
    return c, total_qubits, n_tri, len(broken_X_global)


def run_point(L, d_each, p, n_shots):
    """Run a single (L, p, n_shots) point with cached primitive."""
    timings = {}
    
    t0 = time.time()
    c, nq, n_tri, n_broken = build_surgery_circuit_cached(L, d_each, p)
    timings['build'] = time.time() - t0
    
    t0 = time.time()
    dem = c.detector_error_model(decompose_errors=True)
    timings['dem'] = time.time() - t0
    n_err_mech = sum(1 for inst in dem if inst.type == 'error')
    
    t0 = time.time()
    matcher = pymatching.Matching.from_detector_error_model(dem)
    timings['match'] = time.time() - t0
    
    t0 = time.time()
    sampler = c.compile_detector_sampler()
    events, flips = sampler.sample(shots=n_shots, separate_observables=True)
    pred = matcher.decode_batch(events)
    n_err = int(np.sum(pred != flips))
    timings['sample'] = time.time() - t0
    
    rate = n_err / n_shots
    return {
        'L': L, 'd_each': d_each, 'p': p, 'n_shots': n_shots,
        'n_err': n_err, 'rate': rate,
        'n_qubits': nq, 'n_triangles': n_tri, 'n_broken_X': n_broken,
        'n_error_mechanisms': n_err_mech,
        **{f't_{k}': v for k, v in timings.items()},
        't_total': sum(timings.values()),
    }


if __name__ == '__main__':
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    p = float(sys.argv[2]) if len(sys.argv) > 2 else 0.005
    n_shots = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    
    print(f"Running L={L}, p={p}, n_shots={n_shots}")
    result = run_point(L, L, p, n_shots)
    print(f"\n  Logical error rate: {result['rate']:.5f} ({result['n_err']}/{n_shots})")
    print(f"  Circuit: {result['n_qubits']} qubits, {result['n_triangles']} triangles, "
          f"{result['n_broken_X']} broken X-stabs")
    print(f"  DEM: {result['n_error_mechanisms']} error mechanisms")
    print(f"  Timing: build={result['t_build']:.1f}s dem={result['t_dem']:.1f}s "
          f"match={result['t_match']:.1f}s sample={result['t_sample']:.1f}s")
    print(f"  Total: {result['t_total']:.1f}s")
    
    # Append to results file
    results_file = '/tmp/surgery_threshold_results.json'
    try:
        with open(results_file) as f:
            existing = json.load(f)
    except FileNotFoundError:
        existing = []
    existing.append(result)
    with open(results_file, 'w') as f:
        json.dump(existing, f, indent=2)

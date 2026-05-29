"""Fast incremental threshold sweep — single p, single L, save to JSON."""
import sys
import json
import time
import numpy as np
import stim
import pymatching

from sheet_code_custom_surgery import build_surgery_circuit_v3


def run_one(L, d_each, p, n_shots):
    """Run a single point, return (rate, time_seconds)."""
    t0 = time.time()
    c, nq, n_tri = build_surgery_circuit_v3(L, d_each, p)
    t_build = time.time() - t0
    
    t0 = time.time()
    dem = c.detector_error_model(decompose_errors=True)
    t_dem = time.time() - t0
    
    t0 = time.time()
    matcher = pymatching.Matching.from_detector_error_model(dem)
    t_match = time.time() - t0
    
    t0 = time.time()
    sampler = c.compile_detector_sampler()
    events, flips = sampler.sample(shots=n_shots, separate_observables=True)
    pred = matcher.decode_batch(events)
    n_err = int(np.sum(pred != flips))
    t_sample = time.time() - t0
    
    rate = n_err / n_shots
    return {
        'L': L, 'd_each': d_each, 'p': p, 'n_shots': n_shots,
        'n_err': n_err, 'rate': rate,
        't_build': t_build, 't_dem': t_dem, 't_match': t_match, 't_sample': t_sample,
        'n_qubits': nq, 'n_triangles': n_tri
    }


if __name__ == '__main__':
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    p = float(sys.argv[2]) if len(sys.argv) > 2 else 0.001
    n_shots = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    
    print(f"Running L={L}, p={p}, n_shots={n_shots}...")
    result = run_one(L, L, p, n_shots)
    
    print(f"\n  Result:")
    print(f"    Logical error rate: {result['rate']:.5f} ({result['n_err']}/{n_shots})")
    print(f"    Timing: build={result['t_build']:.1f}s dem={result['t_dem']:.1f}s "
          f"match={result['t_match']:.1f}s sample={result['t_sample']:.1f}s")
    print(f"    Total time: {sum([result['t_build'], result['t_dem'], result['t_match'], result['t_sample']]):.1f}s")
    
    # Append to running results file
    results_file = '/home/claude/surgery_threshold_results.json'
    try:
        with open(results_file, 'r') as f:
            existing = json.load(f)
    except FileNotFoundError:
        existing = []
    existing.append(result)
    with open(results_file, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"  Appended to {results_file} (now {len(existing)} entries)")

"""Threshold sweep with the custom 2-sheet surgery circuit (v3)."""
import sys
import numpy as np
import stim
import pymatching

from sheet_code_custom_surgery import build_surgery_circuit_v3


def run_surgery(L, d_each, p, n_shots):
    c, nq, n_tri = build_surgery_circuit_v3(L, d_each, p)
    sampler = c.compile_detector_sampler()
    events, flips = sampler.sample(shots=n_shots, separate_observables=True)
    dem = c.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    pred = matcher.decode_batch(events)
    n_err = int(np.sum(pred != flips))
    return n_err / n_shots


def sweep():
    p_values = [0.001, 0.002, 0.003, 0.005, 0.008]
    distances = [4, 6]
    n_shots = 3000
    
    print(f"\n=== Custom surgery circuit threshold sweep ===")
    print(f"  d_each per phase = L; 3 phases total")
    print(f"  Z-basis joint observable; {n_shots} shots/point\n")
    
    print(f"  p (%)    L=4 (d_each=4)    L=6 (d_each=6)")
    
    for p in p_values:
        rates = []
        for L in distances:
            try:
                rate = run_surgery(L, L, p, n_shots)
                rates.append(rate)
            except Exception as e:
                rates.append(None)
                print(f"    Error at L={L}, p={p}: {str(e).splitlines()[0]}")
        rate_strs = [f"{r:.5f}" if r is not None else "ERR" for r in rates]
        print(f"  {p*100:5.2f}    " + "          ".join(rate_strs))


if __name__ == '__main__':
    sweep()

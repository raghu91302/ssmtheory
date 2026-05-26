"""Reproduce Section 7: threshold simulation."""
import numpy as np
import stim
import pymatching


def run_memory(d, p, n_shots, rounds_multiplier=1.0):
    rounds = max(int(d * rounds_multiplier), d)
    c = stim.Circuit.generated(
        code_task='surface_code:rotated_memory_z',
        distance=d, rounds=rounds,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
    )
    sampler = c.compile_detector_sampler()
    events, flips = sampler.sample(shots=n_shots, separate_observables=True)
    dem = c.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    pred = matcher.decode_batch(events)
    return int(np.sum(pred != flips)) / n_shots


def main():
    p_values = [0.001, 0.002, 0.003, 0.005, 0.008]
    distances = [3, 5, 7, 9]
    n_shots = 30000

    print("Standard memory experiment (d rounds):")
    print(f"  p (%)  " + "  ".join(f"d={d}" for d in distances))
    std_rates = {}
    for p in p_values:
        rates = [run_memory(d, p, n_shots, 1.0) for d in distances]
        std_rates[p] = rates
        print(f"  {p*100:5.2f}  " + "  ".join(f"{r:.5f}" for r in rates))

    print("\nSurgery-extended experiment (3d rounds):")
    print(f"  p (%)  " + "  ".join(f"d={d}" for d in distances))
    for p in p_values:
        rates = [run_memory(d, p, n_shots, 3.0) for d in distances]
        ratios = [s/r if r > 0 else float('inf')
                  for s, r in zip(rates, std_rates[p])]
        print(f"  {p*100:5.2f}  " + "  ".join(f"{r:.5f}" for r in rates) +
              "  ratios: " + "  ".join(f"{r:4.2f}x" for r in ratios))


if __name__ == '__main__':
    main()

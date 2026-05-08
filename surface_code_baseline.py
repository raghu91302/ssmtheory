"""
surface_code_baseline.py: Surface code [[16, 1, 4]] per-logical error rate
under matched circuit-level noise. Reproduces the Section 7 comparison
"a single [[16, 1, 4]] code block has a per-logical error rate near
6 x 10^-4 at p = 0.001."

Why this script exists:
  Section 7 of the manuscript compares the FCC three-sheet code's per-logical
  error rate (5.2e-3 at p=0.001, from 3sheet_distance_scaling.py) against
  the rotated surface code at d=4. To make that comparison fair, the surface
  code must be simulated with the same circuit-level noise model:

    - Stim's built-in surface_code:rotated_memory_z circuit
    - after_clifford_depolarization = p
    - before_measure_flip_probability = p
    - after_reset_flip_probability = p
    - before_round_data_depolarization = 0  (single block, no multiplex idle)

  This is the same noise model as 3sheet_distance_scaling.py except for the
  multiplex idle term, which does not apply to a single surface code block.

Usage:
    python surface_code_baseline.py

Output:
    Per-logical error rate vs p for the rotated surface code at d = 4,
    sweeping p in the same range as the FCC distance-scaling experiment.
    The d = 4 row at p = 1e-3 is the number cited in Section 7.
"""
from __future__ import annotations
import numpy as np
import stim
import pymatching
import time


P_SWEEP = [0.0001, 0.0003, 0.0005, 0.001, 0.002, 0.003, 0.005]
DISTANCE = 4
SHOTS = 30_000


def make_surface_circuit(d: int, p: float) -> stim.Circuit:
    """Stim's rotated surface code at distance d with our standard
    single-block circuit-level noise model."""
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=d,
        after_clifford_depolarization=p,
        before_round_data_depolarization=0.0,   # no multiplex idle for single block
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )


def simulate_one(d: int, p: float, num_shots: int):
    circ = make_surface_circuit(d, p)
    dem = circ.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    sampler = circ.compile_detector_sampler()

    detection_events, observable_flips = sampler.sample(
        num_shots, separate_observables=True
    )
    predicted = matcher.decode_batch(detection_events)
    # rotated_memory_z encodes 1 logical qubit; observable axis is length 1
    # block error == per-logical error in this single-logical case
    n_errors = int(np.sum(np.any(predicted != observable_flips, axis=1)))
    return n_errors / num_shots


def main():
    print("=" * 70)
    print(f"Surface code [[16, 1, 4]] baseline (d = {DISTANCE})")
    print(f"Stim's rotated_memory_z, single block, no multiplex idle")
    print(f"{SHOTS} shots per p")
    print("=" * 70)
    print()
    print(f"  {'p':>8}    {'p_logical':>12}    {'errors / shots':>18}")
    print("  " + "-" * 50)

    results = {}
    for p in P_SWEEP:
        t0 = time.time()
        per_log = simulate_one(DISTANCE, p, SHOTS)
        n_err = int(round(per_log * SHOTS))
        elapsed = time.time() - t0
        print(f"  {p:>8.4f}    {per_log:>12.3e}    {n_err:>9d} / {SHOTS:<6d}    [{elapsed:>5.1f}s]")
        results[p] = per_log

    print()
    print("=" * 70)
    p_target = 0.001
    val = results.get(p_target)
    if val is not None:
        print(f"At p = {p_target:.4f}: surface code [[16, 1, 4]] per-logical "
              f"error = {val:.2e}")
        print(f"Manuscript Section 7 cites: ~5 x 10^-4 (1M-shot reference value: 4.8e-4)")
        print(f"FCC three-sheet code at L = 4 (from 3sheet_distance_scaling.py): 5.2e-3")
        print(f"Ratio (FCC / surface): {5.2e-3 / val:.1f}x  (manuscript cites ~11x)")
    print("=" * 70)
    print()
    print("NOTE: 30,000 shots gives a noisy estimate at small p_logical.")
    print("For the high-statistics value cited in the manuscript, increase")
    print("SHOTS to 1,000,000 (run-time ~1-2 seconds).")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

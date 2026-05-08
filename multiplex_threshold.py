"""
multiplex_threshold.py: Reproduce the 3-sheet time-multiplexed threshold
claim of Paper 3 (≈ 0.42%), and compare to single-sheet operation (≈ 0.63%).

Model (per Paper 3, Section 9.3):
  - Each layer of the FCC sheet code is an independent 2D rotated surface
    code at distance d = L.
  - The full L-layer sheet code's logical error rate is taken as the
    union bound p_sheet = 1 - (1 - p_layer)^L.
  - Three-sheet time-multiplexing introduces idle decoherence: while
    one sheet is being measured (one round = time t), data qubits in
    the OTHER two sheets idle for 2t. From the active sheet's point of
    view, that corresponds to DEPOLARIZE1(2p) on every data qubit
    before each round of stabilizer extraction (because between two
    consecutive rounds of THIS sheet's measurement, 2t of idle time has
    elapsed).
  - We turn idle noise on (multiplex) or off (single-sheet) by setting
    Stim's `before_round_data_depolarization` parameter accordingly.

Decoder: PyMatching MWPM on Stim's detector error model.
"""

from __future__ import annotations
import numpy as np
import stim
import pymatching
import time
import os

OUT = "/home/claude/multiplex_results"
os.makedirs(OUT, exist_ok=True)

DISTANCES = [3, 5, 7, 9, 11]   # paper uses these
P_SWEEP = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010, 0.012]
SHOTS = 30_000


def make_circuit(d: int, p: float, p_idle: float) -> stim.Circuit:
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=d,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p_idle,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )


def simulate(d: int, p: float, p_idle: float, num_shots: int):
    """Returns (per-layer-error-rate, errors, shots)."""
    circ = make_circuit(d, p, p_idle)
    dem = circ.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    sampler = circ.compile_detector_sampler()
    det, obs = sampler.sample(num_shots, separate_observables=True)
    pred = matcher.decode_batch(det)
    errors = np.any(pred != obs, axis=1).sum()
    return errors / num_shots, int(errors), num_shots


def sweep(label: str, p_idle_factor: float):
    """p_idle_factor: 0 for single-sheet, 2 for multiplex (sets p_idle = factor * p)."""
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"  (p_idle = {p_idle_factor} × p)")
    print(f"{'='*72}")
    print(f"{'p (%)':>7}  ", end="")
    for d in DISTANCES:
        print(f"   d={d:<2d}        ", end="")
    print()

    results = {}  # (d, p) -> per-layer rate
    for p in P_SWEEP:
        print(f"{100*p:>7.3f}  ", end="", flush=True)
        for d in DISTANCES:
            t0 = time.time()
            rate, errs, n = simulate(d, p, p_idle_factor * p, SHOTS)
            results[(d, p)] = rate
            tag = f"{rate:.3e}"
            if errs == 0:
                tag = f"<{1/n:.0e}"
            print(f"  {tag:>11s}  ", end="", flush=True)
        print()

    return results


print(f"Stim {stim.__version__}, PyMatching {pymatching.__version__}")
print(f"Distances: {DISTANCES}")
print(f"Shots/point: {SHOTS}")

t_total = time.time()
single = sweep("Single-sheet operation (no idle noise)", p_idle_factor=0.0)
multi  = sweep("3-sheet multiplexing (idle noise = 2p)", p_idle_factor=2.0)
print(f"\nTotal runtime: {time.time() - t_total:.0f} s")


# ---------------------------------------------------------------
# Save raw data
# ---------------------------------------------------------------
def save_table(results, fn):
    with open(fn, "w") as f:
        f.write(f"{'p':>10s}")
        for d in DISTANCES:
            f.write(f"  d={d:<3d}_layer  ")
        for d in DISTANCES:
            f.write(f"  d={d:<3d}_sheet  ")
        f.write("\n")
        for p in P_SWEEP:
            f.write(f"{p:>10.5f}")
            for d in DISTANCES:
                rate = results[(d, p)]
                f.write(f"  {rate:>11.4e}  ")
            for d in DISTANCES:
                # Sheet rate = 1 - (1 - per_layer)^d  (approximation from paper)
                pL = results[(d, p)]
                sheet_rate = 1 - (1 - pL) ** d
                f.write(f"  {sheet_rate:>11.4e}  ")
            f.write("\n")

save_table(single, f"{OUT}/single_sheet.txt")
save_table(multi,  f"{OUT}/multiplex.txt")
print(f"Tables saved to {OUT}/")


# ---------------------------------------------------------------
# Make plots
# ---------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 11,
    "lines.linewidth": 1.8, "lines.markersize": 6.5,
    "axes.grid": True, "grid.alpha": 0.25,
})

def plot_curves(results, title, threshold_hint, fn):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(DISTANCES)))

    # Per-layer
    for c, d in zip(cmap, DISTANCES):
        rates = [results[(d, p)] for p in P_SWEEP]
        ax1.loglog([100*p for p in P_SWEEP], rates, "o-", color=c, label=f"d = {d}")
    ax1.set_xlabel("Physical error rate (%)")
    ax1.set_ylabel("Per-layer logical error rate")
    ax1.set_title(f"{title} — per layer")
    ax1.axvline(100*threshold_hint, ls=":", color="firebrick", alpha=0.5)
    ax1.text(100*threshold_hint, 1e-5, f"  paper:\n  {100*threshold_hint:.2f}%",
             color="firebrick", fontsize=9, va="bottom")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.set_ylim(5e-6, 1)

    # Sheet (any-layer failure)
    for c, d in zip(cmap, DISTANCES):
        rates = []
        for p in P_SWEEP:
            pL = results[(d, p)]
            rates.append(1 - (1 - pL) ** d)
        ax2.loglog([100*p for p in P_SWEEP], rates, "s-", color=c, label=f"L = {d}")
    ax2.set_xlabel("Physical error rate (%)")
    ax2.set_ylabel("Sheet-code logical error rate")
    ax2.set_title(f"{title} — full L-layer sheet")
    ax2.axvline(100*threshold_hint, ls=":", color="firebrick", alpha=0.5)
    ax2.legend(loc="lower right", fontsize=9)
    ax2.set_ylim(5e-5, 1)

    fig.tight_layout()
    fig.savefig(fn, dpi=180)
    plt.close(fig)
    print(f"Saved: {fn}")

plot_curves(single, "Single-sheet operation", 0.0063, f"{OUT}/single_sheet.png")
plot_curves(multi,  "3-sheet multiplexing",   0.0042, f"{OUT}/multiplex.png")


# ---------------------------------------------------------------
# Threshold crossing analysis
# ---------------------------------------------------------------
def find_threshold(results):
    """Locate p where sheet-rate curves at consecutive L stop separating."""
    crossings = []
    for i, p in enumerate(P_SWEEP):
        # Compute sheet rates at each L
        sheet_rates = [(d, 1 - (1 - results[(d, p)]) ** d) for d in DISTANCES]
        # Check if rates are increasing with L (above threshold) or decreasing (below)
        is_above = all(sheet_rates[k][1] >= sheet_rates[k-1][1]
                       for k in range(1, len(sheet_rates)))
        is_below = all(sheet_rates[k][1] <= sheet_rates[k-1][1]
                       for k in range(1, len(sheet_rates)))
        crossings.append((p, sheet_rates, is_above, is_below))
    return crossings

print("\nThreshold crossing analysis:")
print("-" * 72)
print("\n[Single-sheet]")
print(f"  {'p (%)':>8s}  {'sheet rates by L':>40s}  ordering")
for p, sheet_rates, above, below in find_threshold(single):
    rates_str = "  ".join(f"L={d}:{r:.1e}" for d, r in sheet_rates)
    tag = ("INCREASING" if above else "DECREASING" if below else "MIXED")
    print(f"  {100*p:>6.3f}    {rates_str}    [{tag}]")

print("\n[3-sheet multiplex]")
print(f"  {'p (%)':>8s}  {'sheet rates by L':>40s}  ordering")
for p, sheet_rates, above, below in find_threshold(multi):
    rates_str = "  ".join(f"L={d}:{r:.1e}" for d, r in sheet_rates)
    tag = ("INCREASING" if above else "DECREASING" if below else "MIXED")
    print(f"  {100*p:>6.3f}    {rates_str}    [{tag}]")

print(f"\nDone. Output in {OUT}/")

#!/usr/bin/env python3
"""Run every verification script in this directory in order, top to bottom.

This is the single entry point referenced in the paper's data-availability section.
If every assertion passes, the script exits with status 0 and prints a summary.
"""
import subprocess
import sys
import os
import time

SCRIPTS = [
    ("01_structure_tensor.py", "Structure tensor S^mn = 12 I, T = 0, FCC sub-lattice"),
    ("02_24cell_triality.py",  "24-cell f-vector, triality 8+8+8, F_box = 36"),
    ("03_d4_css_code.py",      "CSS code [[1536, 1282, >=3]] at L=4"),
    ("04_mass_spectrum.py",    "All six rest-mass predictions and deviations"),
]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    print("=" * 78)
    print("Mass-Energy-Information Equivalence II  --  verification suite")
    print("=" * 78)
    total = 0.0
    for fname, label in SCRIPTS:
        path = os.path.join(here, fname)
        print(f"\n>>> {fname} -- {label}")
        print("-" * 78)
        t0 = time.time()
        r = subprocess.run([sys.executable, path], capture_output=False)
        dt = time.time() - t0
        total += dt
        if r.returncode != 0:
            print(f"\n!!! FAILED in {fname} (exit code {r.returncode})")
            sys.exit(r.returncode)
        print(f"--- {fname} OK in {dt:.2f}s")
    print("\n" + "=" * 78)
    print(f"All verifications passed in {total:.2f}s total.")
    print("=" * 78)


if __name__ == '__main__':
    main()

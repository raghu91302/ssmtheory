#!/usr/bin/env python3
"""
Master runner for the FCC + D4 Doubler paper verification suite.

Runs both verification scripts in sequence and reports the cumulative result.

Usage:
    python3 doubler_run_all.py
"""
import os
import sys
import subprocess
import time

SCRIPTS = [
    ("doubler_fcc_verification.py", "FCC Bond-Direction Dirac Operator (3D)"),
    ("doubler_d4_verification.py",  "D4 Bond-Direction Dirac Operator (4D)"),
]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    print("=" * 78)
    print("Fermion Chirality from Non-Bipartite Topology --- verification suite")
    print("=" * 78)

    t0 = time.time()
    for script, label in SCRIPTS:
        path = os.path.join(here, script)
        if not os.path.exists(path):
            print(f"\n[skip] {script}: file not found.")
            continue
        print(f"\n>>> {label}")
        print(f"    ({script})")
        t1 = time.time()
        result = subprocess.run([sys.executable, path], capture_output=True, text=True)
        dt = time.time() - t1
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            print(f"--- {script} FAILED in {dt:.2f}s")
            sys.exit(1)
        # Print just the last summary block
        for line in result.stdout.splitlines()[-3:]:
            print(f"    {line}")
        print(f"--- {script} OK in {dt:.2f}s")

    total = time.time() - t0
    print("\n" + "=" * 78)
    print(f"All verifications passed in {total:.2f}s total.")
    print("=" * 78)


if __name__ == "__main__":
    main()

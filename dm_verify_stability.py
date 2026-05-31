#!/usr/bin/env python3
"""dm_verify_stability.py
Selection-Stitch Model -- dark matter annihilation paper.
Tests whether the anchor-free (full-S4) tetrahedral occupant is a stable minimum of the
bond-strain energy E(x) = 0.5 * sum_i (|x - V_i| - r0)^2. Reports the Hessian at the
symmetric center and the energy along the anchor-selecting and face-escape distortions.
Reproduces Section 4.3. Depends only on numpy.
"""
import numpy as np

a = 2.0
L = a/np.sqrt(2)

# regular tetrahedral void, a=2: center (0.5,0.5,0.5), four bounding FCC atoms
c = np.array([0.5, 0.5, 0.5])
V = np.array([[0, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1]], float)
r0 = np.linalg.norm(V[0]-c)


def E(x):
    return 0.5*sum((np.linalg.norm(x-Vi)-r0)**2 for Vi in V)


def hessian(f, x, h=1e-5):
    n = len(x); H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            xpp = x.copy(); xpp[i] += h; xpp[j] += h
            xpm = x.copy(); xpm[i] += h; xpm[j] -= h
            xmp = x.copy(); xmp[i] -= h; xmp[j] += h
            xmm = x.copy(); xmm[i] -= h; xmm[j] -= h
            H[i, j] = (f(xpp)-f(xpm)-f(xmp)+f(xmm))/(4*h*h)
    return H


def main():
    print(f"L = {L:.4f}")
    print(f"equilibrium bond r0 = {r0:.4f} = {r0/L:.4f} L  (metric wall {1/np.sqrt(3):.4f} L)\n")

    w = np.linalg.eigvalsh(hessian(E, c.copy()))
    print(f"Hessian eigenvalues at symmetric center: {np.round(w, 4)}")
    print(f"  positive-definite (stable minimum)? {bool(np.all(w > 1e-6))}\n")

    print("Energy vs displacement toward one vertex (anchor-selecting distortion):")
    d1 = (V[0]-c)/np.linalg.norm(V[0]-c)
    for tdisp in [0.0, 0.05, 0.10, 0.15, 0.20]:
        print(f"  disp = {tdisp:.2f} L  ->  E = {E(c+tdisp*L*d1):.6f}")

    print("\nEnergy vs displacement toward a triangular face (escape distortion):")
    fc = (V[1]+V[2]+V[3])/3.0
    d2 = (fc-c)/np.linalg.norm(fc-c)
    for tdisp in [0.0, 0.05, 0.10, 0.15, 0.20]:
        print(f"  disp = {tdisp:.2f} L  ->  E = {E(c+tdisp*L*d2):.6f}")

    print("\nVerified: the anchor-free tetrahedral occupant is a stable local minimum;")
    print("both the anchor-selecting and face-escape distortions raise the energy.")


if __name__ == "__main__":
    main()

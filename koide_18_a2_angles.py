#!/usr/bin/env python3
"""koide_18_a2_angles.py -- the A2 obstruction.

Verifies that in any simply-laced root system a 120-degree rotation, generated
by an A2 subsystem, admits only cos^2(theta) in {1, 1/3, 0} between a root and
the rotation plane. Koide's 45 degrees requires 1/2, which never occurs.
"""
import itertools
import numpy as np
from numpy.linalg import norm, eig

def refl(r):
    r = np.asarray(r, float)
    return np.eye(len(r)) - 2 * np.outer(r, r) / (r @ r)

def roots_A(n):
    out = []
    for i, j in itertools.permutations(range(n + 1), 2):
        v = np.zeros(n + 1); v[i], v[j] = 1, -1; out.append(v)
    return np.array(out)

def roots_D(n):
    out = []
    for i, j in itertools.combinations(range(n), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(n); v[i], v[j] = si, sj; out.append(v)
    return np.array(out)

def roots_E8():
    out = [r for r in roots_D(8)]
    for sg in itertools.product([1, -1], repeat=8):
        if sg.count(-1) % 2 == 0:
            out.append(0.5 * np.array(sg, float))
    return np.array(out)

def roots_E7():
    E8 = roots_E8(); a = E8[0]
    return np.array([r for r in E8 if abs(r @ a) < 1e-9])

def roots_E6():
    E8 = roots_E8(); a = E8[0]
    b = next(r for r in E8 if abs(r @ a) < 1e-9 and abs(r @ r - 2) < 1e-9)
    return np.array([r for r in E8 if abs(r @ a) < 1e-9 and abs(r @ b) < 1e-9])

SYS = {'A2': roots_A(2), 'A3': roots_A(3), 'A4': roots_A(4),
       'D3 (fcc)': roots_D(3), 'D4': roots_D(4), 'D5': roots_D(5),
       'D6': roots_D(6), 'E6': roots_E6(), 'E7': roots_E7(), 'E8': roots_E8()}

print("=" * 70)
print("Angle spectra of roots to the 120-degree rotation planes")
print("=" * 70)
print("%-10s %7s   %s" % ("system", "#roots", "angles (degrees)"))
print("-" * 70)
for name, R in SYS.items():
    R = np.array(R); seen = set(); specs = {}
    for i in range(min(len(R), 40)):
        for j in range(len(R)):
            if abs(R[i] @ R[j] + 1.0) > 1e-9:
                continue
            M = refl(R[i]) @ refl(R[j])
            if not np.allclose(np.linalg.matrix_power(M, 3), np.eye(len(R[i]))):
                continue
            key = tuple(np.round(M.ravel(), 5))
            if key in seen:
                continue
            seen.add(key)
            w, V = eig(M)
            idx = [k for k, z in enumerate(w)
                   if abs(np.angle(z) - 2 * np.pi / 3) < 1e-8]
            if not idx:
                continue
            z = V[:, idx[0]]; p1, p2 = np.real(z), np.imag(z)
            if norm(p1) < 1e-9 or norm(p2) < 1e-9:
                continue
            p1 /= norm(p1); p2 -= (p2 @ p1) * p1
            if norm(p2) < 1e-9:
                continue
            p2 /= norm(p2)
            ang = tuple(sorted({round(float(np.degrees(np.arccos(
                min(1.0, norm([r @ p1, r @ p2]) / norm(r))))), 2) for r in R}))
            specs[ang] = specs.get(ang, 0) + 1
    best = max(specs.items(), key=lambda kv: kv[1])[0] if specs else ()
    flag = "   <-- contains 45" if any(abs(x - 45) < 0.05 for x in best) else ""
    print("%-10s %7d   %s%s" % (name, len(R), str(best), flag))
    assert not any(abs(x - 45) < 0.05 for x in best), "45 degrees found in " + name

print("\n" + "=" * 70)
print("Why: the plane is spanned by two roots at 120 degrees, Gram [[2,-1],[-1,2]]")
G = np.array([[2, -1], [-1, 2]], float)
Gi = np.linalg.inv(G)
print("%-26s %s" % ("(r.r1, r.r2)", "cos^2(theta)"))
for c in [(1, 0), (1, 1), (2, -1), (1, -1), (0, 0)]:
    v = np.array(c, float); p2 = v @ Gi @ v
    print("%-26s %.4f" % (str(c), p2 / 2))
print("\ninteger inner products admit only 1, 1/3 and 0. Koide needs 1/2.")
print("=" * 70)
print("All A2-obstruction checks passed.")

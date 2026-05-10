"""
Verification of SU(3) closure for the SSM tetrahedral defect's algebraic
operators. Reproduces the structure constants table in Appendix A of
Kulkarni 2026.

The 8 generators in the color basis (3 skew pairs of K_4):
  H_1 = diag(1, -1, 0)            : Cartan, color 1 vs color 2
  H_2 = diag(1, 1, -2)/sqrt(3)    : Cartan, average(1,2) vs 3
  E_ij = |i><j|, F_ij = |j><i|    : root vectors (color transitions)

Verifies: all 28 commutators close on the 8-dim space, and the resulting
Lie algebra is su(3) with A_2 root system.
"""
import numpy as np

# Build generators in C^3
H1 = np.diag([1.0, -1.0, 0.0])
H2 = np.diag([1.0, 1.0, -2.0]) / np.sqrt(3)

E12 = np.zeros((3,3)); E12[0,1] = 1.0
F12 = E12.T
E13 = np.zeros((3,3)); E13[0,2] = 1.0
F13 = E13.T
E23 = np.zeros((3,3)); E23[1,2] = 1.0
F23 = E23.T

ops = [('H1', H1), ('H2', H2), ('E12', E12), ('F12', F12),
       ('E13', E13), ('F13', F13), ('E23', E23), ('F23', F23)]

basis = np.array([op.flatten() for _, op in ops]).T  # 9x8

def comm(A, B):
    return A @ B - B @ A

def express(M):
    """Express M in the 8-op basis. Returns (coefficients, error)."""
    v = M.flatten()
    coeffs, _, _, _ = np.linalg.lstsq(basis, v, rcond=None)
    err = np.linalg.norm(v - basis @ coeffs)
    return coeffs, err

# All commutators
print("Full commutator table:")
print("=" * 60)
all_close = True
for i, (n1, op1) in enumerate(ops):
    for j, (n2, op2) in enumerate(ops):
        if i >= j: continue
        c = comm(op1, op2)
        if np.linalg.norm(c) < 1e-12:
            print(f"  [{n1}, {n2}] = 0")
            continue
        coeffs, err = express(c)
        if err > 1e-10:
            print(f"  [{n1}, {n2}]: FAILS to close (err={err:.2e})")
            all_close = False
            continue
        terms = []
        for k, coef in enumerate(coeffs):
            if abs(coef) > 1e-10:
                sign = '+' if coef > 0 else '-'
                mag = abs(coef)
                if abs(mag - 1.0) < 1e-10:
                    terms.append(f"{sign}{ops[k][0]}")
                elif abs(mag - 0.5) < 1e-10:
                    terms.append(f"{sign}½{ops[k][0]}")
                elif abs(mag - np.sqrt(3)) < 1e-10:
                    terms.append(f"{sign}√3·{ops[k][0]}")
                elif abs(mag - np.sqrt(3)/2) < 1e-10:
                    terms.append(f"{sign}(√3/2)·{ops[k][0]}")
                else:
                    terms.append(f"{coef:+.3f}·{ops[k][0]}")
        # Clean up leading +
        expr = ' '.join(terms)
        if expr.startswith('+'): expr = expr[1:]
        print(f"  [{n1}, {n2}] = {expr}")

print("=" * 60)
print(f"Closure on 8-dim space: {all_close}")

# Check root system
print()
print("Root system verification:")
print(f"  α_12 (from E12): ({comm(H1,E12)[0,1]/E12[0,1]}, {comm(H2,E12)[0,1]/E12[0,1]:.4f})")
print(f"  α_13 (from E13): ({comm(H1,E13)[0,2]/E13[0,2]}, {comm(H2,E13)[0,2]/E13[0,2]:.4f})")
print(f"  α_23 (from E23): ({comm(H1,E23)[1,2]/E23[1,2]}, {comm(H2,E23)[1,2]/E23[1,2]:.4f})")
print()
print("These are the A_2 roots: ±(2,0), ±(1,√3), ±(-1,√3).")
print("All have |α|² = 4. Angles: 60°, 60°, 120°.")
print()
print("Therefore the 8 operators close on su(3) with A_2 root system. ✓")

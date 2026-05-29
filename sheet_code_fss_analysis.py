"""Finite-size scaling analysis for the surgery threshold.

Standard FSS fit: at threshold, p_L is independent of L.
Below threshold, p_L decreases with L (faster suppression).
Above threshold, p_L increases with L.

We can fit:
    p_L(p) = A + B * (p - p_th) * L^(1/nu) + C * ((p - p_th) * L^(1/nu))^2

or use the simpler crossover analysis.
"""
import json
import numpy as np
from scipy.optimize import curve_fit, minimize

with open('/home/claude/surgery_threshold_results.json') as f:
    results = json.load(f)

# Merge by (L, p)
merged = {}
for r in results:
    key = (r['L'], r['p'])
    merged.setdefault(key, {'n_err': 0, 'n_shots': 0})
    merged[key]['n_err'] += r['n_err']
    merged[key]['n_shots'] += r['n_shots']

# Group by L
by_L = {}
for (L, p), v in merged.items():
    rate = v['n_err'] / v['n_shots']
    se = np.sqrt(max(rate*(1-rate), 1e-9) / v['n_shots'])
    by_L.setdefault(L, []).append((p, rate, se, v['n_err'], v['n_shots']))
for L in by_L:
    by_L[L].sort()

print("=== Surgery threshold data (merged) ===")
print(f"  {'p (%)':>8}  {'L=4':>12}  {'L=6':>12}  {'L=8':>12}")
all_p = sorted({p for L in by_L for (p, _, _, _, _) in by_L[L]})
for p in all_p:
    row = [f"{p*100:>8.3f}"]
    for L in (4, 6, 8):
        pts = [(rate, se, ne, ns) for pp, rate, se, ne, ns in by_L.get(L, []) if abs(pp - p) < 1e-9]
        if pts:
            rate, se, ne, ns = pts[0]
            row.append(f"{rate:.4f}±{se:.4f} ({ne}/{ns})")
        else:
            row.append("—")
    print("  " + "  ".join(f"{c:<25}" for c in row))

print("\n=== Crossover analysis ===")
def find_cross(L1, L2):
    pts_L1 = {pp: (rate, se) for pp, rate, se, _, _ in by_L[L1]}
    pts_L2 = {pp: (rate, se) for pp, rate, se, _, _ in by_L[L2]}
    common = sorted(set(pts_L1) & set(pts_L2))
    crosses = []
    for i in range(len(common)-1):
        p_a, p_b = common[i], common[i+1]
        d_a = pts_L1[p_a][0] - pts_L2[p_a][0]
        d_b = pts_L1[p_b][0] - pts_L2[p_b][0]
        if d_a * d_b < 0:  # sign change
            # Linear interpolation
            p_cross = p_a + (p_b - p_a) * d_a / (d_a - d_b)
            crosses.append(p_cross)
    return crosses

for pair in [(4,6), (4,8), (6,8)]:
    c = find_cross(*pair)
    if c:
        print(f"  L={pair[0]} vs L={pair[1]}: crossing(s) at p = {', '.join(f'{x*100:.3f}%' for x in c)}")

print("\n=== Joint finite-size scaling fit ===")
# Collect all data points within a sensible range (excluding super-low-p where rate is 0)
fit_data = []
for L in (4, 6, 8):
    for p, rate, se, n_err, n_shots in by_L[L]:
        # Include points with reasonable statistics
        if n_err >= 5 and rate < 0.5:
            fit_data.append((p, L, rate, se))

if len(fit_data) < 6:
    print(f"  Not enough data points for FSS fit ({len(fit_data)})")
else:
    print(f"  Using {len(fit_data)} points for FSS fit")
    
    # Try a 1-parameter scaling form: p_L = A + B*x + C*x^2 where x = (p-p_th)*L^(1/nu)
    def scaling(params, p, L):
        p_th, nu, A, B, C = params
        x = (p - p_th) * L**(1.0/nu)
        return A + B*x + C*x**2
    
    def neg_loglik(params):
        p_th, nu, A, B, C = params
        if nu <= 0:
            return 1e10
        total = 0
        for p, L, rate, se in fit_data:
            pred = scaling(params, p, L)
            total += ((pred - rate) / se)**2
        return total
    
    # Initial guess
    x0 = [0.010, 1.5, 0.2, 0.05, 0.001]
    
    from scipy.optimize import minimize
    res = minimize(neg_loglik, x0, method='Nelder-Mead', options={'maxiter': 10000})
    
    if res.success:
        p_th, nu, A, B, C = res.x
        print(f"  Fit results:")
        print(f"    p_th = {p_th*100:.4f}%")
        print(f"    nu   = {nu:.3f}")
        print(f"    A    = {A:.4f}")
        print(f"    B    = {B:.4f}")
        print(f"    C    = {C:.4f}")
        print(f"    chi^2 = {res.fun:.2f} (DOF = {len(fit_data) - 5})")
    else:
        print(f"  Fit did not converge")

# Save aggregated data as CSV for the paper
import csv
with open('/home/claude/threshold_summary.csv', 'w') as f:
    w = csv.writer(f)
    w.writerow(['L', 'p', 'n_err', 'n_shots', 'rate', 'standard_error'])
    for L in sorted(by_L):
        for p, rate, se, ne, ns in by_L[L]:
            w.writerow([L, p, ne, ns, rate, se])
print(f"\n  Saved threshold_summary.csv")

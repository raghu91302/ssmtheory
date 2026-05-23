"""Quick simulation - free boundary, larger lattice, more cycles for stats."""
import numpy as np
from scipy.spatial import cKDTree

def fcc(L):
    pts = [[x,y,z] for x in range(L) for y in range(L) for z in range(L) if (x+y+z)%2==0]
    return np.array(pts, dtype=float) / np.sqrt(2)

L_eq = 1.0
R_min = 1.0 / np.sqrt(3)
L_crit = 2.0 / np.sqrt(3)
TARGET = np.sqrt(3) / 2**(2/3)

def run_sim(L_size=4, n_cycles=10, seed=0):
    np.random.seed(seed)
    pts = fcc(L_size)
    N0 = len(pts)
    tree = cKDTree(pts)
    bonds = list(tree.query_pairs(1.05))
    extent0 = (pts.max(0)-pts.min(0)).mean()
    
    history = [(0, 1.0, N0, extent0)]
    cum = 1.0
    
    for c in range(n_cycles):
        # Cosmic stretch
        factor = L_crit / L_eq
        pts = pts * factor
        cum *= factor
        
        # Insert one per node
        for _ in range(5):
            over = []
            for k, (i,j) in enumerate(bonds):
                d = np.linalg.norm(pts[i]-pts[j])
                if d > L_crit + 1e-9:
                    over.append((k,i,j,d))
            if not over: break
            over.sort(key=lambda x:-x[3])
            used=set(); new_pts=[]; to_remove=[]; to_add=[]
            for (_,i,j,d) in over:
                if i in used or j in used: continue
                mid = (pts[i]+pts[j])/2
                new_idx = len(pts) + len(new_pts)
                new_pts.append(mid)
                to_remove.append(tuple(sorted([i,j])))
                to_add.extend([(i,new_idx),(j,new_idx)])
                used.add(i); used.add(j)
            if not new_pts: break
            pts = np.vstack([pts, np.array(new_pts)])
            rs = set(to_remove)
            bonds = [b for b in bonds if tuple(sorted(b)) not in rs]
            bonds.extend(to_add)
        
        # Spring relax
        for it in range(800):
            forces = np.zeros_like(pts)
            max_strain = 0
            for (i,j) in bonds:
                dv = pts[j]-pts[i]
                d = np.linalg.norm(dv)
                if d < 1e-12: continue
                strain = (d-L_eq)/L_eq
                f = strain * dv / d
                forces[i] += f
                forces[j] -= f
                max_strain = max(max_strain, abs(strain))
            pts = pts + 0.02 * forces
            if max_strain < 1e-5: break
        
        extent = (pts.max(0)-pts.min(0)).mean()
        history.append((c+1, cum, len(pts), extent))
    
    return history, N0, extent0

# Run with L_size=4 (faster)
h, N0, e0 = run_sim(L_size=4, n_cycles=10, seed=0)
print(f"L_size=4, N0={N0}, extent0={e0:.3f}")
print(f"{'Cycle':>5} {'Stretch':>8} {'N':>5} {'Extent':>8} {'PerCycle':>9} {'Cumul':>8}")
for i, (c, s, N, e) in enumerate(h):
    a_eff = e/e0
    cum_r = a_eff/s if s>0 else 1
    if i > 0:
        prev_e = h[i-1][3]
        prev_s = h[i-1][1]
        pcr = (e/prev_e) / (s/prev_s)
    else:
        pcr = 1.0
    print(f"{c:>5d} {s:>8.4f} {N:>5d} {e:>8.3f} {pcr:>9.4f} {cum_r:>8.4f}")

# Compute mean per-cycle ratio (cycles 2 onwards)
ratios = []
for i in range(2, len(h)):
    e = h[i][3]
    e_prev = h[i-1][3]
    s = h[i][1]
    s_prev = h[i-1][1]
    r = (e/e_prev) / (s/s_prev)
    ratios.append(r)
print(f"\nMean per-cycle ratio (cycles 2+): {np.mean(ratios):.4f} ± {np.std(ratios):.4f}")
print(f"Target: sqrt(3)/2^(2/3) = {TARGET:.4f}")

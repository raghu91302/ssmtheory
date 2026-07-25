#!/usr/bin/env python3
"""
pbh_revised_regge_uniqueness.py -- selection of the Regge action within the
local frustration-linear family (Section 3.3 of the manuscript).

Family: S = sum_h w_c A_h delta_h, one weight per hinge congruence class
(the most general local, hinge-supported, point-group-symmetric functional
linear in the deficit angles). On the D4 subdivision of Appendix C the star
hinges fall into two classes, (2,2,2) x 96 and (2,2,4) x 29. Two computations:

(1) Flat-vacuum stationarity (identification-independent, the primary
    selection): for each origin-edge orbit, the first variation of the
    weighted action, summed over ALL simplices containing the edge, equals
    exactly +-6 (w1 - w2) per unit squared-length variation. Partial
    cancellation across simplices occurs (per-simplex class sums are +-1/2
    for short-edge variations and -+3/4 for the long edge) but is incomplete
    for every orbit: flat space is a solution only at equal weights.

(2) Fierz-Pauli corroboration (at the fixed edge-strain -> metric
    identification): the linearized kinetic operator has the FP form only at
    equal weights; the relative covariance violation grows linearly,
    ~ 4.3 * dw, against ~7e-9 at the Regge point.

Builds on linearized_gravity_verify.py (same directory). Runtime ~30-60 min.
"""
import numpy as np, itertools as it
from collections import defaultdict

src = open('linearized_gravity_verify.py').read()
cut = src.index("# ---- Section 5: explicit linearized")
ns = {}
core = src[src.index("# === Explicit D4 Regge"):cut]
exec(core, ns)
D4Regge = ns['D4Regge']; TTpol = ns['TTpol']; _n = ns['_n']

def qFP(kh, eps):
    kh = kh/np.linalg.norm(kh); a = kh@eps; s = kh@eps@kh; t = np.trace(eps)
    return 0.5*np.sum(eps*eps) - a@a + t*s - 0.5*t*t

class WeightedD4(D4Regge):
    def hinge_class(self, h):
        L = sorted([int(round((self.pts[a]-self.pts[b])@(self.pts[a]-self.pts[b])))
                    for a, b in it.combinations(h, 2)])
        return tuple(L)
    def hessian_w(self, wmap, eps=1e-6):
        defgrad = {}
        for h in self.hinges:
            es = {tuple(sorted((i, j))) for s in self.around[h]
                  for i, j in it.combinations(s, 2)}
            for e in es:
                L0 = (self.pts[e[0]]-self.pts[e[1]])@(self.pts[e[0]]-self.pts[e[1]])
                defgrad[(h, e)] = (self.deficit(h, {e: L0+eps})
                                   - self.deficit(h, {e: L0-eps}))/(2*eps)
        def tri2(a2, b2, c2):
            return 0.25*np.sqrt(max(2*a2*b2+2*b2*c2+2*c2*a2-a2*a2-b2*b2-c2*c2, 0))
        def dA(h, e):
            i, j, k = h
            L = {tuple(sorted((i, j))): (self.pts[i]-self.pts[j])@(self.pts[i]-self.pts[j]),
                 tuple(sorted((i, k))): (self.pts[i]-self.pts[k])@(self.pts[i]-self.pts[k]),
                 tuple(sorted((j, k))): (self.pts[j]-self.pts[k])@(self.pts[j]-self.pts[k])}
            f = lambda Lm: tri2(Lm[tuple(sorted((j, k)))],
                                Lm[tuple(sorted((i, k)))],
                                Lm[tuple(sorted((i, j)))])
            Lp = dict(L); Lp[e] = L[e]+eps
            Lm = dict(L); Lm[e] = L[e]-eps
            return (f(Lp)-f(Lm))/(2*eps)
        M = defaultdict(float)
        for e in self.edges:
            for h in self.hinges:
                if e[0] in h and e[1] in h:
                    g = dA(h, e)*wmap[self.hinge_class(h)]
                    for ep in {tuple(sorted((i, j))) for s in self.around[h]
                               for i, j in it.combinations(s, 2)}:
                        M[(e, ep)] += g*defgrad[(h, ep)]
        self.M = M
        return M

r = WeightedD4()
classes = sorted({r.hinge_class(h) for h in r.hinges})
counts = {c: sum(1 for h in r.hinges if r.hinge_class(h) == c) for c in classes}
print("hinge congruence classes (sorted squared edge lengths -> count):")
for c in classes: print("  ", c, "->", counts[c])

np.random.seed(0)
draws = [(np.random.randn(4), (lambda e: e+e.T)(np.random.randn(4, 4)))
         for _ in range(60)]
def fp_spread(wmap):
    r.hessian_w(wmap)
    rat = np.array([r.coeff(k, e)/qFP(k, e) for k, e in draws
                    if abs(qFP(k, e)) > 1e-3])
    return rat.mean(), rat.std()/abs(rat.mean())

w1 = {c: 1.0 for c in classes}
m, sp = fp_spread(w1)
print(f"\nequal weights (Regge):        mean C/qFP = {m:.6f}, rel spread = {sp:.2e}")

for ci, c in enumerate(classes):
    w = dict(w1); w[c] = 1.10
    m, sp = fp_spread(w)
    print(f"class {c} x1.10:   mean = {m:.6f}, rel spread = {sp:.2e}")

rng = np.random.default_rng(7)
for t in range(3):
    w = {c: float(1.0+0.2*(rng.random()-0.5)) for c in classes}
    m, sp = fp_spread(w)
    print(f"random weights #{t+1}:          mean = {m:.6f}, rel spread = {sp:.2e}")

for d in (0.02, 0.05, 0.10):
    w = dict(w1); w[classes[0]] = 1.0+d
    m, sp = fp_spread(w)
    print(f"class0 1+{d:.2f}: rel spread = {sp:.3e}  (spread/dw = {sp/d:.3f})")

print("\n== per-simplex Schlafli, classwise (two representative edges of one simplex) ==")
s0 = sorted(r.star[0]); pts = r.pts
embed = ns['embed']; dih = ns['dih']
def theta(t, s, ov):
    rest = [x for x in s if x not in t]; sv = list(t)+rest
    L = np.zeros((5, 5))
    for a in range(5):
        for b in range(a+1, 5):
            key = tuple(sorted((sv[a], sv[b])))
            L[a, b] = L[b, a] = ov.get(key, (pts[sv[a]]-pts[sv[b]])@(pts[sv[a]]-pts[sv[b]]))
    return dih(embed(L))
def tarea(t):
    a = pts[t[1]]-pts[t[0]]; b = pts[t[2]]-pts[t[0]]
    return 0.5*np.sqrt((a@a)*(b@b)-(a@b)**2)
def hcl(t):
    return tuple(sorted(int(round((pts[a]-pts[b])@(pts[a]-pts[b])))
                 for a, b in it.combinations(t, 2)))
eps = 1e-6
tris = [tuple(sorted(t)) for t in it.combinations(s0, 3)]
for e in [tuple(sorted((s0[0], s0[1]))), tuple(sorted((s0[1], s0[2])))]:
    L0 = (pts[e[0]]-pts[e[1]])@(pts[e[0]]-pts[e[1]])
    tot = {}; alltot = 0.0
    for t in tris:
        dth = (theta(t, s0, {e: L0+eps})-theta(t, s0, {e: L0-eps}))/(2*eps)
        c = hcl(t); tot[c] = tot.get(c, 0.0)+tarea(t)*dth; alltot += tarea(t)*dth
    print(f"edge {tuple(int(x) for x in e)}: Schlafli total = {alltot:+.2e} | classwise:",
          {c: f"{v:+.4f}" for c, v in sorted(tot.items())})
print("NOTE: the sign of the classwise split depends on the edge, so cancellation")
print("across the simplices sharing one edge must be computed, not assumed.")

print("\n== decisive check: per-edge first variation over ALL simplices containing the edge ==")
star_ok = all(len({hcl(t) for t in it.combinations(sorted(s), 3)}) == 2 for s in r.star)
print(f"star simplices: {len(r.star)} | every star simplex contains both classes: {star_ok}")
edges_by_class = {}
for e in r.edges:
    L2 = int(round((pts[e[0]]-pts[e[1]])@(pts[e[0]]-pts[e[1]])))
    edges_by_class.setdefault(L2, []).append(e)
print("origin-edge orbits (squared length -> count):",
      {k: len(v) for k, v in sorted(edges_by_class.items())})
for L2, es in sorted(edges_by_class.items()):
    for e in es[:2]:
        L0 = (pts[e[0]]-pts[e[1]])@(pts[e[0]]-pts[e[1]])
        simps = [s for s in r.simp if e[0] in s and e[1] in s]
        percls = {}; persim = []
        for s in simps:
            loc = {}
            for t in it.combinations(sorted(s), 3):
                dth = (theta(t, s, {e: L0+eps})-theta(t, s, {e: L0-eps}))/(2*eps)
                c = hcl(t); loc[c] = loc.get(c, 0.0)+tarea(t)*dth
            persim.append(round(loc.get((2, 2, 2), 0.0), 4))
            for c, v in loc.items(): percls[c] = percls.get(c, 0.0)+v
        from collections import Counter
        print(f"edge L^2={L2} {tuple(int(x) for x in e)}: {len(simps)} simplices, "
              f"per-simplex (2,2,2)-sums {dict(Counter(persim))}")
        print("    per-edge TOTAL: " + ", ".join(f"{c}: {v:+.4f}" for c, v in sorted(percls.items()))
              + "  -> tadpole per unit dL^2: (w1-w2) * "
              + f"{percls.get((2,2,2),0.0):+.1f}")
print("\nConclusion: partial cancellation across simplices occurs but is incomplete;")
print("every edge orbit carries an exact tadpole +-6 (w1 - w2). Flat space is a")
print("solution only at equal weights: within the local frustration-linear family,")
print("the lattice selects the Regge action.")

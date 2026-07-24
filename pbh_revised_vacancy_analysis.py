#!/usr/bin/env python3
"""Punctured FCC CSS code study (independent reimplementation).
Build the [[3L^3, 2L^3+2, 3]] code of arXiv:2603.20294, carve spherical
vacancies, and compute the static code-theoretic quantities vs radius."""
import numpy as np
import itertools as it

def gf2_rank(M):
    M = M.copy() % 2
    r = 0
    rows, cols = M.shape
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if M[i, c]:
                piv = i; break
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(rows):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        r += 1
        if r == rows:
            break
    return r

def gf2_in_rowspace(M, v):
    A = np.vstack([M, v]) % 2
    return gf2_rank(A) == gf2_rank(M)

NN = []
for i in range(3):
    for j in range(3):
        if i < j:
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(3, int); v[i] = si; v[j] = sj
                    NN.append(v)
E6 = [np.eye(3, dtype=int)[i] * s for i in range(3) for s in (1, -1)]

class FCCCode:
    def __init__(self, L):
        self.L = L
        self.nodes = [np.array(p) for p in it.product(range(L), repeat=3)
                      if sum(p) % 2 == 0]
        self.nidx = {tuple(p): i for i, p in enumerate(self.nodes)}
        self.octs = [np.array(p) for p in it.product(range(L), repeat=3)
                     if sum(p) % 2 == 1]
        self.edges = []
        self.eidx = {}
        for p in self.nodes:
            for d in NN:
                q = (p + d) % L
                key = tuple(sorted((tuple(p), tuple(q))))
                if key not in self.eidx:
                    self.eidx[key] = len(self.edges)
                    self.edges.append(key)
        n = len(self.edges)
        self.HZ = np.zeros((len(self.nodes), n), dtype=np.uint8)
        for iv, p in enumerate(self.nodes):
            for d in NN:
                q = (p + d) % L
                self.HZ[iv, self.eidx[tuple(sorted((tuple(p), tuple(q))))]] = 1
        self.HX = np.zeros((len(self.octs), n), dtype=np.uint8)
        for io, c in enumerate(self.octs):
            surr = [tuple((c + e) % L) for e in E6]
            cnt = 0
            for a, b in it.combinations(surr, 2):
                key = tuple(sorted((a, b)))
                if key in self.eidx:
                    da = np.abs(np.array(a) - np.array(b))
                    da = np.minimum(da, self.L - da)
                    if int(np.sum(da * da)) == 2:
                        self.HX[io, self.eidx[key]] = 1
                        cnt += 1
            assert cnt == 12, (tuple(c), cnt)

    def params(self):
        n = len(self.edges)
        rz, rx = gf2_rank(self.HZ), gf2_rank(self.HX)
        return n, n - rz - rx, rz, rx

def torus_dist2(a, b, L):
    d = np.abs(np.array(a) - np.array(b))
    d = np.minimum(d, L - d)
    return int(np.sum(d * d))

def puncture(code, R_int2):
    L = code.L
    removed_nodes = {tuple(p) for p in code.nodes
                     if torus_dist2(p, (0, 0, 0), L) < R_int2}
    removed_edges = set()
    severed = 0
    for kk, (a, b) in enumerate(code.edges):
        ain, bin_ = a in removed_nodes, b in removed_nodes
        if ain or bin_:
            removed_edges.add(kk)
            if ain != bin_:
                severed += 1
    keep = np.array([kk for kk in range(len(code.edges))
                     if kk not in removed_edges])
    zrows = [i for i, p in enumerate(code.nodes)
             if tuple(p) not in removed_nodes]
    HZp = code.HZ[np.ix_(zrows, keep)]
    xrows = []
    for io, c in enumerate(code.octs):
        surr = [tuple((c + e) % L) for e in E6]
        if not any(s in removed_nodes for s in surr):
            xrows.append(io)
    HXp = code.HX[np.ix_(xrows, keep)]
    return HZp, HXp, keep, removed_nodes, removed_edges, severed

def low_weight_logicals(HZp, HXp):
    out = {}
    for name, Hk, Hr in (("X", HZp, HXp), ("Z", HXp, HZp)):
        zero_cols = np.where(~Hk.any(axis=0))[0]
        w1 = 0
        for c in zero_cols:
            v = np.zeros(Hk.shape[1], dtype=np.uint8); v[c] = 1
            if not gf2_in_rowspace(Hr, v):
                w1 += 1
        colkeys = {}
        w2 = 0
        for c in range(Hk.shape[1]):
            key = Hk[:, c].tobytes()
            if key in colkeys:
                v = np.zeros(Hk.shape[1], dtype=np.uint8)
                v[c] = 1; v[colkeys[key]] = 1
                if not gf2_in_rowspace(Hr, v):
                    w2 += 1
            else:
                colkeys[key] = c
        out[name] = (w1, w2)
    return out

def exterior_triangle_logical(code, keep, removed_nodes, HXp):
    L = code.L
    kmap = {kk: i for i, kk in enumerate(keep)}
    best = None; bestd = -1
    for p in code.nodes:
        tp = tuple(p)
        if tp in removed_nodes:
            continue
        d2 = torus_dist2(p, (0, 0, 0), L)
        q = tuple((p + np.array([1, 1, 0])) % L)
        r = tuple((p + np.array([1, 0, 1])) % L)
        if q in removed_nodes or r in removed_nodes:
            continue
        try:
            eks = [kmap[code.eidx[tuple(sorted((tp, q)))]],
                   kmap[code.eidx[tuple(sorted((tp, r)))]],
                   kmap[code.eidx[tuple(sorted((q, r)))]]]
        except KeyError:
            continue
        if d2 > bestd:
            bestd = d2; best = eks
    v = np.zeros(len(keep), dtype=np.uint8)
    for ek in best:
        v[ek] = 1
    return (not gf2_in_rowspace(HXp, v)), np.sqrt(bestd / 2.0)

def conversion_step_stats(code, removed_nodes):
    L = code.L
    stats = []
    for p in code.nodes:
        tp = tuple(p)
        if tp in removed_nodes:
            continue
        nbrs = [tuple((p + d) % L) for d in NN]
        if not any(nb in removed_nodes for nb in nbrs):
            continue
        surv_edges = sum(1 for nb in nbrs if nb not in removed_nodes)
        z_affected = 1 + surv_edges
        octs = {tuple((p + e) % L) for e in E6}
        x_affected = sum(1 for c in octs if not any(
            tuple((np.array(c) + e) % L) in removed_nodes for e in E6))
        stats.append((surv_edges, z_affected, x_affected))
    a = np.array(stats)
    return a.mean(axis=0), a.max(axis=0), len(stats)

if __name__ == "__main__":
    for L in (6, 8):
        code = FCCCode(L)
        n, k, rz, rx = code.params()
        print(f"L={L}: n={n}, k={k} (expect {2*L**3+2}), rank HZ={rz}, rank HX={rx}")
    code = FCCCode(8)
    n0, k0, _, _ = code.params()
    print(f"\nvacancy study at L=8 (n0={n0}, k0={k0}):")
    rows = []
    for RL0 in (1.5, 2.0, 2.5, 3.0, 3.5):
        R_int2 = 2.0 * RL0**2
        HZp, HXp, keep, rn, re, severed = puncture(code, R_int2)
        npr = len(keep)
        kp = npr - gf2_rank(HZp) - gf2_rank(HXp)
        lw = low_weight_logicals(HZp, HXp)
        tri, td = exterior_triangle_logical(code, keep, rn, HXp)
        (m_e, m_z, m_x), (M_e, M_z, M_x), nb = conversion_step_stats(code, rn)
        A = 4 * np.pi * RL0**2
        d3 = tri and all(v == (0, 0) for v in lw.values())
        print(f" R={RL0}: nodes_rm={len(rn)}, edges_rm={len(re)}, severed={severed},"
              f" n'={npr}, k'={kp}, deficit={k0-kp},"
              f" deficit/edges_rm={(k0-kp)/len(re):.3f}")
        print(f"   w<=2 logicals {lw} | ext triangle nontrivial={tri}"
              f" (dist {td:.2f} L0) => d'=3: {d3}")
        print(f"   boundary nodes={nb}: surv edges mean={m_e:.1f},"
              f" Z-checks max={M_z}, X-checks max={M_x},"
              f" severed/A={severed/A:.3f} vs 3sqrt2={3*np.sqrt(2):.3f}")
        rows.append((RL0, len(rn), len(re), severed, npr, kp, k0 - kp,
                     m_e, M_z, M_x, int(d3)))
    np.save("/tmp/vacancy_rows.npy", np.array(rows))

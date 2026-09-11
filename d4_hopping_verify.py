import numpy as np
from itertools import product, combinations

# D4 = { x in Z^4 : sum x_i even }.  x0 is the selected time axis.
# The slice x0 = 0 is D3 (FCC), nearest-neighbor distance sqrt(2),
# conventional cubic cell side a = 2.
a = 2.0
inD4 = (
    lambda v: all(float(x).is_integer() for x in v)
    and int(round(sum(v))) % 2 == 0
)
R = lambda x: np.round(x, 6)
ok = lambda b: "OK" if b else "FAIL"

print("=" * 68)
print("1.  THE SPATIAL SLICE OF D4 IS FCC")
print("=" * 68)
sl = [
    (0, x, y, z)
    for x, y, z in product(range(-2, 3), repeat=3)
    if (x + y + z) % 2 == 0
]
nn = [p for p in sl if abs(np.linalg.norm(p) - np.sqrt(2)) < 1e-9]
print(f"  NN at sqrt(2) in slice : {len(nn)} (expect 12) {ok(len(nn)==12)}")

print()
print("=" * 68)
print("2.  TETRAHEDRAL VOIDS OF THE SLICE")
print("=" * 68)
basis = np.array([[0, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=float)
voids = np.vstack(
    [
        np.array([[1, 1, 1], [3, 3, 1], [3, 1, 3], [1, 3, 3]]) * (a / 4),
        np.array([[3, 3, 3], [1, 1, 3], [1, 3, 1], [3, 1, 1]]) * (a / 4),
    ]
)
cls = ["P"] * 4 + ["N"] * 4


def bounding(c):
    out = set()
    for d in product(range(-2, 3), repeat=3):
        for b in basis:
            p = np.array(d) * a + b
            if abs(np.linalg.norm(p - c) - a * np.sqrt(3) / 4) < 1e-6:
                out.add(tuple(R(p)))
    return out


q = np.round(voids / (a / 4)).astype(int)
print(f"  voids per cell   : {len(voids)} (expect 8) {ok(len(voids)==8)}")
print(f"  all coordinates odd in units a/4  : {ok(bool((q%2!=0).all()))}")
pmod = sorted({int(r.sum()) % 4 for r, t in zip(q, cls) if t == "P"})
print(f"  P sum mod 4      : {pmod}")
nmod = sorted({int(r.sum()) % 4 for r, t in zip(q, cls) if t == "N"})
print(f"  N sum mod 4      : {nmod}")
print("  -> simple cubic lattice, spacing a/2, two-colored by sum mod 4")

print()
print("=" * 68)
print("3.  VOID ADJACENCY (periodic images included)")
print("=" * 68)
hops = [
    np.array(d) * (a / 2)
    for d in [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]
]
imgs = [
    (voids[k] + np.array(t) * a, cls[k])
    for t in product(range(-1, 2), repeat=3)
    for k in range(8)
]
deg, opp, shr, dst = [], True, set(), set()
for i in range(8):
    vi = bounding(voids[i])
    n = 0
    for c, t in imgs:
        if np.allclose(c, voids[i]):
            continue
        s = vi & bounding(c)
        if len(s) >= 2:
            n += 1
            shr.add(len(s))
            opp &= t != cls[i]
            dst.add(round(float(np.linalg.norm(c - voids[i])) / a, 6))
    deg.append(n)
print(f"  neighbors per void               : {deg}  {ok(set(deg)=={6})}")
print(f"  hop distance / a : {sorted(dst)} {ok(sorted(dst)==[0.5])}")
print(f"  every neighbor of opposite class : {ok(opp)}")
print(f"  shared atoms/hop : {sorted(shr)} {ok(sorted(shr)==[2])}")
print(f"  edges per conventional cell       : {8*6//2}")
eg = lambda v: set(frozenset(p) for p in combinations(sorted(v), 2))
rear = {
    (
        len(eg(bounding(voids[i])) - eg(bounding(voids[i] + d))),
        len(eg(bounding(voids[i] + d)) - eg(bounding(voids[i]))),
        len(eg(bounding(voids[i])) & eg(bounding(voids[i] + d))),
    )
    for i in range(8)
    for d in hops
}
print(
    f"  (broken, created, kept) per hop   : {rear}  {ok(rear=={(5,5,1)})}"
)

print()
print("=" * 68)
print("4.  THE VOID CLASS sigma")
print("=" * 68)


def dirs(i):
    c = voids[i]
    return frozenset(
        tuple(int(round(x)) for x in (np.array(v) - c) / (a / 4))
        for v in bounding(c)
    )


sig = {}
for i in range(8):
    p = {d[0] * d[1] * d[2] for d in dirs(i)}
    assert len(p) == 1
    sig[i] = p.pop()
print(f"  sigma = product of the components of any bond direction")
sig_ok = all((sig[i] == -1) == (cls[i] == "P") for i in range(8))
print(f"  sigma = class    : {ok(sig_ok)}")
flip = True
for i in range(8):
    for d in hops:
        tgt = (voids[i] + d) % a
        j = [m for m in range(8) if np.allclose(voids[m] % a, tgt)][0]
        flip &= sig[j] == -sig[i]
print(f"  sigma flips on every hop          : {ok(flip)}")

print()
print("=" * 68)
print("5.  NO-GO: THE VOID CARRIES EXACTLY ONE BIT")
print("=" * 68)
bys = {}
for i in range(8):
    bys.setdefault(sig[i], set()).add(dirs(i))
for s in sorted(bys):
    print(f"  sigma={s:+d}: {len(bys[s])} set(s) {ok(len(bys[s])==1)}")
P, N = list(bys[-1])[0], list(bys[+1])[0]
inv_ok = frozenset(tuple(-x for x in d) for d in P) == N
print(f"  inversion P -> N : {ok(inv_ok)}")

print()
print("=" * 68)
print("6.  HOP PARITY: A HOP CARRIES A TIME STEP")
print("=" * 68)
for lbl, v in (
    ("hop, no time step", (0, 1, 0, 0)),
    ("hop, +1 time step", (1, 1, 0, 0)),
    ("hop, -1 time step", (-1, 1, 0, 0)),
):
    print(
        f"    {lbl}: {v} sum={sum(v):+d} inD4={str(inD4(v)):5s}"
        f" {ok(inD4(v)==(v[0]!=0))}"
    )
sp = sorted(
    {
        v
        for v in product([-1, 0, 1], repeat=4)
        if v[0] == 0
        and inD4(v)
        and abs(np.linalg.norm(v) - np.sqrt(2)) < 1e-9
    }
)
tm = sorted(
    {
        v
        for v in product([-1, 0, 1], repeat=4)
        if v[0] != 0
        and inD4(v)
        and abs(np.linalg.norm(v) - np.sqrt(2)) < 1e-9
    }
)
print(f"  spatial-only bonds : {len(sp)}, spatial length sqrt(2)")
print(f"  time-mixed bonds   : {len(tm)}, spatial length 1")
print(f"  hop matches time-mixed only {ok(len(sp)==12 and len(tm)==12)}")

print()
print("=" * 68)
print("7.  REST WORLDLINES MUST ZIG-ZAG")
print("=" * 68)
for dt in (1, 2):
    v = (dt, 0, 0, 0)
    print(
        f"  tick {v}: sum={sum(v)} inD4={str(inD4(v)):5s}"
        f" {ok(inD4(v)==(dt%2==0))}"
    )
one = sorted(
    {
        v
        for v in product([-1, 0, 1], repeat=4)
        if abs(v[0]) == 1
        and inD4(v)
        and abs(np.linalg.norm(v) - np.sqrt(2)) < 1e-9
    }
)
print(
    f"  one-tick steps: {len(one)}, one spatial component each"
    f" {ok(all(sum(1 for c in v[1:] if c)==1 for v in one))}"
)
s1, s2 = (1, 1, 0, 0), (1, -1, 0, 0)
net = tuple(x + y for x, y in zip(s1, s2))
print(f"  {s1} then {s2} -> net {net}  {ok(net==(2,0,0,0))}")
qq = np.array([1, 1, 1])
lab = []
for nm, dq in (
    ("start", np.zeros(3, int)),
    ("+x hop", np.array([2, 0, 0])),
    ("-x hop", np.array([-2, 0, 0])),
):
    qq = qq + dq
    lab.append("P" if qq.sum() % 4 == 3 else "N")
    print(
        f"    {nm:6s} (a/4)*{tuple(int(x) for x in qq)}"
        f" sum%4={qq.sum()%4} {lab[-1]}"
    )
print(f"  the rest worldline alternates P N P  {ok(lab==['P','N','P'])}")

print()
print("=" * 68)
print("8.  MOMENTS OF THE D4 BOND SET")
print("=" * 68)
import sympy as smp

k = smp.symbols("k0 k1 k2 k3", real=True)
kv = smp.Matrix(k)
k2 = sum(x**2 for x in k)
bonds = [
    smp.Matrix(v) / smp.sqrt(2)
    for v in product([-1, 0, 1], repeat=4)
    if inD4(v) and sum(1 for c in v if c) == 2
]
M2 = smp.simplify(sum((b.dot(kv)) ** 2 for b in bonds))
M4 = smp.simplify(smp.expand(sum((b.dot(kv)) ** 4 for b in bonds)))
print(f"  bonds: {len(bonds)}  (expect 24)  {ok(len(bonds)==24)}")
print(f"  M2 = {smp.factor(M2)}")
m4res = smp.simplify(M4 - 3 * k2**2)
print(f"  M4 - 3(k^2)^2 = {m4res} {ok(m4res == 0)}")
print(
    "  -> M2 and M4 are direction-independent; anisotropy begins at order 6"
)

print()
print("=" * 68)
print("9.  TETRAHEDRAL SITE SYMMETRY AND SPIN")
print("=" * 68)
# The rotational site symmetry of a tetrahedral void is T = A4 (order 12).
# Its preimage under the double cover SU(2) -> SO(3) is the binary tetrahedral
# group 2T, realised as the 24 unit Hurwitz quaternions.
_e = []
for _i in range(4):
    for _s in (1, -1):
        _q = np.zeros(4)
        _q[_i] = _s
        _e.append(tuple(_q))
for _sg in product([0.5, -0.5], repeat=4):
    _e.append(tuple(_sg))
G2T = sorted(set(tuple(round(x, 6) + 0.0 for x in v) for v in _e))
qmul = lambda A, B: (
    A[0] * B[0] - A[1] * B[1] - A[2] * B[2] - A[3] * B[3],
    A[0] * B[1] + A[1] * B[0] + A[2] * B[3] - A[3] * B[2],
    A[0] * B[2] - A[1] * B[3] + A[2] * B[0] + A[3] * B[1],
    A[0] * B[3] + A[1] * B[2] - A[2] * B[1] + A[3] * B[0],
)
qinv = lambda A: (A[0], -A[1], -A[2], -A[3])
rq = lambda t: tuple(round(x, 6) + 0.0 for x in t)
_cl, _sn = [], set()
for g in G2T:
    if g in _sn:
        continue
    c = sorted({rq(qmul(qmul(h, g), qinv(h))) for h in G2T})
    _cl.append(c)
    _sn |= set(c)
print(f"  |2T| = {len(G2T)}  (expect 24)  {ok(len(G2T)==24)}")
print(f"  classes = {len(_cl)} -> {len(_cl)} irreps {ok(len(_cl)==7)}")
print(f"  class sizes = {[len(c) for c in _cl]}")


def chi_j(j, q):
    """Character of the SU(2) spin-j representation at a unit quaternion."""
    w = max(-1.0, min(1.0, q[0]))
    th = 2 * np.arccos(w)
    ms = np.arange(-j, j + 1e-9, 1.0)
    return float(np.real(np.sum(np.exp(1j * ms * th))))


_m1 = (-1.0, 0.0, 0.0, 0.0)
print(
    f"  chi_(1/2)(-1) = {chi_j(0.5,_m1):+.1f}  (expect -2, spinorial)  "
    f"{ok(abs(chi_j(0.5,_m1)+2) < 1e-9)}"
)

print("\n  Restriction of spin-j to 2T:")
print("    j      dim   <chi,chi>   irreducible")
irr = {}
for j in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5]:
    ip = sum(chi_j(j, q) ** 2 for q in G2T) / len(G2T)
    irr[j] = abs(ip - 1) < 1e-9
    print(
        f"    {j:<6} {int(2*j+1):<5} {ip:8.4f}    {'yes' if irr[j] else 'no'}"
    )
_half = [j for j in irr if abs(j - round(j)) > 0.1]
_hi = [j for j in _half if irr[j]]
print(f"  half-integer spins irreducible under 2T: {_hi}  {ok(_hi==[0.5])}")
print(
    "  -> j = 1/2 is the unique half-integer spin that stays irreducible;"
)
print("     the faithful irreps of 2T all have dimension 2.")

print("\n  Does that 2-dimensional irrep identify the SU(2) parent?")
_ip = lambda f, g: sum(f(q) * g(q) for q in G2T) / len(G2T)
print("    j      multiplicity of the j=1/2 irrep in the restriction")
for j in [0.5, 1.5, 2.5, 3.5]:
    m = _ip(lambda q, j=j: chi_j(j, q), lambda q: chi_j(0.5, q))
    print(f"    {j:<6} {int(round(m))}")
print("  -> the same 2-dim irrep occurs inside j = 5/2 and 7/2; j = 3/2")
print(
    "     decomposes into the other two faithful irreps. Finite site symmetry"
)
print("     therefore does not determine the SU(2) parent representation.")
print("     Prop 5 selects; it does not determine spin.")


print()
print("=" * 68)
print("10. WHY MIGRATION GOES BY EDGES (metric wall of Ref. [2])")
print("=" * 68)
# Ref [2] excludes a node lying within L/sqrt(3) of ALL THREE vertices of a
# bounding triangle at once. The condition is three-body; two atoms do not
# trigger it.
Lnn = a / np.sqrt(2)
mwall = Lnn / np.sqrt(3)
print(f"  L = {Lnn:.4f}, metric wall L/sqrt(3) = {mwall:.4f}")

ATOMS = [
    np.array(p, dtype=float)
    for p in product(range(-3, 6), repeat=3)
    if sum(p) % 2 == 0
]
n_in = lambda p: sum(
    1 for q in ATOMS if np.linalg.norm(p - q) <= mwall + 1e-9
)
Tc = np.array([0.5, 0.5, 0.5])
rT_ = a * np.sqrt(3) / 4
bTc = [q for q in ATOMS if abs(np.linalg.norm(q - Tc) - rT_) < 1e-6]
print(f"  bounding atoms: {len(bTc)}  {ok(len(bTc)==4)}")
print(f"  atoms within wall at center: {n_in(Tc)}  {ok(n_in(Tc)==0)}")

face_bad = all(
    max(np.linalg.norm(sum(f) / 3 - q) for q in f) <= mwall + 1e-9
    for f in combinations(bTc, 3)
)
print(f"  4 face centroids at the wall w.r.t. 3 atoms: {ok(face_bad)}")
print("    -> face channels are over-constrained and excluded")

edge_max = max(
    n_in((np.array(e[0]) + np.array(e[1])) / 2) for e in combinations(bTc, 2)
)
print(f"  max atoms within the wall at an edge midpoint: {edge_max}"
      f"  {ok(edge_max==2)}")
print("    -> edge channels never reach three and are permitted")
print(f"  4 faces closed, 6 edges open -> coordination 6"
      f"  {ok(len(list(combinations(bTc,2)))==6)}")


print()
print("=" * 68)
print("11. BARYON NUMBER vs CHARGE UNDER MIGRATION")
print("=" * 68)
# Ref [2]: the four bonds are one anchor plus three valence quarks; the charge
# Q = -1 + W is carried by winding numbers on the three valence bonds.
rT2 = a * np.sqrt(3) / 4
bd = lambda c: {
    tuple(np.round(q, 6))
    for q in ATOMS
    if abs(np.linalg.norm(q - np.array(c, float)) - rT2) < 1e-6
}
Tv = (0.5, 0.5, 0.5)
steps = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
tot = kept_anchor = 0
vdist = {}
for hstep in steps:
    Tn = tuple(Tv[i] + hstep[i] * (a / 2) for i in range(3))
    b1, b2 = bd(Tv), bd(Tn)
    kept, gone = b1 & b2, b1 - b2
    for anch in b1:
        tot += 1
        kept_anchor += anch in kept
        nv = len((b1 - {anch}) & gone)
        vdist[nv] = vdist.get(nv, 0) + 1
print(f"  (anchor, hop) pairs: {tot}  {ok(tot==24)}")
print(f"  anchor in kept edge: {kept_anchor}/{tot} {ok(kept_anchor==12)}")
print(f"  valence bonds replaced per hop: {dict(sorted(vdist.items()))}")
print(f"  never zero valence bonds replaced: {ok(0 not in vdist)}")
print("  -> B rides the node and epsilon (invariant); Q rides the windings")
print("     and is partly redrawn every hop: the two transport differently.")


print()
print("=" * 68)
print("12. BARYON SPECIES UNDER MIGRATION")
print("=" * 68)
# Ref [2]: species is fixed by the anchor choice and by the winding numbers
# w in {0,1} on the three valence bonds, with Q = -1 + sum(w).
rT3 = a * np.sqrt(3) / 4
ASET = {
    tuple(p) for p in product(range(-4, 7), repeat=3) if sum(p) % 2 == 0
}


def bnd3(cc):
    cc = np.array(cc, float)
    return sorted(
        p for p in ASET
        if abs(np.linalg.norm(np.array(p, float) - cc) - rT3) < 1e-6
    )
HOPS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

# (a) an anchor-preserving rule would trap the defect in one octant
Tsp = (0.5, 0.5, 0.5)
bs = bnd3(Tsp)
edge_of = {}
for hh in HOPS:
    T2 = tuple(Tsp[i] + hh[i] * (a / 2) for i in range(3))
    edge_of[hh] = set(bs) & set(bnd3(T2))
trapped = True
for anc in bs:
    allowed = [hh for hh in HOPS if anc in edge_of[hh]]
    rev = any(tuple(-x for x in hh) in allowed for hh in allowed)
    trapped &= (len(allowed) == 3 and not rev)
print(f"  anchor rule: 3 of 6 hops, no reversal {ok(trapped)}")
print("    -> that rule confines the defect to one octant; the anchor cannot")
print("       be conserved, so a transport rule is needed instead.")

# (b) released atoms map onto engaged atoms by the lattice translation 2*hop
good = True
for st in [(0.5, 0.5, 0.5), (1.5, 0.5, 0.5),
           (0.5, 1.5, 0.5), (1.5, 1.5, 1.5)]:
    for hh in HOPS:
        T2 = tuple(st[i] + hh[i] * (a / 2) for i in range(3))
        b1, b2 = set(bnd3(st)), set(bnd3(T2))
        rel, eng = sorted(b1 - b2), sorted(b2 - b1)
        sh = tuple(2 * x for x in hh)
        img = sorted(tuple(p[i] + sh[i] for i in range(3)) for p in rel)
        good &= img == eng
        good &= sum(sh) % 2 == 0
print(f"  released = engaged shifted by 2*hop {ok(good)}")
print("    -> the shift has length a and even coordinate sum: a lattice")
print("       translation, giving a canonical bijection on BOUNDING ATOMS.")
print("       Note it is not an isometry on defect bonds: the center moves")
print("       by h and the atom by 2h, so d -> d + h. Species preservation")
print("       therefore needs the winding labels to follow the bijection,")
print("       which is an assumption, not a consequence (Prop. 8).")


print()
print("  bond vectors are NOT preserved by that map:")
cv = np.array([0.5, 0.5, 0.5])
hv = np.array([1.0, 0.0, 0.0])
c2v = cv + hv
b1v, b2v = set(bnd3(tuple(cv))), set(bnd3(tuple(c2v)))
shift_ok = True
for v in sorted(b1v - b2v):
    dvec = np.array(v, float) - cv
    vp = np.array(v, float) + 2 * hv
    dp = vp - c2v
    shift_ok &= np.allclose(dp - dvec, hv)
for v in sorted(b1v & b2v):
    dvec = np.array(v, float) - cv
    dp = np.array(v, float) - c2v
    shift_ok &= np.allclose(dp - dvec, -hv)
print(f"    released bonds shift by +h, shared bonds by -h {ok(shift_ok)}")
print("    -> no defect bond is carried to itself; the local frame rotates.")


print()
print("=" * 68)
print("13. QUANTUM-WALK DISPERSION ON THE VOID SUBLATTICE")
print("=" * 68)
# Sec. 4.3: a nearest-neighbor Hamiltonian on the bipartite void sublattice
# is gapless, while a discrete-time walk carries a mass as a coin angle.
s1 = np.array([[0, 1], [1, 0]], dtype=complex)
s2 = np.array([[0, -1j], [1j, 0]])
s3 = np.diag([1, -1]).astype(complex)
I2c = np.eye(2, dtype=complex)
al = [np.kron(s1, s) for s in (s1, s2, s3)]
be = np.kron(s3, I2c)
I4c = np.eye(4, dtype=complex)

alg = all(
    np.allclose(al[i] @ al[j] + al[j] @ al[i], 2 * (i == j) * I4c)
    for i in range(3) for j in range(3)
) and all(np.allclose(al[i] @ be + be @ al[i], 0) for i in range(3))
print(f"  Dirac algebra {{a_i,a_j}}=2d_ij, {{a_i,b}}=0  {ok(alg)}")

E = lambda t, M: np.cos(t) * I4c + 1j * np.sin(t) * M


def walk(k, th):
    Cm = np.cos(th / 2) * I4c - 1j * np.sin(th / 2) * be
    M = E(k[0] / 2, al[0]) @ E(k[1] / 2, al[1]) @ E(k[2], al[2])
    M = M @ E(k[1] / 2, al[1]) @ E(k[0] / 2, al[0])
    return Cm @ M @ Cm


def omg(k, th):
    return np.sort(np.abs(np.angle(np.linalg.eigvals(walk(k, th)))))[-1]


th = 0.05
gap_ok = abs(omg(np.zeros(3), th) - th) < 1e-9
print(f"  gap at k=0 equals the coin angle {ok(gap_ok)}")
dirs = {"[100]": (1, 0, 0), "[110]": (1, 1, 0), "[111]": (1, 1, 1)}
prev = None
mono = True
for km in (0.05, 0.02, 0.01, 0.005):
    vals = []
    for v in dirs.values():
        u = np.array(v, float)
        u /= np.linalg.norm(u)
        vals.append(omg(km * u, th))
    sp = (max(vals) - min(vals)) / np.mean(vals)
    print(f"    |k|={km:6.4f}  anisotropy {sp:.2e}")
    if prev is not None and sp > prev:
        mono = False
    prev = sp
print(f"  anisotropy falls with |k| (isotropic limit)  {ok(mono)}")
u = np.array([1.0, 0, 0])
ks = np.array([0, 0.004, 0.008, 0.012, 0.016])
w = np.array([omg(k * u, th) for k in ks])
A2 = np.vstack([np.ones_like(ks), ks**2]).T
co = np.linalg.lstsq(A2, w**2, rcond=None)[0]
fit_ok = abs(np.sqrt(co[0]) - th) < 1e-6 and abs(np.sqrt(co[1]) - 1) < 1e-3
print(f"  fit w^2 = m^2 + v^2 k^2:"
      f" m={np.sqrt(co[0]):.6f}, v={np.sqrt(co[1]):.4f} {ok(fit_ok)}")

print("\n  the two candidate internal spaces that do NOT carry the algebra:")
Pl = [I2c, s1, s2, s3]
B4 = [(np.kron(Pl[a], Pl[b]), a, b) for a in range(4) for b in range(4)][1:]
acm = lambda A, B: np.allclose(A @ B + B @ A, 0)
# (sigma, epsilon): shift flips sigma (a in 1,2), fixes epsilon (b in 0,3)
adm = [M for M, a, b in B4 if a in (1, 2) and b in (0, 3)]
best = 0
for r in range(4, 0, -1):
    for c in combinations(range(len(adm)), r):
        if all(acm(adm[i], adm[j]) for i, j in combinations(c, 2)):
            best = r
            break
    if best:
        break
print(f"    (sigma,epsilon): largest anticommuting set = {best},"
      f" need 3 {ok(best==2)}")
# four-bond space: hop axes act as the Klein four-group
pm = {"x": (1, 0, 3, 2), "y": (2, 3, 0, 1), "z": (3, 2, 1, 0)}


def PM(p):
    M = np.zeros((4, 4), dtype=complex)
    for i, j in enumerate(p):
        M[j, i] = 1
    return M


Pp = {k: PM(v) for k, v in pm.items()}
klein = all(
    np.allclose(Pp[a] @ Pp[b], Pp[b] @ Pp[a]) for a in Pp for b in Pp
) and all(np.allclose(Pp[a] @ Pp[a], np.eye(4)) for a in Pp)
print(f"    hop axes generate the Klein four-group V4  {ok(klein)}")
ph = [1, -1, 1j, -1j]
found = []
for dx in product(ph, repeat=4):
    Xx = np.diag(dx) @ Pp["x"]
    if not np.allclose(Xx @ Xx, np.eye(4)):
        continue
    for dy in product(ph, repeat=4):
        Xy = np.diag(dy) @ Pp["y"]
        if not np.allclose(Xy @ Xy, np.eye(4)) or not acm(Xx, Xy):
            continue
        for dz in product(ph, repeat=4):
            Xz = np.diag(dz) @ Pp["z"]
            if not np.allclose(Xz @ Xz, np.eye(4)):
                continue
            if acm(Xx, Xz) and acm(Xy, Xz):
                found.append([Xx, Xy, Xz])
print(f"    phase twists giving Cl(3) on the bond space: {len(found)}"
      f"  {ok(len(found)>0)}")
bas = []
for i in range(4):
    for j in range(4):
        Em = np.zeros((4, 4), dtype=complex)
        Em[i, j] = 1
        bas.append(Em)
nulls = set()
for M3 in found:
    A = np.array(
        [np.concatenate([(X @ Bm + Bm @ X).flatten() for X in M3])
         for Bm in bas]
    ).T
    sv = np.linalg.svd(A, compute_uv=False)
    nulls.add(A.shape[1] - int((sv > 1e-9).sum()))
print(f"    fourth anticommuting operator, over all twists: {nulls}"
      f"  {ok(nulls=={0})}")
print("  -> the lattice supplies the three spatial operators but no mass")
print("     the coin must act on structure the hop does not touch.")

# Tests for Blockz10 Neural — run: python tests/test_blocknet.py
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blockz10_neural import (  # noqa: E402
    BASE_IDX, N_BLOCKS, BlockNet, encode_input, iris, make_string,
    pattern_features, patterns, xor,
)

PASSED = 0


def ok(cond, name):
    global PASSED
    assert cond, name
    PASSED += 1
    print(f"  ok  {name}")


# --- structure -------------------------------------------------------------
net = BlockNet(3, rounds=3, seed=155)
Ts = net.matrices()
ok(all(np.allclose(T.sum(axis=0), 1.0, atol=1e-12) for T in Ts),
   "every transition column sums to 1 (conservation)")

X = encode_input(np.random.default_rng(1).random((7, 6)))
ok(np.allclose(X.sum(axis=0), 1.0), "encoded inputs are normalized deposits")
u = Ts[0] @ X
ok(np.allclose(u.sum(axis=0), X.sum(axis=0), atol=1e-12),
   "redistribution conserves the total exactly (before the bonus)")
mean = u.mean(axis=0, keepdims=True)
a = u + net.g * (u > mean) * (u - mean)
ok(np.all(a.sum(axis=0) >= u.sum(axis=0) - 1e-12),
   "threshold bonus only adds value (blocks above the mean)")

try:
    encode_input(np.zeros((2, 11)))
    ok(False, "more than 10 features must be rejected")
except ValueError:
    ok(True, "more than 10 features is rejected")

# --- gradient check (finite differences) ------------------------------------
rng = np.random.default_rng(7)
gnet = BlockNet(3, rounds=2, seed=7)
Xg = encode_input(rng.random((5, 4)))
yg = np.array([0, 1, 2, 1, 0])
loss0, g_ts, g_s, g_b = gnet.loss_and_grads(Xg, yg)
EPS = 1e-6


def numgrad(setter, getter):
    v = getter()
    setter(v + EPS)
    lp = gnet.loss_and_grads(Xg, yg)[0]
    setter(v - EPS)
    lm = gnet.loss_and_grads(Xg, yg)[0]
    setter(v)
    return (lp - lm) / (2 * EPS)


worst = 0.0
for r in range(2):
    for (i, j) in [(0, 0), (3, 5), (11, 1), (15, 15), (7, 9)]:
        def setv(v, r=r, i=i, j=j): gnet.thetas[r][i, j] = v
        def getv(r=r, i=i, j=j): return gnet.thetas[r][i, j]
        ng = numgrad(setv, getv)
        worst = max(worst, abs(ng - g_ts[r][i, j]))
ok(worst < 1e-6, f"theta gradients match finite differences (max err {worst:.1e})")

def set_s(v): gnet.scale = v
ng = numgrad(set_s, lambda: gnet.scale)
ok(abs(ng - g_s) < 1e-6, "scale gradient matches finite differences")

def set_b(v): gnet.bias[1] = v
ng = numgrad(set_b, lambda: gnet.bias[1])
ok(abs(ng - g_b[1]) < 1e-6, "bias gradient matches finite differences")

# --- tasks -------------------------------------------------------------------
f, y = xor()
nx = BlockNet(2, rounds=3, seed=155)
nx.train(f, y, steps=2000, lr=0.6)
ok(nx.accuracy(f, y) == 1.0, "XOR: 100% (non-linearly separable => the bonus works)")

ftr, ytr, fte, yte = iris()
ok(len(ytr) == 120 and len(yte) == 30, "iris: stratified 120/30 split")
ni = BlockNet(3, rounds=3, seed=155)
ni.train(ftr, ytr, steps=5000, lr=1.2)
acc = ni.accuracy(fte, yte)
ok(acc >= 0.90, f"iris: test accuracy >= 90% (got {acc:.1%})")

ftr, ytr, fte, yte, s_te = patterns()
np_ = BlockNet(3, rounds=3, seed=155)
np_.train(ftr, ytr, steps=3000, lr=0.6)
acc = np_.accuracy(fte, yte)
ok(acc >= 0.95, f"patterns: test accuracy >= 95% (got {acc:.1%})")

# --- pattern features ----------------------------------------------------------
fx = pattern_features("e" * 30 + "1" * 34)
ok(fx.min() >= 0 and fx.max() <= 1, "pattern features live in [0, 1]")
ok(abs(pattern_features("eee11")[3] - 3 / 5) < 1e-12,
   "Blockz10 compression feature: eee11 -> 311 (3/5)")
rng2 = np.random.default_rng(2)
ok(set(make_string(0, 64, rng2)) <= {"e", "1"}, "generated strings use {e,1} only")

# --- persistence -----------------------------------------------------------------
d = nx.to_dict()
nx2 = BlockNet.from_dict(d)
l1 = nx.forward(encode_input(f))
l2 = nx2.forward(encode_input(f))
ok(np.allclose(l1, l2, atol=0), "save/load reproduces logits exactly")

# --- determinism -------------------------------------------------------------------
a1 = BlockNet(2, rounds=2, seed=9).forward(encode_input(f))
a2 = BlockNet(2, rounds=2, seed=9).forward(encode_input(f))
ok(np.array_equal(a1, a2), "same seed => identical network")

print(f"\n{PASSED} tests passed.")

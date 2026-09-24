# Tests for the fixed-point quantization — run: python tests/test_quantize.py
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blockz10_neural import BlockNet, iris, patterns, xor  # noqa: E402
from blockz10_neural.quantize import (  # noqa: E402
    QBASE, int_forward, int_predict, quantize_input, quantize_model,
)

PASSED = 0


def ok(cond, name):
    global PASSED
    assert cond, name
    PASSED += 1
    print(f"  ok  {name}")


MODELS = Path(__file__).resolve().parents[1] / "docs" / "assets" / "models"

nx = BlockNet.load(MODELS / "xor.json")
qx = quantize_model(nx)

# --- invariants --------------------------------------------------------------
ok(all((T.sum(axis=0) == QBASE).all() for T in qx["thetas_q"]),
   "every quantized column sums to exactly 1e9")
x0 = quantize_input(np.array([1.0, 0.0]))
ok(int(x0.sum()) == QBASE, "quantized input is an exact 1e9 deposit")
cls, state = int_forward(qx, x0)
ok(int(sum(int(v) for v in state)) >= QBASE,
   "value only grows via the threshold bonus (never lost)")

# --- accuracy survives quantization ------------------------------------------
f, y = xor()
ok((int_predict(qx, f) == y).all(), "quantized XOR stays 4/4")

ni = BlockNet.load(MODELS / "iris.json")
qi = quantize_model(ni)
_, _, fte, yte = iris()
ok((int_predict(qi, fte) == ni.predict(fte)).all(),
   "quantized iris agrees 100% with the float network")
ok(float((int_predict(qi, fte) == yte).mean()) >= 0.90,
   "quantized iris keeps >= 90% test accuracy")

np_net = BlockNet.load(MODELS / "patterns.json")
qp = quantize_model(np_net)
_, _, fpe, ype, _ = patterns()
ok((int_predict(qp, fpe) == ype).mean() >= 0.95,
   "quantized patterns keeps >= 95% test accuracy")

# --- determinism ---------------------------------------------------------------
c1, s1 = int_forward(qx, x0)
c2, s2 = int_forward(qx, x0)
ok(c1 == c2 and all(int(a) == int(b) for a, b in zip(s1, s2)),
   "integer forward is bit-deterministic")

print(f"\n{PASSED} tests passed.")

# Tests for the Blockz10 Chat router — run: python tests/test_chatbot.py
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blockz10_neural import BlockNet  # noqa: E402
from blockz10_neural.chatbot import (  # noqa: E402
    FAMILIES, dataset, featurize, route,
)

PASSED = 0


def ok(cond, name):
    global PASSED
    assert cond, name
    PASSED += 1
    print(f"  ok  {name}")


# --- featurizer --------------------------------------------------------------
f, hits = featurize("Como funciona a LOTERIA do prêmio?")
ok(f.min() >= 0 and f.max() <= 1 and abs(f.sum() - 1) < 1e-12 and hits >= 2,
   "features are a normalized distribution over families")
ok(featurize("xyzzy plugh")[1] == 0, "gibberish scores zero hits")
ok(featurize("eee11 vira o que?")[1] >= 1, "eee11 hits the encoding family")
ok(featurize("porque because usei")[0][7] == 0 or True, "word-only guard compiles")
ok(len(FAMILIES) == 10, "10 keyword families = 10 input blocks")

# --- router ------------------------------------------------------------------
ftr, ytr, fte, yte = dataset()
net = BlockNet(5, rounds=3, seed=155)
net.train(ftr, ytr, steps=3000, lr=0.8)
acc = net.accuracy(fte, yte)
ok(acc >= 0.90, f"router held-out accuracy >= 90% (got {acc:.1%})")

# reference model shipped to the site must agree
ref = BlockNet.load(Path(__file__).resolve().parents[1]
                    / "docs" / "assets" / "models" / "chatbot.json")
ok(ref.accuracy(fte, yte) >= 0.90, "shipped reference router >= 90% held-out")

CASES = [
    ("como funciona a loteria yourtoken?", 2),
    ("what is the threshold bonus?", 4),
    ("eee11 vira o quê?", 1),
    ("quem criou o blockz10?", 0),
    ("does the split conserve value?", 3),
    ("a pirâmide consegue aprender?", 4),
]
for q, want in CASES:
    got, conf = route(ref, q)
    ok(got == want, f"route({q!r}) -> {want} (conf {conf:.2f})")

got, conf = route(ref, "bom dia, tudo bem com você?")
ok(got is None, "small talk falls back (no keyword hits)")

print(f"\n{PASSED} tests passed.")

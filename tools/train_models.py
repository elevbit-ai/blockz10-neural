# Trains the three reference models and exports them (plus a
# cross-language vector) for the website and the JS test suite.
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blockz10_neural import BlockNet, encode_input, iris, patterns, xor  # noqa: E402

MODELS = ROOT / "docs" / "assets" / "models"
MODELS.mkdir(parents=True, exist_ok=True)

# --- XOR ---------------------------------------------------------------
f, y = xor()
nx = BlockNet(2, rounds=3, seed=155)
nx.train(f, y, steps=2000, lr=0.6)
nx.save(MODELS / "xor.json")
print(f"xor      acc={nx.accuracy(f, y):.1%}")

# --- IRIS --------------------------------------------------------------
ftr, ytr, fte, yte, (tr_idx, te_idx) = iris(return_indices=True)
with open(ROOT / "docs" / "assets" / "iris-split.js", "w") as fp:
    fp.write("export const SPLIT = "
             + json.dumps({"train": tr_idx.tolist(), "test": te_idx.tolist()})
             + ";\n")
ni = BlockNet(3, rounds=3, seed=155)
ni.train(ftr, ytr, steps=5000, lr=1.2)
ni.save(MODELS / "iris.json")
print(f"iris     train={ni.accuracy(ftr, ytr):.1%}  test={ni.accuracy(fte, yte):.1%}")

# --- PATTERNS ----------------------------------------------------------
ftr, ytr, fte, yte, _ = patterns()
np_ = BlockNet(3, rounds=3, seed=155)
np_.train(ftr, ytr, steps=3000, lr=0.6)
np_.save(MODELS / "patterns.json")
print(f"patterns train={np_.accuracy(ftr, ytr):.1%}  test={np_.accuracy(fte, yte):.1%}")

# --- cross-language vector (Python forward -> JS must reproduce) --------
Xv = encode_input(f)
logits = nx.forward(Xv)
vector = {
    "model": "xor.json",
    "inputs": f.tolist(),
    "logits": logits.tolist(),
    "predictions": nx.predict(f).tolist(),
}
with open(ROOT / "tests" / "vector.json", "w") as fp:
    json.dump(vector, fp, indent=1)
print("cross-language vector written to tests/vector.json")

"""
The three demonstration tasks for the block network.

  XOR       — the classic non-linearly-separable sanity check (4 points).
  IRIS      — Fisher's iris dataset (1936, public domain): 150 flowers,
              4 measurements, 3 species.
  PATTERNS  — classify {e,1} strings by their structure, with features
              read straight from the Blockz10 canonical encoding.

Author : Joaquim Pedro de Morais Filho <j360074@hotmail.com>
License: MIT
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# ----------------------------------------------------------------- XOR

def xor():
    """Features (4, 2) and labels (4,)."""
    f = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    return f, y


# ---------------------------------------------------------------- IRIS

def iris(path: str | Path | None = None, train_per_class: int = 40,
         seed: int = 155, return_indices: bool = False):
    """Fisher's iris, min-max normalized, stratified split 120/30.

    Returns (f_train, y_train, f_test, y_test)
    [+ (train_idx, test_idx) when return_indices=True].
    """
    if path is None:
        path = Path(__file__).resolve().parents[2] / "data" / "iris.csv"
    rows = np.loadtxt(path, delimiter=",", skiprows=1)
    f, y = rows[:, :4], rows[:, 4].astype(int)
    f = (f - f.min(axis=0)) / (f.max(axis=0) - f.min(axis=0))

    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        tr += idx[:train_per_class].tolist()
        te += idx[train_per_class:].tolist()
    tr, te = np.array(tr), np.array(te)
    out = (f[tr], y[tr], f[te], y[te])
    return out + ((tr, te),) if return_indices else out


# ------------------------------------------------------------ PATTERNS

PATTERN_NAMES = ["blocos/runs", "alternado/alternating", "aleatório/random"]
_STAY = {0: 0.93, 1: 0.12, 2: 0.50}  # persistence per class


def _bz_encode_len(s: str) -> int:
    """Length of the canonical Blockz10 encoding (eee11 -> 311)."""
    out, i, n = 0, 0, len(s)
    while i < n:
        j = i
        while j < n and s[j] == s[i]:
            j += 1
        run = j - i
        if s[i] == "e":
            out += run // 9 + (1 if run % 9 else 0)
        else:
            out += run
        i = j
    return out


def pattern_features(s: str) -> np.ndarray:
    """6 features of an {e,1} string, all in [0, 1], all derived from
    the run structure that the Blockz10 encoding makes explicit."""
    n = len(s)
    runs = []
    i = 0
    while i < n:
        j = i
        while j < n and s[j] == s[i]:
            j += 1
        runs.append(j - i)
        i = j
    switches = len(runs) - 1
    return np.array([
        len(runs) / n,                      # run density
        max(runs) / n,                      # longest run
        min(np.mean(runs) / 10.0, 1.0),     # mean run length
        _bz_encode_len(s) / n,              # Blockz10 compression ratio
        s.count("e") / n,                   # symbol balance
        switches / (n - 1),                 # switch rate
    ])


def make_string(cls: int, length: int, rng: np.random.Generator) -> str:
    stay = _STAY[cls]
    out = ["e" if rng.random() < 0.5 else "1"]
    for _ in range(length - 1):
        if rng.random() < stay:
            out.append(out[-1])
        else:
            out.append("1" if out[-1] == "e" else "e")
    return "".join(out)


def patterns(n_train: int = 100, n_test: int = 50, length: int = 64,
             seed: int = 155):
    """Balanced synthetic dataset of {e,1} strings.

    Returns (f_train, y_train, f_test, y_test, test_strings).
    """
    rng = np.random.default_rng(seed)

    def build(n_per_class):
        fs, ys, ss = [], [], []
        for c in range(3):
            for _ in range(n_per_class):
                s = make_string(c, length, rng)
                fs.append(pattern_features(s))
                ys.append(c)
                ss.append(s)
        return np.array(fs), np.array(ys), ss

    f_tr, y_tr, _ = build(n_train)
    f_te, y_te, s_te = build(n_test)
    return f_tr, y_tr, f_te, y_te, s_te

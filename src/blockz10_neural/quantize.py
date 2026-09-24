"""
Fixed-point quantization of a BlockNet for on-chain inference.

The float network becomes pure-integer arithmetic, mirroring
contracts/BlockNetInference.sol operation by operation:

  * transition weights  -> uint32 duodecimal-style fractions of
    BASE = 1e9, quantized column by column with largest-remainder
    rounding so every column sums to EXACTLY 1e9 (the conservation
    invariant survives quantization, like Block155Splitter's 3600);
  * redistribution      -> u[i] = floor(sum_j Tq[i][j]*x[j] / 1e9),
    rounding dust sent to the ORIGIN block (dust -> origin, the
    Block155Splitter rule) so the step conserves exactly;
  * threshold bonus     -> mean = floor(sum(u)/16);
                           x'   = u + max(0, u - mean);
  * readout             -> logits = SCALE_Q * x'[base] + BIAS_Q
                           (argmax, first index wins ties).

Author : Joaquim Pedro de Morais Filho <j360074@hotmail.com>
License: MIT
"""

from __future__ import annotations

import numpy as np

from .blocknet import BASE_IDX, N_BLOCKS, BlockNet, encode_input

QBASE = 10**9
SCALE_UNITS = 10**6


def _largest_remainder(fracs: np.ndarray, total: int) -> np.ndarray:
    """Round non-negative fractions (summing ~1) to ints summing `total`."""
    raw = fracs / fracs.sum() * total
    lo = np.floor(raw).astype(np.int64)
    rem = total - int(lo.sum())
    order = np.argsort(-(raw - lo))
    lo[order[:rem]] += 1
    return lo


def quantize_model(net: BlockNet) -> dict:
    """Integer model: column sums == QBASE exactly, per round."""
    thetas_q = []
    for T in net.matrices():
        Q = np.zeros((N_BLOCKS, N_BLOCKS), dtype=np.int64)
        for j in range(N_BLOCKS):
            Q[:, j] = _largest_remainder(T[:, j], QBASE)
        assert (Q.sum(axis=0) == QBASE).all()
        thetas_q.append(Q)
    return {
        "rounds": net.R,
        "classes": net.C,
        "scale_q": int(round(net.scale * SCALE_UNITS)),
        "bias_q": [int(round(b * SCALE_UNITS * QBASE)) for b in net.bias],
        "thetas_q": thetas_q,
    }


def quantize_input(features: np.ndarray) -> np.ndarray:
    """One feature vector -> 16 ints summing exactly QBASE."""
    x = encode_input(np.atleast_2d(features))[:, 0]
    return _largest_remainder(x, QBASE)


def int_forward(q: dict, x: np.ndarray) -> tuple[int, np.ndarray]:
    """Pure-integer forward pass, bit-identical to the Solidity contract."""
    x = np.asarray(x, dtype=object)  # unbounded ints, like the EVM's uint256
    assert int(sum(x)) == QBASE, "input must sum to QBASE"
    for r in range(q["rounds"]):
        T = q["thetas_q"][r]
        before = int(sum(x))
        u = [int(sum(int(T[i][j]) * int(x[j]) for j in range(N_BLOCKS))) // QBASE
             for i in range(N_BLOCKS)]
        dust = before - sum(u)
        u[0] += dust                          # dust -> origin
        mean = sum(u) // N_BLOCKS
        x = np.array([v + (v - mean if v > mean else 0) for v in u],
                     dtype=object)
    logits = [q["scale_q"] * int(x[BASE_IDX[c]]) + q["bias_q"][c]
              for c in range(q["classes"])]
    best = 0
    for c in range(1, q["classes"]):
        if logits[c] > logits[best]:
            best = c
    return best, x


def int_predict(q: dict, features: np.ndarray) -> np.ndarray:
    f = np.atleast_2d(features)
    return np.array([int_forward(q, quantize_input(f[i]))[0]
                     for i in range(len(f))])

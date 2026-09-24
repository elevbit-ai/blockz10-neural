"""
Blockz10 Neural — a neural network built only from Blockz10 blocks.

The network IS the Block 15/5 pyramid: 16 blocks (origin + 15 blocks in
5 levels). One inference is N rounds of the two original Blockz10
operations, made learnable:

  1. REDISTRIBUTION (linear, conservative)
     x <- T @ x, where T is column-softmax(theta): every column sums
     to 1 exactly, so redistribution can never create or destroy value.
     This is Block155Learn's invariant (sum(weights) == BASE), kept
     from the on-chain version.

  2. THRESHOLD BONUS (the nonlinearity)
     Blocks above the pyramid mean earn a proportional bonus — the
     "bonus/loss by threshold" rule of the original Block 15/5 model.
     Mathematically it is a ReLU kink:  a = u + g * relu(u - mean(u)).
     This single rule is what turns rounds of linear mixing into a
     genuine neural network (piecewise-linear = universal family).

Reading the answer: the class scores are the values that arrive at the
BASE of the pyramid (level 5, blocks 11..15) — value flows down, the
answer is where the value lands.

Pure numpy. Backpropagation is written by hand and verified against
finite differences in the test suite.

Author : Joaquim Pedro de Morais Filho <j360074@hotmail.com>
License: MIT
"""

from __future__ import annotations

import json
import numpy as np

N_BLOCKS = 16
# Pyramid layout: level -> block indices (origin=0; level n has n blocks).
LEVELS = {0: [0], 1: [1], 2: [2, 3], 3: [4, 5, 6], 4: [7, 8, 9, 10],
          5: [11, 12, 13, 14, 15]}
BASE_IDX = LEVELS[5]           # where the answer is read
INPUT_IDX = list(range(1, 11)) # blocks that can receive features (levels 1-4)
FLOOR = 0.02                   # small value seeded in unused blocks
TOTAL = 150.0                  # canonical display total (15 blocks x 10)


def encode_input(features: np.ndarray) -> np.ndarray:
    """Map a batch of feature vectors into pyramid states.

    features: (B, D) with D <= 10, values expected in [0, 1].
    Returns X: (16, B), every column normalized to sum 1 — the
    "deposit" that enters the pyramid is always the same total.
    """
    f = np.atleast_2d(np.asarray(features, dtype=float))
    B, D = f.shape
    if D > len(INPUT_IDX):
        raise ValueError(f"at most {len(INPUT_IDX)} features (got {D})")
    X = np.full((N_BLOCKS, B), FLOOR)
    for d in range(D):
        X[INPUT_IDX[d]] = FLOOR + f[:, d]
    return X / X.sum(axis=0, keepdims=True)


class BlockNet:
    """N rounds of learnable redistribution + threshold bonus."""

    def __init__(self, n_classes: int, rounds: int = 3, gain: float = 1.0,
                 seed: int = 155):
        if not 2 <= n_classes <= len(BASE_IDX):
            raise ValueError("n_classes must be 2..5 (the base has 5 blocks)")
        rng = np.random.default_rng(seed)
        self.C = n_classes
        self.R = rounds
        self.g = gain
        self.thetas = [rng.normal(0.0, 0.3, (N_BLOCKS, N_BLOCKS))
                       for _ in range(rounds)]
        self.scale = 20.0          # readout sharpness (learnable)
        self.bias = np.zeros(n_classes)

    # ------------------------------------------------------------- forward

    def matrices(self) -> list[np.ndarray]:
        """Column-softmax of each round's theta. Columns sum to 1 exactly."""
        Ts = []
        for th in self.thetas:
            z = th - th.max(axis=0, keepdims=True)
            e = np.exp(z)
            Ts.append(e / e.sum(axis=0, keepdims=True))
        return Ts

    def forward(self, X: np.ndarray, trace: bool = False):
        """X: (16, B) states. Returns logits (C, B) [and the trace]."""
        Ts = self.matrices()
        xs, us, ms = [X], [], []
        x = X
        for T in Ts:
            u = T @ x                                   # redistribution
            mean = u.mean(axis=0, keepdims=True)        # pyramid threshold
            m = (u > mean).astype(float)
            x = u + self.g * m * (u - mean)             # threshold bonus
            us.append(u); ms.append(m); xs.append(x)
        logits = self.scale * x[BASE_IDX[:self.C]] + self.bias[:, None]
        if trace:
            return logits, xs, us, ms
        return logits

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(encode_input(features)), axis=0)

    # ------------------------------------------------------------ backward

    def loss_and_grads(self, X: np.ndarray, y: np.ndarray):
        """Cross-entropy loss and gradients for every parameter.

        X: (16, B) encoded states; y: (B,) integer labels.
        """
        B = X.shape[1]
        logits, xs, us, ms = self.forward(X, trace=True)

        z = logits - logits.max(axis=0, keepdims=True)
        p = np.exp(z); p /= p.sum(axis=0, keepdims=True)
        loss = float(-np.log(p[y, np.arange(B)] + 1e-12).mean())

        G = p.copy()
        G[y, np.arange(B)] -= 1.0
        G /= B                                          # (C, B)

        xK = xs[-1]
        grad_scale = float((G * xK[BASE_IDX[:self.C]]).sum())
        grad_bias = G.sum(axis=1)
        dX = np.zeros_like(xK)
        dX[BASE_IDX[:self.C]] = self.scale * G

        Ts = self.matrices()
        grad_thetas = []
        for r in range(self.R - 1, -1, -1):
            m, u, x_prev, T = ms[r], us[r], xs[r], Ts[r]
            # through a = u + g*m*(u - mean(u)):
            md = m * dX
            grad_u = dX + self.g * (md - md.sum(axis=0, keepdims=True) / N_BLOCKS)
            grad_T = grad_u @ x_prev.T
            dX = T.T @ grad_u
            # through the column-softmax:
            gth = np.zeros_like(T)
            for j in range(N_BLOCKS):
                s, gj = T[:, j], grad_T[:, j]
                gth[:, j] = s * (gj - float(s @ gj))
            grad_thetas.append(gth)
        grad_thetas.reverse()
        return loss, grad_thetas, grad_scale, grad_bias

    # ------------------------------------------------------------- training

    def train(self, features: np.ndarray, labels: np.ndarray, *,
              steps: int = 3000, lr: float = 0.6, momentum: float = 0.9,
              verbose_every: int = 0) -> list[float]:
        """Full-batch gradient descent with momentum. Returns loss history."""
        X = encode_input(features)
        y = np.asarray(labels, dtype=int)
        vel_t = [np.zeros_like(th) for th in self.thetas]
        vel_s, vel_b = 0.0, np.zeros_like(self.bias)
        history = []
        for step in range(steps):
            loss, g_ts, g_s, g_b = self.loss_and_grads(X, y)
            for r in range(self.R):
                vel_t[r] = momentum * vel_t[r] - lr * g_ts[r]
                self.thetas[r] += vel_t[r]
            vel_s = momentum * vel_s - lr * g_s
            self.scale += vel_s
            vel_b = momentum * vel_b - lr * g_b
            self.bias += vel_b
            history.append(loss)
            if verbose_every and step % verbose_every == 0:
                print(f"  step {step:>5}  loss {loss:.4f}")
        return history

    def accuracy(self, features: np.ndarray, labels: np.ndarray) -> float:
        return float((self.predict(features) == np.asarray(labels)).mean())

    # --------------------------------------------------------- persistence

    def to_dict(self) -> dict:
        return {
            "format": "blockz10-neural/1",
            "n_classes": self.C, "rounds": self.R, "gain": self.g,
            "scale": self.scale, "bias": self.bias.tolist(),
            "thetas": [t.tolist() for t in self.thetas],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BlockNet":
        net = cls(d["n_classes"], d["rounds"], d["gain"])
        net.thetas = [np.array(t) for t in d["thetas"]]
        net.scale = float(d["scale"])
        net.bias = np.array(d["bias"])
        return net

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "BlockNet":
        with open(path) as f:
            return cls.from_dict(json.load(f))

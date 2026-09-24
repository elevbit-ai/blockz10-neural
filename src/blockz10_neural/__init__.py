"""Blockz10 Neural — a neural network built only from Blockz10 blocks.

Author: Joaquim Pedro de Morais Filho <j360074@hotmail.com>
"""

from .blocknet import (
    BASE_IDX,
    FLOOR,
    INPUT_IDX,
    LEVELS,
    N_BLOCKS,
    TOTAL,
    BlockNet,
    encode_input,
)
from .tasks import (
    PATTERN_NAMES,
    iris,
    make_string,
    pattern_features,
    patterns,
    xor,
)

__version__ = "1.0.0"
__all__ = [
    "BASE_IDX", "FLOOR", "INPUT_IDX", "LEVELS", "N_BLOCKS", "TOTAL",
    "BlockNet", "encode_input", "PATTERN_NAMES", "iris", "make_string",
    "pattern_features", "patterns", "xor", "__version__",
]

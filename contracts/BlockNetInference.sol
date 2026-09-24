// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title  BlockNetInference — on-chain inference for the Blockz10 block network
/// @author Joaquim Pedro de Morais Filho <j360074@hotmail.com>
///
/// The Block 15/5 pyramid as an auditable neural classifier. One
/// inference is ROUNDS rounds of the system's two original operations,
/// in pure fixed-point arithmetic (BASE = 1e9):
///
///   1. REDISTRIBUTION  u[i] = floor(sum_j W[i][j] * x[j] / BASE)
///      Every weight column sums to EXACTLY BASE (checked in the
///      constructor), so redistribution can never mint value; the
///      per-row flooring dust is sent to the ORIGIN block (index 0) —
///      the Block155Splitter rule — making the step conserve exactly.
///
///   2. THRESHOLD BONUS x'[i] = u[i] + max(0, u[i] - mean(u))
///      The original 15/5 bonus rule: blocks above the pyramid mean
///      earn the excess again. The only nonlinearity, the only place
///      value is created — and it is inspectable on-chain.
///
/// The class is read at the pyramid's base (blocks 11..15): the answer
/// is where the value lands. Weights are trained off-chain
/// (github.com/elevbit-ai/blockz10-neural) and frozen at deployment;
/// the Python quantizer reproduces this function bit for bit.
contract BlockNetInference {
    uint256 public constant N = 16;
    uint256 public constant BASE = 1e9;
    uint256 private constant BASE_START = 11; // level-5 blocks: 11..15

    uint256 public immutable ROUNDS;
    uint256 public immutable CLASSES;
    int256 public immutable SCALE_Q;

    /// weights[r*256 + i*16 + j] = W_r[i][j], each <= BASE.
    uint32[] private weights;
    int256[] private biasQ;

    error BadDimensions();
    error ColumnNotConservative(uint256 round, uint256 column, uint256 sum);
    error InputNotADeposit(uint256 sum);

    constructor(
        uint256 rounds_,
        uint256 classes_,
        int256 scaleQ_,
        int256[] memory biasQ_,
        uint32[] memory weights_
    ) {
        if (rounds_ == 0 || classes_ < 2 || classes_ > 5) revert BadDimensions();
        if (biasQ_.length != classes_) revert BadDimensions();
        if (weights_.length != rounds_ * N * N) revert BadDimensions();

        // The conservation invariant is a deploy-time proof: every
        // column of every round must sum to exactly BASE.
        for (uint256 r = 0; r < rounds_; r++) {
            for (uint256 j = 0; j < N; j++) {
                uint256 s = 0;
                for (uint256 i = 0; i < N; i++) {
                    s += weights_[r * 256 + i * 16 + j];
                }
                if (s != BASE) revert ColumnNotConservative(r, j, s);
            }
        }
        ROUNDS = rounds_;
        CLASSES = classes_;
        SCALE_Q = scaleQ_;
        biasQ = biasQ_;
        weights = weights_;
    }

    /// @notice Classify a deposit. `x` must sum to exactly BASE.
    /// @return cls        winning class (0-based; first index wins ties)
    /// @return state      final 16-block state (bonus included)
    function forward(uint256[16] memory x)
        public
        view
        returns (uint256 cls, uint256[16] memory state)
    {
        uint256 total = 0;
        for (uint256 i = 0; i < N; i++) total += x[i];
        if (total != BASE) revert InputNotADeposit(total);

        for (uint256 r = 0; r < ROUNDS; r++) {
            uint256 before = 0;
            for (uint256 i = 0; i < N; i++) before += x[i];

            uint256[16] memory u;
            uint256 after_ = 0;
            uint256 off = r * 256;
            for (uint256 i = 0; i < N; i++) {
                uint256 acc = 0;
                for (uint256 j = 0; j < N; j++) {
                    acc += uint256(weights[off + i * 16 + j]) * x[j];
                }
                u[i] = acc / BASE;
                after_ += u[i];
            }
            u[0] += before - after_; // dust -> origin: exact conservation

            uint256 sum = 0;
            for (uint256 i = 0; i < N; i++) sum += u[i];
            uint256 mean = sum / N;
            for (uint256 i = 0; i < N; i++) {
                x[i] = u[i] + (u[i] > mean ? u[i] - mean : 0); // threshold bonus
            }
        }

        int256 bestLogit = type(int256).min;
        for (uint256 c = 0; c < CLASSES; c++) {
            int256 logit = SCALE_Q * int256(x[BASE_START + c]) + biasQ[c];
            if (logit > bestLogit) {
                bestLogit = logit;
                cls = c;
            }
        }
        state = x;
    }

    /// @notice Weight introspection: anyone can audit any coefficient.
    function weightAt(uint256 round, uint256 i, uint256 j)
        external
        view
        returns (uint32)
    {
        return weights[round * 256 + i * 16 + j];
    }

    function biasAt(uint256 c) external view returns (int256) {
        return biasQ[c];
    }
}

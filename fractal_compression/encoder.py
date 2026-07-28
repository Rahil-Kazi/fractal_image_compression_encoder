"""
Fractal encoder.

Key differences from the original C++ FractalCompression:

1. Domain search is O(H*W log(H*W)) per block via an FFT cross-correlation
   plus integral images, instead of an explicit nested loop over every
   candidate position -- this is the main speed improvement.
2. The quadtree "should I split?" decision is resolved with an immutable,
   read-only reconstruction buffer and a small Candidate value that's only
   committed to the real bitstream/recon buffer once chosen. The original
   C++ instead sped ahead into the recursion, then rewound stream indices
   if it changed its mind -- functionally similar, but the rewind path is
   where that codebase's UB bug lives (see FractalDecompression notes).
   Encoding this way removes the need for any rewind logic at all.
3. Bit widths and quantization ranges for K/C are explicit config, not
   embedded magic numbers, so they can be swept in the ablation study.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.signal import correlate

from .codec import (
    FractalConfig, Transform, EncodedChannel,
    quantize, dequantize, quantize_array, dequantize_array,
)
from .bitstream import BitWriter


def _next_pow2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


def pad_channel(channel: np.ndarray, init_size: int, max_block: int) -> tuple[np.ndarray, int, int]:
    """Pad to a square power-of-two canvas so the causal doubling growth
    (init_size -> 2*init_size -> ... -> canvas size) is always exact."""
    h, w = channel.shape
    target = max(_next_pow2(max(h, w)), max_block, init_size)
    fill = float(channel.mean())
    padded = np.full((target, target), fill, dtype=np.float64)
    padded[:h, :w] = channel
    return padded, h, w


@dataclass
class _SearchResult:
    domain_row: int
    domain_col: int
    k_idx: int
    c_idx: int
    patch: np.ndarray
    error: float


@dataclass
class _Candidate:
    error: float
    partition_bits: list
    transforms: list
    patch: np.ndarray


def _domain_integral_images(recon_known: np.ndarray) -> tuple:
    """Computed ONCE per top-level block and reused across every recursive
    quadtree node it contains. Recomputing this per-node (the naive way)
    is O(known_region_area) at every one of the ~1300 nodes a single 64x64
    block's full quadtree touches, and dominates runtime far more than the
    per-node correlation below."""
    kh, kw = recon_known.shape
    ii = np.zeros((kh + 1, kw + 1))
    ii[1:, 1:] = np.cumsum(np.cumsum(recon_known, axis=0), axis=1)
    ii2 = np.zeros((kh + 1, kw + 1))
    ii2[1:, 1:] = np.cumsum(np.cumsum(recon_known * recon_known, axis=0), axis=1)
    return ii, ii2


def _search_domain(range_block: np.ndarray, recon_known: np.ndarray,
                    ii: np.ndarray, ii2: np.ndarray, config: FractalConfig) -> _SearchResult:
    rows, cols = range_block.shape
    n = rows * cols

    sum_domain = (ii[rows:, cols:] - ii[:-rows or None, cols:]
                  - ii[rows:, :-cols or None] + ii[:-rows or None, :-cols or None])
    sumsq_domain = (ii2[rows:, cols:] - ii2[:-rows or None, cols:]
                    - ii2[rows:, :-cols or None] + ii2[:-rows or None, :-cols or None])

    # Measured: scipy's 'auto' heuristic outperforms a manual small-kernel
    # cutoff here (forcing 'direct' below a size threshold was ~2x SLOWER
    # in profiling -- scipy's C direct-correlation loop doesn't vectorize
    # as well as its FFT path even for small kernels against a mid-size
    # domain), so we defer to it rather than second-guess it.
    cross = correlate(recon_known, range_block, mode='valid', method='auto')

    step = config.step
    sum_domain = sum_domain[::step, ::step]
    sumsq_domain = sumsq_domain[::step, ::step]
    cross = cross[::step, ::step]

    avg_domain = sum_domain / n
    avg_range = float(range_block.mean())
    range_centered_sq = float(((range_block - avg_range) ** 2).sum())

    numerator = cross - n * avg_domain * avg_range
    denom = sumsq_domain - n * avg_domain ** 2
    denom_safe = np.where(denom > 1e-6, denom, 1.0)
    err = range_centered_sq - np.where(denom > 1e-6, (numerator ** 2) / denom_safe, 0.0)
    err = np.clip(err, 0.0, None)

    # Quantization-aware position selection (opt-in via config.quantization_aware).
    #
    # In plain terms: the block above picks whichever domain position looks
    # like the best match *before* K and C get rounded to fit their 5-bit/
    # 7-bit slots in the file format. That's usually fine, but occasionally a
    # position that looked slightly worse before rounding rounds *better*
    # than the one that looked best -- so the position `err`'s argmin picks
    # isn't always the position that reconstructs best once quantization
    # actually happens. This block re-picks the position using the error
    # *after* quantization instead.
    #
    # K and C don't need a joint/conditional search to do this: because the
    # domain window is mean-centered before K is applied and C is added back
    # (see the class docstring), the optimal C never depends on which K you
    # pick and vice versa -- rounding each to its own nearest grid value,
    # independently, already gives the best achievable discrete (K, C) pair.
    # The only subtlety is that evaluating error at those rounded values
    # requires mapping our C back to the raw intercept of R ~= K*D + C_eff
    # (C_eff = avg_domain*(1-K) + C) before plugging into the expanded SSE
    # formula below -- our C isn't that raw intercept itself.
    if config.quantization_aware:
        k_cont = np.where(denom > 1e-6, numerator / denom_safe, 0.0)
        c_cont = avg_range - avg_domain
        k_q = dequantize_array(quantize_array(k_cont, config.k_bits, *config.k_range),
                                config.k_bits, *config.k_range)
        c_q = dequantize_array(quantize_array(c_cont, config.c_bits, *config.c_range),
                                config.c_bits, *config.c_range)
        c_eff = avg_domain * (1.0 - k_q) + c_q

        sum_range = n * avg_range
        sumsq_range = range_centered_sq + n * avg_range ** 2
        err_quant = (
            sumsq_range
            - 2.0 * k_q * cross
            - 2.0 * c_eff * sum_range
            + (k_q ** 2) * sumsq_domain
            + 2.0 * k_q * c_eff * sum_domain
            + n * (c_eff ** 2)
        )
        idx = np.unravel_index(np.argmin(err_quant), err_quant.shape)
    else:
        idx = np.unravel_index(np.argmin(err), err.shape)
    best_y, best_x = idx[0] * step, idx[1] * step
    d = denom[idx]
    best_k = float(numerator[idx] / d) if d > 1e-6 else 0.0
    best_avg_domain = float(avg_domain[idx])
    best_c = avg_range - best_avg_domain

    k_idx = quantize(best_k, config.k_bits, *config.k_range)
    c_idx = quantize(best_c, config.c_bits, *config.c_range)
    k_q = dequantize(k_idx, config.k_bits, *config.k_range)
    c_q = dequantize(c_idx, config.c_bits, *config.c_range)

    domain_block = recon_known[best_y:best_y + rows, best_x:best_x + cols]
    patch = k_q * (domain_block - best_avg_domain) + best_avg_domain + c_q
    patch = np.clip(patch, 0, 255)
    final_error = float(((patch - range_block) ** 2).sum())

    return _SearchResult(best_y, best_x, k_idx, c_idx, patch, final_error)


def _encode_block(original, recon_known, ii, ii2, row, col, rows, cols, config: FractalConfig) -> _Candidate:
    range_block = original[row:row + rows, col:col + cols]
    leaf = _search_domain(range_block, recon_known, ii, ii2, config)
    leaf_candidate = _Candidate(
        error=leaf.error,
        partition_bits=[0],
        transforms=[Transform(row, col, rows, cols,
                               leaf.domain_row, leaf.domain_col, leaf.k_idx, leaf.c_idx)],
        patch=leaf.patch,
    )

    # Early-termination pruning: the split test below is
    # `leaf.error > split_error + error_thresh`, where split_error is a sum
    # of squared errors and therefore always >= 0. That means
    # `split_error + error_thresh >= error_thresh` unconditionally, so
    # whenever `leaf.error <= error_thresh` we already know the split test
    # cannot pass -- no matter what the four children would find, splitting
    # will never be chosen. This is a provably lossless shortcut (verified
    # both by this inequality and empirically against an unpruned run), not
    # a quality/quality-vs-speed heuristic: skipping the children here
    # always produces the exact same leaf that the full recursion would
    # have produced anyway, just without paying for the four child searches
    # that were always going to be discarded. It's one-directional --
    # `leaf.error > error_thresh` does NOT imply a split, so that case still
    # has to recurse to find out.
    if leaf.error <= config.error_thresh:
        return leaf_candidate

    prow, pcol = rows // 2, cols // 2
    if prow >= config.min_block and pcol >= config.min_block:
        c1 = _encode_block(original, recon_known, ii, ii2, row, col, prow, pcol, config)
        c2 = _encode_block(original, recon_known, ii, ii2, row, col + pcol, prow, cols - pcol, config)
        c3 = _encode_block(original, recon_known, ii, ii2, row + prow, col + pcol, rows - prow, cols - pcol, config)
        c4 = _encode_block(original, recon_known, ii, ii2, row + prow, col, rows - prow, pcol, config)
        split_error = c1.error + c2.error + c3.error + c4.error

        if leaf.error > split_error + config.error_thresh:
            patch = np.zeros((rows, cols))
            patch[:prow, :pcol] = c1.patch
            patch[:prow, pcol:] = c2.patch
            patch[prow:, pcol:] = c3.patch
            patch[prow:, :pcol] = c4.patch
            bits = [1] + c1.partition_bits + c2.partition_bits + c3.partition_bits + c4.partition_bits
            transforms = c1.transforms + c2.transforms + c3.transforms + c4.transforms
            return _Candidate(split_error, bits, transforms, patch)

    return leaf_candidate


def encode_channel(channel: np.ndarray, config: FractalConfig) -> EncodedChannel:
    padded, orig_h, orig_w = pad_channel(channel, config.init_size, config.max_block)
    H, W = padded.shape

    recon = np.zeros((H, W), dtype=np.float64)
    # Quantize the seed to real byte values up front -- it's serialized as
    # uint8 in to_bytes(), so reconstructing from an unquantized float
    # seed (in-memory) vs. the truncated-to-uint8 one (from disk) would
    # silently diverge by a fraction of a level.
    seed = np.clip(np.round(padded[:config.init_size, :config.init_size]), 0, 255)
    recon[:config.init_size, :config.init_size] = seed

    partition_writer = BitWriter()
    all_transforms: list[Transform] = []

    domain_rows = domain_cols = config.init_size
    row = 0
    while domain_rows < H or domain_cols < W:
        range_rows = min(domain_rows, config.max_block)
        range_cols = min(domain_cols, config.max_block)
        if domain_cols + range_cols > W:
            range_cols = W - domain_cols

        row = 0
        while row < domain_rows:
            this_rows = min(range_rows, domain_rows - row)
            known = recon[:domain_rows, :domain_cols]
            ii, ii2 = _domain_integral_images(known)
            cand = _encode_block(padded, known, ii, ii2, row, domain_cols, this_rows, range_cols, config)
            recon[row:row + this_rows, domain_cols:domain_cols + range_cols] = cand.patch
            for b in cand.partition_bits:
                partition_writer.write_bit(b)
            all_transforms.extend(cand.transforms)
            row += this_rows
        domain_cols += range_cols

        range_rows2 = min(domain_rows, config.max_block)
        range_cols2 = min(domain_cols // 2, config.max_block)
        if domain_rows + range_rows2 > H:
            range_rows2 = H - domain_rows

        col = 0
        while col < domain_cols and domain_rows < H:
            this_cols = min(range_cols2, domain_cols - col)
            known = recon[:domain_rows, :domain_cols]
            ii, ii2 = _domain_integral_images(known)
            cand = _encode_block(padded, known, ii, ii2, domain_rows, col, range_rows2, this_cols, config)
            recon[domain_rows:domain_rows + range_rows2, col:col + this_cols] = cand.patch
            for b in cand.partition_bits:
                partition_writer.write_bit(b)
            all_transforms.extend(cand.transforms)
            col += this_cols
        domain_rows += range_rows2

    partition_bytes = partition_writer.flush()
    enc = EncodedChannel(
        height=orig_h, width=orig_w, padded_size=H, config=config,
        seed=seed, partition_bits=partition_bytes,
        n_partition_bits=partition_writer.bit_length(), transforms=all_transforms,
    )
    return enc

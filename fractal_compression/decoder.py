"""
Fractal decoder.

Mirrors the encoder's recursion shape exactly: same growth schedule
(causal L-shaped doubling), same quadtree split/leaf decisions (read
from the partition-bit stream instead of decided locally), same
domain-source rule (always the buffer being reconstructed, never the
original -- there is no original at decode time).

Because every domain reference is guaranteed to point at pixels already
written earlier in this same traversal, one pass suffices -- there is no
analogue of the iterate-to-convergence loop classical fractal decoders
need.
"""
from __future__ import annotations
from itertools import islice
import numpy as np

from .codec import FractalConfig, EncodedChannel, Transform, dequantize
from .bitstream import BitReader


def _bit_stream(data: bytes, n_bits: int):
    reader = BitReader(data)
    for _ in range(n_bits):
        yield reader.read_bit()


def _decode_block(recon_known, row, col, rows, cols, bit_iter, transform_iter, config: FractalConfig):
    bit = next(bit_iter)
    prow, pcol = rows // 2, cols // 2
    if bit == 1:
        p1 = _decode_block(recon_known, row, col, prow, pcol, bit_iter, transform_iter, config)
        p2 = _decode_block(recon_known, row, col + pcol, prow, cols - pcol, bit_iter, transform_iter, config)
        p3 = _decode_block(recon_known, row + prow, col + pcol, rows - prow, cols - pcol, bit_iter, transform_iter, config)
        p4 = _decode_block(recon_known, row + prow, col, rows - prow, pcol, bit_iter, transform_iter, config)
        patch = np.zeros((rows, cols))
        patch[:prow, :pcol] = p1
        patch[:prow, pcol:] = p2
        patch[prow:, pcol:] = p3
        patch[prow:, :pcol] = p4
        return patch

    t: Transform = next(transform_iter)
    domain_block = recon_known[t.domain_row:t.domain_row + rows, t.domain_col:t.domain_col + cols]
    avg_domain = float(domain_block.mean())
    k = dequantize(t.k_idx, config.k_bits, *config.k_range)
    c = dequantize(t.c_idx, config.c_bits, *config.c_range)
    patch = k * (domain_block - avg_domain) + avg_domain + c
    return np.clip(patch, 0, 255)


def decode_channel(enc: EncodedChannel) -> np.ndarray:
    config = enc.config
    H = enc.padded_size
    recon = np.zeros((H, H), dtype=np.float64)
    recon[:config.init_size, :config.init_size] = enc.seed

    bit_iter = _bit_stream(enc.partition_bits, enc.n_partition_bits)
    transform_iter = iter(enc.transforms)

    domain_rows = domain_cols = config.init_size
    while domain_rows < H or domain_cols < H:
        range_rows = min(domain_rows, config.max_block)
        range_cols = min(domain_cols, config.max_block)
        if domain_cols + range_cols > H:
            range_cols = H - domain_cols

        row = 0
        while row < domain_rows:
            this_rows = min(range_rows, domain_rows - row)
            known = recon[:domain_rows, :domain_cols]
            patch = _decode_block(known, row, domain_cols, this_rows, range_cols, bit_iter, transform_iter, config)
            recon[row:row + this_rows, domain_cols:domain_cols + range_cols] = patch
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
            patch = _decode_block(known, domain_rows, col, range_rows2, this_cols, bit_iter, transform_iter, config)
            recon[domain_rows:domain_rows + range_rows2, col:col + this_cols] = patch
            col += this_cols
        domain_rows += range_rows2

    return recon[:enc.height, :enc.width]


def derive_leaf_block_sizes(partition_bits: bytes, n_partition_bits: int, padded_size: int,
                             init_size: int, max_block: int, min_block: int) -> list:
    """Replays only the quadtree *shape* (no pixel data) to recover each
    leaf's (rows, cols) in traversal order -- needed to deserialize a
    payload that was written with EncodedChannel.to_bytes(), since the
    payload itself doesn't repeat block sizes."""
    bit_iter = _bit_stream(partition_bits, n_partition_bits)
    sizes: list = []

    def walk(rows, cols):
        bit = next(bit_iter)
        prow, pcol = rows // 2, cols // 2
        if bit == 1:
            walk(prow, pcol)
            walk(prow, cols - pcol)
            walk(rows - prow, cols - pcol)
            walk(rows - prow, pcol)
        else:
            sizes.append((rows, cols))

    H = padded_size
    domain_rows = domain_cols = init_size
    while domain_rows < H or domain_cols < H:
        range_rows = min(domain_rows, max_block)
        range_cols = min(domain_cols, max_block)
        if domain_cols + range_cols > H:
            range_cols = H - domain_cols
        row = 0
        while row < domain_rows:
            this_rows = min(range_rows, domain_rows - row)
            walk(this_rows, range_cols)
            row += this_rows
        domain_cols += range_cols

        range_rows2 = min(domain_rows, max_block)
        range_cols2 = min(domain_cols // 2, max_block)
        if domain_rows + range_rows2 > H:
            range_rows2 = H - domain_rows
        col = 0
        while col < domain_cols and domain_rows < H:
            this_cols = min(range_cols2, domain_cols - col)
            walk(range_rows2, this_cols)
            col += this_cols
        domain_rows += range_rows2

    return sizes

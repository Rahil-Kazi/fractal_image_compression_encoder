import inspect
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fractal_compression import FractalConfig, encode_channel, decode_channel
import fractal_compression.encoder as enc_mod

PRUNE_MARKER = "if leaf.error <= config.error_thresh:"


def _synthetic_image(size=48, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 4 * np.pi, size)
    grid = np.sin(x)[:, None] * np.cos(x)[None, :] * 60 + 128
    grid += rng.normal(0, 5, size=(size, size))
    return np.clip(grid, 0, 255)


def _unpruned_encode_channel():
    """Derives an unpruned twin of encode_channel directly from the live
    source of _encode_block (stripping only the early-termination
    if/return) and encode_channel, via inspect.getsource + exec. This is
    used instead of a hand-transcribed duplicate so the reference can never
    silently drift out of sync with the real implementation, and so a typo
    in a hand copy can't masquerade as a behavioral difference."""
    block_src = inspect.getsource(enc_mod._encode_block)
    lines = block_src.splitlines()
    out, skipping = [], False
    for line in lines:
        if PRUNE_MARKER in line:
            skipping = "if"
            continue
        if skipping == "if" and "return leaf_candidate" in line:
            skipping = False
            continue
        if skipping == "if":
            continue
        out.append(line)
    block_src_stripped = "\n".join(out).replace(
        "def _encode_block(", "def _encode_block_unpruned(")
    assert PRUNE_MARKER not in block_src_stripped, \
        "failed to strip the early-termination check -- PRUNE_MARKER is stale"

    channel_src = inspect.getsource(enc_mod.encode_channel).replace(
        "def encode_channel(", "def encode_channel_unpruned(").replace(
        "_encode_block(", "_encode_block_unpruned(")

    ns = dict(enc_mod.__dict__)
    exec(compile(block_src_stripped, "<unpruned _encode_block>", "exec"), ns)
    exec(compile(channel_src, "<unpruned encode_channel>", "exec"), ns)
    return ns["encode_channel_unpruned"]


def test_early_termination_matches_unpruned_recursion_exactly():
    img = _synthetic_image(48)
    cfg = FractalConfig(error_thresh=100)

    pruned_enc = encode_channel(img, cfg)
    unpruned_enc = _unpruned_encode_channel()(img, cfg)

    pruned_t = [(t.row, t.col, t.rows, t.cols, t.domain_row, t.domain_col, t.k_idx, t.c_idx)
                for t in pruned_enc.transforms]
    unpruned_t = [(t.row, t.col, t.rows, t.cols, t.domain_row, t.domain_col, t.k_idx, t.c_idx)
                  for t in unpruned_enc.transforms]
    assert pruned_t == unpruned_t
    assert pruned_enc.n_partition_bits == unpruned_enc.n_partition_bits
    assert pruned_enc.partition_bits == unpruned_enc.partition_bits

    assert np.array_equal(decode_channel(pruned_enc), decode_channel(unpruned_enc))


def test_early_termination_matches_unpruned_recursion_multiple_configs():
    # Different error thresholds exercise the pruning check at very
    # different points in the tree (loose threshold prunes near the root,
    # tight threshold rarely prunes at all).
    for seed in [0, 1, 2]:
        for error_thresh in [20, 100, 1000]:
            img = _synthetic_image(32, seed=seed)
            cfg = FractalConfig(error_thresh=error_thresh, max_block=32)
            pruned_enc = encode_channel(img, cfg)
            unpruned_enc = _unpruned_encode_channel()(img, cfg)
            assert pruned_enc.n_partition_bits == unpruned_enc.n_partition_bits
            assert pruned_enc.partition_bits == unpruned_enc.partition_bits
            assert len(pruned_enc.transforms) == len(unpruned_enc.transforms)


def test_early_termination_roundtrip_quality_unchanged():
    img = _synthetic_image(48)
    cfg = FractalConfig(error_thresh=100, quantization_aware=True)
    enc = encode_channel(img, cfg)
    recon = decode_channel(enc)
    assert recon.shape == img.shape
    from fractal_compression.metrics import psnr
    assert psnr(img, recon) > 20


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

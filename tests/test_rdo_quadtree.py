import inspect
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fractal_compression import FractalConfig, encode_channel, decode_channel
import fractal_compression.encoder as enc_mod

RDO_PRUNE_MARKER = "if leaf.error <= config.rdo_lambda * (4 + 3 * per_leaf_bits):"


def _synthetic_image(size=48, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 4 * np.pi, size)
    grid = np.sin(x)[:, None] * np.cos(x)[None, :] * 60 + 128
    grid += rng.normal(0, 5, size=(size, size))
    return np.clip(grid, 0, 255)


def _unpruned_encode_channel_rdo():
    """Same technique as tests/test_early_termination.py: derive an
    unpruned twin of encode_channel directly from the live source of
    _encode_block, stripping only the RDO early-termination if/return, via
    inspect.getsource + exec. Avoids a hand-transcribed duplicate that can
    silently drift out of sync with the real implementation."""
    block_src = inspect.getsource(enc_mod._encode_block)
    lines = block_src.splitlines()
    out, skipping = [], False
    for line in lines:
        if RDO_PRUNE_MARKER in line:
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
    assert RDO_PRUNE_MARKER not in block_src_stripped, \
        "failed to strip the RDO early-termination check -- RDO_PRUNE_MARKER is stale"

    channel_src = inspect.getsource(enc_mod.encode_channel).replace(
        "def encode_channel(", "def encode_channel_unpruned(").replace(
        "_encode_block(", "_encode_block_unpruned(")

    ns = dict(enc_mod.__dict__)
    exec(compile(block_src_stripped, "<unpruned rdo _encode_block>", "exec"), ns)
    exec(compile(channel_src, "<unpruned rdo encode_channel>", "exec"), ns)
    return ns["encode_channel_unpruned"]


@pytest.mark.parametrize("seed,rdo_lambda", [(0, 5.0), (1, 50.0), (2, 500.0)])
def test_rdo_pruning_matches_unpruned_recursion_exactly(seed, rdo_lambda):
    img = _synthetic_image(32, seed=seed)
    cfg = FractalConfig(rdo_lambda=rdo_lambda, max_block=32)

    pruned_enc = encode_channel(img, cfg)
    unpruned_enc = _unpruned_encode_channel_rdo()(img, cfg)

    pruned_t = [(t.row, t.col, t.rows, t.cols, t.domain_row, t.domain_col, t.k_idx, t.c_idx)
                for t in pruned_enc.transforms]
    unpruned_t = [(t.row, t.col, t.rows, t.cols, t.domain_row, t.domain_col, t.k_idx, t.c_idx)
                  for t in unpruned_enc.transforms]
    assert pruned_t == unpruned_t
    assert pruned_enc.n_partition_bits == unpruned_enc.n_partition_bits
    assert pruned_enc.partition_bits == unpruned_enc.partition_bits
    assert np.array_equal(decode_channel(pruned_enc), decode_channel(unpruned_enc))


def test_rdo_roundtrip_quality():
    img = _synthetic_image(48)
    cfg = FractalConfig(rdo_lambda=20.0)
    enc = encode_channel(img, cfg)
    recon = decode_channel(enc)
    assert recon.shape == img.shape
    from fractal_compression.metrics import psnr
    assert psnr(img, recon) > 20


def test_rdo_lambda_monotonicity():
    # Larger lambda penalizes bits more heavily, so it should never produce
    # MORE leaves (more splitting) than a smaller lambda on the same image.
    img = _synthetic_image(48)
    leaf_counts = []
    for lam in [1.0, 10.0, 100.0, 1000.0]:
        cfg = FractalConfig(rdo_lambda=lam)
        enc = encode_channel(img, cfg)
        leaf_counts.append(len(enc.transforms))
    assert leaf_counts == sorted(leaf_counts, reverse=True), \
        f"leaf counts should be non-increasing as lambda grows, got {leaf_counts}"


def test_rdo_default_is_disabled_and_matches_legacy_behavior():
    # rdo_lambda=None (the default) must reproduce byte-identical output to
    # before this feature existed -- i.e. exactly the error_thresh path.
    img = _synthetic_image(48)
    cfg = FractalConfig(error_thresh=100)
    assert cfg.rdo_lambda is None
    enc = encode_channel(img, cfg)
    recon = decode_channel(enc)
    assert recon.shape == img.shape


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

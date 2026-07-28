import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fractal_compression import FractalConfig, encode_channel, decode_channel
from fractal_compression.entropy import (
    _build_huffman_lengths, _canonical_codes, _code_stream, _decode_stream,
    encode_channel_entropy, decode_channel_entropy, entropy_bits_per_pixel,
)
from fractal_compression.bitstream import BitWriter, BitReader


def _synthetic_image(size=64, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 4 * np.pi, size)
    grid = np.sin(x)[:, None] * np.cos(x)[None, :] * 60 + 128
    grid += rng.normal(0, 5, size=(size, size))
    return np.clip(grid, 0, 255)


def test_huffman_roundtrip_skewed_symbols():
    rng = np.random.default_rng(2)
    # heavily skewed distribution over an 8-symbol alphabet
    probs = np.array([0.5, 0.2, 0.1, 0.08, 0.05, 0.03, 0.02, 0.02])
    symbols = rng.choice(8, size=500, p=probs).tolist()

    writer = BitWriter()
    _code_stream(writer, symbols, alphabet_bits=3)
    data = writer.flush()

    reader = BitReader(data)
    decoded = _decode_stream(reader, alphabet_bits=3, n_symbols=len(symbols))
    assert decoded == symbols


def test_huffman_single_distinct_symbol():
    symbols = [7] * 20
    writer = BitWriter()
    _code_stream(writer, symbols, alphabet_bits=4)
    data = writer.flush()

    reader = BitReader(data)
    decoded = _decode_stream(reader, alphabet_bits=4, n_symbols=len(symbols))
    assert decoded == symbols


def test_channel_entropy_roundtrip_is_lossless():
    img = _synthetic_image(64)
    cfg = FractalConfig(error_thresh=60)
    enc = encode_channel(img, cfg)
    direct = decode_channel(enc)

    data = encode_channel_entropy(enc)
    enc2 = decode_channel_entropy(data, cfg)
    from_entropy = decode_channel(enc2)

    assert np.array_equal(direct, from_entropy)


def test_entropy_coding_reduces_bits_with_enough_leaves():
    img = _synthetic_image(64)
    cfg = FractalConfig(error_thresh=10)  # low threshold -> many leaves, tables amortize
    enc = encode_channel(img, cfg)
    assert len(enc.transforms) > 50
    assert entropy_bits_per_pixel(enc) <= enc.bits_per_pixel()


def test_entropy_coding_never_regresses_even_with_few_leaves():
    # High threshold -> very few leaves, where per-stream Huffman tables
    # would often cost more than they save; the per-stream fixed-width
    # fallback should keep entropy coding a no-op rather than a net loss.
    img = _synthetic_image(64)
    cfg = FractalConfig(error_thresh=5000)
    enc = encode_channel(img, cfg)
    fixed_bytes = len(enc.to_bytes())
    entropy_bytes = len(encode_channel_entropy(enc))
    assert entropy_bytes <= fixed_bytes + 1  # at most 1 bit/stream flag overhead, rounded up


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

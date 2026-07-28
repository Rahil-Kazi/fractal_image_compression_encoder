"""
Shared pieces between the encoder and decoder: configuration, the
quantizers for the contrast (K) and brightness (C) coefficients, and the
container for one channel's compressed bitstream (with real byte
serialization, so "compression ratio" is measured on actual bytes rather
than an estimate).

Design notes (what's different from the original C++ codebase, and why):

- Domain search is vectorized (see encoder.py) instead of a per-position
  Python loop -- the main encode-speed win.
- Coordinate fields use a FIXED bit width (ceil(log2(padded_size))) for
  every transform, rather than the original's incrementally-growing
  width. That original trick saves a handful of bits early on, but it
  makes the bitstream format state-dependent in a way that's easy to get
  subtly wrong; a fixed width trades a small amount of compactness for a
  format that's simple to serialize, parse, and reason about.
- K and C are quantized with an explicit, documented range instead of
  magic constants, and both are configurable so the ablation study in
  benchmarks/ can sweep them.
- Domain source is always the *reconstructed* (lossy) buffer, never the
  original -- this is what makes causal, single-pass, non-iterative
  reconstruction correct, and is preserved from the original design.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .bitstream import BitWriter, BitReader


@dataclass
class FractalConfig:
    init_size: int = 4            # raw seed block side length
    min_block: int = 2            # smallest quadtree leaf
    max_block: int = 64           # largest block considered in one go
    error_thresh: float = 100.0   # summed-squared-error split threshold
    step: int = 1                 # domain search stride (>1 trades quality for speed)
    k_bits: int = 5
    c_bits: int = 7
    k_range: tuple = (-1.5, 1.5)
    c_range: tuple = (-255.0, 255.0)


@dataclass
class Transform:
    """One quadtree leaf's fractal code."""
    row: int
    col: int
    rows: int
    cols: int
    domain_row: int
    domain_col: int
    k_idx: int
    c_idx: int


def quantize(value: float, bits: int, vmin: float, vmax: float) -> int:
    levels = (1 << bits) - 1
    t = (value - vmin) / (vmax - vmin)
    return int(round(np.clip(t, 0.0, 1.0) * levels))


def dequantize(idx: int, bits: int, vmin: float, vmax: float) -> float:
    levels = (1 << bits) - 1
    return vmin + (idx / levels) * (vmax - vmin)


@dataclass
class EncodedChannel:
    height: int                # original (pre-padding) height
    width: int                 # original (pre-padding) width
    padded_size: int           # square canvas side actually coded
    config: FractalConfig
    seed: np.ndarray
    partition_bits: bytes
    n_partition_bits: int
    transforms: list = field(default_factory=list)

    @property
    def coord_bits(self) -> int:
        return max(1, int(np.ceil(np.log2(self.padded_size))))

    def total_bits(self) -> int:
        per_leaf = 2 * self.coord_bits + self.config.k_bits + self.config.c_bits
        return self.seed.size * 8 + self.n_partition_bits + len(self.transforms) * per_leaf

    def bits_per_pixel(self) -> float:
        return self.total_bits() / (self.height * self.width)

    def compression_ratio(self) -> float:
        original_bits = self.height * self.width * 8
        return original_bits / self.total_bits()

    # ---------------- real byte serialization ----------------

    def to_bytes(self) -> bytes:
        cb = self.coord_bits
        header = BitWriter()
        header.write_uint(self.height, 16)
        header.write_uint(self.width, 16)
        header.write_uint(self.padded_size, 16)
        header.write_uint(self.config.init_size, 8)
        header.write_uint(len(self.partition_bits), 32)
        header.write_uint(self.n_partition_bits, 32)
        header.write_uint(len(self.transforms), 32)
        header_bytes = header.flush()

        payload = BitWriter()
        for t in self.transforms:
            payload.write_uint(t.domain_row, cb)
            payload.write_uint(t.domain_col, cb)
            payload.write_uint(t.k_idx, self.config.k_bits)
            payload.write_uint(t.c_idx, self.config.c_bits)
        payload_bytes = payload.flush()

        return (header_bytes + self.seed.astype(np.uint8).tobytes()
                + self.partition_bits + payload_bytes)

    @classmethod
    def from_bytes(cls, data: bytes, config: FractalConfig,
                   block_sizes: list) -> "EncodedChannel":
        """block_sizes: the (rows, cols) for each leaf in traversal order.
        This is fully determined by the partition bits (a pure function of
        the quadtree shape), so the decoder derives it itself while it
        walks the tree -- see decoder.decode_channel. Passed in here so
        this class stays independent of the decoder module.
        """
        header_reader = BitReader(data)
        height = header_reader.read_uint(16)
        width = header_reader.read_uint(16)
        padded_size = header_reader.read_uint(16)
        init_size = header_reader.read_uint(8)
        n_partition_bytes = header_reader.read_uint(32)
        n_partition_bits = header_reader.read_uint(32)
        n_transforms = header_reader.read_uint(32)

        header_bit_len = 16 + 16 + 16 + 8 + 32 + 32 + 32
        header_byte_len = (header_bit_len + 7) // 8

        offset = header_byte_len
        seed = np.frombuffer(data, dtype=np.uint8,
                              count=init_size * init_size,
                              offset=offset).reshape(init_size, init_size).astype(np.float64)
        offset += init_size * init_size

        partition_bits = data[offset:offset + n_partition_bytes]
        offset += n_partition_bytes

        payload_reader = BitReader(data[offset:])
        cb = max(1, int(np.ceil(np.log2(padded_size))))
        transforms = []
        for i in range(n_transforms):
            dr = payload_reader.read_uint(cb)
            dc = payload_reader.read_uint(cb)
            k_idx = payload_reader.read_uint(config.k_bits)
            c_idx = payload_reader.read_uint(config.c_bits)
            rows, cols = block_sizes[i]
            transforms.append(Transform(0, 0, rows, cols, dr, dc, k_idx, c_idx))

        return cls(height=height, width=width, padded_size=padded_size, config=config,
                    seed=seed, partition_bits=partition_bits,
                    n_partition_bits=n_partition_bits, transforms=transforms)

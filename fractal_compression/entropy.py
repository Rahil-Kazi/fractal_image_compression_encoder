"""
Entropy coding on top of the fixed-width bitstream from codec.py.

The fixed-width format (EncodedChannel.to_bytes) spends a constant number of
bits per K index, C index, coordinate, and partition bit regardless of their
actual distribution. Empirically (see the plan behind this module) those
distributions are far from uniform -- c_idx in particular uses under half its
nominal entropy -- so a canonical Huffman code recovers real bits without
touching the encoder's search/quantization at all.

This module re-packs an already-encoded EncodedChannel's Transform list and
partition bits into a smaller, losslessly-decodable byte string. It does not
change what gets encoded, only how the same (row/col/k_idx/c_idx) values are
serialized.

Design choices worth noting:
- Coordinates are Huffman-coded as raw domain_row/domain_col values, not as
  deltas from the range block's own position -- deltas measured *worse* here,
  because the causal-region restriction means many domain matches are not
  spatially near their range block.
- Partition bits are grouped into 4-bit tuples before coding, since a binary
  alphabet cannot beat 1 bit/symbol under Huffman no matter how skewed.
- Huffman tables are sparse (only used symbols, as explicit (symbol, length)
  pairs) rather than one dense length-per-possible-symbol slot, since most
  streams here use well under half their nominal alphabet in a single image
  and a dense table would often cost more than it saves.
- Each of the 5 streams picks Huffman-vs-fixed-width independently (a 1-bit
  flag per stream) based on which is actually smaller once table overhead is
  counted. Measured on camera_128: at typical leaf counts, only c_idx (the
  most skewed stream) has few enough unique values to make the sparse table
  pay for itself -- domain_row/domain_col/k_idx/nibbles frequently do NOT,
  because they use a large fraction of their alphabet relative to the number
  of leaves. Coding every stream unconditionally would make entropy coding a
  net loss on low-leaf-count images; the per-stream fallback makes it a
  (small) win or a no-op instead.
"""
from __future__ import annotations
import heapq
import itertools
import numpy as np

from .bitstream import BitWriter, BitReader
from .codec import EncodedChannel, FractalConfig, Transform
from .decoder import derive_leaf_block_sizes

_LENGTH_BITS = 6  # per-symbol Huffman code length field width in the sparse table


def _build_huffman_lengths(counts: dict) -> dict:
    """symbol -> counts (only symbols that occur) -> symbol -> code length."""
    symbols = list(counts.keys())
    if len(symbols) == 1:
        return {symbols[0]: 1}

    tie = itertools.count()
    heap = [[cnt, next(tie), [sym]] for sym, cnt in counts.items()]
    heapq.heapify(heap)
    lengths = {sym: 0 for sym in symbols}

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for sym in lo[2]:
            lengths[sym] += 1
        for sym in hi[2]:
            lengths[sym] += 1
        heapq.heappush(heap, [lo[0] + hi[0], next(tie), lo[2] + hi[2]])

    return lengths


def _canonical_codes(lengths: dict) -> dict:
    """symbol -> length -> symbol -> (code, length), assigned in the
    canonical (length, symbol) sorted order."""
    ordered = sorted(lengths.items(), key=lambda kv: (kv[1], kv[0]))
    codes = {}
    code = 0
    prev_length = ordered[0][1]
    for i, (sym, length) in enumerate(ordered):
        if i > 0:
            code = (code + 1) << (length - prev_length)
        codes[sym] = (code, length)
        prev_length = length
    return codes


def _write_sparse_table(writer: BitWriter, lengths: dict, symbol_bits: int) -> None:
    ordered = sorted(lengths.items(), key=lambda kv: (kv[1], kv[0]))
    writer.write_uint(len(ordered), 16)
    for sym, length in ordered:
        assert length < (1 << _LENGTH_BITS), "Huffman code length overflowed the table field"
        writer.write_uint(sym, symbol_bits)
        writer.write_uint(length, _LENGTH_BITS)


def _read_sparse_table(reader: BitReader, symbol_bits: int) -> dict:
    n_used = reader.read_uint(16)
    lengths = {}
    for _ in range(n_used):
        sym = reader.read_uint(symbol_bits)
        length = reader.read_uint(_LENGTH_BITS)
        lengths[sym] = length
    return lengths


def _write_code(writer: BitWriter, code: int, length: int) -> None:
    for i in reversed(range(length)):
        writer.write_bit((code >> i) & 1)


def _encode_symbols(writer: BitWriter, symbols: list, codes: dict) -> None:
    for sym in symbols:
        code, length = codes[sym]
        _write_code(writer, code, length)


def _decode_symbols(reader: BitReader, lengths: dict, n_symbols: int) -> list:
    codes = _canonical_codes(lengths)
    by_length_code = {(length, code): sym for sym, (code, length) in codes.items()}
    max_length = max(lengths.values())

    out = []
    for _ in range(n_symbols):
        code = 0
        length = 0
        while length <= max_length:
            code = (code << 1) | reader.read_bit()
            length += 1
            sym = by_length_code.get((length, code))
            if sym is not None:
                out.append(sym)
                break
    return out


def _huffman_cost_bits(lengths: dict, symbols: list, alphabet_bits: int) -> int:
    table_bits = 16 + len(lengths) * (alphabet_bits + _LENGTH_BITS)
    payload_bits = sum(lengths[s] for s in symbols)
    return table_bits + payload_bits


def _code_stream(writer: BitWriter, symbols: list, alphabet_bits: int) -> None:
    """Huffman-codes `symbols` if that's actually smaller than fixed-width
    once the sparse table overhead is counted; otherwise falls back to
    plain fixed-width, so per-stream entropy coding never costs more than 1
    extra bit (the mode flag) versus the original format."""
    counts: dict = {}
    for s in symbols:
        counts[s] = counts.get(s, 0) + 1
    lengths = _build_huffman_lengths(counts)
    huffman_bits = _huffman_cost_bits(lengths, symbols, alphabet_bits)
    fixed_bits = len(symbols) * alphabet_bits

    if huffman_bits < fixed_bits:
        writer.write_bit(1)
        _write_sparse_table(writer, lengths, alphabet_bits)
        codes = _canonical_codes(lengths)
        _encode_symbols(writer, symbols, codes)
    else:
        writer.write_bit(0)
        for s in symbols:
            writer.write_uint(s, alphabet_bits)


def _decode_stream(reader: BitReader, alphabet_bits: int, n_symbols: int) -> list:
    use_huffman = reader.read_bit()
    if use_huffman:
        lengths = _read_sparse_table(reader, alphabet_bits)
        return _decode_symbols(reader, lengths, n_symbols)
    return [reader.read_uint(alphabet_bits) for _ in range(n_symbols)]


def _partition_bits_to_nibbles(partition_bits: bytes, n_partition_bits: int) -> list:
    reader = BitReader(partition_bits)
    bits = [reader.read_bit() for _ in range(n_partition_bits)]
    pad = (-len(bits)) % 4
    bits.extend([0] * pad)
    nibbles = []
    for i in range(0, len(bits), 4):
        group = bits[i:i + 4]
        nibbles.append(sum(b << j for j, b in enumerate(group)))
    return nibbles


def _nibbles_to_partition_bytes(nibbles: list, n_partition_bits: int) -> bytes:
    writer = BitWriter()
    written = 0
    for nibble in nibbles:
        for j in range(4):
            if written >= n_partition_bits:
                break
            writer.write_bit((nibble >> j) & 1)
            written += 1
    return writer.flush()


def encode_channel_entropy(enc: EncodedChannel) -> bytes:
    cfg = enc.config
    cb = enc.coord_bits

    nibbles = _partition_bits_to_nibbles(enc.partition_bits, enc.n_partition_bits)
    domain_rows = [t.domain_row for t in enc.transforms]
    domain_cols = [t.domain_col for t in enc.transforms]
    k_idxs = [t.k_idx for t in enc.transforms]
    c_idxs = [t.c_idx for t in enc.transforms]

    header = BitWriter()
    header.write_uint(enc.height, 16)
    header.write_uint(enc.width, 16)
    header.write_uint(enc.padded_size, 16)
    header.write_uint(cfg.init_size, 8)
    header.write_uint(enc.n_partition_bits, 32)
    header.write_uint(len(enc.transforms), 32)
    header_bytes = header.flush()

    body = BitWriter()
    _code_stream(body, nibbles, 4)
    _code_stream(body, domain_rows, cb)
    _code_stream(body, domain_cols, cb)
    _code_stream(body, k_idxs, cfg.k_bits)
    _code_stream(body, c_idxs, cfg.c_bits)
    body_bytes = body.flush()

    return header_bytes + enc.seed.astype("uint8").tobytes() + body_bytes


def decode_channel_entropy(data: bytes, config: FractalConfig) -> EncodedChannel:
    header_reader = BitReader(data)
    height = header_reader.read_uint(16)
    width = header_reader.read_uint(16)
    padded_size = header_reader.read_uint(16)
    init_size = header_reader.read_uint(8)
    n_partition_bits = header_reader.read_uint(32)
    n_transforms = header_reader.read_uint(32)

    header_bit_len = 16 + 16 + 16 + 8 + 32 + 32
    header_byte_len = (header_bit_len + 7) // 8

    offset = header_byte_len
    seed = np.frombuffer(data, dtype=np.uint8,
                          count=init_size * init_size,
                          offset=offset).reshape(init_size, init_size).astype(np.float64)
    offset += init_size * init_size

    cb = max(1, int(np.ceil(np.log2(padded_size))))
    n_nibble_groups = (n_partition_bits + 3) // 4

    body_reader = BitReader(data[offset:])
    nibbles = _decode_stream(body_reader, 4, n_nibble_groups)
    domain_rows = _decode_stream(body_reader, cb, n_transforms)
    domain_cols = _decode_stream(body_reader, cb, n_transforms)
    k_idxs = _decode_stream(body_reader, config.k_bits, n_transforms)
    c_idxs = _decode_stream(body_reader, config.c_bits, n_transforms)

    partition_bytes = _nibbles_to_partition_bytes(nibbles, n_partition_bits)
    block_sizes = derive_leaf_block_sizes(partition_bytes, n_partition_bits, padded_size,
                                           init_size, config.max_block, config.min_block)

    transforms = [
        Transform(0, 0, rows, cols, dr, dc, k_idx, c_idx)
        for (rows, cols), dr, dc, k_idx, c_idx in
        zip(block_sizes, domain_rows, domain_cols, k_idxs, c_idxs)
    ]

    return EncodedChannel(height=height, width=width, padded_size=padded_size, config=config,
                           seed=seed, partition_bits=partition_bytes,
                           n_partition_bits=n_partition_bits, transforms=transforms)


def entropy_bits_per_pixel(enc: EncodedChannel) -> float:
    return len(encode_channel_entropy(enc)) * 8 / (enc.height * enc.width)

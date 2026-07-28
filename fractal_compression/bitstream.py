"""
Minimal LSB-first bit packing/unpacking, used for the partition-scheme
stream and the per-leaf coordinate/K/C streams.

This replaces the C++ FractalCompression::putbit / dec2bin / getbit /
bin2dec quartet with a small, well-tested pair of classes.
"""
from __future__ import annotations


class BitWriter:
    def __init__(self):
        self._bytes = bytearray()
        self._cur = 0
        self._nbits = 0

    def write_bit(self, bit: int) -> None:
        if bit:
            self._cur |= (1 << self._nbits)
        self._nbits += 1
        if self._nbits == 8:
            self._bytes.append(self._cur)
            self._cur = 0
            self._nbits = 0

    def write_uint(self, value: int, bits: int) -> None:
        """Write `value` as an unsigned integer using exactly `bits` bits, LSB first."""
        for i in range(bits):
            self.write_bit((value >> i) & 1)

    def flush(self) -> bytes:
        if self._nbits > 0:
            self._bytes.append(self._cur)
            self._cur = 0
            self._nbits = 0
        return bytes(self._bytes)

    def bit_length(self) -> int:
        return len(self._bytes) * 8 + self._nbits


class BitReader:
    def __init__(self, data: bytes):
        self._data = data
        self._byte_idx = 0
        self._bit_idx = 0

    def read_bit(self) -> int:
        byte = self._data[self._byte_idx]
        bit = (byte >> self._bit_idx) & 1
        self._bit_idx += 1
        if self._bit_idx == 8:
            self._bit_idx = 0
            self._byte_idx += 1
        return bit

    def read_uint(self, bits: int) -> int:
        value = 0
        for i in range(bits):
            value |= (self.read_bit() << i)
        return value

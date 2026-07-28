"""
Whole-image encode/decode: grayscale passes straight through to the
single-channel codec; color images are converted to YCbCr and each plane
coded independently, with the option to subsample the two chroma planes
2x2 the way JPEG's 4:2:0 mode does (chroma is far less bit-hungry to get
"good enough", so this is the main rate/quality knob for color).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .codec import FractalConfig
from .encoder import encode_channel
from .decoder import decode_channel
from . import color as colorspace


@dataclass
class EncodedImage:
    is_color: bool
    channels: list          # list[EncodedChannel], length 1 (gray) or 3 (Y, Cb, Cr)
    chroma_subsampled: bool
    chroma_shape: tuple = None  # (h, w) of Cb/Cr before subsampling, for upsampling on decode

    def total_bits(self) -> int:
        return sum(c.total_bits() for c in self.channels)

    def bits_per_pixel(self, height: int, width: int) -> float:
        return self.total_bits() / (height * width)

    def compression_ratio(self, height: int, width: int, n_channels: int) -> float:
        original_bits = height * width * n_channels * 8
        return original_bits / self.total_bits()


def encode_image(img: np.ndarray, config: FractalConfig, chroma_subsample: bool = True) -> EncodedImage:
    img = img.astype(np.float64)
    if img.ndim == 2:
        enc = encode_channel(img, config)
        return EncodedImage(is_color=False, channels=[enc], chroma_subsampled=False)

    ycbcr = colorspace.rgb_to_ycbcr(img)
    y, cb, cr = ycbcr[..., 0], ycbcr[..., 1], ycbcr[..., 2]
    chroma_shape = cb.shape

    if chroma_subsample:
        cb = colorspace.subsample_chroma(cb)
        cr = colorspace.subsample_chroma(cr)

    enc_y = encode_channel(y, config)
    enc_cb = encode_channel(cb, config)
    enc_cr = encode_channel(cr, config)
    return EncodedImage(is_color=True, channels=[enc_y, enc_cb, enc_cr],
                         chroma_subsampled=chroma_subsample, chroma_shape=chroma_shape)


def decode_image(enc: EncodedImage) -> np.ndarray:
    if not enc.is_color:
        return decode_channel(enc.channels[0])

    y = decode_channel(enc.channels[0])
    cb = decode_channel(enc.channels[1])
    cr = decode_channel(enc.channels[2])

    if enc.chroma_subsampled:
        cb = colorspace.upsample_chroma(cb, enc.chroma_shape)
        cr = colorspace.upsample_chroma(cr, enc.chroma_shape)

    ycbcr = np.stack([y, cb, cr], axis=-1)
    return colorspace.ycbcr_to_rgb(ycbcr)

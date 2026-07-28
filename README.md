# Fractal Image Compression (Python)

A from-scratch Python rewrite of the causal, quadtree-partitioned, non-iterative
fractal image codec originally prototyped in C++/OpenCV, built for benchmarking
and (potential) publication rather than just as a port.

## What changed vs. the original C++ codebase

**Fixed / redesigned:**
- The quadtree split-vs-leaf decision is resolved with a read-only reconstruction
  snapshot and a `Candidate` value that's only committed once chosen, instead of
  speculatively recursing and rewinding stream indices on a change of mind. The
  original's rewind path is also where its decoder had a real bug (a `bit==1`
  branch that fell off the end of a non-`void` function without returning).
- `FractalDecompression`'s `Width(Height)` constructor-order bug (width silently
  forced equal to height) has no equivalent here — width and height are threaded
  through explicitly.
- Domain search is vectorized (integral images for domain sum/sum-of-squares,
  computed once per top-level block and reused across the whole quadtree it
  contains, plus a single cross-correlation call per node) instead of an
  explicit nested loop over every candidate position.
- Coordinate fields use a fixed bit width (`ceil(log2(padded_size))`) instead of
  the original's incrementally-growing width — slightly less compact, but the
  bitstream format is then a pure function of image size, not encode-time state.
- Color support (RGB → YCbCr, optional 4:2:0 chroma subsampling) — the original
  was grayscale-only.
- A real byte-level bitstream (`EncodedChannel.to_bytes/from_bytes`) rather than
  just an in-memory transform list, so "compression ratio" is measured on actual
  serialized bytes.

**Deliberately unchanged:** the core idea — same-size domain/range blocks,
searched only in an already-known ("causal") region, giving a single-pass,
non-iterative decoder — is kept as-is. See "Publication angle" below for why
that idea by itself isn't new.

## Layout

```
fractal_compression/
  bitstream.py      LSB-first bit packing (BitWriter/BitReader)
  codec.py          FractalConfig, Transform, EncodedChannel (+ serialization)
  encoder.py        quadtree encoder, vectorized domain search
  decoder.py        single-pass causal decoder
  color.py          RGB <-> YCbCr, 4:2:0 chroma subsampling
  image_codec.py    whole-image (color-aware) encode/decode
  entropy.py        per-stream canonical Huffman coding on top of an
                    already-encoded channel (K/C indices, coordinates,
                    partition bits), with fixed-width fallback per stream
  baseline_jpeg.py  JPEG rate-distortion baseline via Pillow
  metrics.py        PSNR / SSIM
benchmarks/
  run_benchmark.py  rate-distortion, ablation, and color experiments
tests/
  test_roundtrip.py pytest suite
results/            CSVs + plots produced by run_benchmark.py
```

## Usage

```python
import numpy as np
from skimage import data
from fractal_compression import FractalConfig, encode_channel, decode_channel

img = data.camera().astype(np.float64)[:128, :128]
cfg = FractalConfig(error_thresh=100, max_block=64)
enc = encode_channel(img, cfg)
recon = decode_channel(enc)

print(enc.bits_per_pixel(), enc.compression_ratio())
```

Color images: use `encode_image` / `decode_image` from the same package instead.

Run the benchmark suite: `python3 benchmarks/run_benchmark.py`
Run tests: `python3 -m pytest tests/ -v`

## GUI

An interactive Streamlit app to upload (or pick a sample) image, tune
`error_thresh`, block sizes, K/C quantization, chroma subsampling, and
entropy coding from sliders, and see the reconstruction, its quadtree
partition, and its real bpp/PSNR/SSIM numbers immediately:

```
streamlit run gui/app.py
```

It's pure UI wiring around the same public API described above — no codec
logic lives in `gui/`.

## Benchmark results (this run)

All on 128x128 grayscale crops (`camera`, `coins` from `skimage.data`),
`max_block=64`, full domain search (`step=1`) unless noted.

### Rate-distortion vs. JPEG (camera_128), with and without entropy coding

| bpp (fixed-width) | bpp (+ entropy coding) | Fractal PSNR | JPEG PSNR (nearest bpp) |
|---|---|---|---|
| ~0.14 | ~0.14 | 40.3 dB | — |
| ~0.30 | ~0.27 | 40.6–40.9 dB | 44.3 dB (q=30, bpp 0.273) |
| ~0.76 | ~0.66 | 40.9 dB | — |

**Entropy coding was added** (`fractal_compression/entropy.py`): a canonical
Huffman code, independently applied per-stream (K indices, C indices, domain
row/col coordinates, partition bits grouped into 4-bit tuples) with a 1-bit
per-stream fallback to the original fixed-width packing whenever the sparse
Huffman table itself would cost more than it saves — this was necessary
because on a single 128x128 image, several streams (domain_row, domain_col,
k_idx, the partition-bit nibbles) frequently use a large enough fraction of
their alphabet that a naive "always Huffman-code" design was a *net loss* at
low leaf counts; only `c_idx` (highest skew: ~57% of its entropy is
redundant) reliably paid for its own table. With the fallback, entropy coding
is a consistent, measured **~8–13% bitrate reduction at matched PSNR, with
zero regressions**, across every `ERROR_THRESH` tested on both `camera_128`
and `coins_128` (see `results/rate_distortion.csv`, `codec=fractal_entropy`).

**Honest finding: entropy coding does not close the JPEG gap.** At matched
bitrate, plain JPEG still beats this codec by roughly 4–6 dB PSNR on natural
photographic content — visually, the `fractal_entropy` curve in
`results/rate_distortion.png` is barely distinguishable from `fractal`, just
shifted slightly left. The fractal codec's PSNR curve is *still* unusually
flat (40.9 → 40.3 dB from 0.76 to 0.14 bpp even after entropy coding), which
confirms the original hypothesis was only half right: entropy coding *does*
reclaim real, measured bits (the redundancy was there, mostly in `c_idx`),
but the flatness itself is not primarily a missing-entropy-coding problem —
it's the fixed 5-bit/7-bit K/C quantization grid itself that caps achievable
quality regardless of how efficiently those quantized values are packed.
Closing the JPEG gap further would need coarser bit allocation at low
bitrate (variable K/C precision, or rate-distortion-optimal quadtree pruning
per Priority 2) rather than better packing of what's already there.

### ERROR_THRESH ablation (camera_128, step=1)

| ERROR_THRESH | leaves | bpp | PSNR |
|---|---|---|---|
| 30 | 486 | 0.819 | 40.91 |
| 100 | 174 | 0.298 | 40.70 |
| 400 | 87 | 0.153 | 40.51 |
| 1200 | 66 | 0.118 | 40.34 |

Leaf count (and thus bitrate) drops ~7x from ERROR_THRESH=30 to 1200 for a PSNR
loss of only ~0.6 dB — most of the quadtree's fine splitting is buying very
little quality on this image, another quantified, paper-worthy observation.

### Color: chroma subsampling (astronaut, 96x96, ERROR_THRESH=100)

| 4:2:0 subsample | bpp | PSNR | SSIM |
|---|---|---|---|
| off | 5.43 | 35.6 dB | 0.948 |
| on | 2.95 | 34.6 dB | 0.942 |

~46% bitrate reduction for ~1 dB PSNR loss — chroma subsampling is a clear win
here, consistent with why JPEG does the same thing by default.

Full CSVs and plots are in `results/`.

## Publication angle — read this before writing a paper

The core mechanism here (causal domain search → non-iterative decode) is
**not novel**; it's a documented technique from the early-to-mid 1990s
(Lepsøy/Øien/Ramstad's fast non-iterative decoder, and later causal-domain-pool
variants). A paper that presents this architecture alone would likely be asked
to cite that prior art and would not clear review at a serious venue.

What *is* realistic: classical fractal coding is a small, mostly dormant niche
now (mainstream compression research has moved to learned/neural codecs), but
smaller IEEE conferences and journals still publish incremental, well-measured
improvements to it. Given the benchmark results above, a legitimate paper-sized
contribution from this codebase would be one of:

1. ~~Close the JPEG gap by adding entropy coding.~~ **Done, and measured not
   to be sufficient on its own** — see the rate-distortion table above.
   Per-stream canonical Huffman coding (`fractal_compression/entropy.py`)
   gives a real, reproducible ~8–13% bitrate reduction at matched PSNR with
   zero regressions, but the JPEG gap and the flat PSNR curve both persist
   essentially unchanged. That's the more interesting finding: it isolates
   the flatness to the K/C quantization grid itself, not to missing entropy
   coding, which sharpens the remaining open question into one a paper could
   credibly answer — variable-precision or rate-distortion-optimal K/C
   quantization (see #2) is now the more promising lever, not better packing
   of what's already there.
2. **Adaptive ERROR_THRESH / block size selection.** The ablation shows most
   fine-grained splitting buys little quality; a content-adaptive threshold
   (or rate-distortion-optimal quadtree pruning, à la RDO in video codecs)
   is a measurable, defensible contribution.
3. **A proper comparison study.** Systematically benchmark this causal/
   non-iterative variant against classical iterative fractal coding (accuracy,
   speed, memory) across a standard test set (Kodak, USC-SIPI) — even without
   a new technique, a rigorous, reproducible comparison with confidence
   intervals is publishable at the venues described above.

Any of these, written up with the CSVs/plots this repo already produces as your
Figures/Tables, is a realistic target for a small IEEE conference or a
regional/student journal — not a top-tier venue without a bigger new idea.

## Known limitations

- Encode is slow (~2.5–3s for 128x128 at step=1) — the recursive quadtree
  visits every node down to `min_block` regardless of whether it ends up
  splitting, and profiling shows scipy's correlation call, not the integral
  images, now dominates. A C extension or early-exit pruning would be the
  next optimization pass; not done here since the priority was a correct,
  benchmarkable reference implementation.
- Entropy coding (`entropy.py`) operates on a single already-encoded channel
  in isolation — each stream's Huffman table is built and stored per image,
  with no cross-image or cross-channel dictionary reuse. It's also
  grayscale-channel-level only; `EncodedImage` (color) doesn't yet expose an
  entropy-coded total-bits helper, though nothing about the per-channel
  design prevents adding one.
- Square, power-of-two-padded canvas internally; non-power-of-two/non-square
  inputs are supported (padded and cropped transparently) but padding uses the
  channel mean, which very slightly biases blocks that straddle the real/padded
  boundary.

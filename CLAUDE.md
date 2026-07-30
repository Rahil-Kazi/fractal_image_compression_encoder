# Fractal Image Compression — project memory

This file exists so a fresh Claude Code session in this repo can pick up
exactly where the last one left off, without re-deriving math or re-trying
approaches that were already tested and rejected. Read this before starting
new work here. It's a supplement to `README.md`, not a replacement — the
README has the architecture, layout, and benchmark tables; this file has the
*history and reasoning* behind the current state.

## Environment

- Python venv at `/tmp/fcvenv` (not committed, recreate with
  `python3 -m venv /tmp/fcvenv && source /tmp/fcvenv/bin/activate && pip install -r requirements.txt`
  if it's gone).
- Tests: `python3 -m pytest tests/ -v` — 15/15 passing as of the last commit.
- Benchmarks: `python3 benchmarks/run_benchmark.py` — writes CSVs/PNGs to `results/`.
- GUI: `streamlit run gui/app.py` (default port 8501).
- Git remote: `origin` → `https://github.com/Rahil-Kazi/fractal_image_compression_encoder.git`,
  branch `main`. Push access already set up; just `git push origin main`.

## What this codebase is

A from-scratch Python port of a causal, quadtree-partitioned, **non-iterative**
fractal image codec (domain blocks only ever reference the already-
reconstructed buffer, never the original — that's what makes single-pass
decode possible; see README's "Deliberately unchanged" section). Originally
ported from a C++/OpenCV prototype (`fractal_c++/` next to this directory)
for benchmarking and possible publication. The `README.md`'s "Publication
angle" section has the honest framing: the core causal/non-iterative
mechanism is 1990s prior art, not novel; the actual contribution has to be
measured improvements on top of it.

## Chronological build log (what's been done, and why)

### 1. Entropy coding (`fractal_compression/entropy.py`)
Fixed-width K/C/coordinate/partition-bit packing had a flat, suspicious
PSNR-vs-bpp curve. Added per-stream canonical Huffman coding with a critical
detail: **each of the 5 streams independently falls back to fixed-width
when its own sparse Huffman table would cost more than it saves** — an
unconditional "always Huffman-code" design was measured to be a *net loss*
on low-leaf-count images (table overhead > savings). Result: consistent
~8–13% bitrate reduction at matched PSNR, zero regressions — but it does
**not** close the JPEG gap; the flat curve turned out to be a quantization
problem, not a packing-efficiency problem (see #3 below).

### 2. Streamlit GUI (`gui/app.py`)
Upload/sample image → tune `FractalConfig` fields via sliders → see
original/reconstructed/quadtree-overlay + real metrics. Pure UI wiring, no
codec logic lives here. Known gotcha already fixed: `PIL.Image.open()` on a
Streamlit `UploadedFile` needs `io.BytesIO(uploaded_file.getvalue())`, not
the raw stream (read-position issue → spurious `UnidentifiedImageError`).
Also registers `pillow-heif` so iPhone HEIC uploads work (very common
failure mode for "large" ~3000×4000 uploads that look like they should be
plain JPEG/PNG but aren't).

### 3. Quantization-aware domain search (`FractalConfig(quantization_aware=True)`) — the big one
**The finding:** `_search_domain` picked the domain position by argmin over
*continuous*-space error, then quantized K/C afterward — so the chosen
position wasn't always the one that reconstructs best post-quantization.

**The math (load-bearing, don't re-derive from scratch):** this codec's
transform is `K·(D − mean(D)) + mean(D) + C` (mean-centered), NOT the naive
`K·D + C`. Because the domain window is mean-centered before K is applied,
**K and C are mathematically decoupled** — the optimal C is always
`avg_range − avg_domain`, independent of K, and vice versa (verified both
algebraically and numerically against brute-force search). This means:
- No joint/conditional K-C search is needed — independent nearest-grid
  rounding of each is already the joint discrete optimum.
- But evaluating error at those quantized values for position-selection
  purposes requires mapping C to the *raw* intercept first:
  `C_eff = avg_domain·(1 − K_q) + C_q`. Using `C_q` directly in the general
  expanded SSE formula is **silently wrong by orders of magnitude** — this
  was caught by direct numerical verification, not inspection. Don't skip
  that verification step if this code is ever touched again.

**Result:** implemented in `_search_domain` (encoder.py) behind
`config.quantization_aware`, off by default. Measured: PSNR higher at
**every** setting tested, up to **+12 dB on coins_128**, +2.9–4.2 dB on a
color case, **+13.8 dB observed live in the GUI** at 256×256. Cost: ~40–50%
slower encode (more elementwise array ops), decode/bitstream format
unaffected. Narrows but doesn't close the JPEG gap (~1 dB of ~3.7 dB closed,
for free in bits).

### 4. Rejected speed optimizations for #3's overhead — don't retry these without new evidence
A long back-and-forth proposed increasingly specific fixes for the ~40–50%
quantization-aware slowdown. **All were tested empirically before being
accepted or rejected — this repo's working norm is "verify, don't assume,"
and it paid off every time here:**

**Important node-profile correction, read this first:** an early measurement
(used to reject the scratch-buffer idea below) sampled only the *first few*
node shapes from an encode and found them tiny (1×1–3×3), attributed to
`min_block=2` forcing deep recursion. That sample was unrepresentative.
Once early-termination pruning (#5) existed and was combined with
`quantization_aware=True`, the *actual* profile for a real encode
(`camera_128`, `error_thresh=100`) is **only 231 total node visits**
(down from ~5451 measured without pruning/quantization-aware), with a
**median valid-search-grid size of ~3249 elements** (roughly 57×57) — medium,
not tiny. The two features compound: better-quantized matches have lower
error, so `leaf.error <= error_thresh` fires far more often, pruning far
more aggressively. **All the rejections below were re-verified against this
corrected, realistic profile** — the conclusions held, but for different
reasons than originally stated in a couple of cases (noted inline).

- **ML feature embeddings + FAISS/ANN nearest-neighbor search** to replace
  the exhaustive correlation search: rejected without implementation. The
  domain pool here is the *causal, growing* reconstructed region, not a
  static pool — an ANN index would need constant rebuilding as the region
  grows, likely costing more than the search it replaces. Also breaks the
  affine-invariance the search needs (K/C already handle contrast/brightness
  invariance; a generic embedding isn't guaranteed to preserve that).
- **Spatial-locality windowing** (restrict the domain search to a local
  neighborhood around the range block instead of the full causal region,
  on the theory that self-similar matches are usually nearby): **directly
  contradicted by measured match-distance data.** On both test images, the
  median distance between a range block and its chosen domain match is
  ~62px on a 128px canvas, and ~47–49% of matches sit beyond 64px (the far
  side of the image). This is consistent with an earlier, independent
  finding from the entropy-coding work (#1): delta-encoding coordinates
  relative to the range block's position measured *worse* (higher entropy)
  than absolute coordinates, for the same underlying reason — this causal
  codec's matches are not spatially clustered near their source blocks.
  Windowing would blind the search to roughly half its real matches; not a
  free speedup, a real quality cost that was never even worth ablating
  given how strongly the data pointed against it.
- **Depth-dependent strided search via `scipy.signal.correlate`** (full
  `step=1` for large blocks, coarser stride for small ones): the *quality*
  side looked safe (existing global `step=1` vs `step=2` ablation in
  `results/ablation_error_thresh.csv` shows near-zero PSNR difference), but
  the *speed* side doesn't deliver what it promises with the current
  architecture. `config.step` already exists, but it slices `correlate()`'s
  **output** *after* the full-resolution FFT/correlate call already ran —
  measured: `step=1` → `step=4` only cut encode time ~15% (0.041s → 0.035s
  on `camera_128`), because the dominant cost (`correlate` itself) is
  untouched. A proposed fix — decimate the **input** (`recon_known`) before
  correlating — was caught as an actual correctness bug, not just
  underwhelming: it silently changes same-size domain-block search into a
  classical *larger-domain-downsampled-to-range-size* search (verified by
  computing both approaches on identical input and showing the outputs
  differ, then tracing why) — a fundamental violation of this codec's
  documented "same-size domain/range blocks" design, not a performance
  tradeoff. A *correct* strided search would need a bespoke sliding-window
  implementation (e.g. `numpy.lib.stride_tricks.sliding_window_view` on a
  decimated candidate-position grid) since `scipy.signal.correlate` has no
  native way to compute only a strided subset of output positions cheaper
  than computing all of them. Not implemented; would need its own
  from-scratch correctness verification against a brute-force reference
  before trusting it, same as everything else in this list.
- **GPU offload via PyTorch (`F.conv2d`) for the correlation step:**
  tested twice, rejected both times, for different reasons.
  - First pass (stale profile, ~5000+ tiny nodes): rejected on the
    reasoning that per-node kernel-launch/host-device-transfer overhead
    would dominate on mostly-tiny arrays. Proposed code also had real bugs
    (squaring a ones-kernel instead of the input array for `sum_sq_D`; a
    `groups=` conv2d batching call with input on the wrong axis — batch vs.
    channel — that would error at runtime for any batch size > 1).
  - Second pass, after the node-profile correction above (231 nodes,
    medium arrays — a much more GPU-favorable shape on paper): **tested
    directly on real hardware (Apple Silicon, MPS backend — no CUDA
    available) against all 231 real node shapes from an actual encode,
    full round-trip cost included (tensor creation, host→device transfer,
    `F.conv2d`, device→host).** Result: **MPS was 8x slower than
    `scipy.signal.correlate`** even after 50 warmup calls and taking the
    best of 3 timed passes (0.58ms/node vs. 0.07ms/node). Root cause is the
    same category as the scratch-buffer finding below — fixed per-call
    dispatch overhead (MPS command-buffer submission, in this case)
    dominating over compute at this array size — just manifesting in the
    GPU driver layer instead of NumPy. **Per-node GPU dispatch is
    conclusively off the table on this hardware at this problem size.**
    The only way GPU could still plausibly help is genuine batching (many
    nodes in one call), which requires restructuring the recursion into a
    level-order/BFS traversal first — not attempted, real engineering
    effort, not a quick swap.
- **Pre-allocated NumPy scratch buffers (`out=` parameter) to avoid
  reallocating arrays per node:** tested twice, rejected both times.
  - First pass (stale profile): 12% slower than the naive vectorized
    version.
  - **Re-tested against the corrected, realistic 231-node/medium-array
    profile specifically because the original rejection's node-shape data
    was stale — still a net loss, 31% slower.** So this isn't a "small
    array" artifact as originally framed; NumPy's per-call dispatch
    overhead dominates allocation cost even at ~3249-element arrays,
    because the buffered version needs ~15 granular `out=` calls vs. one
    compact vectorized expression. **Don't retry `out=` buffer reuse in
    `_search_domain` again without a fundamentally different array-size
    regime than what's been tested** (both the ~1–9 element and
    ~3249-element regimes have now been checked and rejected).

### 5. Early-termination quadtree pruning (`_encode_block` in `encoder.py`)
**Proof (load-bearing):** the split rule is
`leaf.error > split_error + error_thresh`, and `split_error` (sum of squared
errors) is always ≥ 0. So `split_error + error_thresh ≥ error_thresh`
unconditionally, meaning `leaf.error ≤ error_thresh` *guarantees* the split
condition can't fire — the four children can be skipped entirely with zero
behavior change. One-directional only: `leaf.error > error_thresh` does
NOT imply a split, that branch still has to recurse to find out.

**Verification pitfall worth remembering:** a hand-transcribed "unpruned
reference" implementation for the test gave *inconsistent* results (788 vs
792 bits on a test case) despite looking byte-identical to the real code on
manual inspection, AST diffing, and line-by-line comparison. The actual bug
was never found in the transcription — instead, the test was rebuilt to
derive the reference via `inspect.getsource()` off the *live* `_encode_block`
source (stripping only the pruning check via text surgery), which can't
drift out of sync with the real implementation the way a hand copy can. If
you need a "reference/unpruned" version of any recursive function in this
codebase for testing again, use this `inspect.getsource`-based pattern from
`tests/test_early_termination.py`, not a hand-typed duplicate.

**Result:** implemented unconditionally (no config flag needed — it's
provably lossless, not a quality/speed tradeoff). Measured impact is real
but modest: only **~0–20% fewer node visits, ~1–8% faster encode** on
`camera_128`/`coins_128` at default settings — NOT a dramatic win, because
`min_block=2` forces deep recursion regardless of pruning. Doesn't claw back
#3's overhead. Kept anyway since it's free and strictly safe.

### 6. A separate theory document (not in this repo)
A standalone "Fractal Image Compression — Theory & Flow" explainer was
published as a Claude Artifact (not a repo file) for an MS EE student
audience — covers self-similarity, range/domain blocks, the affine
transform, the quadtree, causal-vs-iterative decoding (with the Banach
fixed-point intuition), the encode/decode pipeline as Mermaid diagrams,
evaluation metrics, and a glossary. No code discussed. If asked to update or
reference it, it is NOT a file in this repo — it would need to be
re-located via the Artifact tool's list action, or re-created if truly lost.

## Current verified state (as of the last commit on `main`)

- 15/15 tests passing (`test_roundtrip.py`, `test_entropy.py`,
  `test_quantization_aware_search.py`, `test_early_termination.py`).
- `results/` has 4 benchmark outputs: `rate_distortion.{csv,png}` (fractal
  vs. fractal+entropy vs. JPEG), `ablation_error_thresh.{csv,png}`,
  `color_chroma.csv`, `quantization_aware.{csv,png}` (continuous- vs.
  quantized-error search, grayscale + one color case).
- `FractalConfig` now has two opt-in flags beyond the original design:
  `quantization_aware` (default `False`) and the always-on early-termination
  pruning (no flag — it's lossless).

## What's still open (from the original task brief, not yet done)

The session that started this work was handed a 5-priority task list. Only
entropy coding (Priority 1) and a chunk of the search-quality work (which
turned into quantization-aware search + early termination, not originally
scoped but a bigger win than Priority 1 alone) are done. Still open:

- **RDO-style adaptive `ERROR_THRESH`/block-size selection** (original
  Priority 2) — a Lagrangian `error + λ·bits` split criterion instead of the
  fixed `error_thresh`. The ablation study already shows most fine-grained
  splitting buys little quality; this is the natural next lever, and ties
  directly into why early-termination's impact was modest (a smarter split
  criterion might prune much more aggressively than the current threshold
  does).
- **A real performance pass** beyond what's been tried — profiling still
  points at `scipy.signal.correlate` as the dominant per-node cost, not
  Python overhead. GPU offload might still be worth it if batched by tree
  *depth* (all same-size sibling nodes across the image at once) rather
  than per-node, but that requires restructuring the recursion into a
  level-order traversal first — not attempted.
- **A classical iterative fractal-coding baseline** (original Priority 4) —
  for a real causal-vs-classical comparison (speed, quality, memory), not
  just implied by the design notes.
- **Paper/write-up scaffolding** (original Priority 5) — a `paper/`
  directory with related-work citations, not started.
- Larger/more test images (256×256, 512×512, or a real test set like Kodak)
  — everything so far is 96–128px crops for speed of iteration.

## Working norms established this session (keep following these)

- **Verify before implementing, especially performance claims.** Every
  optimization proposal in this session was tested empirically before being
  accepted; the scratch-buffer idea looked reasonable and was measurably
  wrong. Don't skip this step for speed-related suggestions.
- **Report honest, sometimes negative/modest results** — this repo's whole
  framing (see README) is built on not overselling findings. Early
  termination's "0–20% node reduction" was reported as-is, not spun.
- **Keep new features opt-in via `FractalConfig` fields when they change
  output** (`quantization_aware`), but implement unconditionally when a
  change is provably lossless (early termination) — don't add a toggle for
  something that's always safe.
- Every new benchmark writes a CSV + plot into `results/`, matching
  existing convention.

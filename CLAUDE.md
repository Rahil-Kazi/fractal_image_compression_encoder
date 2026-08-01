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
- **Variance-bucket pre-filter before the `quantization_aware` refinement**
  (prompted by three independent prior-art papers all using cheap-scalar-
  statistic domain-block pruning — see "Prior art literature review"
  below): domain-block variance is `denom/n` in `_search_domain`, already
  computed for free via the integral images before the quantization-aware
  refinement's extra elementwise ops run. Tested masking the refinement
  (`quantize_array`/`dequantize_array`/`err_quant`) down to the ~50% of
  positions whose variance is closest to the range block's, instead of
  running it over the whole valid grid. **Rejected — verified, not just
  reasoned by analogy to the prior art.** On `camera_128`
  (`quantization_aware=True`, `error_thresh=100`): unmasked baseline
  0.041s/41.54dB; masked variants (mask_fraction 0.25/0.5/0.75) all came
  out *slower* (0.72–0.79x, i.e. 25–38% slower), with negligible quality
  change (-0.01 to -0.04dB — confirms the variance signal is a real, nearly
  free-to-compute proxy for match quality, just not exploitable for speed
  here). Root cause isolated directly: masking + gathering costs ~48µs/node
  (~22µs quantile+compare, ~26µs for the boolean-index gathers on 5–6
  arrays), which is comparable to or larger than the entire per-node
  compute budget (~70µs, per the GPU-rejection profiling above) it's trying
  to shrink — same "fixed per-call dispatch overhead dominates at this
  array-size regime" pattern as the `out=`-buffer and per-node-GPU
  rejections above, just showing up as NumPy fancy-indexing overhead this
  time. **Don't retry array-masking/subsetting inside `_search_domain`
  without first checking whether the overhead problem here has actually
  changed** (e.g. a fundamentally coarser-grained masking scheme that
  gathers once instead of per-array, or an array-size regime not yet
  tested).

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

### 6. RDO quadtree split (`FractalConfig(rdo_lambda=...)`)
Replaces the fixed `error_thresh` split rule with the standard Lagrangian
`error + λ·bits` criterion (same idea as HEVC/AV1 CTU splitting): split iff
it lowers the combined score, using the *real* bit cost of leaf-vs-split
(not a flat error quantity). Opt-in via `rdo_lambda: float | None = None`
on `FractalConfig` — `None` (default) is byte-identical to the pre-existing
`error_thresh` path (tested directly).

**Math worked out and load-bearing (see the design-note comment above the
branch in `_encode_block`):** since this codec's bitstream is fixed-width,
`per_leaf_bits` (bit cost of one leaf) is a **constant**, independent of
block size or content. That gives two things:
- A candidate's total `bits` is trackable through the recursion exactly
  like `error` already was (`_Candidate.bits`, computed bottom-up:
  `1 + per_leaf_bits` for a leaf, `1 + sum(children.bits)` for a split).
- The existing early-termination proof **generalizes cleanly**: every child
  contributes ≥ `1 + per_leaf_bits` bits regardless of content, so
  `split_bits ≥ 1 + 4·(1+per_leaf_bits)` unconditionally, and combined with
  `split_error ≥ 0`, splitting is provably never chosen whenever
  `leaf.error ≤ λ·(4 + 3·per_leaf_bits)` — same shape as the original
  proof, re-derived for the new criterion. Verified exactly lossless the
  same way as #5 (`inspect.getsource`-derived reference, not a hand copy) —
  `tests/test_rdo_quadtree.py`.

**Honest finding: RDO and the fixed threshold land on nearly the same
rate-distortion curve** (`results/rdo_quadtree.{csv,png}`) — not a clear
win. This isn't a bug or a tuning issue, it's structural: because
`per_leaf_bits` is constant, the Lagrangian test algebraically reduces (for
any node whose children don't split further) to
`leaf.error − split_error > λ·(1 + 3·per_leaf_bits)` — literally the same
*form* as `error_thresh`, just reparameterized. RDO's real advantage over a
flat threshold — correctly weighing splits whose bit cost genuinely
varies — doesn't get to show up in this codec because leaf bit cost never
varies. **If this is revisited, the version worth trying is feeding
entropy coding's *actual*, non-uniform realized bit costs (`c_idx` measurably
costs fewer real bits than `k_idx`, per `entropy.py`'s findings) into
`λ·bits` instead of the fixed-width estimate** — that's where genuine
per-leaf bit-cost variation actually exists in this codebase. Not
implemented.

Calibration note: `error_thresh` and `λ` are different units (error vs.
bits), not comparable a priori — the benchmark sweep range
(`[0.2, 0.5, 1, 2, 5, 10, 20, 50]`) was picked by first running a quick
calibration pass to find λ values spanning a comparable bpp range to the
existing `error_thresh` sweep, the same "check empirically before
committing to a design" pattern used for the entropy-coding streams (#1).

### 7. A separate theory document (not in this repo)
A standalone "Fractal Image Compression — Theory & Flow" explainer was
published as a Claude Artifact (not a repo file) for an MS EE student
audience — covers self-similarity, range/domain blocks, the affine
transform, the quadtree, causal-vs-iterative decoding (with the Banach
fixed-point intuition), the encode/decode pipeline as Mermaid diagrams,
evaluation metrics, and a glossary. No code discussed. If asked to update or
reference it, it is NOT a file in this repo — it would need to be
re-located via the Artifact tool's list action, or re-created if truly lost.

## Prior art literature review (IEEE papers in `fractal_compression/prior_art_literature/`)

Reviewing 8 recent IEEE conference papers one at a time for related-work
citations and any techniques worth stealing/testing. Source PDFs have
generic filenames (`paper1.pdf`–`paper8.pdf`); some (2, 4, 5) have garbled
`pdftotext` output from embedded font subsetting — render with `pdftoppm`
and read the pages as images instead of trusting text extraction.

### paper2 — Sakai et al., "Performance Evaluation of Fractal Image
Compression on Bit-Depth Control and GPU Parallelization" (ICCE-Taiwan 2025)

Architecturally different from our codec: fixed-grid RB(4×4)/DB(8×8) blocks
(no adaptive quadtree), and a discrete affine-transform-index + separately
encoded DC component instead of our continuous quantized K/C. Their domain
pool is pre-clustered via K-means on quantized-DCT AC components before RB
processing — implies a static, non-causal domain pool (built from the
original image), unlike our causal reconstructed-buffer-only design.

**GPU finding (Table II), relevant to our open "real performance pass"
item:** CPU 58,478 ms → their clustered method on GPU ~11–15 ms (~4000–5000x).
Doesn't contradict our own MPS result (8x *slower* than
`scipy.signal.correlate` per-node) — different problem shape. Their speedup
comes from (1) fixed same-size blocks, so every RB's search is embarrassingly
parallel with no recursive/adaptive dependency, and (2) K-means
pre-clustering shrinking each RB's candidate set before any GPU work. This
is an existence proof that **batching across same-size siblings** (the
untried "GPU batched by tree depth" idea already in the open-items list
above) is the right shape to try, not per-node dispatch. Corroborating,
not new: their own table also shows 16→32 threads getting *slower* for both
GPU variants (799→2499 ms, and 1.1→15 ms) — same oversubscription/dispatch
pattern behind our `out=`-buffer and per-node-MPS rejections.

**Quality finding worth reusing in our own benchmarking narrative:** their
FIC loses to JPEG(q95) at 8-bit depth but **wins at 4-bit depth** — JPEG
loses high-frequency content entirely under heavy quantization while FIC
can still use all AC components with fewer distinct block/transform types
needed. Suggests our "narrows but doesn't close the JPEG gap" story might
be better framed around **low-bit-depth/heavily-quantized regimes**
specifically, rather than full 8-bit parity — not yet tested on our codec.

**Caveats:** 2-page short/workshop paper — scatter plots only, no numeric
PSNR/CE tables, no error bars, single 256×256 test-image class implied, and
the "CT" metric is defined in text as "number of comparisons" but reported
in the table as milliseconds (inconsistent). Treat exact numbers as
indicative, not rigorous.

### paper3 — Xie et al., "Optimization Method For Fractal Image Compression
Based on the Maximum Inter-class Variance Method" (IUCC 2024)

Classical global-search FIC (Barnsley/Jacquin style — RB 4×4 non-overlapping,
DB 8×8, 8 isometries, domain block down-sampled to range size). Their
speedup (FCBO): build a "reconstructed image" from each domain block's mean
gray level, run **Otsu's method** on that reduced image to get one threshold
`Td`, bucket every domain block (and every range block) into category T/B
by whether its mean ≥ `Td`, and restrict search to same-category blocks
only. Result: ~42.4% encode-time reduction, <0.5 dB PSNR drop, tested on
VOC (natural images) and BreakHis (breast tissue microscopy), 1000 images
each.

**Relevance:** this is the same family of idea as paper2's K-means/DCT
clustering — pre-filter the domain pool by a cheap scalar statistic before
the expensive match search — now corroborated by a second, independent
technique/paper (~40%+ time savings, <1 dB cost is a repeated pattern
across both). Doesn't change our existing conclusion on the ANN/FAISS
rejection above, though: **both papers assume a static, precomputed domain
pool**, classified once before any range-block processing starts — exactly
the assumption our causal, growing-domain-pool design breaks. Even
architecturally, our search is one vectorized `scipy.signal.correlate` call
over the whole valid grid, not an iterate-and-filter loop — restricting it
to a same-category subset means masking/slicing before correlating, the
same shape of idea (and same per-call-dispatch-overhead problem) as the
already-rejected strided/scratch-buffer experiments in item #4 above. Not
a free win for us the way it is for their per-block-loop implementation.

**Independent, more useful takeaway:** their baseline (unoptimized) FIC
got ~13 dB higher PSNR on microscopy images than on generic VOC natural
images (40.4 dB vs 27.6 dB) — cheap, strong evidence that self-similarity-
heavy image classes are FIC's actual sweet spot. Worth considering a
texture/microscopy-like test image in our own benchmark suite (currently
only 96–128px camera/coins crops) if we want a "here's where this approach
genuinely wins" result rather than only chasing the JPEG-gap number on
generic photos.

### paper5 — Panigrahy et al., "Space Limited Quadtree based Fractal Image
Encoder" (AESPC 2024)

Classical Jacquin/PIFS FIC (non-overlapping range B×B, overlapping domain
2B×2B contracted by 2×2 averaging), "no-search" lineage (Furao & Hasegawa).
Their contribution: restrict domain candidates to a fixed **3B×3B
neighborhood centered on the range block** (≤5 fixed positions — center +
4 neighbors at B/2 offset, fewer near edges), falling back to quadtree
subdivision (retried in the same restricted neighborhood) if none meet the
error threshold. Eq. 3–5's mean-centered decoupled-scale/offset derivation
is the same math as our `quantization_aware` work — independent
confirmation this part really is standard (their ref [20] is Fisher's 1994
book), matching CLAUDE.md's "not novel" framing.

**Results (512×512 cameraman/boat):** their "QT full-search" baseline takes
11,908s / 6,230s per image (clearly an unvectorized/looped implementation).
Their proposed 5-neighbor method is 1150–1900x faster than that baseline,
but at a real, large quality cost: ~8 dB worse PSNR than full search on
cameraman, up to 14 dB worse (worst case) on boat.

**Relevance — direct, useful contrast to our own rejected "spatial-locality
windowing" idea (item #4 above):** we rejected windowing based on measured
data showing ~47–49% of our best matches sit beyond 64px on a 128px canvas
(non-local). This paper **independently corroborates that exact risk** —
restricting to immediate neighbors costs them 5–14 dB, a steep real price,
not a free lunch. Confirms rather than contradicts our earlier rejection.
Also worth noting: their "1000x+ speedup" is against an unvectorized,
loop-based full search (hours per 512×512 image) — our own
`scipy.signal.correlate`-based full search over the causal region already
runs in a fraction of a second per node, i.e. we may not need this kind of
restriction at all since their "slow baseline" doesn't resemble our actual
starting point. No technique adopted from this paper; strengthens
confidence in a prior decision instead.

### paper6 — Kulkarni et al., "Fractal Image Encoding: A Comparative Study
of Compression Techniques" (ICSCC 2024)

Largely a literature survey stitched with a small MATLAB experiment, not a
novel technique — re-implements three known variants (basic FIC, DCT-domain
"no-search" FIC, quadtree partitioning) and a conceptual GPU-parallel design
that's the same "one thread per range block, uniform fixed-size blocks"
pattern as paper2. Table 1 (256×256 Lena): Traditional Fractal 3h
encode/10s decode/CR 8:1/PSNR 28dB; DCT-based 1h/5s/10:1/32dB; Quadtree
45min/3s/12:1/30dB.

**Relevance:** PSNR numbers here (28–32 dB) are noticeably weaker than our
own codec's baseline results, and the 3-hour "traditional" encode time for
one 256×256 image is the *third* independent paper in this batch (after
paper5's 11,908s and paper2's 58,478ms un-clustered baselines) showing
naive/looped full-search FIC implementations in the literature are
extremely slow — our vectorized `correlate`-based search is already well
ahead of what a lot of this literature treats as its baseline; worth
stating plainly in any future related-work section rather than assuming we
need to catch up on raw speed. No new technique to adopt here. **Practical
value is its reference list** as a seed bibliography for the still-open
`paper/` write-up item — worth a closer look later: [7] "Fast Full-Search
Algorithm of Fractal Image Compression," [11] "Similarity based
optimization to fractal image encoding based on multithreading
parallelization," [14] "Enhanced Image Compression Using Fractal and Tree
Seed-Bio Inspired Algorithm," [10] a fixed-partition medical-image fractal
compression paper.

### paper8 — Tiwari et al., "Enhanced Image Compression Using Fractals
and Principle Component Analysis" (IHCSP 2023)

Hybrid: (1) DCT, keeping only low-frequency coefficients (5000 in their
comparisons) and discarding the rest; (2) fractal coding on the
high-frequency residual, with a **variance-based domain-block pre-filter**
(restrict search to domain blocks whose variance is close to the range
block's — stated in the paper as `|V_R − V_D'| > Th`, which reads backwards
from the obviously-intended "select blocks whose variance is *close*";
likely a paper typo, unverified against original rendering); (3) PCA/
eigenvector fusion for final reconstruction, trading PSNR vs. CR via PC
count.

**Results (256×256 Lenna/Tajmahal/Pepper, range 16×16, DCT coeff=5000,
PC=100):** DCT alone 0.55–0.73s/CR~2.1–2.3/PSNR 28.3–31.8dB; Conventional
fractal 17.6–19.0s/CR 33.0–36.5/PSNR 17.2–20.3dB; Hybrid (DCT+Fractal)
7.2–10.3s/CR 41.5–50.5/PSNR 24.4–26.7dB; **their proposed (+PCA)**
6.5–9.5s/CR 34–47.4/**PSNR 33.4–36.4dB** — best speed and PSNR of the four,
at a real CR cost vs. plain DCT+Fractal hybrid. Abstract's headline
percentages (3.66%/15.07%/44.67%) don't cleanly reconcile with these table
numbers when checked directly — treat those specific figures with
skepticism.

**Relevance:** the variance-based filter is a **third independent
instance** of the "prune domain search by a cheap scalar statistic" family
(after paper2's K-means/DCT-AC clustering and paper3's Otsu threshold) —
same static-pool objection applies, doesn't map cleanly onto our causal
growing-pool + single vectorized `correlate()` design. **The genuinely new
idea here**: DCT low-pass extraction *before* fractal coding, with fractal
applied only to the high-frequency residual, rather than treating DCT and
fractal as competing alternatives. Since JPEG is fundamentally DCT-based,
this kind of frequency-domain hybrid is a more principled way to attack our
"narrows but doesn't close the JPEG gap" framing than a head-to-head
comparison. **Update: implemented and tested end-to-end on `camera_128` —
rejected for our specific architecture, verified rather than assumed** (see
"Evaluating prior-art idea #3" below for the full sweep and root cause).
Their "Conventional fractal" baseline PSNR (17.2–20.3 dB) is the fourth
paper in this batch (after paper2, paper5, paper6) with a strikingly weak
traditional-FIC baseline vs. our own already-tuned codec — same running
observation, worth remembering for framing our own results.

### paper7 — Li et al., "Research on Fractal Image Compression Algorithm
Based on Generalized Mandelbrot Set" (CISCE 2023)

Genuinely different approach from the rest of this batch: instead of
searching the image itself for self-similar domain blocks, builds a
**fixed, precomputed, image-independent dictionary** of 8×8 blocks
generated synthetically from a "generalized Mandelbrot set"
(`F(z) = (x²−y²+p)cos(α) + i(2xy+q)cos(β)`, varying α/β 0–180° across many
escape-time renderings, quantized down to 8×8 entries). Encoding matches
each range block against this static universal dictionary (mean-difference
+ 8 isometries), never against the image's own content at all.

**Results (512×512 Peppers, single test):** Jacquin classical baseline
20.14 dB / CR 31.35 / 407s; proposed Mandelbrot-dictionary method 19.26 dB
/ CR 29.54 / **58s** — ~7x faster, ~0.9 dB PSNR cost, slightly lower CR.

**Relevance:** breaks the core premise of self-referential fractal coding
entirely — closer to vector quantization with a fractal-flavored,
precomputed codebook than to "exploit *this* image's self-similarity,"
which is the central story both classical FIC and our own codec are built
on (per README's framing). Not architecturally adoptable without
abandoning that framing. Also doesn't address a bottleneck we actually
have: their Jacquin baseline (407s for one 512×512 image) is the *fifth*
instance in this batch (after paper2/5/6/8) of a naive/slow baseline our
own vectorized `correlate()`-based search already outperforms by a wide
margin. **Thin evidence base** worth flagging plainly: one test image, one
baseline comparison, single run, no error bars — the 7x/~1dB tradeoff
shouldn't be read as a general result. Useful only as a related-work
citation showing the field's range of philosophies (adaptive/self-
referential vs. fixed/universal dictionaries), not as a technique to test.

### Evaluating prior-art idea #2: GPU/CPU batching by same-size siblings

Prior-art papers 2 and 6 both describe GPU parallelism as one-thread-per-
range-block over a fixed, non-adaptive grid — architecturally different
from our recursive quadtree, but the underlying idea (batch same-size
sibling searches instead of dispatching per-node) is exactly the "GPU
batched by depth" item already flagged as open/untried above. Tested the
CPU-side version of this first, since GPU per-node dispatch was already
verified to lose (8x slower via MPS, see item #4 above).

**Step 1 — is there actually redundant computation to eliminate?** Yes,
verified directly: instrumented a real `camera_128` encode
(`quantization_aware=True`, `error_thresh=100`) and found 231 total
`_search_domain` node visits share only **15 distinct `recon_known`
buffers** (median 13 nodes/buffer, max 73) — because `recon_known` is
passed down unchanged through an entire top-level block's recursion (only
row/col/block-size change per node, not the domain). `scipy.signal.correlate`
recomputes the domain-side FFT fresh at every node visit despite the domain
being identical across all of them (confirmed 212/231 real calls use the
`fft` method via `scipy.signal.choose_conv_method`, not `direct`).

**Step 2 — first attempt was wrong, caught by the correctness-then-speed
discipline this repo already follows.** First prototype: one shared FFT
per distinct domain buffer, padded to fit the *largest* kernel size used
against it, reused across every kernel size. Correctness-verified against
`scipy.signal.correlate`'s own FFT-path output (max abs error ~1e-8 across
kernel sizes 2×2–64×64) — but timed 0.91x (i.e. *slower*) on the real
73-node kernel-size mix from that encode. Root cause: forcing small kernels
(e.g. 2×2) into an FFT sized for the group's largest kernel (64×64) throws
away the efficiency scipy's own `auto` heuristic gets from matching FFT
size (or using `direct`) to each kernel individually.

**Step 3 — corrected version: reuse only within same-kernel-size groups,
each at its own appropriately-sized FFT.** Using the real per-kernel-size
counts from the 73-node group (`{2:24, 4:12, 8:16, 16:16, 32:4, 64:1}`):
correctness-verified again (same ~1e-8 tolerance), then timed —
**baseline 6.53ms vs. reused 3.90ms, a verified 1.67x speedup** on the
dominant per-node cost (97.7% of total encode time).

**What this does and doesn't prove:** the FFT-reuse math is verified in
isolation (a standalone `DomainFFTCache` prototype), not wired into
`_search_domain`/`encoder.py`. Realizing it for real requires restructuring
`_encode_block` from depth-first to level-order/BFS traversal — process
all nodes at one tree depth together (so same-size siblings can share one
FFT), decide splits, then descend — which is a real, scoped engineering
task, not a drop-in change. Early-termination pruning's existing
lossless-correctness proof is unaffected by this reordering (it's still
decided node-by-node; only *when* each node's search happens changes, not
the split decision itself). This is a genuinely new, verified lead — not
just a re-confirmation of something already known — and it doesn't need
GPU at all, which changes the calculus on how much GPU offload is worth
pursuing further.

### Evaluating prior-art idea #3: DCT-low-pass-then-fractal-on-residual
(paper8) — tested end-to-end, rejected for this architecture

Unlike items #1/#2, this required an actual working prototype (not just a
timing experiment) since it's a pipeline architecture question, not a
search-speed question. Built: full-image 2D DCT (`scipy.fft.dctn`), keep
only a low-frequency F×F coefficient box, inverse-transform to get a
low-frequency approximation, `residual = original - low_freq_approx`,
encode the residual with our existing fractal codec (offset by +128 first
— our encoder/decoder clip to [0,255] in several places, e.g. `encoder.py`
lines 167/292 and `decoder.py`:50, which assumes unsigned pixel-range
input and can't take a signed residual directly), decode, add the
low-frequency layer back.

**First pass (camera_128, keep=32, `error_thresh=100`, matching the rest
of this session's benchmarks):** hybrid PSNR 43.47dB at 1.17 bpp
(0.5 bpp generous flat-rate estimate for the DCT box + 0.67 bpp for the
fractal-coded residual), vs. our plain fractal-only baseline's 41.54dB at
0.19 bpp. Before concluding anything, caught a real calibration issue first:
our `error_thresh` is an *absolute* SSE cutoff tuned for full 0-255-range
images, but the residual's dynamic range is far smaller (std≈2.7 for
keep=32 vs. the original's full range) — using the same threshold value
forces the quadtree to recurse to small blocks just to clear an absolute
error bar on low-magnitude content, inflating bit cost independent of
whether the hybrid idea itself is sound.

**Corrected: swept `error_thresh` properly for the residual (0.5 to 100).**
Best point found: th=50, hybrid PSNR 44.26dB at 1.38 bpp total. For
comparison, plain JPEG at the *same* PSNR (44.27dB) costs only **0.27 bpp**
— roughly 5x fewer bits — and our own plain fractal-only codec needs only
0.19 bpp for 41.54dB, meaning the hybrid needs ~7x more bits for ~2.7dB
more PSNR than not doing the hybrid at all. The DCT-box cost estimate used
throughout (flat 8 bits/coefficient, no entropy coding) is itself generous
to the hybrid — a properly entropy-coded DCT layer would only widen this
gap, not close it.

**Root cause, not just a bad number:** the residual does have some real,
exploitable structure — quantization-aware search's PSNR lift was actually
*larger* on the residual than on the original (+1.62dB vs. +0.87dB), so
this isn't "the residual is pure noise." It fails because of how this
codec spends bits specifically: **the bitstream is fixed-width per leaf**
(same point already established in the RDO section above — leaf cost is
constant regardless of content), so encoding fine low-magnitude residual
detail costs the same per-leaf bits as encoding a full-contrast range
block would, even though the residual leaf carries far less information.
That structural mismatch, not a tuning mistake, is why this doesn't
translate into a win for our specific architecture — a codec with
content-adaptive/entropy-proportional leaf cost might see a different
result, but that isn't what we have.

## Current verified state (as of the last commit on `main`)

- 21/21 tests passing (`test_roundtrip.py`, `test_entropy.py`,
  `test_quantization_aware_search.py`, `test_early_termination.py`,
  `test_rdo_quadtree.py`).
- `results/` has 5 benchmark outputs: `rate_distortion.{csv,png}` (fractal
  vs. fractal+entropy vs. JPEG), `ablation_error_thresh.{csv,png}`,
  `color_chroma.csv`, `quantization_aware.{csv,png}` (continuous- vs.
  quantized-error search, grayscale + one color case), `rdo_quadtree.{csv,png}`
  (Lagrangian vs. fixed-threshold split).
- `FractalConfig` now has three opt-in-or-safe additions beyond the original
  design: `quantization_aware` (default `False`), `rdo_lambda` (default
  `None`, falls back to `error_thresh`), and the always-on early-termination
  pruning (no flag needed for the legacy path — it's lossless; RDO mode has
  its own analogous always-on pruning too).

## What's still open (from the original task brief, not yet done)

The session that started this work was handed a 5-priority task list.
Entropy coding (Priority 1), the search-quality work (quantization-aware
search + early termination, not originally scoped but a bigger win than
Priority 1 alone), and Priority 2 (RDO quadtree, though it landed as a
mostly-negative/structural finding rather than a win) are done. Still open:

- ~~A real performance pass~~ **Done for the CPU-side domain-FFT-reuse angle
  — implemented, verified, and committed (`3e67a7a`).** Profiling had
  pointed at `scipy.signal.correlate` as the dominant per-node cost (97.7%
  of encode time on `camera_128`), and within one top-level block's
  recursion `recon_known` (the domain buffer) turned out to be shared
  across many nodes (real profile: 231 node visits, only 15 distinct domain
  buffers, largest shared by 73 nodes) — `correlate`'s FFT path was
  redundantly recomputing the domain-side FFT at every visit despite the
  domain being unchanged. `_DomainFFTCache` (`encoder.py`) memoizes it per
  (domain, kernel-size) pair instead. **Turned out not to need the BFS/
  level-order recursion restructuring originally thought necessary** — a
  cache keyed by kernel size works fine with the existing depth-first
  recursion unchanged, since reuse only depends on having seen that
  (domain, size) pair before, not on visit order. Measured end-to-end:
  8–23% faster encode (camera_128 ~19%, coins_128 ~17%, camera_256 ~8% —
  the smaller gain at larger size is a real, not-yet-explained pattern
  worth another look if this gets revisited), 21/21 tests still pass.
  **Important caveat, not swept under the rug:** unlike early-termination
  and RDO pruning, this is *not* provably bit-exact — in a small fraction
  of blocks (0–0.44% across three test images), a near-exact tie between
  two candidate domain positions gets broken differently due to
  floating-point rounding differences between the cached-FFT path and
  scipy's own internal one. Effect measured as negligible in every case
  found (identical bit cost always; PSNR differences from identical-to-14-
  decimals up to 0.0002 dB worst case) but it is a real, if narrow, change
  in this codec's correctness guarantee for the default code path. GPU
  offload remains a separate, likely lower-priority question now that a
  verified CPU-only win exists on the dominant cost.
- **A classical iterative fractal-coding baseline** (original Priority 4) —
  for a real causal-vs-classical comparison (speed, quality, memory), not
  just implied by the design notes.
- **Paper/write-up scaffolding** (original Priority 5) — a `paper/`
  directory with related-work citations, not started. A draft abstract
  exists (produced in conversation, not yet saved to a file) — explicitly
  deferred, not saved as of this note.
- ~~Larger/more test images~~ **Done — the real 24-image Kodak test set
  (native 512×768/768×512, standard PCD source
  `https://r0k.us/graphics/kodak/`) is downloaded into `data/kodak/`
  (gitignored — kept local-only, matching the same local-only decision made
  for `prior_art_literature/`), and `benchmarks/run_kodak_benchmark.py`
  (new, standalone script, deliberately not wired into `run_benchmark.py`'s
  `__main__` — see runtime discussion below) ran a single fixed config
  (`quantization_aware=True, error_thresh=100`) across all 24 images.
  Results in `results/kodak_benchmark.csv`/`.png`. **Headline numbers:
  mean PSNR 46.27dB, mean entropy-coded bpp 4.42**, across real native-
  resolution photos rather than 96–256px crops.

  **Important finding #1 — scaling:** encode time scales much worse than
  linearly with image size. One Kodak image (padded to a 1024×1024 canvas
  internally) took 366.7s/400.2s (quantization_aware=False/True
  respectively) for a *single* `error_thresh` setting, vs. ~4s for
  `camera_256`'s 256×256 canvas (16x fewer pixels, nowhere near 16x
  faster). This is why the existing full parameter sweep (7 `error_thresh`
  values × 2 `quantization_aware` settings) wasn't run across all 24 images
  at native resolution — estimated ~35 hours, not attempted; the single-
  config run above took ~2.7–5.4 hours instead (see finding #2 for why that
  range is wide).

  **Important finding #2 — a real, but non-reproducible, timing anomaly,
  investigated and resolved:** in the first full run, 22/24 images encoded
  in the normal 130–510s range, but kodim19 took 11,263.7s (~3.1 hours) and
  kodim20 took 1,930.9s (~32min) — 68% of the entire run's wall time, with
  no content-based explanation (kodim19's transform count, 59,430, was
  unremarkable — kodim13 has *more* transforms, 91,350, yet only took
  511s). Re-ran kodim19 alone afterward to check: it reproduced in 415.64s
  — completely normal, with the exact same transform count (59,430) and
  PSNR (46.93dB) as the original anomalous run, confirming the *encode
  result itself* is deterministic and correct; the extra ~10,850s in the
  original run was a one-off environmental slowdown during the unattended
  multi-hour background job (competing process, thermal throttling, OS
  scheduling, sleep/wake — not narrowed down further, and not a codec bug).
  kodim20 was not independently re-verified but is presumed the same
  category of transient hiccup, given the same run conditions — an
  assumption, not confirmed. **If this benchmark is ever re-run and an
  image takes wildly longer than similar-transform-count peers again,
  suspect the environment before the algorithm** — this exact pattern has
  now been checked once and cleared.

  **Process lesson worth keeping:** the original run was launched without
  `python3 -u`, so `print()` output only flushed at the very end rather
  than incrementally — checking progress mid-run had to be estimated from
  CPU time and the known per-image rate rather than read directly from the
  log (this is also part of why the anomaly wasn't caught live). Use `-u`
  (or `PYTHONUNBUFFERED=1`) for any future long-running background script
  if live progress-checking matters.
- 2 of the 8 IEEE prior-art papers (paper1, paper4 — both on FIC-based
  image encryption) were explicitly deprioritized and will not be
  reviewed — not relevant to this project's compression-quality/
  performance focus, per direct decision, not an oversight.

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

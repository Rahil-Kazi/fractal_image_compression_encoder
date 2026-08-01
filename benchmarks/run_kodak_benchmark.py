"""
Real-world-scale benchmark: the standard 24-image Kodak test set, at native
resolution (768x512 or 512x768), instead of the 96-256px crops used
elsewhere in this repo.

Kept as its own script (not wired into run_benchmark.py's __main__) because
of scale: a single image at one config takes several minutes at this
resolution (measured directly: ~367-400s for one 512x768 image, padded
internally to a 1024x1024 canvas -- see CLAUDE.md for the finding that this
scales worse than linearly with pixel count, not just proportionally).
Running the existing benchmark suite's full parameter sweep (7 error_thresh
values x 2 quantization_aware settings) across all 24 images would take
roughly 35 hours; this script deliberately runs a single, minimal config
(quantization_aware=True, error_thresh=100 -- this codebase's best-known
setting) across all 24 images instead, to get real bpp/PSNR/encode-time
numbers at native resolution without an unbounded runtime.

Images are the standard Kodak PCD set from https://r0k.us/graphics/kodak/,
downloaded into data/kodak/ (not committed by default -- see CLAUDE.md).

Produces:
  results/kodak_benchmark.csv   per-image bpp/PSNR/SSIM/encode time
  results/kodak_benchmark.png   PSNR-vs-bpp scatter across the 24 images

Run with:  python3 benchmarks/run_kodak_benchmark.py
"""
import csv
import glob
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fractal_compression import FractalConfig, encode_channel, decode_channel
from fractal_compression.entropy import entropy_bits_per_pixel
from fractal_compression.metrics import psnr, ssim_score

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "kodak")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_kodak_benchmark():
    print("== Kodak test set: native-resolution grayscale encode ==")
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.png")))
    if not files:
        print(f"  no images found in {DATA_DIR} -- download the Kodak set first")
        return []

    cfg = FractalConfig(quantization_aware=True, error_thresh=100, max_block=64, step=1)
    rows = []
    suite_t0 = time.time()

    for i, path in enumerate(files, 1):
        name = os.path.splitext(os.path.basename(path))[0]
        img = np.array(Image.open(path).convert("L")).astype(np.float64)

        t0 = time.time()
        enc = encode_channel(img, cfg)
        enc_time = time.time() - t0
        t0 = time.time()
        recon = decode_channel(enc)
        dec_time = time.time() - t0

        recon_c = recon[:img.shape[0], :img.shape[1]]
        image_psnr = psnr(img, recon_c)
        image_ssim = ssim_score(img, recon_c)
        fixed_bpp = len(enc.to_bytes()) * 8 / (img.shape[0] * img.shape[1])
        entropy_bpp = entropy_bits_per_pixel(enc)

        rows.append({
            "image": name, "height": img.shape[0], "width": img.shape[1],
            "bpp": fixed_bpp, "entropy_bpp": entropy_bpp,
            "psnr": image_psnr, "ssim": image_ssim,
            "n_transforms": len(enc.transforms),
            "encode_s": enc_time, "decode_s": dec_time,
        })
        elapsed = time.time() - suite_t0
        print(f"  [{i:2d}/{len(files)}] {name}: bpp={fixed_bpp:.3f} "
              f"(entropy bpp={entropy_bpp:.3f}) PSNR={image_psnr:.2f} "
              f"SSIM={image_ssim:.4f} enc={enc_time:.1f}s dec={dec_time:.2f}s "
              f"[suite elapsed {elapsed/60:.1f}min]")

    with open(os.path.join(RESULTS_DIR, "kodak_benchmark.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.scatter([r["bpp"] for r in rows], [r["psnr"] for r in rows])
    for r in rows:
        ax1.annotate(r["image"].replace("kodim", ""), (r["bpp"], r["psnr"]),
                     fontsize=7, alpha=0.7)
    ax1.set_xlabel("bits per pixel")
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title("Kodak set: bpp vs PSNR (quantization_aware=True, error_thresh=100)")
    ax1.grid(alpha=0.3)

    ax2.bar([r["image"].replace("kodim", "") for r in rows],
            [r["encode_s"] for r in rows])
    ax2.set_xlabel("Kodak image")
    ax2.set_ylabel("encode time (s)")
    ax2.set_title("Encode time per image (native resolution)")
    ax2.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "kodak_benchmark.png"), dpi=130)

    total_time = time.time() - suite_t0
    print(f"\nDone: {len(rows)} images in {total_time/60:.1f} minutes total.")
    print(f"  mean PSNR={np.mean([r['psnr'] for r in rows]):.2f}dB  "
          f"mean bpp={np.mean([r['bpp'] for r in rows]):.3f}  "
          f"mean encode={np.mean([r['encode_s'] for r in rows]):.1f}s")
    print("  saved kodak_benchmark.csv / .png")
    return rows


if __name__ == "__main__":
    run_kodak_benchmark()

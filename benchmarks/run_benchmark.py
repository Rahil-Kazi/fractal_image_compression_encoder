"""
Paper-ready benchmark suite for the fractal codec.

Produces:
  results/rate_distortion.csv        fractal vs JPEG, PSNR/SSIM vs bpp
  results/rate_distortion.png        the corresponding plot
  results/ablation_error_thresh.csv  quality/size/time vs ERROR_THRESH
  results/ablation_error_thresh.png
  results/color_chroma.csv           chroma-subsampling on/off comparison

Run with:  python3 benchmarks/run_benchmark.py
"""
import csv
import os
import time
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import data

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fractal_compression import FractalConfig, encode_channel, decode_channel, encode_image, decode_image
from fractal_compression.entropy import entropy_bits_per_pixel
from fractal_compression.metrics import psnr, ssim_score
from fractal_compression.baseline_jpeg import jpeg_rate_distortion

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_test_images():
    return {
        "camera_128": data.camera().astype(np.float64)[:128, :128],
        "coins_128": data.coins().astype(np.float64)[:128, :128],
    }


def run_rate_distortion():
    print("== Rate-distortion: fractal vs JPEG ==")
    rows = []
    error_threshs = [30, 60, 100, 150, 250, 400, 700]

    for name, img in get_test_images().items():
        for et in error_threshs:
            cfg = FractalConfig(error_thresh=et, max_block=64, step=1)
            t0 = time.time()
            enc = encode_channel(img, cfg)
            enc_time = time.time() - t0
            t0 = time.time()
            recon = decode_channel(enc)
            dec_time = time.time() - t0
            image_psnr = psnr(img, recon)
            image_ssim = ssim_score(img, recon)
            # Use real serialized bytes (to_bytes), not the header-less
            # bits_per_pixel() estimate, so the comparison against
            # fractal_entropy (which is necessarily measured on real bytes,
            # header included) is apples-to-apples.
            fixed_bpp = len(enc.to_bytes()) * 8 / (enc.height * enc.width)
            rows.append({
                "image": name, "codec": "fractal", "param": et,
                "bpp": fixed_bpp, "psnr": image_psnr,
                "ssim": image_ssim,
                "encode_s": enc_time, "decode_s": dec_time,
            })
            entropy_bpp = entropy_bits_per_pixel(enc)
            rows.append({
                "image": name, "codec": "fractal_entropy", "param": et,
                "bpp": entropy_bpp, "psnr": image_psnr,
                "ssim": image_ssim,
                "encode_s": enc_time, "decode_s": dec_time,
            })
            print(f"  {name} fractal ET={et:>4} bpp={fixed_bpp:.3f} "
                  f"(entropy bpp={entropy_bpp:.3f}) "
                  f"PSNR={image_psnr:.2f} SSIM={image_ssim:.4f} "
                  f"enc={enc_time:.2f}s dec={dec_time:.4f}s")

        for r in jpeg_rate_distortion(img):
            rows.append({
                "image": name, "codec": "jpeg", "param": r["quality"],
                "bpp": r["bpp"], "psnr": psnr(img, r["decoded"]),
                "ssim": ssim_score(img, r["decoded"]),
                "encode_s": None, "decode_s": None,
            })

    with open(os.path.join(RESULTS_DIR, "rate_distortion.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # plot
    fig, axes = plt.subplots(1, len(get_test_images()), figsize=(6 * len(get_test_images()), 5))
    if len(get_test_images()) == 1:
        axes = [axes]
    for ax, name in zip(axes, get_test_images().keys()):
        for codec, marker in [("fractal", "o-"), ("fractal_entropy", "^-"), ("jpeg", "s--")]:
            pts = [(r["bpp"], r["psnr"]) for r in rows if r["image"] == name and r["codec"] == codec]
            pts.sort()
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker, label=codec)
        ax.set_xlabel("bits per pixel")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(name)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "rate_distortion.png"), dpi=130)
    print(f"  saved rate_distortion.csv / .png\n")
    return rows


def run_ablation():
    print("== Ablation: ERROR_THRESH and search step ==")
    img = data.camera().astype(np.float64)[:128, :128]
    rows = []
    for et in [30, 60, 100, 150, 250, 400, 700, 1200]:
        for step in [1, 2]:
            cfg = FractalConfig(error_thresh=et, max_block=64, step=step)
            t0 = time.time()
            enc = encode_channel(img, cfg)
            enc_time = time.time() - t0
            recon = decode_channel(enc)
            rows.append({
                "error_thresh": et, "step": step, "n_leaves": len(enc.transforms),
                "bpp": enc.bits_per_pixel(), "psnr": psnr(img, recon),
                "ssim": ssim_score(img, recon), "encode_s": enc_time,
            })
            print(f"  ET={et:>4} step={step} leaves={len(enc.transforms):>4} "
                  f"bpp={enc.bits_per_pixel():.3f} PSNR={psnr(img, recon):.2f} enc={enc_time:.2f}s")

    with open(os.path.join(RESULTS_DIR, "ablation_error_thresh.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    fig, ax1 = plt.subplots(figsize=(7, 5))
    for step in [1, 2]:
        pts = [(r["bpp"], r["psnr"]) for r in rows if r["step"] == step]
        pts.sort()
        xs, ys = zip(*pts)
        ax1.plot(xs, ys, "o-", label=f"step={step}")
    ax1.set_xlabel("bits per pixel")
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title("ERROR_THRESH sweep (camera_128)")
    ax1.legend()
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "ablation_error_thresh.png"), dpi=130)
    print("  saved ablation_error_thresh.csv / .png\n")
    return rows


def run_color_experiment():
    print("== Color: chroma subsampling on vs off ==")
    img = data.astronaut().astype(np.float64)[:96, :96, :]
    rows = []
    for subsample in [True, False]:
        for et in [100, 300]:
            cfg = FractalConfig(error_thresh=et, max_block=64, step=1)
            t0 = time.time()
            enc = encode_image(img, cfg, chroma_subsample=subsample)
            enc_time = time.time() - t0
            recon = decode_image(enc)
            recon_c = np.clip(recon, 0, 255)
            rows.append({
                "chroma_subsample": subsample, "error_thresh": et,
                "bpp": enc.bits_per_pixel(*img.shape[:2]),
                "psnr": psnr(img, recon_c),
                "ssim": ssim_score(img.astype(np.uint8), recon_c.astype(np.uint8)),
                "encode_s": enc_time,
            })
            print(f"  subsample={subsample} ET={et} bpp={rows[-1]['bpp']:.3f} "
                  f"PSNR={rows[-1]['psnr']:.2f} SSIM={rows[-1]['ssim']:.4f} enc={enc_time:.2f}s")

    with open(os.path.join(RESULTS_DIR, "color_chroma.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("  saved color_chroma.csv\n")
    return rows


if __name__ == "__main__":
    run_rate_distortion()
    run_ablation()
    run_color_experiment()
    print("Done. See results/ for CSVs and plots.")

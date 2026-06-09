"""
test_anomalib.py — Standalone benchmark: anomalib PatchCore for conveyor product matching.

Trains PatchCore on ref*.jpg images, then scores good*/bad* images.
Uses anomaly_map.mean() as the continuous similarity score (NOT the binary pred_score).
Shows the heatmap overlay on each image so you can see WHERE the difference is detected.

Folder layout:
  ref*.jpg / ref*.png   — training images (treated as "good" by PatchCore)
  good*.jpg / good*.png — should score LOW anomaly (i.e. match)
  bad*.jpg  / bad*.png  — should score HIGH anomaly (i.e. not match)

Usage:
  python3 test_anomalib.py --folder ./objects
  python3 test_anomalib.py --folder ./objects --size 256 --thresh 0.5

Install:
  pip install anomalib
"""

import argparse
import glob
import os
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_images(folder, pattern):
    return sorted(os.path.abspath(p) for p in glob.glob(os.path.join(folder, pattern)))


def load_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None and img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3:4] / 255.0
        return (img[:, :, :3] * alpha).astype(np.uint8)
    return cv2.imread(path)


def separation(scores_low, scores_high):
    ml = float(np.mean(scores_low))
    mh = float(np.mean(scores_high))
    gap = mh - ml
    overlap = max(0.0, max(scores_low) - min(scores_high)) if scores_low and scores_high else 0.0
    verdict = "GOOD" if gap > 0.05 and overlap == 0 else ("OK" if gap > 0.02 else "POOR")
    return ml, mh, gap, overlap, verdict


# ─────────────────────────────────────────────────────────────────────────────
# PatchCore — train
# ─────────────────────────────────────────────────────────────────────────────

def train_patchcore(ref_paths: list, image_size: int, backbone: str, work_dir: Path):
    import torch
    from anomalib.data import Folder
    from anomalib.models import Patchcore
    from anomalib.engine import Engine
    from torchvision.transforms import v2 as T

    normal_dir = work_dir / "normal"
    normal_dir.mkdir(parents=True, exist_ok=True)
    for p in ref_paths:
        shutil.copy2(p, normal_dir / Path(p).name)

    aug = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ])

    datamodule = Folder(
        name="product",
        normal_dir=str(normal_dir),
        root=str(work_dir),
        augmentations=aug,
        train_batch_size=16,
        eval_batch_size=1,
        num_workers=0,
        test_split_mode="none",
        val_split_mode="from_train",
        val_split_ratio=0.2,
    )

    model = Patchcore(
        backbone=backbone,
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
    )

    engine = Engine(
        default_root_dir=str(work_dir / "lightning"),
        max_epochs=1,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_progress_bar=True,
    )

    print(f"\n[PatchCore] Training on {len(ref_paths)} reference images...")
    t0 = time.time()
    engine.fit(model=model, datamodule=datamodule)
    train_sec = time.time() - t0
    print(f"[PatchCore] Training done — {train_sec:.1f}s")
    return engine, model


# ─────────────────────────────────────────────────────────────────────────────
# Score — returns (raw_map_mean, anomaly_map_numpy)
# ─────────────────────────────────────────────────────────────────────────────

def score_image(engine, model, image_path: str):
    """
    Returns (score: float, heatmap: np.ndarray HxW float32).

    score    = mean of the raw anomaly_map — continuous, NOT the binary pred_score.
               Higher = more different from training images.
    heatmap  = the spatial anomaly map resized to original image dimensions.
               Shows WHERE the difference is — bright = anomalous region.
    """
    t0      = time.time()
    results = engine.predict(model=model, data_path=image_path, return_predictions=True)
    infer_ms = (time.time() - t0) * 1000

    if not results:
        return 0.0, None, infer_ms

    batch = results[0]
    amap  = batch.anomaly_map   # Mask tensor [1, H, W]
    if amap is None:
        return 0.0, None, infer_ms

    amap_np = amap.squeeze().cpu().numpy().astype(np.float32)  # H x W
    score   = float(amap_np.mean())
    return score, amap_np, infer_ms


def _normalise_scores(scores: list) -> list:
    if not scores:
        return scores
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.5] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def _rotate_bgr(bgr: np.ndarray, deg: float) -> np.ndarray:
    """Rotate image around its centre. Corners filled with mean border colour."""
    h, w = bgr.shape[:2]
    # estimate border/background colour from image edges
    border = np.concatenate([
        bgr[0, :].reshape(-1, 3),
        bgr[-1, :].reshape(-1, 3),
        bgr[:, 0].reshape(-1, 3),
        bgr[:, -1].reshape(-1, 3),
    ]).mean(axis=0).tolist()
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(bgr, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=[int(b) for b in border])



def _heatmap_overlay(bgr: np.ndarray, amap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay jet heatmap on BGR image. amap is H×W float, any range."""
    if amap is None:
        return bgr
    h, w = bgr.shape[:2]
    amap_r = cv2.resize(amap, (w, h))
    # normalise to 0-255 for colormap
    mn, mx = amap_r.min(), amap_r.max()
    if mx > mn:
        amap_u8 = ((amap_r - mn) / (mx - mn) * 255).astype(np.uint8)
    else:
        amap_u8 = np.zeros((h, w), dtype=np.uint8)
    heat_bgr = cv2.applyColorMap(amap_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr, 1 - alpha, heat_bgr, alpha, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Visual grid — image + heatmap overlay side by side
# ─────────────────────────────────────────────────────────────────────────────

def show_grid(targets, norm_scores, rot_results, thresh, folder):
    """
    Two columns per image: original@0° + heatmap@best_angle.
    If best_angle != 0, the heatmap panel shows the rotated image.
    """
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    n       = len(targets)
    per_row = min(n, 3)
    cols    = per_row * 2
    rows    = (n + per_row - 1) // per_row

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, min(4 * rows, 12)))
    axes = np.array(axes).reshape(rows, cols)

    for idx, ((tpath, label), norm, rr) in enumerate(zip(targets, norm_scores, rot_results)):
        row = idx // per_row
        col = (idx % per_row) * 2

        # original image at 0°
        bgr0    = load_bgr(tpath)
        img_rgb = cv2.cvtColor(bgr0, cv2.COLOR_BGR2RGB)

        # best angle image + heatmap
        best_bgr   = rr["best_bgr"]
        best_amap  = rr["best_amap"]
        best_angle = rr["best_angle"]
        best_norm  = norm   # norm is already based on best_score

        is_anomalous = best_norm >= thresh
        result_str   = "NOT MATCH" if is_anomalous else "MATCH"
        color        = "#e74c3c" if is_anomalous else "#2ecc71"

        overlay_rgb = cv2.cvtColor(_heatmap_overlay(best_bgr, best_amap), cv2.COLOR_BGR2RGB)

        ax_img  = axes[row, col]
        ax_heat = axes[row, col + 1]

        ax_img.imshow(img_rgb)
        ax_img.axis("off")
        ax_img.set_title(
            f"{os.path.basename(tpath)} [{label}]\n"
            f"score={best_norm:.3f} → {result_str}",
            fontsize=7, color=color, fontweight="bold", pad=2)

        angle_note = f"best angle: {best_angle}°" if best_angle != 0 else "angle: 0°"
        ax_heat.imshow(overlay_rgb)
        ax_heat.axis("off")
        ax_heat.set_title(
            f"heatmap @ {angle_note}\n(bright=different)",
            fontsize=7, color="#555555", pad=2)

        for ax in (ax_img, ax_heat):
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
                spine.set_visible(True)

    for idx in range(n, rows * per_row):
        r = idx // per_row
        c = (idx % per_row) * 2
        axes[r, c].set_visible(False)
        axes[r, c + 1].set_visible(False)

    fig.suptitle(f"PatchCore: product match  (thresh={thresh}, rot_step=10°)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(folder, "test_anomalib.png")
    plt.savefig(out, bbox_inches="tight", dpi=120)
    print(f"\n  Grid saved → {out}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(folder: str, image_size: int, backbone: str, thresh: float, use_rotation: bool = False):
    refs  = find_images(folder, "ref*.jpg") + find_images(folder, "ref*.png")
    goods = find_images(folder, "good*.jpg") + find_images(folder, "good*.png")
    bads  = find_images(folder, "bad*.jpg")  + find_images(folder, "bad*.png")

    if not refs:
        raise FileNotFoundError(f"No ref*.jpg/png found in {folder}")
    if not goods and not bads:
        raise FileNotFoundError(f"No good*/bad* images found in {folder}")

    print(f"\n  References  : {len(refs)}")
    print(f"  Good targets: {len(goods)}")
    print(f"  Bad targets : {len(bads)}")
    print(f"  Backbone    : {backbone}")
    print(f"  Image size  : {image_size}x{image_size}")
    print(f"  Threshold   : {thresh}\n")

    with tempfile.TemporaryDirectory(prefix="anomalib_test_") as tmp:
        work_dir = Path(tmp)
        engine, model = train_patchcore(refs, image_size, backbone, work_dir)

        targets     = [(p, "GOOD") for p in goods] + [(p, "BAD") for p in bads]
        rot_results = []

        # ── Pass 1: score everything at 0° to get the raw score range ──────────
        print(f"\n  Pass 1 — scoring {len(targets)} images at 0°...")
        pass1_scores = []
        pass1_amaps  = []
        for tpath, label in targets:
            s, amap, ms = score_image(engine, model, tpath)
            pass1_scores.append(s)
            pass1_amaps.append(amap)
            print(f"    {os.path.basename(tpath):<25} [{label}]  raw@0°={s:.4f}  ({ms:.0f}ms)")

        # derive the raw cutoff from the normalised threshold
        lo, hi = min(pass1_scores), max(pass1_scores)
        if hi > lo:
            raw_thresh = lo + thresh * (hi - lo)
        else:
            raw_thresh = lo  # all same — nothing to reject

        rejected_0 = sum(1 for s in pass1_scores if s > raw_thresh)
        print(f"\n  Raw threshold (unnormalised): {raw_thresh:.4f}  "
              f"({rejected_0}/{len(targets)} rejected at 0° → will try rotations)")

        # ── Pass 2: rotation loop only for rejected images ───────────────────
        print(f"\n  Pass 2 — rotation search (step 10°) for rejected images...")
        for i, (tpath, label) in enumerate(targets):
            s0    = pass1_scores[i]
            amap0 = pass1_amaps[i]
            bgr0  = load_bgr(tpath)

            rr = {
                "score0":     s0,
                "amap0":      amap0,
                "best_score": s0,
                "best_angle": 0,
                "best_amap":  amap0,
                "best_bgr":   bgr0,
                "rotated":    False,
                "all_angles": [(0, s0)],
            }

            if s0 > raw_thresh and use_rotation:
                print(f"    {os.path.basename(tpath):<25} rejected@0° ({s0:.4f}) → trying rotations...",
                      end="", flush=True)
                rr["rotated"] = True
                for deg in range(10, 360, 10):
                    rot_bgr  = _rotate_bgr(bgr0, deg)
                    tmp_path = tpath + f".__rot{deg}.jpg"
                    cv2.imwrite(tmp_path, rot_bgr)
                    try:
                        s, amap, _ms = score_image(engine, model, tmp_path)
                    finally:
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass
                    rr["all_angles"].append((deg, s))
                    if s < rr["best_score"]:
                        rr["best_score"] = s
                        rr["best_angle"] = deg
                        rr["best_amap"]  = amap
                        rr["best_bgr"]   = rot_bgr
                changed = (f" best={rr['best_score']:.4f} at {rr['best_angle']}°"
                           if rr["best_angle"] != 0 else f" still {rr['best_score']:.4f} (no improvement)")
                print(changed)
            elif s0 > raw_thresh:
                print(f"    {os.path.basename(tpath):<25} rejected@0° ({s0:.4f}) — rotation disabled")
            else:
                print(f"    {os.path.basename(tpath):<25} OK@0° ({s0:.4f}) — skipping rotation")

            rot_results.append(rr)

    # use best_score (lowest across all rotations) for the table and separation
    raw_scores  = [rr["best_score"] for rr in rot_results]
    norm_scores = _normalise_scores(raw_scores)

    W = 90
    print(f"\n{'Image':<25} {'Label':<6} {'@0° raw':>10} {'Best raw':>10} {'Best°':>6} {'Norm':>8} {'Decision':>12}")
    print("-" * W)
    for (tpath, label), rr, norm in zip(targets, rot_results, norm_scores):
        decision = "NOT MATCH" if norm >= thresh else "MATCH"
        mark = "✓" if (label == "GOOD") == (norm < thresh) else "✗"
        print(f"  {os.path.basename(tpath):<23} {label:<6} "
              f"{rr['score0']:>10.4f} {rr['best_score']:>10.4f} {rr['best_angle']:>6}° "
              f"{norm:>8.4f} {decision:>12}  {mark}")

    good_scores = [n for (_, lbl), n in zip(targets, norm_scores) if lbl == "GOOD"]
    bad_scores  = [n for (_, lbl), n in zip(targets, norm_scores) if lbl == "BAD"]

    print(f"\n{'='*W}")
    print("  SEPARATION ANALYSIS  (using best score across all rotation angles)")
    print(f"{'='*W}")
    print(f"  {'Metric':<22} {'Good mean':>10} {'Bad mean':>10} {'Gap':>8} {'Overlap':>9} {'Verdict':>8}")
    print(f"  {'-'*(W-2)}")

    if good_scores and bad_scores:
        ml, mh, gap, ov, verd = separation(good_scores, bad_scores)
        print(f"  {'PatchCore (rot-best)':<22} {ml:>10.4f} {mh:>10.4f} {gap:>+8.4f} {ov:>9.4f} {verd:>8}")
        mid = (ml + mh) / 2.0
        print(f"\n  Suggested threshold (midpoint): {mid:.3f}")
        print(f"  → re-run with: --thresh {mid:.2f}")

    print(f"{'='*W}")
    print("  Gap > 0.05 with 0 overlap = GOOD    Gap > 0.02 = OK    else POOR")
    print("  ✓ = correctly classified    ✗ = wrong decision at current threshold\n")

    show_grid(targets, norm_scores, rot_results, thresh, folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PatchCore product match test")
    parser.add_argument("--folder",   default="objects",         help="Folder with ref*/good*/bad* images")
    parser.add_argument("--size",     type=int, default=256,     help="Image resize (default 256)")
    parser.add_argument("--backbone", default="wide_resnet50_2", help="Torchvision backbone")
    parser.add_argument("--thresh",   type=float, default=0.5,   help="Anomaly threshold 0–1 (default 0.5)")
    parser.add_argument("--rotate",   action="store_true",        help="Enable rotation search for rejected images (10° steps up to 360°)")
    args = parser.parse_args()
    run(args.folder, args.size, args.backbone, args.thresh, args.rotate)

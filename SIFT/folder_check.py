"""
Folder Label Checker — SIFT Align + Two-Scale SSIM
===================================================
Batch check all images in a folder against master reference.
Shows 4-panel preview for each image.

How it works:
  Fine SSIM   (blur=3)  → detects text/print/logo pixel differences
  Coarse SSIM (blur=31) → detects structural/position differences;
                          hologram reflection noise disappears at coarse scale

Controls:
    SPACE / N   = next image
    P           = previous image
    S           = save current 4-panel result image
    ESC / Q     = quit

Usage:
    python3 folder_check.py --folder labels/
    python3 folder_check.py --folder labels/ --ref reference/master.jpeg
    python3 folder_check.py --folder labels/ --diff 6 --cdiff 8
"""

import cv2
import numpy as np
import argparse
import os
import time
from skimage.metrics import structural_similarity as ssim_fn

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE        = (800, 600)
SSIM_PASS       = 0.73
DIFF_AREA_FAIL  = 6.0
CDIFF_AREA_FAIL = 8.0
FINE_BLUR       = 3
COARSE_BLUR     = 31
RATIO_TEST      = 0.75
MIN_INLIERS     = 80
IMG_EXTS        = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")


# ── SIFT align ────────────────────────────────────────────────────────────────

def sift_align(master_gray, test_gray, sift):
    kp1, des1 = sift.detectAndCompute(master_gray, None)
    kp2, des2 = sift.detectAndCompute(test_gray,   None)

    if des1 is None or des2 is None or len(des1) < 8 or len(des2) < 8:
        return None, 0

    index_params  = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    try:
        raw = flann.knnMatch(des1, des2, k=2)
    except cv2.error:
        return None, 0

    good = [m for m, n in raw if m.distance < RATIO_TEST * n.distance]
    if len(good) < 8:
        return None, 0

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    matrix, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    if matrix is None:
        return None, 0

    inliers = int(mask.sum()) if mask is not None else 0
    aligned = cv2.warpPerspective(test_gray, matrix, IMG_SIZE)
    return aligned, inliers


# ── Scoring ───────────────────────────────────────────────────────────────────

def compute_scores(master_gray, aligned):
    if aligned.shape != master_gray.shape:
        aligned = cv2.resize(aligned, (master_gray.shape[1], master_gray.shape[0]))

    m_fine = cv2.GaussianBlur(master_gray, (FINE_BLUR, FINE_BLUR), 0)
    t_fine = cv2.GaussianBlur(aligned,     (FINE_BLUR, FINE_BLUR), 0)
    score, diff_fine = ssim_fn(m_fine, t_fine, full=True)
    fine_mask = (diff_fine < 0.5)
    fine_pct  = fine_mask.sum() / fine_mask.size * 100

    m_coarse = cv2.GaussianBlur(master_gray, (COARSE_BLUR, COARSE_BLUR), 0)
    t_coarse = cv2.GaussianBlur(aligned,     (COARSE_BLUR, COARSE_BLUR), 0)
    _, diff_coarse = ssim_fn(m_coarse, t_coarse, full=True)
    coarse_mask = (diff_coarse < 0.5)
    coarse_pct  = coarse_mask.sum() / coarse_mask.size * 100

    return score, fine_pct, coarse_pct, diff_fine, fine_mask


# ── 4-panel preview ───────────────────────────────────────────────────────────

def draw_4panel(master_gray, aligned, diff, fine_mask,
                ssim_score, fine_pct, coarse_pct, inliers,
                result, ssim_ok, fine_ok, coarse_ok,
                filename, idx, total):
    color = (0, 220, 0) if result == "GOOD" else (0, 0, 220)
    h     = master_gray.shape[0]

    # Panel 1 — Master
    p1 = cv2.cvtColor(master_gray, cv2.COLOR_GRAY2BGR)
    cv2.putText(p1, "MASTER", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    # Panel 2 — Aligned test
    p2 = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)
    cv2.putText(p2, "ALIGNED TEST", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    # Panel 3 — Fine diff pixels
    p3 = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)
    p3[fine_mask] = (0, 0, 220)
    cv2.putText(p3, f"FINE DIFF  {fine_pct:.1f}%", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2)

    # Panel 4 — SSIM heatmap
    p4 = cv2.applyColorMap((diff * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.putText(p4, "HEATMAP (BLUE=diff)", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    panel_h = 280
    def rs(img):
        s = panel_h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * s), panel_h))

    grid = np.vstack([np.hstack([rs(p1), rs(p2)]),
                      np.hstack([rs(p3), rs(p4)])])

    # Header bar
    bar_h = 110
    bar   = np.zeros((bar_h, grid.shape[1], 3), dtype=np.uint8)
    bar[:] = (25, 25, 25)

    # File name + index
    cv2.putText(bar, f"[{idx}/{total}]  {filename}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(bar, f"Inliers: {inliers}",
                (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

    ssim_col   = (0, 220, 0) if ssim_ok   else (0, 0, 220)
    fine_col   = (0, 220, 0) if fine_ok   else (0, 0, 220)
    coarse_col = (0, 220, 0) if coarse_ok else (0, 0, 220)
    cv2.putText(bar, f"SSIM: {ssim_score:.3f}",
                (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ssim_col, 2)
    cv2.putText(bar, f"Fine: {fine_pct:.1f}%",
                (210, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fine_col, 2)
    cv2.putText(bar, f"Coarse: {coarse_pct:.1f}%",
                (390, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.7, coarse_col, 2)
    cv2.putText(bar, f"->  {result}",
                (590, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

    cv2.putText(bar, "SPACE/N = next    P = prev    S = save    ESC = quit",
                (10, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1)

    return np.vstack([bar, grid])


def draw_error_panel(filename, idx, total, message, grid_w=1120):
    bar_h = 110
    panel_h = 560
    canvas = np.zeros((bar_h + panel_h, grid_w, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)
    cv2.putText(canvas, f"[{idx}/{total}]  {filename}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
    cv2.putText(canvas, message,
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
    cv2.putText(canvas, "SPACE/N = next    P = prev    ESC = quit",
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1)
    return canvas


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True,  help="Folder of test images")
    parser.add_argument("--ref",    default="reference/technova-label-master.jpeg")
    parser.add_argument("--ssim",   type=float, default=SSIM_PASS)
    parser.add_argument("--diff",   type=float, default=DIFF_AREA_FAIL)
    parser.add_argument("--cdiff",  type=float, default=CDIFF_AREA_FAIL)
    args = parser.parse_args()

    # ── Load master ────────────────────────────────────────────────────
    master_bgr = cv2.imread(args.ref)
    if master_bgr is None:
        print(f"ERROR: reference not found: {args.ref}")
        return
    master_gray = cv2.cvtColor(master_bgr, cv2.COLOR_BGR2GRAY)
    master_gray = cv2.resize(master_gray, IMG_SIZE)
    sift = cv2.SIFT_create(nfeatures=5000)

    # ── Load images ────────────────────────────────────────────────────
    images = sorted(f for f in os.listdir(args.folder)
                    if f.lower().endswith(IMG_EXTS))
    if not images:
        print(f"No images found in {args.folder}")
        return

    total = len(images)
    print(f"\nFolder : {args.folder}  ({total} images)")
    print(f"Ref    : {args.ref}")
    print(f"Thresh : SSIM>={args.ssim}  Fine<{args.diff}%  Coarse<{args.cdiff}%\n")

    # Cache results so revisiting is instant
    cache  = {}
    idx    = 0
    WIN    = "Folder Check — Label Verifier"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    while True:
        fname = images[idx]
        fpath = os.path.join(args.folder, fname)

        if fname not in cache:
            img = cv2.imread(fpath)
            if img is None:
                cache[fname] = {"error": "Cannot read image"}
            else:
                test_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                test_gray = cv2.resize(test_gray, IMG_SIZE)
                aligned, inliers = sift_align(master_gray, test_gray, sift)
                if aligned is None or inliers < MIN_INLIERS:
                    cache[fname] = {"error": f"Low inliers ({inliers}) — reposition or retake"}
                else:
                    ssim_score, fine_pct, coarse_pct, diff_map, fine_mask = \
                        compute_scores(master_gray, aligned)
                    ssim_ok   = ssim_score >= args.ssim
                    fine_ok   = fine_pct   <  args.diff
                    coarse_ok = coarse_pct <  args.cdiff
                    result    = "GOOD" if (ssim_ok and fine_ok and coarse_ok) else "BAD"
                    tag = "✅" if result == "GOOD" else "❌"
                    print(f"[{idx+1:>3}/{total}] {fname:<30} "
                          f"SSIM={ssim_score:.3f}  Fine={fine_pct:.1f}%  "
                          f"Coarse={coarse_pct:.1f}%  {tag} {result}")
                    cache[fname] = dict(
                        aligned=aligned, diff_map=diff_map, fine_mask=fine_mask,
                        ssim_score=ssim_score, fine_pct=fine_pct,
                        coarse_pct=coarse_pct, inliers=inliers,
                        result=result, ssim_ok=ssim_ok,
                        fine_ok=fine_ok, coarse_ok=coarse_ok
                    )

        c = cache[fname]
        if "error" in c:
            panel = draw_error_panel(fname, idx + 1, total, c["error"])
        else:
            panel = draw_4panel(
                master_gray, c["aligned"], c["diff_map"], c["fine_mask"],
                c["ssim_score"], c["fine_pct"], c["coarse_pct"], c["inliers"],
                c["result"], c["ssim_ok"], c["fine_ok"], c["coarse_ok"],
                fname, idx + 1, total
            )

        cv2.imshow(WIN, panel)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord(' '), ord('n'), ord('N')):   # next
            idx = (idx + 1) % total
        elif key in (ord('p'), ord('P')):            # previous
            idx = (idx - 1) % total
        elif key in (ord('s'), ord('S')):            # save panel
            if "error" not in c:
                out = f"result_{fname}"
                cv2.imwrite(out, panel)
                print(f"Saved: {out}")
        elif key in (27, ord('q'), ord('Q')):        # quit
            break

    cv2.destroyAllWindows()

    # Summary
    good = sum(1 for v in cache.values() if v.get("result") == "GOOD")
    bad  = sum(1 for v in cache.values() if v.get("result") == "BAD")
    print(f"\n── Summary ──────────────────────")
    print(f"GOOD : {good}")
    print(f"BAD  : {bad}")
    print(f"Total: {good + bad} / {total}")


if __name__ == "__main__":
    main()

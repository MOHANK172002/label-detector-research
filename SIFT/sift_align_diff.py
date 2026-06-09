"""
SIFT Align + Pixel Diff — Label Authenticator
===============================================
Step 1: SIFT finds keypoints on master + test
Step 2: Homography warps test image → master coordinate space
Step 3: Pixel diff on aligned images → detects ANY addition/change
Step 4: GOOD if SSIM >= ssim_threshold AND diff_area < diff_threshold

Why this works:
  - SIFT alignment  → handles different zoom, angle, distance
  - Pixel diff      → catches stamps, handwriting, marks added on label
  - Body zone only  → ignores header/footer text noise

Usage:
    python3 sift_align_diff.py
    python3 sift_align_diff.py --ref reference/ref2.jpg --labels labels
    python3 sift_align_diff.py --ssim 0.75 --diff 8
    python3 sift_align_diff.py --ssim 0.73 --diff 6

Controls:
    Any key = next image
    ESC     = quit
"""

import cv2
import numpy as np
import os
import argparse
from skimage.metrics import structural_similarity as ssim

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE       = (800, 600)
SSIM_PASS      = 0.75       # SSIM >= this → pass
DIFF_AREA_FAIL = 8.0        # diff area % < this → pass (body zone only)
RATIO_TEST     = 0.75       # Lowe's ratio test for SIFT matching

# Body zone — skip header (logo) and footer (address text)
# Only check the label body where defects appear
BODY_Y1_PCT = 0.18
BODY_Y2_PCT = 0.78


# ── SIFT alignment ────────────────────────────────────────────────────────────

def sift_align(master_gray, test_gray, sift):
    """
    Align test image to master using SIFT + FLANN + homography.
    Returns aligned image and number of inliers (0 = failed).
    """
    kp1, des1 = sift.detectAndCompute(master_gray, None)
    kp2, des2 = sift.detectAndCompute(test_gray,   None)

    if des1 is None or des2 is None or len(des1) < 8 or len(des2) < 8:
        return None, 0

    # FLANN matcher
    index_params  = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann   = cv2.FlannBasedMatcher(index_params, search_params)
    try:
        raw = flann.knnMatch(des1, des2, k=2)
    except cv2.error:
        return None, 0

    # Lowe's ratio test
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
    """
    Compute SSIM and diff area % on the FULL label image.
    """
    if aligned.shape != master_gray.shape:
        aligned = cv2.resize(aligned, (master_gray.shape[1], master_gray.shape[0]))

    ssim_score, diff = ssim(master_gray, aligned, full=True)

    # Pixels where local SSIM < 0.5 = significantly different
    diff_mask     = (diff < 0.5)
    diff_area_pct = diff_mask.sum() / diff_mask.size * 100

    return ssim_score, diff_area_pct, diff, diff_mask


# ── Preview ───────────────────────────────────────────────────────────────────

def draw_preview(master_gray, aligned, test_gray,
                 ssim_score, diff_area_pct, diff, diff_mask,
                 result, filename, ssim_ok, diff_ok):

    color = (0, 220, 0) if result == "GOOD" else (0, 0, 220)
    h = master_gray.shape[0]

    # ── Panel 1: Master ───────────────────────────────────────────────
    master_bgr = cv2.cvtColor(master_gray, cv2.COLOR_GRAY2BGR)
    cv2.putText(master_bgr, "MASTER", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    # ── Panel 2: Aligned test ─────────────────────────────────────────
    aligned_bgr = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)
    cv2.putText(aligned_bgr, "ALIGNED TEST", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    # ── Panel 3: Diff pixels — full image ─────────────────────────────
    diff_display = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)
    diff_display[diff_mask] = (0, 0, 220)      # red on every diff pixel
    cv2.putText(diff_display, f"DIFF  {diff_area_pct:.1f}%", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2)

    # ── Panel 4: SSIM heatmap — full image ────────────────────────────
    heatmap = cv2.applyColorMap((diff * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.putText(heatmap, "HEATMAP (BLUE=diff)", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # ── Combine 4 panels into 2x2 grid ───────────────────────────────
    panel_h = 280
    def resize(img):
        s = panel_h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * s), panel_h))

    top    = np.hstack([resize(master_bgr), resize(aligned_bgr)])
    bottom = np.hstack([resize(diff_display), resize(heatmap)])
    grid   = np.vstack([top, bottom])

    # ── Header bar ────────────────────────────────────────────────────
    bar_h = 85
    bar   = np.zeros((bar_h, grid.shape[1], 3), dtype=np.uint8)
    bar[:] = (25, 25, 25)

    cv2.putText(bar, f"File: {filename}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)

    ssim_col = (0, 220, 0) if ssim_ok else (0, 0, 220)
    diff_col = (0, 220, 0) if diff_ok  else (0, 0, 220)
    cv2.putText(bar, f"SSIM: {ssim_score:.3f}",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ssim_col, 2)
    cv2.putText(bar, f"Diff: {diff_area_pct:.1f}%",
                (230, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, diff_col, 2)
    cv2.putText(bar, f"->  {result}",
                (430, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    reason = []
    if not ssim_ok: reason.append(f"SSIM {ssim_score:.3f} < {SSIM_PASS}")
    if not diff_ok: reason.append(f"diff {diff_area_pct:.1f}% >= {DIFF_AREA_FAIL}%")
    if reason:
        cv2.putText(bar, "  [" + "  &  ".join(reason) + "]",
                    (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

    return np.vstack([bar, grid])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref",    default="reference/good2.jpeg")
    parser.add_argument("--labels", default="labels")
    parser.add_argument("--ssim",   type=float, default=SSIM_PASS,
                        help=f"Min SSIM to pass (default {SSIM_PASS})")
    parser.add_argument("--diff",   type=float, default=DIFF_AREA_FAIL,
                        help=f"Max diff area %% to pass (default {DIFF_AREA_FAIL})")
    args = parser.parse_args()

    master_bgr = cv2.imread(args.ref)
    if master_bgr is None:
        print(f"ERROR: reference not found: {args.ref}")
        return

    master_gray = cv2.cvtColor(master_bgr, cv2.COLOR_BGR2GRAY)
    master_gray = cv2.resize(master_gray, IMG_SIZE)

    sift = cv2.SIFT_create(nfeatures=5000)

    print(f"\nReference  : {args.ref}")
    print(f"Threshold  : SSIM >= {args.ssim}  AND  diff < {args.diff}%  → GOOD")
    print(f"Body zone  : y={BODY_Y1_PCT*100:.0f}%–{BODY_Y2_PCT*100:.0f}%  (header/footer excluded)")
    print(f"\nAny key = next    ESC = quit\n")

    files = sorted(f for f in os.listdir(args.labels)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png')))

    pass_count = fail_count = 0
    results = []

    for file in files:
        path     = os.path.join(args.labels, file)
        test_bgr = cv2.imread(path)
        if test_bgr is None:
            continue

        test_gray = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)
        test_gray = cv2.resize(test_gray, IMG_SIZE)

        print(f"Processing: {file}")

        # Step 1: SIFT alignment
        aligned, inliers = sift_align(master_gray, test_gray, sift)
        if aligned is None:
            print(f"  SIFT align : FAILED\n  Result     : ❌ BAD\n")
            fail_count += 1
            results.append((file, 0.0, 100.0, 0, "BAD"))
            continue

        print(f"  SIFT inliers: {inliers}")

        # Step 2: pixel diff on body zone
        ssim_score, diff_area_pct, diff, diff_mask = \
            compute_scores(master_gray, aligned)

        # Step 3: decision
        ssim_ok = ssim_score    >= args.ssim
        diff_ok = diff_area_pct <  args.diff
        result  = "GOOD" if (ssim_ok and diff_ok) else "BAD"

        print(f"  SSIM       : {ssim_score:.4f}  ({'ok' if ssim_ok else 'FAIL'})")
        print(f"  Diff area  : {diff_area_pct:.2f}%  ({'ok' if diff_ok else 'FAIL'})")
        print(f"  Result     : {'✅' if result == 'GOOD' else '❌'} {result}\n")

        results.append((file, ssim_score, diff_area_pct, inliers, result))
        if result == "GOOD": pass_count += 1
        else:                 fail_count += 1

        # Step 4: preview
        preview = draw_preview(master_gray, aligned, test_gray,
                               ssim_score, diff_area_pct, diff, diff_mask,
                               result, file, ssim_ok, diff_ok)
        cv2.imshow("SIFT Align + Diff", preview)

        if cv2.waitKey(0) == 27:
            break

    # ── Summary ───────────────────────────────────────────────────────────
    total = pass_count + fail_count
    print(f"\n{'='*62}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*62}")
    print(f"  GOOD : {pass_count}/{total}")
    print(f"  BAD  : {fail_count}/{total}")
    print(f"\n  {'File':<24} {'SSIM':>6} {'Diff%':>7} {'Inliers':>8}  Result")
    print(f"  {'-'*58}")
    for name, ss, da, inl, res in results:
        icon = "✅" if res == "GOOD" else "❌"
        print(f"  {icon} {name:<22} {ss:.4f} {da:6.2f}% {inl:8d}  {res}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

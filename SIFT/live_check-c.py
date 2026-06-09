"""
Live Label Checker — SIFT Align + Two-Scale SSIM
=================================================
Real-time webcam label verification against master reference.

How it works:
  1. Load master reference image
  2. Open camera — show live feed
  3. Press SPACE → capture frame → run SIFT align + diff
  4. Fine SSIM   (blur=3)  → detects text/print/logo pixel differences
  5. Coarse SSIM (blur=31) → detects structural/position differences
                              hologram reflection noise disappears at coarse scale
  6. No zone setup needed — works for any label automatically

Controls:
    SPACE   = capture current frame and check
    S       = save cropped label image
    R       = reset / clear result
    ESC     = quit

Usage:
    python3 live_check.py
    python3 live_check.py --ref reference/1.jpeg
    python3 live_check.py --diff 6 --cdiff 8
"""

import cv2
import numpy as np
import argparse
import json
import os
import time
from skimage.metrics import structural_similarity as ssim_fn

REGION_FILE = "region-test.json"

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE         = (800, 600)
SSIM_PASS        = 0.73
DIFF_AREA_FAIL   = 6.0    # fine diff threshold  %
CDIFF_AREA_FAIL  = 8.0    # coarse diff threshold %
FINE_BLUR        = 3      # small blur  — fine scale (text/print)
COARSE_BLUR      = 31     # large blur  — coarse scale (structure/hologram position)
RATIO_TEST       = 0.75
MIN_INLIERS      = 80


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
    """
    Two-scale SSIM diff — no zone needed.
      Fine  (blur=3)  → catches text/print/logo pixel differences
      Coarse(blur=31) → catches structural/position differences;
                        hologram reflection noise disappears at this scale
    """
    if aligned.shape != master_gray.shape:
        aligned = cv2.resize(aligned, (master_gray.shape[1], master_gray.shape[0]))

    # Fine scale — small blur
    m_fine = cv2.GaussianBlur(master_gray, (FINE_BLUR, FINE_BLUR), 0)
    t_fine = cv2.GaussianBlur(aligned,     (FINE_BLUR, FINE_BLUR), 0)
    score, diff_fine = ssim_fn(m_fine, t_fine, full=True)
    fine_mask        = (diff_fine < 0.5)
    fine_pct         = fine_mask.sum() / fine_mask.size * 100

    # Coarse scale — large blur (hologram reflection averages out)
    m_coarse = cv2.GaussianBlur(master_gray, (COARSE_BLUR, COARSE_BLUR), 0)
    t_coarse = cv2.GaussianBlur(aligned,     (COARSE_BLUR, COARSE_BLUR), 0)
    _, diff_coarse = ssim_fn(m_coarse, t_coarse, full=True)
    coarse_mask      = (diff_coarse < 0.5)
    coarse_pct       = coarse_mask.sum() / coarse_mask.size * 100

    return score, fine_pct, coarse_pct, diff_fine, fine_mask


# ── 4-panel preview ───────────────────────────────────────────────────────────

def draw_4panel(master_gray, aligned, diff, diff_mask,
                ssim_score, fine_pct, coarse_pct, inliers,
                result, ssim_ok, fine_ok, coarse_ok,
                timestamp=""):
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
    p3[diff_mask] = (0, 0, 220)
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

    # Header bar — same layout as folder_check
    bar_h = 110
    bar   = np.zeros((bar_h, grid.shape[1], 3), dtype=np.uint8)
    bar[:] = (25, 25, 25)

    label = f"Captured: {timestamp}" if timestamp else "Live capture"
    cv2.putText(bar, label,
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
    cv2.putText(bar, "SPACE = check again    R = reset    S = save    ESC = quit",
                (10, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1)

    return np.vstack([bar, grid])


def draw_idle_overlay(frame):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 45), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, "SPACE = Check label    ESC = Quit",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    return frame


def draw_align_warning(frame, inliers):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (20, 20, 80), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    cv2.putText(frame, "HOLD LABEL CLOSER & FLAT",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 200, 255), 3)
    cv2.putText(frame, f"Inliers: {inliers}  (need >= {MIN_INLIERS})",
                (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 255), 1)
    cv2.rectangle(frame, (6, 6), (w - 6, h - 6), (0, 140, 255), 6)
    return frame


# ── Main ──────────────────────────────────────────────────────────────────────

def run_folder(args, master_gray, sift):
    """Batch mode — process all images in a folder."""
    exts   = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    images = sorted(f for f in os.listdir(args.folder)
                    if f.lower().endswith(exts))
    if not images:
        print(f"No images found in {args.folder}")
        return

    print(f"\nFolder : {args.folder}  ({len(images)} images)")
    print(f"{'File':<30} {'SSIM':>6} {'Fine%':>7} {'Coarse%':>9} {'Inliers':>8}  Result")
    print("─" * 72)

    good_count = bad_count = 0

    for fname in images:
        fpath = os.path.join(args.folder, fname)
        img   = cv2.imread(fpath)
        if img is None:
            print(f"{fname:<30}  ERROR: cannot read")
            continue

        test_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        test_gray = cv2.resize(test_gray, IMG_SIZE)

        aligned, inliers = sift_align(master_gray, test_gray, sift)
        if aligned is None or inliers < MIN_INLIERS:
            print(f"{fname:<30}  LOW INLIERS ({inliers}) — skip")
            continue

        ssim_score, fine_pct, coarse_pct, _, _ = compute_scores(master_gray, aligned)
        ssim_ok   = ssim_score >= args.ssim
        fine_ok   = fine_pct   <  args.diff
        coarse_ok = coarse_pct <  args.cdiff
        result    = "GOOD" if (ssim_ok and fine_ok and coarse_ok) else "BAD"

        tag = "✅" if result == "GOOD" else "❌"
        print(f"{fname:<30} {ssim_score:>6.3f} {fine_pct:>6.1f}% {coarse_pct:>8.1f}%"
              f" {inliers:>8}  {tag} {result}")

        if result == "GOOD":
            good_count += 1
        else:
            bad_count += 1

    total = good_count + bad_count
    print("─" * 72)
    print(f"GOOD: {good_count}/{total}   BAD: {bad_count}/{total}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref",    default="reference/master.jpeg")
    parser.add_argument("--camera", type=int,   default=0)
    parser.add_argument("--ssim",   type=float, default=SSIM_PASS)
    parser.add_argument("--diff",   type=float, default=DIFF_AREA_FAIL)
    parser.add_argument("--cdiff",  type=float, default=CDIFF_AREA_FAIL,
                        help="Coarse diff threshold %% (hologram/structure, default %(default)s)")
    parser.add_argument("--folder", default=None,
                        help="Batch mode — path to folder of test images")
    args = parser.parse_args()

    # ── Load master ────────────────────────────────────────────────────
    master_bgr = cv2.imread(args.ref)
    if master_bgr is None:
        print(f"ERROR: reference not found: {args.ref}")
        return

    master_gray = cv2.cvtColor(master_bgr, cv2.COLOR_BGR2GRAY)
    master_gray = cv2.resize(master_gray, IMG_SIZE)
    sift = cv2.SIFT_create(nfeatures=5000)

    # ── Batch folder mode ──────────────────────────────────────────────
    if args.folder:
        run_folder(args, master_gray, sift)
        return

    # ── Load region ────────────────────────────────────────────────────
    region = None
    if os.path.exists(REGION_FILE):
        with open(REGION_FILE) as f:
            region = json.load(f)
        print(f"Region    : x={region['x']} y={region['y']} "
              f"w={region['w']} h={region['h']}  (from {REGION_FILE})")
    else:
        print(f"No {REGION_FILE} found — using full frame.")
        print(f"Run python3 setup_region.py first.")

    print(f"\nReference : {args.ref}")
    print(f"Threshold : SSIM >= {args.ssim}  AND  fine diff < {args.diff}%"
          f"  AND  coarse diff < {args.cdiff}%  → GOOD")
    print(f"Camera    : {args.camera}")
    print(f"\nSPACE = check    S = save    R = reset    ESC = quit\n")


    url = (
        "tcp://192.168.1.11:8888"
        "?fflags=nobuffer"
        "&flags=low_delay"
        "&framedrop=1"
    )

    # ── Open camera ────────────────────────────────────────────────────
    # cap = cv2.VideoCapture(args.camera)
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera {args.camera}")
        return

    # State
    result        = None
    ssim_score    = 0.0
    fine_pct      = 0.0
    coarse_pct    = 0.0
    inliers       = 0
    align_warning = False
    aligned_img   = None
    diff_map      = None
    diff_mask_map = None
    ssim_ok       = False
    fine_ok       = False
    coarse_ok     = False
    capture_time  = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: cannot read from camera")
            break

        display = frame.copy()

        # Green box — label region
        if region:
            rx, ry, rw, rh = region['x'], region['y'], region['w'], region['h']
            cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
            cv2.putText(display, "Label area", (rx, ry - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

        if align_warning:
            display = draw_align_warning(display, inliers)
            h = display.shape[0]
            cv2.putText(display, "SPACE = try again    ESC = quit",
                        (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (180, 180, 180), 1)
        else:
            display = draw_idle_overlay(display)

        cv2.imshow("Live Feed", display)

        # 4-panel result window
        if result is not None and aligned_img is not None:
            panel = draw_4panel(
                master_gray, aligned_img, diff_map, diff_mask_map,
                ssim_score, fine_pct, coarse_pct, inliers,
                result, ssim_ok, fine_ok, coarse_ok,
                timestamp=capture_time
            )
            cv2.imshow("Result — Master | Aligned | Diff | Heatmap", panel)

        key = cv2.waitKey(1) & 0xFF

        # ── SPACE: capture and check ───────────────────────────────────
        if key == ord(' '):
            print("Checking...")
            result        = None
            align_warning = False
            ssim_score    = 0.0
            fine_pct      = 0.0
            coarse_pct    = 0.0
            inliers       = 0
            capture_time  = time.strftime("%Y-%m-%d %H:%M:%S")

            if region:
                rx, ry, rw, rh = region['x'], region['y'], region['w'], region['h']
                frame_crop = frame[ry:ry+rh, rx:rx+rw]
            else:
                frame_crop = frame

            test_gray = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY)
            test_gray = cv2.resize(test_gray, IMG_SIZE)

            aligned, inliers = sift_align(master_gray, test_gray, sift)

            if aligned is None or inliers < MIN_INLIERS:
                align_warning = True
                print(f"  SIFT inliers: {inliers}  — too low, reposition label")
            else:
                ssim_score, fine_pct, coarse_pct, diff_map, diff_mask_map = \
                    compute_scores(master_gray, aligned)
                aligned_img = aligned
                ssim_ok   = ssim_score >= args.ssim
                fine_ok   = fine_pct   <  args.diff
                coarse_ok = coarse_pct <  args.cdiff
                result    = "GOOD" if (ssim_ok and fine_ok and coarse_ok) else "BAD"

                print(f"  SSIM        : {ssim_score:.4f}")
                print(f"  Fine diff   : {fine_pct:.2f}%   (text/print)")
                print(f"  Coarse diff : {coarse_pct:.2f}%   (structure/hologram position)")
                print(f"  Inliers     : {inliers}")
                print(f"  Result      : {'✅' if result == 'GOOD' else '❌'} {result}\n")

        # ── S: save cropped region ─────────────────────────────────────
        elif key == ord('s') or key == ord('S'):
            fname = f"saved_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            if region:
                rx, ry, rw, rh = region['x'], region['y'], region['w'], region['h']
                save_frame = frame[ry:ry+rh, rx:rx+rw]
            else:
                save_frame = frame.copy()
            cv2.imwrite(fname, save_frame)
            print(f"Saved: {fname}")

        # ── R: reset ──────────────────────────────────────────────────
        elif key == ord('r') or key == ord('R'):
            result        = None
            align_warning = False
            aligned_img   = None
            diff_map      = None
            diff_mask_map = None
            cv2.destroyWindow("Result — Master | Aligned | Diff | Heatmap")

        # ── ESC: quit ─────────────────────────────────────────────────
        elif key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

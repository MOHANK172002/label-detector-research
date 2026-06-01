"""
Live Label Checker — SIFT Align + Pixel Diff
=============================================
Real-time webcam label verification against master reference.

How it works:
  1. Load master reference image
  2. Open camera — show live feed
  3. Press SPACE → capture frame → run SIFT align + diff
  4. Show GOOD / BAD result on screen

Controls:
    SPACE   = capture current frame and check
    R       = reset / clear result
    ESC     = quit

Usage:
    python3 live_check.py
    python3 live_check.py --ref reference/good2.jpeg
    python3 live_check.py --ref reference/good2.jpeg --camera 0
    python3 live_check.py --ssim 0.73 --diff 10
"""

import cv2
import numpy as np
import argparse
import json
import os
from skimage.metrics import structural_similarity as ssim_fn

REGION_FILE = "region-technova.json"
MASK_FILE   = "mask-technova.json"

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE        = (800, 600)
SSIM_PASS       = 0.73
DIFF_AREA_FAIL  = 6.0
RATIO_TEST      = 0.75
MIN_INLIERS     = 80    # below this → alignment too weak → ask to reposition


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

def compute_scores(master_gray, aligned, hologram_mask_rect=None):
    if aligned.shape != master_gray.shape:
        aligned = cv2.resize(aligned, (master_gray.shape[1], master_gray.shape[0]))

    score, diff = ssim_fn(master_gray, aligned, full=True)
    diff_mask   = (diff < 0.5)

    # Apply hologram exclusion zone — zero out that rectangle in the diff mask
    if hologram_mask_rect:
        mx, my, mw, mh = hologram_mask_rect
        # Clamp to image bounds
        H, W = diff_mask.shape
        y1, y2 = max(0, my), min(H, my + mh)
        x1, x2 = max(0, mx), min(W, mx + mw)
        diff_mask[y1:y2, x1:x2] = False

    diff_area_pct = diff_mask.sum() / diff_mask.size * 100
    return score, diff_area_pct, diff, diff_mask


# ── 4-panel preview (same as sift_align_diff.py) ─────────────────────────────

def draw_4panel(master_gray, aligned, diff, diff_mask,
                ssim_score, diff_area_pct, inliers, result, ssim_ok, diff_ok):
    """
    4-panel grid:
      Top-left  : Master
      Top-right : Aligned test
      Bot-left  : Diff pixels (red = different)
      Bot-right : SSIM heatmap
    """
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

    # Panel 3 — Diff pixels
    p3 = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)
    p3[diff_mask] = (0, 0, 220)
    cv2.putText(p3, f"DIFF  {diff_area_pct:.1f}%", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2)

    # Panel 4 — SSIM heatmap
    p4 = cv2.applyColorMap((diff * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.putText(p4, "HEATMAP (BLUE=diff)", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Resize each panel to fixed height
    panel_h = 280
    def rs(img):
        s = panel_h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * s), panel_h))

    grid = np.vstack([np.hstack([rs(p1), rs(p2)]),
                      np.hstack([rs(p3), rs(p4)])])

    # Header bar — separate, no overlap
    bar_h = 85
    bar   = np.zeros((bar_h, grid.shape[1], 3), dtype=np.uint8)
    bar[:] = (25, 25, 25)
    cv2.putText(bar, f"Inliers: {inliers}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)

    ssim_col = (0, 220, 0) if ssim_ok else (0, 0, 220)
    diff_col = (0, 220, 0) if diff_ok  else (0, 0, 220)
    cv2.putText(bar, f"SSIM: {ssim_score:.3f}",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.75, ssim_col, 2)
    cv2.putText(bar, f"Diff: {diff_area_pct:.1f}%",
                (230, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.75, diff_col, 2)
    cv2.putText(bar, f"->  {result}",
                (430, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

    return np.vstack([bar, grid])


def draw_idle_overlay(frame):
    """Overlay shown while waiting for capture."""
    h, w = frame.shape[:2]

    # Bottom hint bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 45), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, "SPACE = Check label    ESC = Quit",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    return frame


def draw_align_warning(frame, inliers):
    """Shown when SIFT cannot align — too few inliers."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (20, 20, 80), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    cv2.putText(frame, "HOLD LABEL CLOSER & FLAT",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 200, 255), 3)
    cv2.putText(frame, f"Inliers: {inliers}  (need >= {MIN_INLIERS}  —  too much background)",
                (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 255), 1)
    # Orange border
    cv2.rectangle(frame, (6, 6), (w - 6, h - 6), (0, 140, 255), 6)
    return frame


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref",    default="reference/1.jpeg")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--ssim",   type=float, default=SSIM_PASS)
    parser.add_argument("--diff",   type=float, default=DIFF_AREA_FAIL)
    args = parser.parse_args()

    # ── Load master ────────────────────────────────────────────────────
    master_bgr = cv2.imread(args.ref)
    if master_bgr is None:
        print(f"ERROR: reference not found: {args.ref}")
        return

    master_gray = cv2.cvtColor(master_bgr, cv2.COLOR_BGR2GRAY)
    master_gray = cv2.resize(master_gray, IMG_SIZE)

    sift = cv2.SIFT_create(nfeatures=5000)

    # ── Load region ────────────────────────────────────────────────────
    region = None
    if os.path.exists(REGION_FILE):
        with open(REGION_FILE) as f:
            region = json.load(f)
        print(f"Region    : x={region['x']} y={region['y']} "
              f"w={region['w']} h={region['h']}  (from {REGION_FILE})")
    else:
        print(f"No {REGION_FILE} found — using full frame.")
        print(f"Run python3 setup_region.py first to define the label area.")

    # ── Load hologram mask ──────────────────────────────────────────────
    # mask.json coords are in camera-crop space; scale to IMG_SIZE for diff use
    holo_mask_cam  = None
    holo_mask_rect = None
    if os.path.exists(MASK_FILE):
        with open(MASK_FILE) as f:
            m = json.load(f)
        holo_mask_cam = (m['x'], m['y'], m['w'], m['h'])
        # Determine crop dimensions (same logic used in SPACE handler)
        if region:
            crop_w, crop_h = region['w'], region['h']
            # mask coords are relative to full frame — subtract region origin
            rel_x = m['x'] - region['x']
            rel_y = m['y'] - region['y']
        else:
            cap_tmp = cv2.VideoCapture(args.camera)
            ret_tmp, frm_tmp = cap_tmp.read()
            crop_h, crop_w = frm_tmp.shape[:2] if ret_tmp else (600, 800)
            cap_tmp.release()
            rel_x, rel_y = m['x'], m['y']
        sx = IMG_SIZE[0] / crop_w
        sy = IMG_SIZE[1] / crop_h
        holo_mask_rect = (int(rel_x * sx), int(rel_y * sy),
                          int(m['w'] * sx), int(m['h'] * sy))
        print(f"Hologram  : mask x={m['x']} y={m['y']} w={m['w']} h={m['h']}"
              f"  scaled→{holo_mask_rect}  (from {MASK_FILE})  — excluded from diff")
    else:
        print(f"No {MASK_FILE} — hologram zone not excluded.")

    print(f"\nReference : {args.ref}")
    print(f"Threshold : SSIM >= {args.ssim}  AND  diff < {args.diff}%  → GOOD")
    print(f"Camera    : {args.camera}")
    print(f"\nSPACE = check label    ESC = quit\n")

    # ── Open camera ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera {args.camera}")
        return

    # State
    result        = None
    ssim_score    = 0.0
    diff_area_pct = 0.0
    inliers       = 0
    align_warning = False
    aligned_img   = None
    diff_map      = None
    diff_mask_map = None
    ssim_ok       = False
    diff_ok       = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: cannot read from camera")
            break

        display = frame.copy()

        # Draw saved region box on live feed
        if region:
            rx, ry, rw, rh = region['x'], region['y'], region['w'], region['h']
            cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
            cv2.putText(display, "Label area", (rx, ry - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

        # Draw hologram mask zone on live feed
        if holo_mask_cam:
            mx, my, mw, mh = holo_mask_cam
            cv2.rectangle(display, (mx, my), (mx + mw, my + mh), (0, 140, 255), 2)
            cv2.putText(display, "Hologram (ignored)", (mx, my - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)

        if align_warning:
            display = draw_align_warning(display, inliers)
            h = display.shape[0]
            cv2.putText(display, "SPACE = try again    ESC = quit",
                        (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (180, 180, 180), 1)
        else:
            display = draw_idle_overlay(display)

        cv2.imshow("Live Feed", display)

        # 4-panel result window — shown separately when result is ready
        if result is not None and aligned_img is not None:
            panel = draw_4panel(master_gray, aligned_img, diff_map, diff_mask_map,
                                ssim_score, diff_area_pct, inliers, result,
                                ssim_ok, diff_ok)
            h_panel = panel.shape[0]
            cv2.putText(panel, "SPACE = check again    R = reset    ESC = quit",
                        (10, h_panel - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (150, 150, 150), 1)
            cv2.imshow("Result — Master | Aligned | Diff | Heatmap", panel)

        key = cv2.waitKey(1) & 0xFF

        # ── SPACE: capture and check ───────────────────────────────────
        if key == ord(' '):
            print("Checking...")
            result        = None
            align_warning = False
            ssim_score    = 0.0
            diff_area_pct = 0.0
            inliers       = 0

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
                ssim_score, diff_area_pct, diff_map, diff_mask_map = \
                    compute_scores(master_gray, aligned, holo_mask_rect)
                aligned_img = aligned
                ssim_ok = ssim_score    >= args.ssim
                diff_ok = diff_area_pct <  args.diff
                result  = "GOOD" if (ssim_ok and diff_ok) else "BAD"
                print(f"  SSIM      : {ssim_score:.4f}")
                print(f"  Diff area : {diff_area_pct:.2f}%")
                print(f"  Inliers   : {inliers}")
                print(f"  Result    : {'✅' if result == 'GOOD' else '❌'} {result}\n")

        # ── S: save cropped region only ───────────────────────────────
        elif key == ord('s') or key == ord('S'):
            import time
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

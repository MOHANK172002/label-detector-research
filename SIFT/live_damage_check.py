"""
SIFT Live Label Damage Checker
================================
Same algorithm as sift_align_diff.py — SIFT align + single-scale SSIM pixel diff —
runs on a live camera/stream with a fixed region and tracker.

Pipeline:
  SETUP (press S — run once):
    Draw a box around the label area in the camera view → saved to region.json
    Then press M to capture the reference master from that region → master.jpg

  INSPECTION (automatic):
    1. Fixed region is cropped from every frame
    2. LabelTracker detects when something stable enters the region
    3. When stable (or press C):
         a. Crop the region
         b. SIFT align → master coordinate space
         c. SSIM pixel diff (pixels where map < 0.5 = different)
         d. GOOD if SSIM >= SSIM_PASS  AND  diff% < DIFF_AREA_FAIL
    4. 4-panel result: Master | Aligned | Diff | Heatmap
    5. Crop saved to SIFT/results/good/ or SIFT/results/bad/

Controls:
  S       = setup region (draw box on live feed)
  X       = draw mask zone to ignore in diff (e.g. hologram, barcode)
  M       = capture current region crop as master reference
  C       = manually check region right now
  R       = reset tracker + last result
  Q / ESC = quit

Usage:
  python3 SIFT/live_damage_check.py
  python3 SIFT/live_damage_check.py --camera 0
  python3 SIFT/live_damage_check.py --ref SIFT/reference/master.jpg
  python3 SIFT/live_damage_check.py --ssim 0.75 --diff 6
"""

import cv2
import numpy as np
import json
import os
import time
import argparse
from skimage.metrics import structural_similarity as ssim_fn

# ── Stream ─────────────────────────────────────────────────────────────────────
DEFAULT_URL = (
    "tcp://192.168.1.11:8888"
    "?fflags=nobuffer&flags=low_delay&framedrop=1"
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
REF_PATH     = os.path.join(BASE_DIR, "reference", "master.jpg")
REGION_FILE  = os.path.join(BASE_DIR, "region-test.json")
MASK_FILE    = os.path.join(BASE_DIR, "mask-test.json")
GOOD_DIR     = os.path.join(BASE_DIR, "results", "good")
BAD_DIR      = os.path.join(BASE_DIR, "results", "bad")

# ── SIFT / SSIM  (same defaults as sift_align_diff.py) ────────────────────────
IMG_SIZE       = (800, 600)
SSIM_PASS      = 0.75       # SSIM >= this → pass
DIFF_AREA_FAIL = 8.0        # diff% < this → pass
RATIO_TEST     = 0.75       # Lowe's ratio test
MIN_INLIERS    = 40         # min RANSAC inliers to accept alignment

# ── Tracker ────────────────────────────────────────────────────────────────────
IOU_SAME       = 0.35
IOU_NEW        = 0.10
CHECK_COOLDOWN = 2.0        # seconds between auto-checks

# ── Auto-check toggle ──────────────────────────────────────────────────────────
AUTO_CHECK = True           # False = only check on C key press


# ══════════════════════════════════════════════════════════════════════════════
# REGION SETUP  (from setup_region.py — Step 1 only, no mask)
# ══════════════════════════════════════════════════════════════════════════════

_drawing  = False
_start_pt = (0, 0)
_end_pt   = (0, 0)

def _mouse_cb(event, x, y, flags, param):
    global _drawing, _start_pt, _end_pt
    if event == cv2.EVENT_LBUTTONDOWN:
        _drawing  = True
        _start_pt = _end_pt = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and _drawing:
        _end_pt = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        _drawing = False
        _end_pt  = (x, y)

def _get_rect(p1, p2):
    x = min(p1[0], p2[0])
    y = min(p1[1], p2[1])
    return x, y, abs(p2[0] - p1[0]), abs(p2[1] - p1[1])

def run_region_setup(cap):
    """
    Draw a box on the live feed to define the label region.
    Returns (x, y, w, h) on confirm, None on cancel.
    Saves result to region.json.
    """
    global _start_pt, _end_pt
    _start_pt = _end_pt = (0, 0)

    WIN = "SETUP — Draw box around LABEL AREA  (ENTER=confirm  R=redraw  ESC=cancel)"
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, _mouse_cb)
    print("\n[Setup] Draw a box around the label area, then press ENTER.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        disp = frame.copy()
        fh, fw = disp.shape[:2]

        x, y, bw, bh = _get_rect(_start_pt, _end_pt)
        if bw > 5 and bh > 5:
            cv2.rectangle(disp, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(disp, f"{bw}x{bh}  ({x},{y})",
                        (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(disp, "Drag box around LABEL AREA",
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(disp, "ENTER/SPACE = confirm    R = redraw    ESC = cancel",
                    (10, fh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow(WIN, disp)
        key = cv2.waitKey(1) & 0xFF

        if key in (13, ord(' ')):
            x, y, bw, bh = _get_rect(_start_pt, _end_pt)
            if bw < 20 or bh < 20:
                print("  Too small — draw again.")
                continue
            cv2.destroyWindow(WIN)
            region = {"x": x, "y": y, "w": bw, "h": bh}
            with open(REGION_FILE, "w") as f:
                json.dump(region, f, indent=2)
            print(f"[Setup] Region saved: x={x} y={y} w={bw} h={bh}  → {REGION_FILE}")
            return region
        elif key in (ord('r'), ord('R')):
            _start_pt = _end_pt = (0, 0)
        elif key == 27:
            cv2.destroyWindow(WIN)
            print("[Setup] Cancelled.")
            return None

    cv2.destroyWindow(WIN)
    return None


def run_mask_setup(cap, region):
    """
    Draw a box over the zone to IGNORE during diff (e.g. hologram, barcode).
    Stored as relative fractions of the region so it scales with any region size.
    Returns mask dict or None on cancel. Saves to mask-test.json.
    """
    global _start_pt, _end_pt
    _start_pt = _end_pt = (0, 0)

    rx, ry, rw, rh = region["x"], region["y"], region["w"], region["h"]
    WIN = "SETUP — Draw MASK zone to IGNORE  (ENTER=confirm  R=redraw  ESC=cancel)"
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, _mouse_cb)
    print("\n[Mask] Draw a box over the zone to IGNORE, then press ENTER.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        disp = frame.copy()
        fh, fw = disp.shape[:2]

        # Always show region reference in green
        cv2.rectangle(disp, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
        cv2.putText(disp, "Label region", (rx, ry - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Current drawn mask box in orange
        x, y, bw, bh = _get_rect(_start_pt, _end_pt)
        if bw > 5 and bh > 5:
            cv2.rectangle(disp, (x, y), (x + bw, y + bh), (0, 140, 255), 2)
            cv2.putText(disp, f"MASK  {bw}x{bh}",
                        (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

        cv2.putText(disp, "Draw box over zone to IGNORE in diff",
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
        cv2.putText(disp, "ENTER/SPACE = confirm    R = redraw    ESC = cancel",
                    (10, fh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow(WIN, disp)
        key = cv2.waitKey(1) & 0xFF

        if key in (13, ord(' ')):
            x, y, bw, bh = _get_rect(_start_pt, _end_pt)
            if bw < 10 or bh < 10:
                print("  Too small — draw again.")
                continue
            cv2.destroyWindow(WIN)
            # Store as relative offsets from region top-left, as fraction of region size
            mask = {
                "rx": (x - rx) / rw,
                "ry": (y - ry) / rh,
                "rw": bw / rw,
                "rh": bh / rh,
            }
            with open(MASK_FILE, "w") as f:
                json.dump(mask, f, indent=2)
            print(f"[Mask] Saved: rx={mask['rx']:.3f} ry={mask['ry']:.3f} "
                  f"rw={mask['rw']:.3f} rh={mask['rh']:.3f}  → {MASK_FILE}")
            return mask
        elif key in (ord('r'), ord('R')):
            _start_pt = _end_pt = (0, 0)
        elif key == 27:
            cv2.destroyWindow(WIN)
            print("[Mask] Cancelled.")
            return None

    cv2.destroyWindow(WIN)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# TRACKER  (inspector.py algorithm — tracks stability inside the region)
# ══════════════════════════════════════════════════════════════════════════════

class LabelTracker:
    """
    Detects when the region content is stable across frames.
    Compares mean brightness of successive crops using a simple diff threshold.
    WAITING → STABLE → CHECKED → WAITING
    """
    def __init__(self):
        self.state      = "WAITING"
        self.last_mean  = None
        self.last_check = 0.0
        self.stable_since = 0.0

    def update(self, crop):
        """
        Pass the current region crop each frame.
        Returns True when the content is stable and ready to check.
        """
        now  = time.time()
        mean = float(crop.mean())

        if self.last_mean is None:
            self.last_mean    = mean
            self.stable_since = now
            self.state        = "WAITING"
            return False

        diff = abs(mean - self.last_mean)
        self.last_mean = mean

        if diff > 2.0:
            # Content changed — reset stability timer
            self.stable_since = now
            self.state        = "WAITING"
            return False

        # Content is stable
        stable_secs = now - self.stable_since
        if stable_secs < 0.5:
            self.state = "WAITING"
            return False

        self.state = "STABLE"

        if (now - self.last_check) > CHECK_COOLDOWN:
            self.state      = "CHECKED"
            self.last_check = now
            return True

        return False

    def get_state(self):
        return self.state

    def reset(self):
        self.state      = "WAITING"
        self.last_mean  = None
        self.last_check = 0.0
        self.stable_since = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# SIFT ALIGN  (sift_align_diff.py — unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def sift_align(master_gray, test_gray, sift):
    kp1, des1 = sift.detectAndCompute(master_gray, None)
    kp2, des2 = sift.detectAndCompute(test_gray,   None)
    if des1 is None or des2 is None or len(des1) < 8 or len(des2) < 8:
        return None, 0
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    try:
        raw = flann.knnMatch(des1, des2, k=2)
    except cv2.error:
        return None, 0
    good = [m for m, n in raw if m.distance < RATIO_TEST * n.distance]
    if len(good) < 8:
        return None, 0
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    matrix, hmask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    if matrix is None:
        return None, 0
    inliers = int(hmask.sum()) if hmask is not None else 0
    aligned = cv2.warpPerspective(test_gray, matrix, IMG_SIZE)
    return aligned, inliers


# ══════════════════════════════════════════════════════════════════════════════
# MASK HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def mask_to_pixels(mask_rel):
    """Convert relative mask offsets to pixel coords on the IMG_SIZE canvas."""
    if mask_rel is None:
        return None
    return {
        "x": max(0, int(mask_rel["rx"] * IMG_SIZE[0])),
        "y": max(0, int(mask_rel["ry"] * IMG_SIZE[1])),
        "w": max(1, int(mask_rel["rw"] * IMG_SIZE[0])),
        "h": max(1, int(mask_rel["rh"] * IMG_SIZE[1])),
    }


def apply_mask_to_pair(img_a, img_b, mask_px):
    """Zero out the mask zone on both IMG_SIZE grayscale images in-place copies."""
    if mask_px is None:
        return img_a, img_b
    a = img_a.copy()
    b = img_b.copy()
    x1 = mask_px["x"]
    y1 = mask_px["y"]
    x2 = min(a.shape[1], x1 + mask_px["w"])
    y2 = min(a.shape[0], y1 + mask_px["h"])
    a[y1:y2, x1:x2] = 0
    b[y1:y2, x1:x2] = 0
    return a, b


# ══════════════════════════════════════════════════════════════════════════════
# SCORING  (sift_align_diff.py algorithm + optional mask)
# ══════════════════════════════════════════════════════════════════════════════

def compute_scores(master_gray, aligned, mask_px=None):
    """
    SSIM pixel diff — pixels where SSIM map < 0.5 counted as different.
    mask_px zone is zeroed on both images before comparison and excluded
    from the diff% count.
    """
    if aligned.shape != master_gray.shape:
        aligned = cv2.resize(aligned, (master_gray.shape[1], master_gray.shape[0]))

    m, a = apply_mask_to_pair(master_gray, aligned, mask_px)
    ssim_score, diff = ssim_fn(m, a, full=True)

    # Exclude mask zone from diff count by setting it to 1.0 (no difference)
    if mask_px is not None:
        x1, y1 = mask_px["x"], mask_px["y"]
        x2 = min(diff.shape[1], x1 + mask_px["w"])
        y2 = min(diff.shape[0], y1 + mask_px["h"])
        diff[y1:y2, x1:x2] = 1.0

    diff_mask     = diff < 0.5
    diff_area_pct = diff_mask.sum() / diff_mask.size * 100
    return ssim_score, diff_area_pct, diff, diff_mask


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

TRACKER_COLOR = {
    "WAITING": (180, 180, 180),
    "STABLE":  (0, 220, 255),
    "CHECKED": (0, 200, 0),
}


def draw_live(frame, region, mask_rel, tracker_state,
              last_verdict, master_loaded, region_loaded):
    out = frame.copy()
    fh, fw = frame.shape[:2]

    if region_loaded:
        rx, ry, rw, rh = region["x"], region["y"], region["w"], region["h"]
        color = TRACKER_COLOR.get(tracker_state, (0, 255, 0))
        cv2.rectangle(out, (rx, ry), (rx + rw, ry + rh), color, 2)
        cv2.putText(out, f"REGION  {tracker_state}",
                    (rx, ry - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Draw mask zone in orange (scaled back to frame coords)
        if mask_rel is not None:
            mx = int(rx + mask_rel["rx"] * rw)
            my = int(ry + mask_rel["ry"] * rh)
            mw = int(mask_rel["rw"] * rw)
            mh = int(mask_rel["rh"] * rh)
            cv2.rectangle(out, (mx, my), (mx + mw, my + mh), (0, 140, 255), 2)
            cv2.putText(out, "MASK (ignored)", (mx, my - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1)
    else:
        cv2.putText(out, "No region — press S to set up",
                    (fw // 2 - 180, fh // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 140, 255), 2)

    # Status bar
    if last_verdict:
        bc = (0, 80, 0) if last_verdict == "GOOD" else (0, 0, 80)
        cv2.rectangle(out, (0, fh - 55), (fw, fh), bc, -1)
        cv2.putText(out, f"LAST: {last_verdict}   S=setup  M=master  C=check  R=reset  Q=quit",
                    (10, fh - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        if not region_loaded:
            msg = "Press S to draw the label region"
            bc  = (50, 30, 10)
        elif not master_loaded:
            msg = "Region set — press M to capture master reference"
            bc  = (60, 20, 20)
        else:
            msg = "Ready — place label in region   S=setup  M=master  C=check  Q=quit"
            bc  = (25, 25, 25)
        cv2.rectangle(out, (0, fh - 38), (fw, fh), bc, -1)
        cv2.putText(out, msg, (10, fh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return out


def draw_result_panel(master_gray, aligned, diff, diff_mask,
                      ssim_score, diff_pct, inliers,
                      verdict, ssim_ok, diff_ok, ssim_thresh, diff_thresh):
    """4-panel result — same layout as sift_align_diff.py draw_preview."""
    color = (0, 220, 0) if verdict == "GOOD" else (0, 0, 220)
    h = master_gray.shape[0]

    p1 = cv2.cvtColor(master_gray, cv2.COLOR_GRAY2BGR)
    cv2.putText(p1, "MASTER", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    p2 = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)
    cv2.putText(p2, "ALIGNED TEST", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    p3 = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)
    p3[diff_mask] = (0, 0, 220)
    cv2.putText(p3, f"DIFF  {diff_pct:.1f}%", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2)

    p4 = cv2.applyColorMap((diff * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.putText(p4, "HEATMAP (BLUE=diff)", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    ph = 280
    def rs(img):
        s = ph / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * s), ph))

    grid = np.vstack([np.hstack([rs(p1), rs(p2)]),
                      np.hstack([rs(p3), rs(p4)])])

    bar = np.full((90, grid.shape[1], 3), 25, dtype=np.uint8)
    sc  = (0, 220, 0) if ssim_ok else (0, 0, 220)
    dc  = (0, 220, 0) if diff_ok  else (0, 0, 220)

    cv2.putText(bar, f"Inliers: {inliers}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    cv2.putText(bar, f"SSIM: {ssim_score:.3f}",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, sc, 2)
    cv2.putText(bar, f"Diff: {diff_pct:.1f}%",
                (230, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, dc, 2)
    cv2.putText(bar, f">>  {verdict}",
                (430, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    reason = []
    if not ssim_ok: reason.append(f"SSIM {ssim_score:.3f} < {ssim_thresh}")
    if not diff_ok: reason.append(f"diff {diff_pct:.1f}% >= {diff_thresh}%")
    if reason:
        cv2.putText(bar, "  [" + "  &  ".join(reason) + "]",
                    (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1)

    return np.vstack([bar, grid])


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════

def save_result(crop_bgr, verdict):
    folder = GOOD_DIR if verdict == "GOOD" else BAD_DIR
    os.makedirs(folder, exist_ok=True)
    ts  = time.strftime("%Y%m%d_%H%M%S")
    ms  = int((time.time() % 1) * 1000)
    out = os.path.join(folder, f"{verdict}_{ts}_{ms:03d}.jpg")
    cv2.imwrite(out, crop_bgr)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# CHECK HELPER
# ══════════════════════════════════════════════════════════════════════════════

def crop_region(frame, region):
    rx, ry, rw, rh = region["x"], region["y"], region["w"], region["h"]
    # Clamp to frame bounds
    x1 = max(0, rx)
    y1 = max(0, ry)
    x2 = min(frame.shape[1], rx + rw)
    y2 = min(frame.shape[0], ry + rh)
    return frame[y1:y2, x1:x2]


def run_check(frame, region, master_gray, sift, ssim_thresh, diff_thresh,
              mask_px=None, tag=""):
    """Crop region from frame, SIFT align, score. Returns (verdict, panel, crop)."""
    crop      = crop_region(frame, region)
    test_gray = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), IMG_SIZE)

    aligned, inliers = sift_align(master_gray, test_gray, sift)
    if aligned is None or inliers < MIN_INLIERS:
        print(f"  {tag}Low inliers ({inliers}) — alignment failed, skip.")
        return None, None, crop

    ssim_score, diff_pct, diff_map, diff_mask = compute_scores(
        master_gray, aligned, mask_px)
    ssim_ok = ssim_score >= ssim_thresh
    diff_ok = diff_pct   <  diff_thresh
    verdict = "GOOD" if (ssim_ok and diff_ok) else "BAD"

    fname = save_result(crop, verdict)
    mask_note = "  [mask active]" if mask_px else ""
    print(f"  {tag}[{verdict}]  SSIM={ssim_score:.3f}  "
          f"Diff={diff_pct:.1f}%  Inliers={inliers}{mask_note}  → {fname}")

    panel = draw_result_panel(
        master_gray, aligned, diff_map, diff_mask,
        ssim_score, diff_pct, inliers,
        verdict, ssim_ok, diff_ok, ssim_thresh, diff_thresh)

    return verdict, panel, crop


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SIFT live label damage checker")
    parser.add_argument("--ref",    default=REF_PATH,
                        help="Master reference image")
    parser.add_argument("--camera", type=int, default=None,
                        help="Webcam index (default: TCP stream)")
    parser.add_argument("--ssim",   type=float, default=SSIM_PASS,
                        help=f"Min SSIM to pass (default {SSIM_PASS})")
    parser.add_argument("--diff",   type=float, default=DIFF_AREA_FAIL,
                        help=f"Max diff%% to pass (default {DIFF_AREA_FAIL})")
    args = parser.parse_args()

    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
    else:
        cap = cv2.VideoCapture(DEFAULT_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("ERROR: cannot open stream/camera")
        return

    sift = cv2.SIFT_create(nfeatures=5000)

    # ── Load region ───────────────────────────────────────────────────
    region        = None
    region_loaded = False
    if os.path.exists(REGION_FILE):
        with open(REGION_FILE) as f:
            region = json.load(f)
        region_loaded = True
        print(f"Region loaded : x={region['x']} y={region['y']} "
              f"w={region['w']} h={region['h']}  ({REGION_FILE})")
    else:
        print("No region — press S to set up the label area.")

    # ── Load mask ─────────────────────────────────────────────────────
    mask_rel = None
    mask_px  = None
    if os.path.exists(MASK_FILE):
        with open(MASK_FILE) as f:
            d = json.load(f)
        if "rx" in d:
            mask_rel = d
        elif region_loaded:
            # Old absolute format — convert to relative using region
            rx, ry, rw, rh = region["x"], region["y"], region["w"], region["h"]
            mask_rel = {
                "rx": (d["x"] - rx) / rw,
                "ry": (d["y"] - ry) / rh,
                "rw": d["w"] / rw,
                "rh": d["h"] / rh,
            }
            with open(MASK_FILE, "w") as f:
                json.dump(mask_rel, f, indent=2)
            print(f"Mask converted from old format → {MASK_FILE}")
        if mask_rel is not None:
            mask_px = mask_to_pixels(mask_rel)
            print(f"Mask loaded   : rx={mask_rel['rx']:.3f} ry={mask_rel['ry']:.3f} "
                  f"rw={mask_rel['rw']:.3f} rh={mask_rel['rh']:.3f}  ({MASK_FILE})")
    else:
        print("No mask — press X to draw a zone to ignore (optional).")

    # ── Load master ───────────────────────────────────────────────────
    master_gray   = None
    master_loaded = False
    if os.path.exists(args.ref):
        m = cv2.imread(args.ref)
        if m is not None:
            master_gray   = cv2.resize(cv2.cvtColor(m, cv2.COLOR_BGR2GRAY), IMG_SIZE)
            master_loaded = True
            print(f"Master loaded : {args.ref}")
    if not master_loaded:
        print("No master — press M to capture region as reference.")

    print(f"Threshold  : SSIM >= {args.ssim}  AND  diff < {args.diff}%  → GOOD")
    print(f"AUTO_CHECK : {AUTO_CHECK}")
    print("S=setup  X=mask  M=master  C=check  R=reset  Q=quit\n")

    tracker      = LabelTracker()
    last_verdict = None
    result_panel = None

    WIN_LIVE   = "SIFT Live Damage Checker"
    WIN_RESULT = "Result — Master | Aligned | Diff | Heatmap"
    cv2.namedWindow(WIN_LIVE,   cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_RESULT, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        # ── Tracker update using region crop ──────────────────────────
        should_check = False
        if region_loaded:
            region_crop  = crop_region(frame, region)
            should_check = tracker.update(region_crop)

        # ── Auto-check ────────────────────────────────────────────────
        if should_check and AUTO_CHECK and master_loaded and region_loaded:
            verdict, panel, _ = run_check(
                frame, region, master_gray, sift, args.ssim, args.diff, mask_px)
            if verdict is not None:
                last_verdict = verdict
                result_panel = panel

        # ── Draw live feed ────────────────────────────────────────────
        display = draw_live(frame, region, mask_rel, tracker.get_state(),
                            last_verdict, master_loaded, region_loaded)
        cv2.imshow(WIN_LIVE, display)

        if result_panel is not None:
            cv2.imshow(WIN_RESULT, result_panel)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break

        elif key in (ord('r'), ord('R')):
            tracker.reset()
            last_verdict = None
            result_panel = None
            print("Reset.")

        elif key in (ord('s'), ord('S')):
            new_region = run_region_setup(cap)
            if new_region is not None:
                region        = new_region
                region_loaded = True
                # Clear mask — it was relative to old region
                mask_rel = None
                mask_px  = None
                if os.path.exists(MASK_FILE):
                    os.remove(MASK_FILE)
                tracker.reset()
                last_verdict  = None
                result_panel  = None
                print("Region updated — press X for mask, M to capture new master.")

        elif key in (ord('x'), ord('X')):
            if not region_loaded:
                print("No region — press S first.")
            else:
                new_mask = run_mask_setup(cap, region)
                if new_mask is not None:
                    mask_rel = new_mask
                    mask_px  = mask_to_pixels(mask_rel)
                    tracker.reset()
                    last_verdict = None
                    result_panel = None

        elif key in (ord('m'), ord('M')):
            if not region_loaded:
                print("No region — press S first.")
            else:
                crop          = crop_region(frame, region)
                master_gray   = cv2.resize(
                    cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), IMG_SIZE)
                master_loaded = True
                os.makedirs(os.path.dirname(args.ref), exist_ok=True)
                cv2.imwrite(args.ref, crop)
                tracker.reset()
                last_verdict  = None
                result_panel  = None
                print(f"Master captured → {args.ref}")

        elif key in (ord('c'), ord('C')):
            if not region_loaded:
                print("No region — press S first.")
            elif not master_loaded:
                print("No master — press M first.")
            else:
                verdict, panel, _ = run_check(
                    frame, region, master_gray, sift,
                    args.ssim, args.diff, mask_px, tag="C/")
                if verdict is not None:
                    last_verdict = verdict
                    result_panel = panel

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

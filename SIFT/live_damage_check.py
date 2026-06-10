"""
SIFT Live Label Damage Checker
================================
Automatically detects the label each frame using HSV sat-suppress (crop_tool.py
algorithm), crops it, SIFT-aligns to a master reference, and scores with SSIM.

No fixed region or manual box draw needed — the label is found automatically.

Pipeline:
  1. Every frame: HSV sat-suppress finds the label bbox
  2. LabelTracker checks the crop is stable across frames
  3. When stable (or press C):
       a. Crop the detected label
       b. SIFT align → master coordinate space
       c. SSIM pixel diff (pixels where map < 0.5 = different)
       d. GOOD if SSIM >= SSIM_PASS  AND  diff% < DIFF_AREA_FAIL
  4. 4-panel result: Master | Aligned | Diff | Heatmap
  5. Crop saved to SIFT/results/good/ or SIFT/results/bad/

Controls:
  T       = toggle detector tuner window (live HSV/morph trackbars)
  M       = capture current detected label crop as master reference
  C       = manually trigger check right now
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REF_PATH = os.path.join(BASE_DIR, "reference", "master.jpg")
GOOD_DIR = os.path.join(BASE_DIR, "results", "good")
BAD_DIR  = os.path.join(BASE_DIR, "results", "bad")

# ── SIFT / SSIM ────────────────────────────────────────────────────────────────
IMG_SIZE       = (800, 600)
SSIM_PASS      = 0.75
DIFF_AREA_FAIL = 8.0
RATIO_TEST     = 0.75
MIN_INLIERS    = 40

# ── Tracker ────────────────────────────────────────────────────────────────────
CHECK_COOLDOWN = 2.0        # seconds between auto-checks
AUTO_CHECK     = True

# ── Label detector (from crop_tool.py) ────────────────────────────────────────
AD_S_MAX        = 241   # max saturation to be considered a label pixel
AD_V_MIN        = 107   # min brightness to be considered a label pixel
AD_MORPH_K      = 7     # morphology kernel size (odd)
AD_MIN_AREA_PCT = 4     # % of frame area — smaller blobs ignored
AD_PADDING      = 6     # pixels of padding added around detected bbox


# ══════════════════════════════════════════════════════════════════════════════
# DETECTOR TUNER  (live trackbar window — press T to open/close)
# ══════════════════════════════════════════════════════════════════════════════

WIN_TUNER = "Detector Tuner  (T=close)"

def _nothing(_):
    pass

def open_tuner():
    cv2.namedWindow(WIN_TUNER, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("S max",     WIN_TUNER, AD_S_MAX,        255, _nothing)
    cv2.createTrackbar("V min",     WIN_TUNER, AD_V_MIN,        255, _nothing)
    cv2.createTrackbar("Morph k",   WIN_TUNER, AD_MORPH_K,       31, _nothing)
    cv2.createTrackbar("Min area%", WIN_TUNER, AD_MIN_AREA_PCT,  50, _nothing)
    cv2.createTrackbar("Padding",   WIN_TUNER, AD_PADDING,       40, _nothing)

def close_tuner():
    try:
        cv2.destroyWindow(WIN_TUNER)
    except Exception:
        pass

def read_tuner():
    """Read current trackbar values. Returns a params dict."""
    k = cv2.getTrackbarPos("Morph k", WIN_TUNER)
    k = max(k + (0 if k % 2 == 1 else 1), 1)   # force odd, min 1
    return {
        "s_max":    cv2.getTrackbarPos("S max",     WIN_TUNER),
        "v_min":    cv2.getTrackbarPos("V min",     WIN_TUNER),
        "morph_k":  k,
        "min_area": cv2.getTrackbarPos("Min area%", WIN_TUNER),
        "padding":  cv2.getTrackbarPos("Padding",   WIN_TUNER),
    }


# ══════════════════════════════════════════════════════════════════════════════
# LABEL DETECTOR  (HSV sat-suppress — crop_tool.py algorithm)
# ══════════════════════════════════════════════════════════════════════════════

def detect_label(frame, params=None):
    """
    Detect the label rectangle in a bottom-lit cyan scene.
    params: dict from read_tuner(), or None to use module-level defaults.
    Returns {x, y, w, h} of the blob closest to the frame centre, or None.
    Also returns the binary mask (for tuner preview).
    """
    s_max    = params["s_max"]    if params else AD_S_MAX
    v_min    = params["v_min"]    if params else AD_V_MIN
    morph_k  = params["morph_k"]  if params else AD_MORPH_K
    min_area_pct = params["min_area"] if params else AD_MIN_AREA_PCT
    padding  = params["padding"]  if params else AD_PADDING

    fh, fw = frame.shape[:2]
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]

    low_sat    = cv2.inRange(s_ch, 0,     s_max)
    bright     = cv2.inRange(v_ch, v_min, 255)
    label_mask = cv2.bitwise_and(low_sat, bright)
    label_mask = cv2.bitwise_not(label_mask)

    k_size  = max(morph_k, 1)
    k_small = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    closed  = cv2.morphologyEx(label_mask, cv2.MORPH_CLOSE, k_small)

    big = max(k_size * 3 + 1, 11)
    if big % 2 == 0:
        big += 1
    k_big = cv2.getStructuringElement(cv2.MORPH_RECT, (big, big))
    solid = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_big)

    cnts, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = (min_area_pct / 100.0) * fh * fw
    cx_f, cy_f = fw // 2, fh // 2
    best_box  = None
    best_dist = float("inf")

    for cnt in cnts:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w > fw * 0.95 or h > fh * 0.95:
            continue
        if w < h * 0.3:
            continue
        dist = ((x + w // 2 - cx_f) ** 2 + (y + h // 2 - cy_f) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_box  = (x, y, w, h)

    if best_box is None:
        return None, solid

    x, y, w, h = best_box
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(fw - x, w + padding * 2)
    h = min(fh - y, h + padding * 2)
    return {"x": x, "y": y, "w": w, "h": h}, solid


# ══════════════════════════════════════════════════════════════════════════════
# TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class LabelTracker:
    """Detects when the region crop is stable across frames."""
    def __init__(self):
        self.state        = "WAITING"
        self.last_mean    = None
        self.last_check   = 0.0
        self.stable_since = 0.0

    def update(self, crop):
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
            self.stable_since = now
            self.state        = "WAITING"
            return False

        if (now - self.stable_since) < 0.5:
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
        self.state        = "WAITING"
        self.last_mean    = None
        self.last_check   = 0.0
        self.stable_since = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# SIFT ALIGN
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
# SCORING
# ══════════════════════════════════════════════════════════════════════════════

def compute_scores(master_gray, aligned):
    if aligned.shape != master_gray.shape:
        aligned = cv2.resize(aligned, (master_gray.shape[1], master_gray.shape[0]))
    ssim_score, diff  = ssim_fn(master_gray, aligned, full=True)
    diff_mask         = diff < 0.5
    diff_area_pct     = diff_mask.sum() / diff_mask.size * 100
    return ssim_score, diff_area_pct, diff, diff_mask


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
# DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

TRACKER_COLOR = {
    "WAITING": (180, 180, 180),
    "STABLE":  (0, 220, 255),
    "CHECKED": (0, 200, 0),
}


def draw_live(frame, region, tracker_state, last_verdict, master_loaded):
    out = frame.copy()
    fh, fw = frame.shape[:2]

    if region is not None:
        rx, ry, rw, rh = region["x"], region["y"], region["w"], region["h"]
        color = TRACKER_COLOR.get(tracker_state, (0, 255, 0))
        cv2.rectangle(out, (rx, ry), (rx + rw, ry + rh), color, 2)
        cv2.putText(out, f"LABEL  {tracker_state}  {rw}x{rh}px",
                    (rx, max(ry - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    else:
        cv2.putText(out, "No label detected",
                    (fw // 2 - 140, fh // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 140, 255), 2)

    if last_verdict:
        bc = (0, 80, 0) if last_verdict == "GOOD" else (0, 0, 80)
        cv2.rectangle(out, (0, fh - 55), (fw, fh), bc, -1)
        cv2.putText(out, f"LAST: {last_verdict}   T=tuner  M=master  C=check  R=reset  Q=quit",
                    (10, fh - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    else:
        if not master_loaded:
            msg = "Point camera at label, then press M to capture master"
            bc  = (60, 20, 20)
        else:
            msg = "Ready — T=tuner  M=master  C=check  R=reset  Q=quit"
            bc  = (25, 25, 25)
        cv2.rectangle(out, (0, fh - 38), (fw, fh), bc, -1)
        cv2.putText(out, msg, (10, fh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return out


def draw_result_panel(master_gray, aligned, diff, diff_mask,
                      ssim_score, diff_pct, inliers,
                      verdict, ssim_ok, diff_ok, ssim_thresh, diff_thresh):
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
# CHECK HELPER
# ══════════════════════════════════════════════════════════════════════════════

def crop_region(frame, region):
    rx, ry, rw, rh = region["x"], region["y"], region["w"], region["h"]
    x1 = max(0, rx)
    y1 = max(0, ry)
    x2 = min(frame.shape[1], rx + rw)
    y2 = min(frame.shape[0], ry + rh)
    return frame[y1:y2, x1:x2]


def run_check(frame, region, master_gray, sift, ssim_thresh, diff_thresh, tag=""):
    crop      = crop_region(frame, region)
    test_gray = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), IMG_SIZE)

    aligned, inliers = sift_align(master_gray, test_gray, sift)
    if aligned is None or inliers < MIN_INLIERS:
        print(f"  {tag}Low inliers ({inliers}) — alignment failed, skip.")
        return None, None, crop

    ssim_score, diff_pct, diff_map, diff_mask = compute_scores(master_gray, aligned)
    ssim_ok = ssim_score >= ssim_thresh
    diff_ok = diff_pct   <  diff_thresh
    verdict = "GOOD" if (ssim_ok and diff_ok) else "BAD"

    fname = save_result(crop, verdict)
    print(f"  {tag}[{verdict}]  SSIM={ssim_score:.3f}  "
          f"Diff={diff_pct:.1f}%  Inliers={inliers}  → {fname}")

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
        print("No master — point camera at label and press M to capture.")

    print(f"Threshold  : SSIM >= {args.ssim}  AND  diff < {args.diff}%  → GOOD")
    print("T=tuner  M=master  C=check  R=reset  Q=quit\n")

    tracker      = LabelTracker()
    last_verdict = None
    result_panel = None
    tuner_open   = False

    WIN_LIVE   = "SIFT Live Damage Checker"
    WIN_RESULT = "Result — Master | Aligned | Diff | Heatmap"
    cv2.namedWindow(WIN_LIVE,   cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_RESULT, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        # ── Read tuner params if window is open ───────────────────────
        params = read_tuner() if tuner_open else None

        # ── Detect label region every frame ───────────────────────────
        region, mask = detect_label(frame, params)

        # ── Show mask preview in tuner window ─────────────────────────
        if tuner_open:
            mask_disp = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            if region is not None:
                rx, ry, rw, rh = region["x"], region["y"], region["w"], region["h"]
                cv2.rectangle(mask_disp, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
            cv2.imshow(WIN_TUNER, mask_disp)

        # ── Tracker update ────────────────────────────────────────────
        should_check = False
        if region is not None:
            region_crop  = crop_region(frame, region)
            should_check = tracker.update(region_crop)
        else:
            tracker.reset()

        # ── Auto-check when stable ────────────────────────────────────
        if should_check and AUTO_CHECK and master_loaded and region is not None:
            verdict, panel, _ = run_check(
                frame, region, master_gray, sift, args.ssim, args.diff)
            if verdict is not None:
                last_verdict = verdict
                result_panel = panel

        # ── Draw ──────────────────────────────────────────────────────
        display = draw_live(frame, region, tracker.get_state(),
                            last_verdict, master_loaded)
        cv2.imshow(WIN_LIVE, display)

        if result_panel is not None:
            cv2.imshow(WIN_RESULT, result_panel)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break

        elif key in (ord('t'), ord('T')):
            tuner_open = not tuner_open
            if tuner_open:
                open_tuner()
                print("Tuner open — adjust S max / V min / Morph k / Min area% / Padding")
            else:
                close_tuner()
                print("Tuner closed.")

        elif key in (ord('r'), ord('R')):
            tracker.reset()
            last_verdict = None
            result_panel = None
            print("Reset.")

        elif key in (ord('m'), ord('M')):
            if region is None:
                print("No label detected — point camera at label first.")
            else:
                crop = crop_region(frame, region)
                master_gray   = cv2.resize(
                    cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), IMG_SIZE)
                master_loaded = True
                os.makedirs(os.path.dirname(args.ref), exist_ok=True)
                cv2.imwrite(args.ref, crop)
                tracker.reset()
                last_verdict = None
                result_panel = None
                print(f"Master captured → {args.ref}")

        elif key in (ord('c'), ord('C')):
            if region is None:
                print("No label detected.")
            elif not master_loaded:
                print("No master — press M first.")
            else:
                verdict, panel, _ = run_check(
                    frame, region, master_gray, sift,
                    args.ssim, args.diff, tag="C/")
                if verdict is not None:
                    last_verdict = verdict
                    result_panel = panel

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

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

try:
    import torch
    _TORCH_GPU = torch.cuda.is_available()
except ImportError:
    _TORCH_GPU = False

if _TORCH_GPU:
    print(f"[GPU] Zone SSIM on {torch.cuda.get_device_name(0)}")
else:
    print("[GPU] PyTorch/CUDA not available — zone SSIM on CPU")

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
DIFF_AREA_FAIL = 3.0
RATIO_TEST     = 0.75
MIN_INLIERS    = 40

# ── Tracker ────────────────────────────────────────────────────────────────────
CHECK_COOLDOWN = 0.3        # seconds between auto-checks
AUTO_CHECK     = True

# ── Feature flags ─────────────────────────────────────────────────────────────
ENABLE_PEN_CHECK     = False   # set True to enable pen/marker detection
ENABLE_WRINKLE_CHECK = False   # set True to enable wrinkle detection
ENABLE_ZONE_CHECK    = True    # detect missing text/logos via zoned SSIM

# ── Pen / mark detection ───────────────────────────────────────────────────────
PEN_SSIM_THRESH  = 0.55
PEN_MIN_AREA     = 400
PEN_MAX_AREA     = 8000
PEN_MIN_ASPECT   = 3.0
PEN_FAIL_COUNT   = 2

# ── Wrinkle detection (LBP texture) ───────────────────────────────────────────
WRINKLE_CELLS    = 16
WRINKLE_FAIL     = 0.30
WRINKLE_CHI_THR  = 0.20

# ── Zone-based missing content detection ──────────────────────────────────────
ZONE_COLS        = 40     # grid columns
ZONE_ROWS        = 40    # grid rows
ZONE_SSIM_THR    = 0.60  # per-zone SSIM below this = content missing in zone
ZONE_FAIL_COUNT  = 1    # how many bad zones before verdict = BAD
ZONE_EDGE_PCT    = 0.02 # fraction of label width/height to ignore at each edge

# ── Label detector (from crop_tool.py) ────────────────────────────────────────
AD_S_MAX        = 255   # max saturation to be considered a label pixel
AD_V_MIN        = 174   # min brightness to be considered a label pixel
AD_MORPH_K      = 7     # morphology kernel size (odd)
AD_MIN_AREA_PCT = 4     # % of frame area — smaller blobs ignored
AD_PADDING      = 0    # pixels of padding added around detected bbox


# ══════════════════════════════════════════════════════════════════════════════
# DETECTOR TUNER  (live trackbar window — press T to open/close)
# ══════════════════════════════════════════════════════════════════════════════

WIN_TUNER = "Detector Tuner  (T=close)"

def _nothing(_):
    pass

def open_tuner():
    cv2.namedWindow(WIN_TUNER, cv2.WINDOW_NORMAL)
    # ── Label detector ──────────────────────────────────────────────────
    cv2.createTrackbar("S max",           WIN_TUNER, AD_S_MAX,                     255, _nothing)
    cv2.createTrackbar("V min",           WIN_TUNER, AD_V_MIN,                     255, _nothing)
    cv2.createTrackbar("Morph k",         WIN_TUNER, AD_MORPH_K,                    31, _nothing)
    cv2.createTrackbar("Min area%",       WIN_TUNER, AD_MIN_AREA_PCT,               50, _nothing)
    cv2.createTrackbar("Padding",         WIN_TUNER, AD_PADDING,                    40, _nothing)
    # ── Pen / mark detection ────────────────────────────────────────────
    cv2.createTrackbar("Pen max area",    WIN_TUNER, PEN_MAX_AREA,               10000, _nothing)
    cv2.createTrackbar("Pen min aspect",  WIN_TUNER, int(PEN_MIN_ASPECT * 10),      80, _nothing)
    cv2.createTrackbar("Pen fail blobs",  WIN_TUNER, PEN_FAIL_COUNT,               10, _nothing)
    # ── Wrinkle detection ───────────────────────────────────────────────
    cv2.createTrackbar("Wrinkle fail%",   WIN_TUNER, int(WRINKLE_FAIL * 100),      100, _nothing)
    cv2.createTrackbar("Wrinkle chi",     WIN_TUNER, int(WRINKLE_CHI_THR * 100),   100, _nothing)

def close_tuner():
    try:
        cv2.destroyWindow(WIN_TUNER)
    except Exception:
        pass

def read_tuner():
    """Read current trackbar values. Returns a params dict."""
    k = cv2.getTrackbarPos("Morph k", WIN_TUNER)
    k = max(k + (0 if k % 2 == 1 else 1), 1)
    return {
        "s_max":          cv2.getTrackbarPos("S max",           WIN_TUNER),
        "v_min":          cv2.getTrackbarPos("V min",           WIN_TUNER),
        "morph_k":        k,
        "min_area":       cv2.getTrackbarPos("Min area%",       WIN_TUNER),
        "padding":        cv2.getTrackbarPos("Padding",         WIN_TUNER),
        "pen_max_area":   cv2.getTrackbarPos("Pen max area",    WIN_TUNER),
        "pen_min_aspect": cv2.getTrackbarPos("Pen min aspect",  WIN_TUNER) / 10.0,
        "pen_fail_count": cv2.getTrackbarPos("Pen fail blobs",  WIN_TUNER),
        "wrinkle_fail":   cv2.getTrackbarPos("Wrinkle fail%",   WIN_TUNER) / 100.0,
        "wrinkle_chi":    cv2.getTrackbarPos("Wrinkle chi",     WIN_TUNER) / 100.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# LABEL DETECTOR  (HSV sat-suppress — crop_tool.py algorithm)
# ══════════════════════════════════════════════════════════════════════════════

def detect_label(frame, params=None):
    """
    Detect the label rectangle using HSV sat-suppress, then fit a minAreaRect
    to get the oriented bounding box.  Returns a region dict with both the
    axis-aligned bbox (for drawing) and the 4-corner oriented quad (for
    perspective crop), plus the binary mask.

    region keys:
      x, y, w, h   – axis-aligned bbox (for drawing the live overlay)
      quad          – np.float32 shape (4,2) ordered TL,TR,BR,BL (for warp)
      quad_w, quad_h – canonical output size for the warp
    """
    s_max        = params["s_max"]    if params else AD_S_MAX
    v_min        = params["v_min"]    if params else AD_V_MIN
    morph_k      = params["morph_k"]  if params else AD_MORPH_K
    min_area_pct = params["min_area"] if params else AD_MIN_AREA_PCT
    padding      = params["padding"]  if params else AD_PADDING

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

    min_area   = (min_area_pct / 100.0) * fh * fw
    cx_f, cy_f = fw // 2, fh // 2
    best_cnt   = None
    best_dist  = float("inf")

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
            best_cnt  = cnt

    if best_cnt is None:
        return None, solid

    # ── Oriented bounding box via minAreaRect ─────────────────────────────────
    rect      = cv2.minAreaRect(best_cnt)   # (centre, (w,h), angle)
    box_pts   = cv2.boxPoints(rect)         # 4 corners, float32
    box_pts   = np.float32(box_pts)

    # Order corners: top-left, top-right, bottom-right, bottom-left
    # Sort by y to split top/bottom pair, then by x within each pair
    s   = box_pts.sum(axis=1)       # smallest sum = TL, largest = BR
    d   = np.diff(box_pts, axis=1)  # smallest diff = TR, largest = BL
    tl  = box_pts[np.argmin(s)]
    br  = box_pts[np.argmax(s)]
    tr  = box_pts[np.argmin(d)]
    bl  = box_pts[np.argmax(d)]
    quad = np.float32([tl, tr, br, bl])

    # Canonical output size = actual oriented rect dimensions (swap if angle rotated >45°)
    rw, rh = rect[1]
    if rw < rh:
        rw, rh = rh, rw
    quad_w = int(rw) + padding * 2
    quad_h = int(rh) + padding * 2

    # Axis-aligned bbox for live overlay rectangle
    ax, ay, aw, ah = cv2.boundingRect(box_pts.astype(np.int32))

    return {
        "x": ax, "y": ay, "w": aw, "h": ah,
        "quad": quad, "quad_w": quad_w, "quad_h": quad_h
    }, solid


def crop_oriented(frame, region):
    """
    Perspective-warp the oriented label quad to a clean upright rectangle.
    No background wedges at corners, no white bleed from axis-aligned edges.
    """
    quad   = region["quad"]
    out_w  = region["quad_w"]
    out_h  = region["quad_h"]

    dst = np.float32([
        [0,         0        ],
        [out_w - 1, 0        ],
        [out_w - 1, out_h - 1],
        [0,         out_h - 1],
    ])
    M    = cv2.getPerspectiveTransform(quad, dst)
    warp = cv2.warpPerspective(frame, M, (out_w, out_h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
    return warp


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
# SCORING — SSIM + PEN MARKS + WRINKLES
# ══════════════════════════════════════════════════════════════════════════════

def detect_pen_marks(aligned, ssim_diff, params=None):
    """
    Detect pen/marker strokes from the SSIM diff map.
    Works on the diff map (not raw pixel difference) so normal printing
    variations and slight misalignment don't trigger false positives.

    Pen strokes show as elongated diff blobs: high aspect ratio bounding rect.
    Returns (stroke_blob_count, overlay_bgr).
    """
    max_area   = params["pen_max_area"]   if params else PEN_MAX_AREA
    min_aspect = params["pen_min_aspect"] if params else PEN_MIN_ASPECT

    # Threshold the SSIM map — pixels where SSIM is LOW = something changed here
    # ssim_diff values: 1.0 = identical, 0.0 = completely different
    diff_bin = (ssim_diff < PEN_SSIM_THRESH).astype(np.uint8) * 255

    # Small close to connect broken stroke pixels
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    diff_bin = cv2.morphologyEx(diff_bin, cv2.MORPH_CLOSE, k)

    pen_mask = np.zeros_like(diff_bin)
    stroke_count = 0
    cnts, _ = cv2.findContours(diff_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < PEN_MIN_AREA or area > max_area:
            continue
        # Fit a rotated rect to get the true aspect ratio regardless of angle
        rect  = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        if rw == 0 or rh == 0:
            continue
        aspect = max(rw, rh) / min(rw, rh)
        if aspect >= min_aspect:
            cv2.drawContours(pen_mask, [cnt], -1, 255, -1)
            stroke_count += 1

    overlay = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)
    overlay[pen_mask > 0] = (255, 0, 255)
    cv2.putText(overlay, f"PEN  {stroke_count} stroke(s)",
                (10, aligned.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    return stroke_count, overlay


def _lbp(gray):
    """Compute a simple 8-neighbour LBP image (no external lib needed)."""
    h, w   = gray.shape
    g      = gray.astype(np.int16)
    lbp    = np.zeros((h, w), dtype=np.uint8)
    angles = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
    for bit, (dy, dx) in enumerate(angles):
        shifted = np.roll(np.roll(g, dy, axis=0), dx, axis=1)
        lbp    |= ((g >= shifted).astype(np.uint8) << bit)
    return lbp


def detect_wrinkles(master_gray, aligned, params=None):
    """
    Compare LBP texture histograms in a grid of cells.
    Wrinkles deform local surface texture even when average intensity is similar.
    Returns (bad_cell_fraction, heatmap_bgr).
    """
    chi_thr = params["wrinkle_chi"] if params else WRINKLE_CHI_THR
    cells   = WRINKLE_CELLS
    h, w    = master_gray.shape
    ch, cw  = h // cells, w // cells

    lbp_m = _lbp(master_gray)
    lbp_a = _lbp(aligned)

    heat   = np.zeros((h, w), dtype=np.float32)
    bad    = 0
    total  = 0

    for r in range(cells):
        for c in range(cells):
            y0, y1 = r * ch, (r + 1) * ch
            x0, x1 = c * cw, (c + 1) * cw
            hm, _  = np.histogram(lbp_m[y0:y1, x0:x1], bins=256, range=(0, 255))
            ha, _  = np.histogram(lbp_a[y0:y1, x0:x1], bins=256, range=(0, 255))
            hm     = hm.astype(np.float32) + 1e-6
            ha     = ha.astype(np.float32) + 1e-6
            hm    /= hm.sum();  ha /= ha.sum()
            # Chi-squared distance
            chi    = float(0.5 * np.sum((hm - ha) ** 2 / (hm + ha)))
            heat[y0:y1, x0:x1] = chi
            total += 1
            if chi > chi_thr:
                bad += 1

    score    = bad / total if total > 0 else 0.0
    heat_n   = np.clip(heat / max(heat.max(), 1e-6), 0, 1)
    # Gaussian blur removes hard block edges from the cell grid
    heat_smooth = cv2.GaussianBlur(heat_n, (31, 31), 0)
    heatmap  = cv2.applyColorMap((heat_smooth * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    cv2.putText(heatmap, f"WRINKLE  {score:.2f}",
                (10, aligned.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return score, heatmap


def detect_zone_missing(master_gray, aligned):
    """
    Divide the label into a grid, compare local SSIM per zone.
    Uses vectorised GPU (PyTorch) when available — 70x faster than CPU loop.
    Falls back to CPU skimage loop automatically if no GPU.
    Returns (bad_zone_count, zone_heatmap_bgr, list_of_bad_zone_rects).
    """
    h, w   = master_gray.shape
    edge_x = int(w * ZONE_EDGE_PCT)
    edge_y = int(h * ZONE_EDGE_PCT)
    cw     = (w - 2 * edge_x) // ZONE_COLS
    ch     = (h - 2 * edge_y) // ZONE_ROWS

    # region covered by the full zone grid
    gw = cw * ZONE_COLS
    gh = ch * ZONE_ROWS

    if _TORCH_GPU and cw >= 2 and ch >= 2:
        # ── GPU path: all 1600 zones in one tensor operation ──────────────
        m_f = torch.from_numpy(
                master_gray[edge_y:edge_y+gh, edge_x:edge_x+gw]
                .astype(np.float32) / 255.0).cuda()
        a_f = torch.from_numpy(
                aligned[edge_y:edge_y+gh, edge_x:edge_x+gw]
                .astype(np.float32) / 255.0).cuda()

        # reshape → (ZONE_ROWS, ZONE_COLS, ch*cw)
        m_b = m_f.reshape(ZONE_ROWS, ch, ZONE_COLS, cw).permute(0,2,1,3).reshape(ZONE_ROWS, ZONE_COLS, -1)
        a_b = a_f.reshape(ZONE_ROWS, ch, ZONE_COLS, cw).permute(0,2,1,3).reshape(ZONE_ROWS, ZONE_COLS, -1)

        mu1 = m_b.mean(-1);  mu2 = a_b.mean(-1)
        s1  = ((m_b - mu1.unsqueeze(-1)) ** 2).mean(-1)
        s2  = ((a_b - mu2.unsqueeze(-1)) ** 2).mean(-1)
        s12 = ((m_b - mu1.unsqueeze(-1)) * (a_b - mu2.unsqueeze(-1))).mean(-1)
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2*mu1*mu2 + C1) * (2*s12 + C2)) / \
                   ((mu1**2 + mu2**2 + C1) * (s1 + s2 + C2))

        ssim_np   = ssim_map.cpu().numpy()          # (ZONE_ROWS, ZONE_COLS)
        bad_mask  = ssim_np < ZONE_SSIM_THR
        bad_zones = int(bad_mask.sum())

        # build heat map on CPU from the GPU result
        heat = np.ones((h, w), dtype=np.float32)    # 1 = identical outside grid
        for r in range(ZONE_ROWS):
            for c in range(ZONE_COLS):
                x0 = edge_x + c * cw;  y0 = edge_y + r * ch
                heat[y0:y0+ch, x0:x0+cw] = ssim_np[r, c]

        bad_rects = [
            (edge_x + c*cw, edge_y + r*ch,
             edge_x + c*cw + cw, edge_y + r*ch + ch)
            for r in range(ZONE_ROWS) for c in range(ZONE_COLS)
            if bad_mask[r, c]
        ]

    else:
        # ── CPU fallback: skimage per-zone loop ───────────────────────────
        heat      = np.ones((h, w), dtype=np.float32)
        bad_zones = 0
        bad_rects = []
        for r in range(ZONE_ROWS):
            for c in range(ZONE_COLS):
                x0 = edge_x + c * cw;  y0 = edge_y + r * ch
                x1 = x0 + cw;          y1 = y0 + ch
                zm, za = master_gray[y0:y1, x0:x1], aligned[y0:y1, x0:x1]
                if zm.shape[0] < 8 or zm.shape[1] < 8:
                    continue
                win = min(zm.shape[0], zm.shape[1])
                win = win if win % 2 == 1 else win - 1
                score = ssim_fn(zm, za, win_size=max(win, 7))
                heat[y0:y1, x0:x1] = score
                if score < ZONE_SSIM_THR:
                    bad_zones += 1
                    bad_rects.append((x0, y0, x1, y1))

    # ── Build heatmap image ───────────────────────────────────────────────
    inverted = np.clip(1.0 - heat, 0, 1)
    hmap     = cv2.applyColorMap((inverted * 255).astype(np.uint8), cv2.COLORMAP_JET)
    for (x0, y0, x1, y1) in bad_rects:
        cv2.rectangle(hmap, (x0, y0), (x1, y1), (0, 0, 255), 2)

    label = f"ZONES  {bad_zones}/{ZONE_COLS * ZONE_ROWS} bad"
    if not ENABLE_ZONE_CHECK:
        label += "  (disabled)"
    cv2.putText(hmap, label, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 255) if bad_zones >= ZONE_FAIL_COUNT else (0, 255, 0), 2)
    return bad_zones, hmap, bad_rects


def compute_scores(master_gray, aligned, params=None):
    if aligned.shape != master_gray.shape:
        aligned = cv2.resize(aligned, (master_gray.shape[1], master_gray.shape[0]))

    ssim_score, diff = ssim_fn(master_gray, aligned, full=True)
    diff_mask        = diff < 0.5
    diff_area_pct    = diff_mask.sum() / diff_mask.size * 100

    pen_pixels,   pen_overlay     = detect_pen_marks(aligned, diff, params)
    wrinkle_score, wrinkle_map    = detect_wrinkles(master_gray, aligned, params)
    bad_zones,    zone_heatmap, _ = detect_zone_missing(master_gray, aligned)

    return (ssim_score, diff_area_pct, diff, diff_mask,
            pen_pixels, pen_overlay, wrinkle_score, wrinkle_map,
            bad_zones, zone_heatmap)


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
        color = TRACKER_COLOR.get(tracker_state, (0, 255, 0))
        if "quad" in region:
            pts = region["quad"].astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)
            rx, ry = int(region["quad"][:, 0].min()), int(region["quad"][:, 1].min())
        else:
            rx, ry, rw, rh = region["x"], region["y"], region["w"], region["h"]
            cv2.rectangle(out, (rx, ry), (rx + rw, ry + rh), color, 2)
        qw, qh = region.get("quad_w", region["w"]), region.get("quad_h", region["h"])
        cv2.putText(out, f"LABEL  {tracker_state}  {qw}x{qh}px",
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


def draw_result_panel(master_gray, aligned, diff_mask,
                      ssim_score, diff_pct, inliers,
                      pen_pixels, pen_overlay, wrinkle_score, wrinkle_map,
                      bad_zones, zone_heatmap,
                      verdict, ssim_ok, diff_ok, pen_ok, wrinkle_ok, zone_ok,
                      ssim_thresh, diff_thresh, pen_fail, wrinkle_fail):
    color = (0, 220, 0) if verdict == "GOOD" else (0, 0, 220)
    h = master_gray.shape[0]

    p1 = cv2.cvtColor(master_gray, cv2.COLOR_GRAY2BGR)
    cv2.putText(p1, "MASTER", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    p2 = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)
    cv2.putText(p2, "ALIGNED TEST", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    # p3: SSIM diff overlay (blue) + pen marks (magenta)
    p3 = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)
    p3[diff_mask] = (0, 0, 220)
    pen_bin = cv2.cvtColor(pen_overlay, cv2.COLOR_BGR2GRAY)
    pen_bin = (pen_bin == 0) & (pen_overlay[:, :, 0] == 255)
    p3[pen_bin] = (255, 0, 255)
    diff_label = f"DIFF {diff_pct:.1f}%"
    if ENABLE_PEN_CHECK:
        diff_label += f"  PEN {pen_pixels} blob(s)"
    cv2.putText(p3, diff_label, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2)

    # p4: zone missing-content heatmap (red = missing, blue = ok)
    p4 = zone_heatmap.copy()

    ph  = 280
    GAP = 6   # black gap between panels in pixels

    def rs(img):
        s = ph / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * s), ph))

    r1, r2, r3, r4 = rs(p1), rs(p2), rs(p3), rs(p4)
    pw = r1.shape[1]

    sep_v = np.zeros((ph,  GAP, 3), dtype=np.uint8)   # vertical divider
    sep_h = np.zeros((GAP, pw * 2 + GAP, 3), dtype=np.uint8)  # horizontal divider

    top_row = np.hstack([r1, sep_v, r2])
    bot_row = np.hstack([r3, sep_v, r4])
    grid    = np.vstack([top_row, sep_h, bot_row])

    W = grid.shape[1]

    # ── Large verdict banner ───────────────────────────────────────────────────
    banner = np.full((120, W, 3),
                     (0, 60, 0) if verdict == "GOOD" else (0, 0, 80),
                     dtype=np.uint8)
    txt_size, _ = cv2.getTextSize(verdict, cv2.FONT_HERSHEY_SIMPLEX, 3.5, 6)
    tx = (W - txt_size[0]) // 2
    cv2.putText(banner, verdict, (tx, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 3.5, color, 6, cv2.LINE_AA)

    # ── Metrics bar ───────────────────────────────────────────────────────────
    sc  = (0, 220, 0) if ssim_ok  else (0, 0, 220)
    dc  = (0, 220, 0) if diff_ok  else (0, 0, 220)
    zc  = (0, 220, 0) if zone_ok  else (0, 80, 255)
    dim = (120, 120, 120)

    bar = np.full((70, W, 3), 25, dtype=np.uint8)
    cv2.putText(bar, f"SSIM: {ssim_score:.3f}",
                (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, sc, 2)
    cv2.putText(bar, f"Diff: {diff_pct:.1f}%",
                (240, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, dc, 2)
    zone_label = f"Zones: {bad_zones}/{ZONE_COLS * ZONE_ROWS}"
    if not ENABLE_ZONE_CHECK:
        zone_label += " [OFF]"
    cv2.putText(bar, zone_label,
                (430, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                dim if not ENABLE_ZONE_CHECK else zc, 2)

    reason = []
    if not ssim_ok: reason.append(f"SSIM {ssim_score:.3f}<{ssim_thresh}")
    if not diff_ok: reason.append(f"diff {diff_pct:.1f}%>={diff_thresh}%")
    if ENABLE_ZONE_CHECK and not zone_ok:
        reason.append(f"missing content {bad_zones} zones>={ZONE_FAIL_COUNT}")
    if ENABLE_PEN_CHECK     and not pen_ok:     reason.append(f"pen {pen_pixels}>={pen_fail}")
    if ENABLE_WRINKLE_CHECK and not wrinkle_ok: reason.append(f"wrinkle {wrinkle_score:.2f}>={wrinkle_fail}")
    if reason:
        cv2.putText(bar, "BAD: " + "  |  ".join(reason),
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 255), 1)

    return np.vstack([banner, bar, grid])


# ══════════════════════════════════════════════════════════════════════════════
# CHECK HELPER
# ══════════════════════════════════════════════════════════════════════════════

def crop_region(frame, region):
    """
    Use the oriented perspective warp when quad is available (new path),
    fall back to simple axis-aligned slice otherwise.
    """
    if "quad" in region:
        return crop_oriented(frame, region)
    rx, ry, rw, rh = region["x"], region["y"], region["w"], region["h"]
    x1 = max(0, rx)
    y1 = max(0, ry)
    x2 = min(frame.shape[1], rx + rw)
    y2 = min(frame.shape[0], ry + rh)
    return frame[y1:y2, x1:x2]


def run_check(frame, region, master_gray, sift, ssim_thresh, diff_thresh,
              params=None, tag=""):
    crop      = crop_region(frame, region)
    test_gray = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), IMG_SIZE)

    aligned, inliers = sift_align(master_gray, test_gray, sift)
    if aligned is None or inliers < MIN_INLIERS:
        print(f"  {tag}Low inliers ({inliers}) — alignment failed, skip.")
        return None, None, crop

    (ssim_score, diff_pct, _, diff_mask,
     pen_pixels, pen_overlay,
     wrinkle_score, wrinkle_map,
     bad_zones, zone_heatmap) = compute_scores(master_gray, aligned, params)

    pen_fail     = params["pen_fail_count"] if params else PEN_FAIL_COUNT
    wrinkle_fail = params["wrinkle_fail"] if params else WRINKLE_FAIL

    ssim_ok    = ssim_score   >= ssim_thresh
    diff_ok    = diff_pct     <  diff_thresh
    pen_ok     = (pen_pixels < pen_fail) if ENABLE_PEN_CHECK     else True
    wrinkle_ok = (wrinkle_score < wrinkle_fail) if ENABLE_WRINKLE_CHECK else True
    zone_ok    = (bad_zones < ZONE_FAIL_COUNT) if ENABLE_ZONE_CHECK     else True
    verdict    = "GOOD" if (ssim_ok and diff_ok and pen_ok and wrinkle_ok and zone_ok) else "BAD"

    fname = save_result(crop, verdict)
    print(f"  {tag}[{verdict}]  SSIM={ssim_score:.3f}  Diff={diff_pct:.1f}%  "
          f"Zones={bad_zones}/{ZONE_COLS*ZONE_ROWS}  Pen={pen_pixels}  "
          f"Wrinkle={wrinkle_score:.2f}  Inliers={inliers}  → {fname}")

    panel = draw_result_panel(
        master_gray, aligned, diff_mask,
        ssim_score, diff_pct, inliers,
        pen_pixels, pen_overlay, wrinkle_score, wrinkle_map,
        bad_zones, zone_heatmap,
        verdict, ssim_ok, diff_ok, pen_ok, wrinkle_ok, zone_ok,
        ssim_thresh, diff_thresh, pen_fail, wrinkle_fail)

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
                if "quad" in region:
                    pts = region["quad"].astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(mask_disp, [pts], True, (0, 255, 0), 2)
                else:
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
                frame, region, master_gray, sift, args.ssim, args.diff, params)
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
                    args.ssim, args.diff, params, tag="C/")
                if verdict is not None:
                    last_verdict = verdict
                    result_panel = panel

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

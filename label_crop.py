"""
Label Auto-Crop from GS Camera (TCP Stream)
============================================
Detects individual VST labels by scanning the SILVER SPINE column.

The silver spine runs the full height of each label on the left edge.
Between labels the spine has a small gap where the green background shows.
Scanning that column for green pixels gives exact label top/bottom boundaries.

Controls:  S = save    D = debug    Q = quit
Usage:
    python3 label_crop.py
    python3 label_crop.py --auto
    python3 label_crop.py --camera 0
    python3 label_crop.py --out crops/
"""

import cv2
import numpy as np
import argparse
import os
import time

DEFAULT_URL = (
    "tcp://192.168.1.11:8888"
    "?fflags=nobuffer"
    "&flags=low_delay"
    "&framedrop=1"
)

# ── HSV ranges (measured from hsv_probe + saved frames) ───────────────────────
# Background green: H=70, S=245, V=122  →  high S is the key
BG_GREEN_LOW  = np.array([55, 150,  40])
BG_GREEN_HIGH = np.array([85, 255, 200])

# White label body: S < 50, V > 120
WHITE_LOW  = np.array([0,   0, 120])
WHITE_HIGH = np.array([180, 50, 255])

# ── Tuning ─────────────────────────────────────────────────────────────────────
CROP_W, CROP_H  = 800, 600
AUTO_COOLDOWN   = 2.5
EDGE_MARGIN     = 8
LABEL_MIN_H     = 80     # minimum label height in pixels
CENTER_ZONE_X   = 0.80
CENTER_ZONE_Y   = 0.80

# Spine scan: how wide a column to average when looking for green gaps
SPINE_SCAN_W    = 20     # px — average this many columns around spine center
# How many consecutive green rows make a real gap (not noise)
GAP_MIN_ROWS    = 4


# ── Find the spine column x-position ──────────────────────────────────────────

def find_spine_x(frame):
    """
    Find the left edge of the label strip by locating the rightmost column
    of the white/label region after masking out the background green.
    Returns the x-coordinate of the spine center, and the strip right edge.
    """
    fh = frame.shape[0]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Background green mask
    bg_mask = cv2.inRange(hsv, BG_GREEN_LOW, BG_GREEN_HIGH)

    # Label region = not background
    label_mask = cv2.bitwise_not(bg_mask)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    label_mask = cv2.morphologyEx(label_mask, cv2.MORPH_CLOSE, k)
    label_mask = cv2.morphologyEx(label_mask, cv2.MORPH_OPEN,  k)

    # Column projection: sum of label pixels per column
    col_sum = label_mask.sum(axis=0).astype(np.float32)

    # The strip occupies a contiguous band of columns with high label-pixel count
    threshold = fh * 255 * 0.3   # column has label pixels in >30% of rows
    label_cols = np.where(col_sum > threshold)[0]

    if len(label_cols) < 10:
        return None, None, label_mask

    strip_x_left  = int(label_cols[0])
    strip_x_right = int(label_cols[-1])

    # Spine is at the left edge of the strip
    spine_x = strip_x_left + SPINE_SCAN_W // 2

    return spine_x, (strip_x_left, strip_x_right), label_mask


# ── Find label boundaries by scanning the spine column for green gaps ──────────

def find_label_bounds(frame, spine_x):
    """
    Scan vertically along the spine column.
    Green pixels = background showing through the gap between labels.
    Clusters of green rows = label boundaries.
    Returns list of (top_y, bottom_y) for each label.
    """
    fh = frame.shape[0]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Average saturation across the spine scan band
    x0 = max(0, spine_x - SPINE_SCAN_W // 2)
    x1 = min(frame.shape[1], spine_x + SPINE_SCAN_W // 2)
    spine_hsv = hsv[:, x0:x1, :]          # shape (fh, band, 3)
    sat_col   = spine_hsv[:, :, 1].mean(axis=1)   # mean saturation per row

    # A row is "green/gap" if mean saturation is high (background green)
    is_gap = sat_col > 100   # background S=245, spine S=30-70

    # Find label regions = runs of non-gap rows
    bounds = []
    in_label   = False
    label_start = 0

    for y in range(fh):
        if not in_label and not is_gap[y]:
            in_label    = True
            label_start = y
        elif in_label and is_gap[y]:
            in_label = False
            label_h  = y - label_start
            if label_h >= LABEL_MIN_H:
                bounds.append((label_start, y))

    # Handle label that reaches bottom of frame
    if in_label:
        label_h = fh - label_start
        if label_h >= LABEL_MIN_H:
            bounds.append((label_start, fh))

    return bounds


# ── Build label boxes ──────────────────────────────────────────────────────────

def build_boxes(bounds, strip_x_left, strip_x_right):
    """Convert (top_y, bottom_y) bounds into (cx, cy, lx, ly, lw, lh) boxes."""
    boxes = []
    lx = strip_x_left
    lw = strip_x_right - strip_x_left
    for (ly, ly2) in bounds:
        lh = ly2 - ly
        cx = lx + lw // 2
        cy = ly + lh // 2
        boxes.append((cx, cy, lx, ly, lw, lh))
    return boxes


# ── Pick the label closest to frame center ────────────────────────────────────

def pick_center_box(boxes, fw, fh):
    cx_f = fw // 2
    cy_f = fh // 2
    zx   = fw * CENTER_ZONE_X
    zy   = fh * CENTER_ZONE_Y

    best, best_dist = None, float("inf")
    for box in boxes:
        cx, cy, lx, ly, lw, lh = box
        if abs(cx - cx_f) > zx or abs(cy - cy_f) > zy:
            continue
        dist = ((cx - cx_f)**2 + (cy - cy_f)**2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best      = box
    return best


# ── Warp crop ─────────────────────────────────────────────────────────────────

def warp_crop(frame, lx, ly, lw, lh, out_w=CROP_W, out_h=CROP_H):
    src = np.array([[lx, ly], [lx+lw, ly], [lx+lw, ly+lh], [lx, ly+lh]],
                   dtype=np.float32)
    dst = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, M, (out_w, out_h))


# ── Overlay ───────────────────────────────────────────────────────────────────

def draw_overlay(frame, boxes, center_box, spine_x, strip_edges, fw, fh):
    out = frame.copy()

    # Spine scan line
    if spine_x:
        cv2.line(out, (spine_x, 0), (spine_x, fh), (0, 220, 220), 1)

    # Strip boundary
    if strip_edges:
        sl, sr = strip_edges
        cv2.line(out, (sl, 0), (sl, fh), (180, 180, 0), 1)
        cv2.line(out, (sr, 0), (sr, fh), (180, 180, 0), 1)

    # All detected label boxes
    for i, (_, _, lx, ly, lw, lh) in enumerate(boxes):
        fully = (lx > EDGE_MARGIN and ly > EDGE_MARGIN and
                 lx+lw < fw-EDGE_MARGIN and ly+lh < fh-EDGE_MARGIN)
        color = (255, 255, 255) if fully else (0, 140, 255)
        tag   = f"#{i+1} {'FULL' if fully else 'PARTIAL'}  {lw}x{lh}px"

        cv2.rectangle(out, (lx, ly), (lx+lw, ly+lh), color, 2)

        # Corner ticks
        t = 14
        for (px, py, dx, dy) in [(lx, ly, 1, 1), (lx+lw, ly, -1, 1),
                                  (lx+lw, ly+lh, -1, -1), (lx, ly+lh, 1, -1)]:
            cv2.line(out, (px, py), (px+dx*t, py),     color, 2)
            cv2.line(out, (px, py), (px, py+dy*t),     color, 2)

        cv2.putText(out, tag, (lx+4, max(ly-5, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Selected center label — bright green
    if center_box:
        _, _, lx, ly, lw, lh = center_box
        cv2.rectangle(out, (lx, ly), (lx+lw, ly+lh), (0, 255, 0), 3)
        t = 20
        for (px, py, dx, dy) in [(lx, ly, 1, 1), (lx+lw, ly, -1, 1),
                                  (lx+lw, ly+lh, -1, -1), (lx, ly+lh, 1, -1)]:
            cv2.line(out, (px, py), (px+dx*t, py),     (0, 255, 0), 3)
            cv2.line(out, (px, py), (px, py+dy*t),     (0, 255, 0), 3)
        cv2.putText(out, f"CROP THIS  {lw}x{lh}px",
                    (lx+4, max(ly-8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    # Frame center crosshair
    cx_f, cy_f = fw//2, fh//2
    cv2.line(out, (cx_f-30, cy_f), (cx_f+30, cy_f), (0, 200, 255), 2)
    cv2.line(out, (cx_f, cy_f-30), (cx_f, cy_f+30), (0, 200, 255), 2)
    cv2.circle(out, (cx_f, cy_f), 5, (0, 200, 255), -1)

    return out


# ── Save ──────────────────────────────────────────────────────────────────────

def save_crop(crop, out_dir, prefix="label"):
    os.makedirs(out_dir, exist_ok=True)
    ts    = time.strftime("%Y%m%d_%H%M%S")
    ms    = int((time.time() % 1) * 1000)
    fname = os.path.join(out_dir, f"{prefix}_{ts}_{ms:03d}.jpg")
    cv2.imwrite(fname, crop)
    return fname


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",    default=DEFAULT_URL)
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--out",    default="crops")
    parser.add_argument("--auto",   action="store_true")
    args = parser.parse_args()

    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        src = f"camera {args.camera}"
    else:
        cap = cv2.VideoCapture(args.url, cv2.CAP_FFMPEG)
        src = args.url

    if not cap.isOpened():
        print(f"ERROR: cannot open {src}")
        return

    print(f"Source : {src}")
    print(f"Output : {args.out}/")
    print(f"\nS = save    D = debug mask    Q = quit\n")

    debug      = False
    debug_open = False
    last_save  = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        fh, fw = frame.shape[:2]

        # ── Detect ────────────────────────────────────────────────────
        spine_x, strip_edges, label_mask = find_spine_x(frame)

        boxes      = []
        center_box = None

        if spine_x and strip_edges:
            sl, sr     = strip_edges
            bounds     = find_label_bounds(frame, spine_x)
            boxes      = build_boxes(bounds, sl, sr)
            center_box = pick_center_box(boxes, fw, fh)

        # ── Display ───────────────────────────────────────────────────
        display = draw_overlay(frame, boxes, center_box, spine_x, strip_edges, fw, fh)

        status = f"Labels: {len(boxes)}"
        if center_box:
            status += "  |  GREEN = ready  (S to save)"
        else:
            status += "  |  No label detected"
        cv2.putText(display, status, (10, fh-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2)

        cv2.imshow("Label Crop", display)

        if debug:
            cv2.imshow("[D] Label mask", label_mask)
            debug_open = True
        elif debug_open:
            cv2.destroyWindow("[D] Label mask")
            debug_open = False

        if center_box:
            _, _, lx, ly, lw, lh = center_box
            cv2.imshow("Crop Preview", warp_crop(frame, lx, ly, lw, lh))

        # ── Auto-save ─────────────────────────────────────────────────
        now = time.time()
        if args.auto and center_box and (now - last_save) > AUTO_COOLDOWN:
            _, _, lx, ly, lw, lh = center_box
            fname = save_crop(warp_crop(frame, lx, ly, lw, lh), args.out)
            print(f"[AUTO] {fname}")
            last_save = now

        # ── Keys ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key in (ord('s'), ord('S')):
            if center_box:
                _, _, lx, ly, lw, lh = center_box
                fname = save_crop(warp_crop(frame, lx, ly, lw, lh), args.out)
                print(f"Saved: {fname}")
                last_save = now
            else:
                print("No label detected.")
        elif key in (ord('d'), ord('D')):
            debug = not debug
            print(f"Debug: {'ON' if debug else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

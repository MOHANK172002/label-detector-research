"""
Label Inspector — Detected BBox Crop + Tracker + Area Check + SIFT/SSIM
========================================================================
Pipeline:

  SETUP (press M):
    1. Draw box around one label on live feed        → region.json
       region.json also stores master_area (w*h px)
    2. Optionally draw hologram mask on that label   → mask.json
       mask.json stores RELATIVE offsets from label top-left corner
       (not absolute frame coords) so it applies to any detected bbox
    3. Crop that region → master.jpg

  INSPECTION (automatic):
    1. V-threshold detects individual label bboxes
    2. Tracker: each detected bbox is tracked across frames by IOU
       When a NEW label enters the center zone and is FULLY visible:
         a. Area check  — compare detected w*h vs master_area (tolerance %)
         b. Crop the detected bbox directly (not fixed region)
         c. Scale the hologram mask relative to this crop size
         d. SIFT align + two-scale SSIM (with mask applied)
         e. GOOD / BAD decision → save to results/

Controls:
  M = setup (draw region + mask, capture master)
  R = reset last result
  Q = quit
"""

import cv2
import numpy as np
import argparse
import json
import os
import time
from skimage.metrics import structural_similarity as ssim_fn

DEFAULT_URL = (
    "tcp://192.168.1.11:8888"
    "?fflags=nobuffer&flags=low_delay&framedrop=1"
)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
REGION_FILE = os.path.join(BASE_DIR, "region.json")
MASK_FILE   = os.path.join(BASE_DIR, "mask.json")
REF_DIR     = os.path.join(BASE_DIR, "reference")
GOOD_DIR    = os.path.join(BASE_DIR, "results", "good")
BAD_DIR     = os.path.join(BASE_DIR, "results", "bad")

# ── V-threshold detection ──────────────────────────────────────────────────────
V_LOW      = 151
V_HIGH     = 255
MORPH_K    = 9
MIN_AREA_P = 3       # % of frame area

# ── Center zone trigger ────────────────────────────────────────────────────────
CENTER_ZONE = 0.35   # fraction of half-frame; label center must be within this

# ── Area tolerance ─────────────────────────────────────────────────────────────
AREA_TOL_PCT = 20.0  # ± % difference from master area to pass area check

# ── SIFT / SSIM ────────────────────────────────────────────────────────────────
IMG_SIZE        = (800, 600)
SSIM_PASS       = 0.73
FINE_FAIL_PCT   = 3.0
COARSE_FAIL_PCT = 5.0
FINE_BLUR       = 3
COARSE_BLUR     = 31
RATIO_TEST      = 0.75
MIN_INLIERS     = 80

# ── Tracker ────────────────────────────────────────────────────────────────────
IOU_SAME    = 0.35   # IoU above this → same label as previous
IOU_NEW     = 0.10   # IoU below this → definitely a new label
CHECK_COOLDOWN = 2.0 # min seconds between checks


# ══════════════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════════════

_drawing  = False
_start_pt = (0, 0)
_end_pt   = (0, 0)

def _mouse_cb(event, x, y, flags, param):
    global _drawing, _start_pt, _end_pt
    if event == cv2.EVENT_LBUTTONDOWN:
        _drawing = True
        _start_pt = _end_pt = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and _drawing:
        _end_pt = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        _drawing = False
        _end_pt = (x, y)

def _get_rect(p1, p2):
    x = min(p1[0], p2[0])
    y = min(p1[1], p2[1])
    return x, y, abs(p2[0]-p1[0]), abs(p2[1]-p1[1])

def _draw_box_step(cap, win_title, color, instruction, hint, existing_box=None):
    global _start_pt, _end_pt
    _start_pt = _end_pt = (0, 0)
    cv2.namedWindow(win_title)
    cv2.setMouseCallback(win_title, _mouse_cb)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        disp = frame.copy()
        fh, fw = disp.shape[:2]
        if existing_box:
            ex, ey, ew, eh = existing_box
            cv2.rectangle(disp, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
            cv2.putText(disp, "Label area", (ex,ey-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        bx, by, bw, bh = _get_rect(_start_pt, _end_pt)
        if bw > 5 and bh > 5:
            cv2.rectangle(disp, (bx,by), (bx+bw,by+bh), color, 2)
            cv2.putText(disp, f"{bw}x{bh}", (bx, by-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(disp, instruction, (10,32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(disp, hint, (10, fh-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        cv2.imshow(win_title, disp)
        key = cv2.waitKey(1) & 0xFF
        if key in (13, ord(' ')):
            bx, by, bw, bh = _get_rect(_start_pt, _end_pt)
            if bw < 20 or bh < 20:
                print("Too small — draw again.")
                continue
            cv2.destroyWindow(win_title)
            return (bx, by, bw, bh)
        elif key in (ord('r'), ord('R')):
            _start_pt = _end_pt = (0, 0)
        elif key == 27:
            cv2.destroyWindow(win_title)
            return None
    cv2.destroyWindow(win_title)
    return None


def run_setup(cap):
    """
    Step 1: draw region box → region.json (includes master_area)
    Step 2: draw hologram mask → mask.json (stored as RELATIVE offsets)
    Step 3: capture master.jpg from that region
    Returns (master_gray, region, mask_rel) or (None, None, None) on cancel.
    """
    print("\n── Step 1: Draw box around one GOOD label ────────────────────")
    rect = _draw_box_step(
        cap,
        win_title   = "SETUP Step 1 — Draw around LABEL",
        color       = (0, 255, 0),
        instruction = "Drag box around the FULL LABEL",
        hint        = "ENTER/SPACE=confirm   R=redraw   ESC=cancel",
    )
    if rect is None:
        print("Setup cancelled.")
        return None, None, None

    x, y, w, h = rect
    region = {"x": x, "y": y, "w": w, "h": h, "master_area": w * h}
    with open(REGION_FILE, "w") as f:
        json.dump(region, f, indent=2)
    print(f"Region saved: {w}x{h} at ({x},{y})  area={w*h}px²  → {REGION_FILE}")

    # Step 2 — hologram mask stored as RELATIVE offset from label top-left
    mask_rel = None
    ans = input("\nDoes this label have a hologram/reflective zone to ignore? (y/n): ").strip().lower()
    if ans in ("y", "yes"):
        print("\n── Step 2: Draw box over the HOLOGRAM zone ──────────────────")
        mrect = _draw_box_step(
            cap,
            win_title    = "SETUP Step 2 — Draw over HOLOGRAM zone",
            color        = (0, 140, 255),
            instruction  = "Drag box over HOLOGRAM (will be IGNORED in diff)",
            hint         = "ENTER/SPACE=confirm   R=redraw   ESC=skip",
            existing_box = (x, y, w, h),
        )
        if mrect:
            mx, my, mw, mh = mrect
            # Store as relative offset from label top-left, as fraction of label size
            mask_rel = {
                "rx": (mx - x) / w,   # left offset as fraction of label width
                "ry": (my - y) / h,   # top offset as fraction of label height
                "rw": mw / w,         # mask width as fraction of label width
                "rh": mh / h,         # mask height as fraction of label height
            }
            with open(MASK_FILE, "w") as f:
                json.dump(mask_rel, f, indent=2)
            print(f"Mask saved (relative): rx={mask_rel['rx']:.3f} ry={mask_rel['ry']:.3f} "
                  f"rw={mask_rel['rw']:.3f} rh={mask_rel['rh']:.3f}  → {MASK_FILE}")
        else:
            print("Hologram mask skipped.")
            if os.path.exists(MASK_FILE):
                os.remove(MASK_FILE)
    else:
        if os.path.exists(MASK_FILE):
            os.remove(MASK_FILE)
        print("No hologram mask.")

    # Step 3 — capture master
    print("\n── Step 3: Capturing master ──────────────────────────────────")
    ret, frame = cap.read()
    if not ret:
        print("ERROR: cannot read frame.")
        return None, region, mask_rel

    crop = frame[y:y+h, x:x+w]
    master_gray = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), IMG_SIZE)
    os.makedirs(REF_DIR, exist_ok=True)
    mpath = os.path.join(REF_DIR, "master.jpg")
    cv2.imwrite(mpath, crop)
    print(f"Master saved: {mpath}  ({w}x{h}px)")
    print("Setup complete.\n")
    return master_gray, region, mask_rel


# ══════════════════════════════════════════════════════════════════════════════
# MASK — dynamic, relative to any detected label bbox
# ══════════════════════════════════════════════════════════════════════════════

def make_mask_for_imgsize(mask_rel):
    """
    Convert relative mask offsets to pixel coords on the IMG_SIZE canvas.
    Returns dict with pixel x,y,w,h at IMG_SIZE scale, or None.
    """
    if mask_rel is None:
        return None
    return {
        "x": int(mask_rel["rx"] * IMG_SIZE[0]),
        "y": int(mask_rel["ry"] * IMG_SIZE[1]),
        "w": max(1, int(mask_rel["rw"] * IMG_SIZE[0])),
        "h": max(1, int(mask_rel["rh"] * IMG_SIZE[1])),
    }


def apply_mask(img, mask_px):
    """Zero-out the mask region on an IMG_SIZE grayscale image."""
    if mask_px is None:
        return img
    out = img.copy()
    mx, my, mw, mh = mask_px["x"], mask_px["y"], mask_px["w"], mask_px["h"]
    # Clamp to image bounds
    x1, y1 = max(0, mx), max(0, my)
    x2, y2 = min(img.shape[1], mx+mw), min(img.shape[0], my+mh)
    out[y1:y2, x1:x2] = 0
    return out


def mask_overlay_on_frame(frame, lx, ly, lw, lh, mask_rel):
    """Draw the mask box on the live frame at the detected label position."""
    if mask_rel is None:
        return
    mx = int(lx + mask_rel["rx"] * lw)
    my = int(ly + mask_rel["ry"] * lh)
    mw = int(mask_rel["rw"] * lw)
    mh = int(mask_rel["rh"] * lh)
    cv2.rectangle(frame, (mx,my), (mx+mw,my+mh), (0,140,255), 2)
    cv2.putText(frame, "mask", (mx, my-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,140,255), 1)


# ══════════════════════════════════════════════════════════════════════════════
# LABEL DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_labels(frame):
    fh, fw = frame.shape[:2]
    v_ch   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]
    mask   = cv2.inRange(v_ch, V_LOW, V_HIGH)
    k      = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_K, MORPH_K))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    flood_m = np.zeros((fh+2, fw+2), dtype=np.uint8)
    flooded = mask.copy()
    cv2.floodFill(flooded, flood_m, (0, 0), 255)
    mask = mask | cv2.bitwise_not(flooded)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = (MIN_AREA_P / 100.0) * fh * fw
    boxes = []
    for cnt in cnts:
        if cv2.contourArea(cnt) < min_area:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bw > fw*0.95 or bh > fh*0.95:
            continue
        if bw < bh * 0.5:
            continue
        boxes.append((bx, by, bw, bh))
    boxes.sort(key=lambda b: b[1])
    return boxes


def pick_center(boxes, fw, fh):
    cx_f, cy_f = fw//2, fh//2
    zx, zy = fw * CENTER_ZONE, fh * CENTER_ZONE
    best_i, best_d = None, float("inf")
    for i, (x, y, w, h) in enumerate(boxes):
        if x < 4 or y < 4 or x+w > fw-4 or y+h > fh-4:
            continue
        cx, cy = x+w//2, y+h//2
        if abs(cx-cx_f) > zx or abs(cy-cy_f) > zy:
            continue
        d = ((cx-cx_f)**2 + (cy-cy_f)**2) ** 0.5
        if d < best_d:
            best_d, best_i = d, i
    return best_i


# ══════════════════════════════════════════════════════════════════════════════
# TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class LabelTracker:
    """
    Tracks a single label across frames using IoU.
    State machine: WAITING → IN_ZONE → CHECKED → WAITING
    """
    def __init__(self):
        self.state      = "WAITING"   # WAITING | IN_ZONE | CHECKED
        self.last_box   = None        # (x,y,w,h) of tracked label
        self.last_check = 0.0

    def _iou(self, a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix = max(0, min(ax+aw, bx+bw) - max(ax, bx))
        iy = max(0, min(ay+ah, by+bh) - max(ay, by))
        inter = ix * iy
        union = aw*ah + bw*bh - inter
        return inter / union if union > 0 else 0.0

    def update(self, center_box):
        """
        Call every frame with the currently detected center box (or None).
        Returns True if this is a NEW label that should be checked now.
        """
        now = time.time()

        if center_box is None:
            # Nothing in center — reset if we were tracking
            if self.state in ("IN_ZONE", "CHECKED"):
                self.state    = "WAITING"
                self.last_box = None
            return False

        if self.state == "WAITING":
            # New label entering the zone
            self.state    = "IN_ZONE"
            self.last_box = center_box
            return False   # wait one frame for stability

        if self.state == "IN_ZONE":
            iou = self._iou(center_box, self.last_box)
            if iou > IOU_SAME:
                # Same label, stable in zone — check it if cooldown passed
                self.last_box = center_box
                if (now - self.last_check) > CHECK_COOLDOWN:
                    self.state      = "CHECKED"
                    self.last_check = now
                    return True
            else:
                # Different label arrived — restart
                self.last_box = center_box
            return False

        if self.state == "CHECKED":
            iou = self._iou(center_box, self.last_box)
            if iou < IOU_NEW:
                # Clearly a new label
                self.state    = "IN_ZONE"
                self.last_box = center_box
            else:
                self.last_box = center_box
            return False

        return False

    def get_state(self):
        return self.state


# ══════════════════════════════════════════════════════════════════════════════
# AREA CHECK
# ══════════════════════════════════════════════════════════════════════════════

def area_check(detected_w, detected_h, master_area, tol_pct=AREA_TOL_PCT):
    """
    Compare detected label area to master area with tolerance.
    Returns (passed, detected_area, area_diff_pct).
    """
    detected_area = detected_w * detected_h
    if master_area <= 0:
        return True, detected_area, 0.0
    diff_pct = abs(detected_area - master_area) / master_area * 100
    return diff_pct <= tol_pct, detected_area, diff_pct


# ══════════════════════════════════════════════════════════════════════════════
# SIFT + SSIM
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
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    M, hmask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    if M is None:
        return None, 0
    aligned = cv2.warpPerspective(test_gray, M, IMG_SIZE)
    return aligned, int(hmask.sum()) if hmask is not None else 0


def compute_scores(master_gray, aligned, mask_px):
    if aligned.shape != master_gray.shape:
        aligned = cv2.resize(aligned, (master_gray.shape[1], master_gray.shape[0]))

    # Fine scale — blur first, then SSIM on full images (no zeroing before compare)
    mf = cv2.GaussianBlur(master_gray, (FINE_BLUR, FINE_BLUR), 0)
    af = cv2.GaussianBlur(aligned,     (FINE_BLUR, FINE_BLUR), 0)
    score, diff_fine = ssim_fn(mf, af, full=True)

    # Zero out mask zone in the diff map AFTER computing (don't count those pixels)
    diff_fine_masked = diff_fine.copy()
    if mask_px:
        x1 = max(0, mask_px["x"])
        y1 = max(0, mask_px["y"])
        x2 = min(diff_fine.shape[1], mask_px["x"] + mask_px["w"])
        y2 = min(diff_fine.shape[0], mask_px["y"] + mask_px["h"])
        diff_fine_masked[y1:y2, x1:x2] = 1.0  # set to 1.0 = perfect match → ignored

    fine_mask = diff_fine_masked < 0.5
    fine_pct  = fine_mask.sum() / fine_mask.size * 100

    # Coarse scale — same approach
    mc = cv2.GaussianBlur(master_gray, (COARSE_BLUR, COARSE_BLUR), 0)
    ac = cv2.GaussianBlur(aligned,     (COARSE_BLUR, COARSE_BLUR), 0)
    _, diff_coarse = ssim_fn(mc, ac, full=True)
    if mask_px:
        diff_coarse[y1:y2, x1:x2] = 1.0
    coarse_pct = (diff_coarse < 0.5).sum() / diff_coarse.size * 100

    return score, fine_pct, coarse_pct, diff_fine, fine_mask


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

def draw_4panel(master_gray, aligned, diff, diff_mask,
                ssim_score, fine_pct, coarse_pct, inliers,
                area_ok, area_diff_pct,
                verdict, ssim_ok, fine_ok, coarse_ok):
    color = (0,220,0) if verdict == "GOOD" else (0,0,220)
    h = master_gray.shape[0]
    p1 = cv2.cvtColor(master_gray, cv2.COLOR_GRAY2BGR)
    cv2.putText(p1, "MASTER",  (10,h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
    p2 = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)
    cv2.putText(p2, "ALIGNED", (10,h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
    p3 = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)
    p3[diff_mask] = (0,0,220)
    cv2.putText(p3, f"DIFF {fine_pct:.1f}%", (10,h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,220), 2)
    p4 = cv2.applyColorMap((diff*255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.putText(p4, "HEATMAP", (10,h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    ph = 260
    def rs(img):
        s = ph / img.shape[0]
        return cv2.resize(img, (int(img.shape[1]*s), ph))
    grid = np.vstack([np.hstack([rs(p1), rs(p2)]),
                      np.hstack([rs(p3), rs(p4)])])
    bar = np.full((110, grid.shape[1], 3), 25, dtype=np.uint8)
    sc = (0,220,0) if ssim_ok  else (0,0,220)
    fc = (0,220,0) if fine_ok  else (0,0,220)
    cc = (0,220,0) if coarse_ok else (0,0,220)
    ac = (0,220,0) if area_ok  else (0,0,220)
    cv2.putText(bar, f"Inliers:{inliers}",           ( 10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)
    cv2.putText(bar, f"Area diff:{area_diff_pct:.1f}%",( 200,22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ac, 1)
    cv2.putText(bar, f"SSIM:{ssim_score:.3f}",       ( 10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, sc, 2)
    cv2.putText(bar, f"Fine:{fine_pct:.1f}%",        (220, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fc, 2)
    cv2.putText(bar, f"Coarse:{coarse_pct:.1f}%",    (400, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cc, 2)
    cv2.putText(bar, f">> {verdict}",                (600, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(bar, "M=setup  R=reset  Q=quit",     ( 10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140,140,140), 1)
    return np.vstack([bar, grid])


TRACKER_COLOR = {
    "WAITING": (180, 180, 180),
    "IN_ZONE": (0, 220, 255),
    "CHECKED": (0, 200, 0),
}

def draw_live(frame, boxes, center_i, tracker_state,
              mask_rel, last_verdict, last_area_diff,
              master_area, fw, fh, master_loaded):
    out = frame.copy()

    # All label boxes
    for i, (x, y, w, h) in enumerate(boxes):
        is_c  = (i == center_i)
        fully = x > 4 and y > 4 and x+w > 4 and y+h > 4 and x+w < fw-4 and y+h < fh-4

        if is_c:
            color = TRACKER_COLOR.get(tracker_state, (0,255,0))
            thick = 3
            tag   = f"{tracker_state}  {w}x{h}px"
        elif fully:
            color, thick, tag = (255,255,255), 1, f"{w}x{h}"
        else:
            color, thick, tag = (0,140,255), 1, "PARTIAL"

        cv2.rectangle(out, (x,y), (x+w,y+h), color, thick)
        t = 18 if is_c else 10
        for px, py, dx, dy in [(x,y,1,1),(x+w,y,-1,1),(x+w,y+h,-1,-1),(x,y+h,1,-1)]:
            cv2.line(out, (px,py), (px+dx*t,py),  color, thick)
            cv2.line(out, (px,py), (px,py+dy*t),  color, thick)
        cv2.putText(out, tag, (x+4, max(y-6,14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 if is_c else 0.4, color, 1)

        # Area info on center label + mask visibility decision
        if is_c and master_area > 0:
            area_diff = abs(w*h - master_area) / master_area * 100
            area_within_tol = area_diff <= AREA_TOL_PCT
            ac = (0,220,0) if area_within_tol else (0,0,220)
            cv2.putText(out, f"area diff {area_diff:.1f}%",
                        (x+4, y+h+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ac, 1)
        else:
            area_within_tol = True  # no master yet — don't suppress mask

        # Draw dynamic mask only when label area matches master (mask position is meaningful)
        if is_c and mask_rel and area_within_tol:
            mask_overlay_on_frame(out, x, y, w, h, mask_rel)

    # Center zone box
    zx, zy = int(fw*CENTER_ZONE), int(fh*CENTER_ZONE)
    cx_f, cy_f = fw//2, fh//2
    cv2.rectangle(out, (cx_f-zx,cy_f-zy), (cx_f+zx,cy_f+zy), (0,200,255), 1)
    cv2.line(out, (cx_f-15,cy_f), (cx_f+15,cy_f), (0,200,255), 2)
    cv2.line(out, (cx_f,cy_f-15), (cx_f,cy_f+15), (0,200,255), 2)

    # Bottom status bar
    if last_verdict:
        bc = (0,160,0) if last_verdict == "GOOD" else (0,0,160)
        cv2.rectangle(out, (0,fh-55), (fw,fh), bc, -1)
        area_str = f"  area_diff={last_area_diff:.1f}%" if last_area_diff is not None else ""
        cv2.putText(out, f"LAST: {last_verdict}{area_str}   M=setup  R=reset  Q=quit",
                    (10,fh-14), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
    else:
        if not master_loaded:
            msg = "NO MASTER — press M to run setup"
            bc  = (60,20,20)
        else:
            msg = "Ready — bring label to CENTER zone   M=setup  R=reset  Q=quit"
            bc  = (25,25,25)
        cv2.rectangle(out, (0,fh-38), (fw,fh), bc, -1)
        cv2.putText(out, msg, (10,fh-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════

def save_result(crop, verdict):
    folder = GOOD_DIR if verdict == "GOOD" else BAD_DIR
    os.makedirs(folder, exist_ok=True)
    ts  = time.strftime("%Y%m%d_%H%M%S")
    ms  = int((time.time() % 1) * 1000)
    out = os.path.join(folder, f"{verdict}_{ts}_{ms:03d}.jpg")
    cv2.imwrite(out, crop)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref",    default=os.path.join(REF_DIR, "master.jpg"))
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--ssim",   type=float, default=SSIM_PASS)
    parser.add_argument("--diff",   type=float, default=FINE_FAIL_PCT)
    parser.add_argument("--cdiff",  type=float, default=COARSE_FAIL_PCT)
    parser.add_argument("--area-tol", type=float, default=AREA_TOL_PCT,
                        dest="area_tol")
    args = parser.parse_args()

    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
    else:
        cap = cv2.VideoCapture(DEFAULT_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("ERROR: cannot open stream")
        return

    sift = cv2.SIFT_create(nfeatures=5000)

    # ── Load existing config ──────────────────────────────────────────
    region = None
    if os.path.exists(REGION_FILE):
        with open(REGION_FILE) as f:
            region = json.load(f)
        print(f"Region loaded: {region['w']}x{region['h']}  "
              f"master_area={region.get('master_area', 'N/A')}px²")

    mask_rel = None
    if os.path.exists(MASK_FILE):
        with open(MASK_FILE) as f:
            d = json.load(f)
        # Support both old absolute format and new relative format
        if "rx" in d:
            mask_rel = d
            print(f"Mask loaded (relative): {mask_rel}")
        else:
            # Old format — convert using region if available
            if region:
                mask_rel = {
                    "rx": (d["x"] - region["x"]) / region["w"],
                    "ry": (d["y"] - region["y"]) / region["h"],
                    "rw": d["w"] / region["w"],
                    "rh": d["h"] / region["h"],
                }
                print(f"Mask loaded (converted from old format): {mask_rel}")

    master_gray   = None
    master_loaded = False
    master_area   = region.get("master_area", 0) if region else 0

    if os.path.exists(args.ref):
        m = cv2.imread(args.ref)
        if m is not None:
            master_gray   = cv2.resize(cv2.cvtColor(m, cv2.COLOR_BGR2GRAY), IMG_SIZE)
            master_loaded = True
            print(f"Master loaded: {args.ref}")

    if not master_loaded:
        print("No master — press M to run setup.\n")

    mask_px = make_mask_for_imgsize(mask_rel)

    # ── State ──────────────────────────────────────────────────────────
    tracker       = LabelTracker()
    last_verdict  = None
    last_area_diff = None
    result_panel  = None

    WIN_LIVE   = "Label Inspector"
    WIN_RESULT = "Result"
    cv2.namedWindow(WIN_LIVE,   cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_RESULT, cv2.WINDOW_NORMAL)
    print("M=setup  R=reset  Q=quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        fh, fw = frame.shape[:2]

        # ── Detect + pick center ───────────────────────────────────────
        boxes    = detect_labels(frame)
        center_i = pick_center(boxes, fw, fh)

        center_box = boxes[center_i] if center_i is not None else None

        # ── Tracker update ─────────────────────────────────────────────
        should_check = tracker.update(center_box)

        # ── Check this label ───────────────────────────────────────────
        if should_check and master_loaded and center_box is not None:
            bx, by, bw, bh = center_box

            # 1. Area check — compare to master
            area_ok, _, area_diff = area_check(
                bw, bh, master_area, args.area_tol)

            # 2. Crop the detected bbox directly
            crop      = frame[by:by+bh, bx:bx+bw]
            test_gray = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), IMG_SIZE)

            # 3. SIFT align
            aligned, inliers = sift_align(master_gray, test_gray, sift)

            if aligned is None or inliers < MIN_INLIERS:
                print(f"  Low inliers ({inliers}) — realigning label")
            else:
                # 4. SSIM — skip mask if area is outside tolerance (size mismatch)
                active_mask = mask_px if area_ok else None
                ssim_score, fine_pct, coarse_pct, diff_map, diff_mask = \
                    compute_scores(master_gray, aligned, active_mask)

                ssim_ok   = ssim_score >= args.ssim
                fine_ok   = fine_pct   <  args.diff
                coarse_ok = coarse_pct <  args.cdiff
                verdict   = "GOOD" if (ssim_ok and fine_ok and coarse_ok and area_ok) \
                            else "BAD"

                last_verdict   = verdict
                last_area_diff = area_diff

                fname = save_result(crop, verdict)
                print(f"  [{verdict}]  SSIM={ssim_score:.3f}  "
                      f"Fine={fine_pct:.1f}%  Coarse={coarse_pct:.1f}%  "
                      f"Area_diff={area_diff:.1f}%  Inliers={inliers}  → {fname}")

                result_panel = draw_4panel(
                    master_gray, aligned, diff_map, diff_mask,
                    ssim_score, fine_pct, coarse_pct, inliers,
                    area_ok, area_diff, verdict,
                    ssim_ok, fine_ok, coarse_ok
                )

        # ── Draw live feed ─────────────────────────────────────────────
        display = draw_live(
            frame, boxes, center_i, tracker.get_state(),
            mask_rel, last_verdict, last_area_diff,
            master_area, fw, fh, master_loaded
        )
        cv2.imshow(WIN_LIVE, display)

        if result_panel is not None:
            cv2.imshow(WIN_RESULT, result_panel)

        # ── Keys ───────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break

        elif key in (ord('r'), ord('R')):
            tracker      = LabelTracker()
            last_verdict = None
            last_area_diff = None
            result_panel = None
            print("Reset.")

        elif key in (ord('m'), ord('M')):
            new_master, new_region, new_mask_rel = run_setup(cap)
            if new_master is not None:
                master_gray   = new_master
                master_loaded = True
                region        = new_region
                mask_rel      = new_mask_rel
                master_area   = region.get("master_area", 0)
                mask_px       = make_mask_for_imgsize(mask_rel)
                tracker       = LabelTracker()
                last_verdict  = None
                last_area_diff = None
                result_panel  = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

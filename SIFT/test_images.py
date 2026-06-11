"""
Image-folder Label Tester
==========================
Test pre-cropped label images against a master reference using the same
SIFT + SSIM + zone-missing pipeline as live_damage_check.py.

No camera, no label detection — just feed it folders of cropped label images.

Usage:
  python3 SIFT/test_images.py --master SIFT/reference/master.jpg --folder path/to/labels/
  python3 SIFT/test_images.py --master SIFT/reference/master.jpg --folder path/to/labels/ --ssim 0.75 --diff 3

Controls (result window focused):
  N / Space  = next image
  P          = previous image
  S          = save current result panel to results/test_output/
  Q / ESC    = quit
"""

import cv2
import numpy as np
import os
import time
import argparse
from skimage.metrics import structural_similarity as ssim_fn

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(BASE_DIR, "results", "test_output")

# ── SIFT / SSIM ────────────────────────────────────────────────────────────────
IMG_SIZE       = (800, 600)
SSIM_PASS      = 0.75
DIFF_AREA_FAIL = 3.0
RATIO_TEST     = 0.75
MIN_INLIERS    = 40

# ── Feature flags ─────────────────────────────────────────────────────────────
ENABLE_PEN_CHECK     = False
ENABLE_WRINKLE_CHECK = False
ENABLE_ZONE_CHECK    = True

# ── Pen / mark detection ───────────────────────────────────────────────────────
PEN_SSIM_THRESH  = 0.55
PEN_MIN_AREA     = 400
PEN_MAX_AREA     = 8000
PEN_MIN_ASPECT   = 3.0
PEN_FAIL_COUNT   = 2

# ── Wrinkle detection ──────────────────────────────────────────────────────────
WRINKLE_CELLS    = 16
WRINKLE_FAIL     = 0.30
WRINKLE_CHI_THR  = 0.20

# ── Zone-based missing content detection ──────────────────────────────────────
ZONE_COLS        = 40
ZONE_ROWS        = 40
ZONE_SSIM_THR    = 0.60
ZONE_FAIL_COUNT  = 1
ZONE_EDGE_PCT    = 0.005

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


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

def detect_pen_marks(aligned, ssim_diff):
    diff_bin = (ssim_diff < PEN_SSIM_THRESH).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    diff_bin = cv2.morphologyEx(diff_bin, cv2.MORPH_CLOSE, k)
    pen_mask = np.zeros_like(diff_bin)
    stroke_count = 0
    cnts, _ = cv2.findContours(diff_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < PEN_MIN_AREA or area > PEN_MAX_AREA:
            continue
        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        if rw == 0 or rh == 0:
            continue
        if max(rw, rh) / min(rw, rh) >= PEN_MIN_ASPECT:
            cv2.drawContours(pen_mask, [cnt], -1, 255, -1)
            stroke_count += 1
    overlay = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)
    overlay[pen_mask > 0] = (255, 0, 255)
    return stroke_count, overlay


def _lbp(gray):
    h, w  = gray.shape
    g     = gray.astype(np.int16)
    lbp   = np.zeros((h, w), dtype=np.uint8)
    for bit, (dy, dx) in enumerate([(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]):
        shifted = np.roll(np.roll(g, dy, axis=0), dx, axis=1)
        lbp    |= ((g >= shifted).astype(np.uint8) << bit)
    return lbp


def detect_wrinkles(master_gray, aligned):
    cells  = WRINKLE_CELLS
    h, w   = master_gray.shape
    ch, cw = h // cells, w // cells
    lbp_m  = _lbp(master_gray)
    lbp_a  = _lbp(aligned)
    heat   = np.zeros((h, w), dtype=np.float32)
    bad, total = 0, 0
    for r in range(cells):
        for c in range(cells):
            y0, y1 = r * ch, (r + 1) * ch
            x0, x1 = c * cw, (c + 1) * cw
            hm, _ = np.histogram(lbp_m[y0:y1, x0:x1], bins=256, range=(0, 255))
            ha, _ = np.histogram(lbp_a[y0:y1, x0:x1], bins=256, range=(0, 255))
            hm = hm.astype(np.float32) + 1e-6;  hm /= hm.sum()
            ha = ha.astype(np.float32) + 1e-6;  ha /= ha.sum()
            chi = float(0.5 * np.sum((hm - ha) ** 2 / (hm + ha)))
            heat[y0:y1, x0:x1] = chi
            total += 1
            if chi > WRINKLE_CHI_THR:
                bad += 1
    score       = bad / total if total > 0 else 0.0
    heat_smooth = cv2.GaussianBlur(np.clip(heat / max(heat.max(), 1e-6), 0, 1), (31, 31), 0)
    heatmap     = cv2.applyColorMap((heat_smooth * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    return score, heatmap


def detect_zone_missing(master_gray, aligned):
    h, w   = master_gray.shape
    edge_x = int(w * ZONE_EDGE_PCT)
    edge_y = int(h * ZONE_EDGE_PCT)
    cw     = (w - 2 * edge_x) // ZONE_COLS
    ch     = (h - 2 * edge_y) // ZONE_ROWS
    heat   = np.zeros((h, w), dtype=np.float32)
    bad_zones, bad_rects = 0, []
    for r in range(ZONE_ROWS):
        for c in range(ZONE_COLS):
            x0, y0 = edge_x + c * cw, edge_y + r * ch
            x1, y1 = x0 + cw, y0 + ch
            zm, za = master_gray[y0:y1, x0:x1], aligned[y0:y1, x0:x1]
            if zm.shape[0] < 8 or zm.shape[1] < 8:
                continue
            win = min(zm.shape[0], zm.shape[1])
            win = win if win % 2 == 1 else win - 1
            win = max(win, 7)
            score = ssim_fn(zm, za, win_size=win)
            heat[y0:y1, x0:x1] = score
            if score < ZONE_SSIM_THR:
                bad_zones += 1
                bad_rects.append((x0, y0, x1, y1))
    inverted = np.clip(1.0 - heat, 0, 1)
    hmap     = cv2.applyColorMap((inverted * 255).astype(np.uint8), cv2.COLORMAP_JET)
    for (x0, y0, x1, y1) in bad_rects:
        cv2.rectangle(hmap, (x0, y0), (x1, y1), (0, 0, 255), 2)
    label = f"ZONES  {bad_zones}/{ZONE_COLS * ZONE_ROWS} bad"
    if not ENABLE_ZONE_CHECK:
        label += "  (disabled)"
    cv2.putText(hmap, label, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 255) if bad_zones >= ZONE_FAIL_COUNT else (0, 255, 0), 2)
    return bad_zones, hmap


def compute_scores(master_gray, aligned):
    if aligned.shape != master_gray.shape:
        aligned = cv2.resize(aligned, (master_gray.shape[1], master_gray.shape[0]))
    ssim_score, diff  = ssim_fn(master_gray, aligned, full=True)
    diff_mask         = diff < 0.5
    diff_area_pct     = diff_mask.sum() / diff_mask.size * 100
    pen_pixels, pen_overlay           = detect_pen_marks(aligned, diff)
    wrinkle_score, wrinkle_map        = detect_wrinkles(master_gray, aligned)
    bad_zones, zone_heatmap           = detect_zone_missing(master_gray, aligned)
    return (ssim_score, diff_area_pct, diff_mask,
            pen_pixels, pen_overlay,
            wrinkle_score, wrinkle_map,
            bad_zones, zone_heatmap)


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

def draw_result_panel(master_gray, aligned, diff_mask,
                      ssim_score, diff_pct, inliers,
                      pen_pixels, pen_overlay, wrinkle_score, wrinkle_map,
                      bad_zones, zone_heatmap,
                      verdict, ssim_ok, diff_ok, pen_ok, wrinkle_ok, zone_ok,
                      ssim_thresh, diff_thresh,
                      filename, idx, total):
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
    pen_bin = cv2.cvtColor(pen_overlay, cv2.COLOR_BGR2GRAY)
    pen_bin = (pen_bin == 0) & (pen_overlay[:, :, 0] == 255)
    p3[pen_bin] = (255, 0, 255)
    diff_label = f"DIFF {diff_pct:.1f}%"
    if ENABLE_PEN_CHECK:
        diff_label += f"  PEN {pen_pixels} blob(s)"
    cv2.putText(p3, diff_label, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2)

    p4 = zone_heatmap.copy()

    ph  = 280
    GAP = 6

    def rs(img):
        s = ph / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * s), ph))

    r1, r2, r3, r4 = rs(p1), rs(p2), rs(p3), rs(p4)
    pw = r1.shape[1]
    sep_v = np.zeros((ph, GAP, 3), dtype=np.uint8)
    sep_h = np.zeros((GAP, pw * 2 + GAP, 3), dtype=np.uint8)
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
    cv2.putText(bar, zone_label, (430, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                dim if not ENABLE_ZONE_CHECK else zc, 2)

    reason = []
    if not ssim_ok: reason.append(f"SSIM {ssim_score:.3f}<{ssim_thresh}")
    if not diff_ok: reason.append(f"diff {diff_pct:.1f}%>={diff_thresh}%")
    if ENABLE_ZONE_CHECK and not zone_ok:
        reason.append(f"missing content {bad_zones} zones>={ZONE_FAIL_COUNT}")
    if ENABLE_PEN_CHECK     and not pen_ok:  reason.append(f"pen {pen_pixels}>={PEN_FAIL_COUNT}")
    if ENABLE_WRINKLE_CHECK and not wrinkle_ok: reason.append(f"wrinkle {wrinkle_score:.2f}>={WRINKLE_FAIL}")
    if reason:
        cv2.putText(bar, "BAD: " + "  |  ".join(reason),
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 255), 1)

    # ── File info bar ─────────────────────────────────────────────────────────
    info = np.full((36, W, 3), 15, dtype=np.uint8)
    name_str  = f"[{idx}/{total}]  {filename}"
    nav_str   = "N/Space=next  P=prev  S=save  Q=quit"
    cv2.putText(info, name_str, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    ns, _ = cv2.getTextSize(nav_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(info, nav_str, (W - ns[0] - 10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

    return np.vstack([banner, bar, info, grid])


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Image-folder label tester")
    parser.add_argument("--master", required=True,
                        help="Master reference image (cropped label)")
    parser.add_argument("--folder", required=True,
                        help="Folder of cropped label images to test")
    parser.add_argument("--ssim",   type=float, default=SSIM_PASS,
                        help=f"Min SSIM to pass (default {SSIM_PASS})")
    parser.add_argument("--diff",   type=float, default=DIFF_AREA_FAIL,
                        help=f"Max diff%% to pass (default {DIFF_AREA_FAIL})")
    args = parser.parse_args()

    # ── Load master ───────────────────────────────────────────────────────────
    m = cv2.imread(args.master)
    if m is None:
        print(f"ERROR: cannot read master: {args.master}")
        return
    master_gray = cv2.resize(cv2.cvtColor(m, cv2.COLOR_BGR2GRAY), IMG_SIZE)
    print(f"Master : {args.master}")

    # ── Collect image list ────────────────────────────────────────────────────
    images = sorted([
        os.path.join(args.folder, f)
        for f in os.listdir(args.folder)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
    ])
    if not images:
        print(f"ERROR: no images found in {args.folder}")
        return
    print(f"Found {len(images)} image(s) in {args.folder}")
    print("N/Space=next  P=prev  S=save  Q/ESC=quit\n")

    sift = cv2.SIFT_create(nfeatures=5000)
    os.makedirs(OUT_DIR, exist_ok=True)

    WIN = "Label Tester — Master | Aligned | Diff | Zones"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    idx     = 0       # current image index
    cache   = {}      # path → result panel (avoid re-computing on prev)
    panel   = None

    def process(path):
        img = cv2.imread(path)
        if img is None:
            print(f"  Cannot read {path}, skipping.")
            return None, None
        test_gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), IMG_SIZE)
        aligned, inliers = sift_align(master_gray, test_gray, sift)
        if aligned is None or inliers < MIN_INLIERS:
            print(f"  [{os.path.basename(path)}] alignment failed (inliers={inliers})")
            return None, None

        (ssim_score, diff_pct, diff_mask,
         pen_pixels, pen_overlay,
         wrinkle_score, wrinkle_map,
         bad_zones, zone_heatmap) = compute_scores(master_gray, aligned)

        ssim_ok    = ssim_score  >= args.ssim
        diff_ok    = diff_pct    <  args.diff
        pen_ok     = (pen_pixels   < PEN_FAIL_COUNT)  if ENABLE_PEN_CHECK     else True
        wrinkle_ok = (wrinkle_score < WRINKLE_FAIL)   if ENABLE_WRINKLE_CHECK else True
        zone_ok    = (bad_zones    < ZONE_FAIL_COUNT) if ENABLE_ZONE_CHECK    else True
        verdict    = "GOOD" if (ssim_ok and diff_ok and pen_ok and wrinkle_ok and zone_ok) else "BAD"

        print(f"  [{os.path.basename(path)}]  {verdict}  "
              f"SSIM={ssim_score:.3f}  Diff={diff_pct:.1f}%  "
              f"Zones={bad_zones}/{ZONE_COLS*ZONE_ROWS}  Inliers={inliers}")

        p = draw_result_panel(
            master_gray, aligned, diff_mask,
            ssim_score, diff_pct, inliers,
            pen_pixels, pen_overlay, wrinkle_score, wrinkle_map,
            bad_zones, zone_heatmap,
            verdict, ssim_ok, diff_ok, pen_ok, wrinkle_ok, zone_ok,
            args.ssim, args.diff,
            os.path.basename(path), idx + 1, len(images))
        return verdict, p

    while True:
        path = images[idx]

        if path not in cache:
            verdict, panel = process(path)
            if panel is not None:
                cache[path] = (verdict, panel)
        else:
            verdict, panel = cache[path]

        if panel is not None:
            cv2.imshow(WIN, panel)
        else:
            blank = np.zeros((300, 700, 3), dtype=np.uint8)
            cv2.putText(blank, f"[{idx+1}/{len(images)}] Alignment failed — {os.path.basename(path)}",
                        (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2)
            cv2.putText(blank, "N=next  P=prev  Q=quit",
                        (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            cv2.imshow(WIN, blank)

        key = cv2.waitKey(0) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key in (ord('n'), ord('N'), ord(' ')):
            idx = (idx + 1) % len(images)
        elif key in (ord('p'), ord('P')):
            idx = (idx - 1) % len(images)
        elif key in (ord('s'), ord('S')):
            if panel is not None:
                ts  = time.strftime("%Y%m%d_%H%M%S")
                out = os.path.join(OUT_DIR,
                                   f"{verdict}_{os.path.splitext(os.path.basename(path))[0]}_{ts}.jpg")
                cv2.imwrite(out, panel)
                print(f"  Saved → {out}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

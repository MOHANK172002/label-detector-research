"""
Label Crop Tool
===============
Opens an image (or live camera/stream), shows HSV trackbars to tune the
background mask, draws bounding boxes on detected labels, and saves crops.

Usage:
    python3 crop_tool.py                              # use saved test image
    python3 crop_tool.py --image frame_20260604_192314.jpg
    python3 crop_tool.py --stream                     # use TCP stream
    python3 crop_tool.py --camera 0                   # use webcam

Controls:
    S          = save all detected label crops to  crops/
    Q / ESC    = quit
"""

import cv2
import numpy as np
import argparse
import os
import time

DEFAULT_URL = (
    "tcp://192.168.1.11:8888"
    "?fflags=nobuffer&flags=low_delay&framedrop=1"
)
DEFAULT_IMAGE = "frame_20260605_143212.jpg"
OUT_DIR       = "crops"

# ── Initial trackbar values ────────────────────────────────────────────────────
# Frame analysis (frame_20260605_124653):
#   Background V=58,  Label V=135-184  → threshold at V~100 separates them
#   H and S are nearly identical everywhere → only V matters here
INIT = dict(
    V_low=100, V_high=255,   # brightness threshold — main knob
    blur=9,                  # morphology kernel size (odd)
    min_area=3,              # % of image area minimum
)


def nothing(_):
    pass


def make_trackbars(win):
    cv2.createTrackbar("V low",    win, INIT["V_low"],   255, nothing)
    cv2.createTrackbar("V high",   win, INIT["V_high"],  255, nothing)
    cv2.createTrackbar("Morph k",  win, INIT["blur"],     31, nothing)
    cv2.createTrackbar("Min area%",win, INIT["min_area"], 50, nothing)


def get_trackbars(win):
    v_lo = cv2.getTrackbarPos("V low",     win)
    v_hi = cv2.getTrackbarPos("V high",    win)
    k    = cv2.getTrackbarPos("Morph k",   win)
    area = cv2.getTrackbarPos("Min area%", win)
    k = max(k + (0 if k % 2 == 1 else 1), 1)
    return v_lo, v_hi, k, area


def detect_labels(frame, v_lo, v_hi, morph_k, min_area_pct):
    """
    Threshold V channel → flood-fill holes → find solid label contours.
    Returns list of (x, y, w, h) boxes, the filled mask, and center box index.
    """
    fh, fw = frame.shape[:2]
    hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v_ch   = hsv[:, :, 2]

    label_mask = cv2.inRange(v_ch, v_lo, v_hi)

    # Morph close: fill interior holes of each label
    k_size = max(morph_k, 1)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    label_mask = cv2.morphologyEx(label_mask, cv2.MORPH_CLOSE, k)
    label_mask = cv2.morphologyEx(label_mask, cv2.MORPH_OPEN,  k)

    # Flood-fill from corner to find true background, subtract from inverted
    # to get only interior holes, then fill them in
    flood_mask = np.zeros((fh + 2, fw + 2), dtype=np.uint8)
    flooded    = label_mask.copy()
    cv2.floodFill(flooded, flood_mask, (0, 0), 255)
    filled_mask = label_mask | cv2.bitwise_not(flooded)

    cnts, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

    min_area = (min_area_pct / 100.0) * fh * fw
    boxes = []
    for cnt in cnts:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w > fw * 0.95 or h > fh * 0.95:   # whole-frame false positive
            continue
        if w < h * 0.5:                        # too narrow — not a label
            continue
        boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[1])  # top-to-bottom

    # Find which box is closest to frame center
    cx_f, cy_f = fw // 2, fh // 2
    center_idx = None
    best_dist  = float("inf")
    for i, (x, y, w, h) in enumerate(boxes):
        dist = ((x + w//2 - cx_f)**2 + (y + h//2 - cy_f)**2) ** 0.5
        if dist < best_dist:
            best_dist  = dist
            center_idx = i

    return boxes, filled_mask, center_idx


def draw_boxes(frame, boxes, center_idx):
    out = frame.copy()
    fh, fw = frame.shape[:2]

    for i, (x, y, w, h) in enumerate(boxes):
        is_center = (i == center_idx)
        fully     = x > 4 and y > 4 and x+w < fw-4 and y+h < fh-4

        if is_center:
            color     = (0, 255, 0)    # bright green = will be cropped
            thickness = 3
            tag       = f"CROP  {w}x{h}px"
        elif fully:
            color     = (255, 255, 255)  # white = full but not center
            thickness = 1
            tag       = f"#{i+1}  {w}x{h}"
        else:
            color     = (0, 140, 255)    # orange = partial
            thickness = 1
            tag       = f"#{i+1} PARTIAL"

        cv2.rectangle(out, (x, y), (x+w, y+h), color, thickness)
        t = 18 if is_center else 12
        for (px, py, dx, dy) in [(x, y, 1, 1), (x+w, y, -1, 1),
                                   (x+w, y+h, -1, -1), (x, y+h, 1, -1)]:
            cv2.line(out, (px, py), (px+dx*t, py),  color, thickness)
            cv2.line(out, (px, py), (px, py+dy*t),  color, thickness)
        cv2.putText(out, tag, (x+4, max(y-6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55 if is_center else 0.45,
                    color, 2 if is_center else 1)

    # Frame center crosshair
    cv2.line(out, (fw//2-20, fh//2), (fw//2+20, fh//2), (0, 200, 255), 2)
    cv2.line(out, (fw//2, fh//2-20), (fw//2, fh//2+20), (0, 200, 255), 2)

    cv2.putText(out, f"Labels: {len(boxes)}   GREEN=crop target   S=save  Q=quit",
                (10, fh-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2)
    return out


def save_crops(frame, boxes, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    saved = []
    for i, (x, y, w, h) in enumerate(boxes):
        crop  = frame[y:y+h, x:x+w]
        fname = os.path.join(out_dir, f"label_{ts}_{i+1}.jpg")
        cv2.imwrite(fname, crop)
        saved.append(fname)
        print(f"  Saved: {fname}")
    return saved


def run(get_frame, is_live=False):
    WIN_MAIN = "Label Crop Tool"
    WIN_MASK = "Mask (background)"

    cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_MASK, cv2.WINDOW_NORMAL)
    make_trackbars(WIN_MAIN)

    frame = None

    while True:
        if is_live:
            f = get_frame()
            if f is not None:
                frame = f
        else:
            frame = get_frame()   # static image, returned every loop

        if frame is None:
            continue

        v_lo, v_hi, k, min_area  = get_trackbars(WIN_MAIN)
        boxes, mask, center_idx  = detect_labels(frame, v_lo, v_hi, k, min_area)
        display                  = draw_boxes(frame, boxes, center_idx)

        cv2.imshow(WIN_MAIN, display)
        cv2.imshow(WIN_MASK, mask)

        key = cv2.waitKey(30 if is_live else 1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key in (ord('s'), ord('S')):
            if center_idx is not None:
                # Save only the center label
                save_crops(frame, [boxes[center_idx]])
            elif boxes:
                save_crops(frame, boxes)
            else:
                print("No labels detected — adjust trackbars.")

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--image",  default=DEFAULT_IMAGE,
                       help="Path to a test image")
    group.add_argument("--stream", action="store_true",
                       help="Use TCP stream")
    group.add_argument("--camera", type=int,
                       help="Webcam index")
    args = parser.parse_args()

    if args.stream or args.camera is not None:
        src = DEFAULT_URL if args.stream else args.camera
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG if args.stream else 0)
        if not cap.isOpened():
            print(f"ERROR: cannot open {src}")
            return
        def get_frame():
            ret, f = cap.read()
            return f if ret else None
        print(f"Streaming from: {src}")
        print("Adjust trackbars to mask the BACKGROUND color.\n")
        run(get_frame, is_live=True)
        cap.release()
    else:
        img = cv2.imread(args.image)
        if img is None:
            print(f"ERROR: cannot read {args.image}")
            return
        print(f"Image: {args.image}")
        print("Adjust trackbars to mask the BACKGROUND color.")
        print("S = save crops   Q = quit\n")
        run(lambda: img, is_live=False)


if __name__ == "__main__":
    main()

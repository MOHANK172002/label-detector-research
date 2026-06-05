"""
Region Setup — Draw label area + optional hologram mask
========================================================
Run this ONCE to define:
  1. Label region  → region.json   (where label appears in camera frame)
  2. Hologram mask → mask.json     (zone to IGNORE during diff — optional)

Usage:
    python3 setup_region.py
    python3 setup_region.py --camera 0

Controls (both steps):
    Drag mouse  = draw box
    ENTER/SPACE = confirm and continue
    R           = redraw
    ESC         = quit without saving
"""

import cv2
import json
import argparse
import os

REGION_FILE = "region-test.json"
MASK_FILE   = "mask-test.json"

drawing  = False
start_pt = (0, 0)
end_pt   = (0, 0)


def mouse_callback(event, x, y, flags, param):
    global drawing, start_pt, end_pt

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing  = True
        start_pt = (x, y)
        end_pt   = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_pt = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_pt  = (x, y)


def get_rect(p1, p2):
    x = min(p1[0], p2[0])
    y = min(p1[1], p2[1])
    w = abs(p2[0] - p1[0])
    h = abs(p2[1] - p1[1])
    return x, y, w, h


def draw_box_step(cap, win_title, box_color, instruction_main, instruction_hint,
                  existing_box=None):
    """
    Generic draw-a-box step.
    Returns (x, y, w, h) on confirm, or None on ESC.
    existing_box: optional (x,y,w,h) already-saved box to show as reference.
    """
    global start_pt, end_pt
    start_pt = (0, 0)
    end_pt   = (0, 0)

    cv2.namedWindow(win_title)
    cv2.setMouseCallback(win_title, mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display    = frame.copy()
        fh, fw     = display.shape[:2]

        # Show already-saved reference box (e.g. label region while drawing mask)
        if existing_box:
            ex, ey, ew, eh = existing_box
            cv2.rectangle(display, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            cv2.putText(display, "Label area", (ex, ey - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Current drawn box
        x, y, bw, bh = get_rect(start_pt, end_pt)
        if bw > 5 and bh > 5:
            cv2.rectangle(display, (x, y), (x + bw, y + bh), box_color, 2)
            cv2.putText(display, f"{bw}x{bh}  ({x},{y})",
                        (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        cv2.putText(display, instruction_main,
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
        cv2.putText(display, instruction_hint,
                    (10, fh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow(win_title, display)
        key = cv2.waitKey(1) & 0xFF

        if key in (13, ord(' ')):       # ENTER or SPACE — confirm
            x, y, bw, bh = get_rect(start_pt, end_pt)
            if bw < 20 or bh < 20:
                print("Box too small — draw a larger area.")
                continue
            cv2.destroyWindow(win_title)
            return (x, y, bw, bh)

        elif key == ord('r') or key == ord('R'):
            start_pt = (0, 0)
            end_pt   = (0, 0)

        elif key == 27:                 # ESC — cancel
            cv2.destroyWindow(win_title)
            return None

    cv2.destroyWindow(win_title)
    return None


def ask_hologram_terminal():
    """Ask user in terminal whether the label has a hologram."""
    while True:
        ans = input("\nDoes your label have a hologram? (y/n): ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please enter y or n.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    
    DEFAULT_URL = (
    "tcp://192.168.1.11:8888"
    "?fflags=nobuffer&flags=low_delay&framedrop=1"
    )
    
    # cap = cv2.VideoCapture(args.camera)
    cap = cv2.VideoCapture(DEFAULT_URL)
    
    if not cap.isOpened():
        print(f"ERROR: cannot open camera {args.camera}")
        return

    # ── Step 1: draw label region ─────────────────────────────────────
    print("\n── Step 1: Label Region ──────────────────────────────────────")
    print("Draw a box around the FULL label area in the camera view.")

    rect = draw_box_step(
        cap,
        win_title        = "Step 1 — Draw box around label area",
        box_color        = (0, 255, 0),
        instruction_main = "Draw box around LABEL AREA",
        instruction_hint = "ENTER/SPACE = confirm    R = redraw    ESC = quit",
    )

    if rect is None:
        print("Cancelled — nothing saved.")
        cap.release()
        return

    x, y, w, h = rect
    region = {"x": x, "y": y, "w": w, "h": h}
    with open(REGION_FILE, "w") as f:
        json.dump(region, f, indent=2)
    print(f"Saved region : x={x} y={y} w={w} h={h}  →  {REGION_FILE}")

    # ── Step 2: hologram mask (optional) ─────────────────────────────
    has_hologram = ask_hologram_terminal()

    if not has_hologram:
        # Remove old mask.json if it exists so live_check doesn't use a stale one
        if os.path.exists(MASK_FILE):
            os.remove(MASK_FILE)
            print(f"Removed old {MASK_FILE}.")
        print("No hologram mask saved.")
        cap.release()
        cv2.destroyAllWindows()
        return

    print("\n── Step 2: Hologram Mask ─────────────────────────────────────")
    print("Draw a box over the HOLOGRAM area on the label.")
    print("This zone will be IGNORED during diff checking.")

    mask_rect = draw_box_step(
        cap,
        win_title        = "Step 2 — Draw box over HOLOGRAM area",
        box_color        = (0, 140, 255),
        instruction_main = "Draw box over HOLOGRAM area  (will be ignored in diff)",
        instruction_hint = "ENTER/SPACE = confirm    R = redraw    ESC = skip",
        existing_box     = (x, y, w, h),   # show label region as reference
    )

    if mask_rect is None:
        print("Hologram mask skipped — no mask.json saved.")
    else:
        mx, my, mw, mh = mask_rect
        mask = {"x": mx, "y": my, "w": mw, "h": mh}
        with open(MASK_FILE, "w") as f:
            json.dump(mask, f, indent=2)
        print(f"Saved mask   : x={mx} y={my} w={mw} h={mh}  →  {MASK_FILE}")

    cap.release()
    cv2.destroyAllWindows()
    print("\nSetup complete. Run:  python3 live_check.py")


if __name__ == "__main__":
    main()

"""
HSV Probe — click anywhere on the live frame to print that pixel's HSV value.
Use this to find the exact HSV range of:
  - The green background
  - The green header bar on the label
  - The silver/white label body

Press Q to quit.
"""

import cv2
import numpy as np

DEFAULT_URL = (
    "tcp://192.168.1.11:8888"
    "?fflags=nobuffer"
    "&flags=low_delay"
    "&framedrop=1"
)

clicked = []

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        frame, hsv = param
        bgr = frame[y, x].tolist()
        h, s, v = hsv[y, x].tolist()
        print(f"  Pixel ({x:4d},{y:4d})  BGR={bgr}  HSV=({h:3d}, {s:3d}, {v:3d})")
        clicked.append((x, y, h, s, v))

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",    default=DEFAULT_URL)
    parser.add_argument("--camera", type=int, default=None)
    args = parser.parse_args()

    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
    else:
        cap = cv2.VideoCapture(args.url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("Cannot open source")
        return

    print("Click on pixels to read HSV values.")
    print("Click on: background green, label green header, label white body, silver strip")
    print("Press Q to quit.\n")

    win = "HSV Probe — click to sample"
    cv2.namedWindow(win)

    frame = None
    hsv   = None
    cv2.setMouseCallback(win, on_mouse, param=[frame, hsv])

    while True:
        ret, f = cap.read()
        if not ret:
            continue

        frame = f
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Update callback param with latest frame
        cv2.setMouseCallback(win, on_mouse, param=[frame, hsv])

        display = frame.copy()

        # Draw all clicked points
        for x, y, h, s, v in clicked:
            cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(display, f"H{h} S{s} V{v}", (x+8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        cv2.putText(display, "Click pixels to sample HSV   Q=quit",
                    (10, display.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

        cv2.imshow(win, display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if clicked:
        print("\n── Summary ──────────────────────────────")
        print(f"{'Region':<20} H    S    V")
        for x, y, h, s, v in clicked:
            print(f"  ({x},{y})           {h:3d}  {s:3d}  {v:3d}")

if __name__ == "__main__":
    main()

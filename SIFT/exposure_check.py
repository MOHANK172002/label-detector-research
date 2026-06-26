"""
Live exposure tuner for the Pi GS camera (rpicam-vid) — winding machine motion blur.

A/D = shutter speed -/+   W/S = gain -/+   R = apply (restart camera)   Q = quit

Tune shutter down until motion blur on the moving label disappears, bump gain
back up to compensate for the resulting darker image.
"""

import cv2
import subprocess
import os
import time

FIFO_PATH = "/tmp/exposure_check.mjpeg"

EXPOSURE  = 19040   # shutter speed, microseconds
GAIN      = 2.0
EXP_STEP  = 500
GAIN_STEP = 0.2


def start_cam(exp, gain):
    if os.path.exists(FIFO_PATH):
        os.remove(FIFO_PATH)
    os.mkfifo(FIFO_PATH)

    cmd = [
        "rpicam-vid",
        "-t", "0",
        "--nopreview",
        "--width", "1280",
        "--height", "720",
        "--framerate", "60",
        "--codec", "mjpeg",
        "--shutter", str(int(exp)),
        "--gain", str(gain),
        "-o", FIFO_PATH,
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cap = cv2.VideoCapture(FIFO_PATH, cv2.CAP_FFMPEG)
    return proc, cap


def stop_cam(proc, cap):
    cap.release()
    proc.kill()
    proc.wait()
    if os.path.exists(FIFO_PATH):
        os.remove(FIFO_PATH)


p, cap = start_cam(EXPOSURE, GAIN)
if not cap.isOpened():
    print("ERROR: could not open camera stream — is rpicam-vid installed and the camera connected?")

print("A/D = shutter -/+   W/S = gain -/+   R = apply   Q = quit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    txt = f"Shutter={EXPOSURE}us  Gain={GAIN:.1f}   A/D shutter  W/S gain  R=apply  Q=quit"
    cv2.putText(frame, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Exposure Test", frame)

    k = cv2.waitKey(1) & 0xFF

    if k == ord('q'):
        break
    elif k == ord('a'):
        EXPOSURE = max(100, EXPOSURE - EXP_STEP)
    elif k == ord('d'):
        EXPOSURE += EXP_STEP
    elif k == ord('w'):
        GAIN += GAIN_STEP
    elif k == ord('s'):
        GAIN = max(1.0, GAIN - GAIN_STEP)
    elif k == ord('r'):
        print(f"Applying shutter={EXPOSURE}us gain={GAIN:.1f} ...")
        stop_cam(p, cap)
        time.sleep(0.3)
        p, cap = start_cam(EXPOSURE, GAIN)

stop_cam(p, cap)
cv2.destroyAllWindows()

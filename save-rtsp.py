import cv2
import time
from datetime import datetime

# Raspberry Pi TCP stream
url = (
    "tcp://192.168.1.11:8888"
    "?fflags=nobuffer"
    "&flags=low_delay"
    "&framedrop=1"
)

# Open stream
cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
if not cap.isOpened():
    print("Cannot open stream")
    exit()

print("Press S to save frame | Press Q to quit")

start_time = time.time()
auto_saved = True  # Ensure auto-save happens only once

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame receive failed")
        break

    # Auto-save once after 2 seconds
    if not auto_saved and (time.time() - start_time) >= 2:
        filename = f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Auto-saved: {filename}")
        auto_saved = True

    # Show frame
    cv2.imshow("Stream Viewer", frame)

    key = cv2.waitKey(1) & 0xFF

    # Save frame on S
    if key == ord('s'):
        filename = f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")

    # Quit on Q
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
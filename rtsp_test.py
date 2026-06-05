import cv2
import zxingcpp

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

print("Press Q to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame receive failed")
        break

    # Decode all QR/Barcodes
    results = zxingcpp.read_barcodes(frame)

    for result in results:

        # Barcode text
        text = result.text

        # Corner points
        points = result.position

        pts = [
            (int(points.top_left.x), int(points.top_left.y)),
            (int(points.top_right.x), int(points.top_right.y)),
            (int(points.bottom_right.x), int(points.bottom_right.y)),
            (int(points.bottom_left.x), int(points.bottom_left.y)),
        ]

        # Draw bounding box
        for i in range(4):
            cv2.line(
                frame,
                pts[i],
                pts[(i + 1) % 4],
                (0, 255, 0),
                2
            )

        # Put decoded text above QR
        cv2.putText(
            frame,
            text,
            (pts[0][0], pts[0][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        print("Detected:", text)

    # Show frame
    cv2.imshow("ZXing QR Scanner", frame)

    # Quit on Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
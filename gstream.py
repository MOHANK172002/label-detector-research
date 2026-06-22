import cv2
import sys
import zxingcpp
import numpy as np
import subprocess

# Use GStreamer pipeline for USB camera
# Works with V4L2 and GStreamer on Linux
# Usage: python3 usb_preview.py [camera_index] [width] [height]
# Example: python3 usb_preview.py 0 1280 720

def gstreamer_pipeline(cam_index=2, width=1920, height=1200, fps=30, format='MJPG'):
    """
    Build a GStreamer pipeline for v4l2src.
    
    Supported formats:
      - 'MJPG': Motion-JPEG (compressed, high bandwidth efficiency)
      - 'YUYV': Raw uncompressed (lower resolutions/fps only)
    
    For this camera, use MJPG for 800x600+ resolutions.
    """
    if format.upper() == 'MJPG':
        # MJPEG: supports up to 1600x1200 @ 30fps
        return (
            f"v4l2src device=/dev/video{cam_index} ! "
            f"image/jpeg, width={width}, height={height}, framerate={fps}/1 ! "
            "jpegdec ! videoconvert ! video/x-raw, format=BGR ! appsink"
        )
    else:
        # YUYV: raw format, limited to lower resolutions at 30fps
        return (
            f"v4l2src device=/dev/video{cam_index} ! "
            f"video/x-raw, width={width}, height={height}, framerate={fps}/1 ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink"
        )


def main():
    # Parse command-line arguments
    cam_index = 2
    width = 1600
    height = 1200
    fmt = 'MJPG'
    
    if len(sys.argv) > 1:
        try:
            cam_index = int(sys.argv[1])
        except ValueError:
            pass
    
    if len(sys.argv) > 3:
        try:
            width = int(sys.argv[2])
            height = int(sys.argv[3])
        except ValueError:
            pass
    
    if len(sys.argv) > 4:
        fmt = sys.argv[4].upper()
    
    pipeline = gstreamer_pipeline(cam_index, width, height, format=fmt)
    print("Opening camera with pipeline:")
    print(f"  Format: {fmt}")
    print(f"  Resolution: {width}x{height}")
    print(f"  {pipeline}\n")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print(f"Failed to open camera at /dev/video{cam_index}")
        print(f"Resolution {width}x{height} with format {fmt} may not be supported.\n")
        print("Supported resolutions for this camera:")
        print("  MJPG: 640x480, 800x600, 1024x768, 1280x720, 1600x1200 (all @ 30fps)")
        print("  YUYV: 640x480, 640x360 (@ 30fps)")
        return

    print(f"✓ Camera opened successfully with {fmt} format at {width}x{height}")
    print("Press 'q' to quit\n")

    # Create window and set to fullscreen
    window_name = f'USB Camera - {fmt} {width}x{height}'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        # Scan frame for barcodes/QR codes
        results = zxingcpp.read_barcodes(frame)
        for result in results:
            print(f"Found barcode: {result.text} ({result.format})")
            
			# Play beep sound
            try:
                subprocess.Popen(["aplay", "beep.wav"])
            except Exception as e:
                print(f"Failed to play beep: {e}")

            # Draw bounding box on frame
            if hasattr(result, 'position') and result.position:
                points = [result.position.top_left, result.position.top_right, result.position.bottom_right, result.position.bottom_left]
                pts = np.array([(int(p.x), int(p.y)) for p in points], dtype=np.int32)
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                cv2.putText(frame, result.text, (pts[0][0], pts[0][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Rotate frame 90 degrees clockwise
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

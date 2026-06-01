"""
SIFT Feature Matching — Label Authenticator
============================================
Uses SIFT (Scale-Invariant Feature Transform) — handles different zoom/distance.

Why SIFT over ORB:
  - ORB: fixed scale steps → fails if camera distance varies
  - SIFT: automatic scale  → works at any zoom level

Dual scoring:
  - match_ratio  = inliers / master_keypoints * 100%  (quantity of matches)
  - avg_distance = average descriptor distance          (quality of matches)
  GOOD only if match_ratio >= ratio_threshold AND avg_distance <= dist_threshold

From test data pattern:
  Good labels : avg_dist ~78–90   (strong, confident matches)
  Bad labels  : avg_dist ~100–120 (weaker, less confident matches)

Usage:
    python3 sift_match.py
    python3 sift_match.py --ref reference/good2.jpeg --labels labels
    python3 sift_match.py --ratio 25 --dist 100

Controls:
    Any key = next image
    ESC     = quit
"""

import cv2
import numpy as np
import os
import argparse

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE        = (800, 600)
RATIO_THRESHOLD = 25.0      # match_ratio % >= this → pass
DIST_THRESHOLD  = 100.0     # avg_distance <= this  → pass (lower = better)
RATIO_TEST      = 0.75      # Lowe's ratio test for FLANN
MAX_DRAW        = 80        # max match lines in preview


# ── SIFT helpers ──────────────────────────────────────────────────────────────

def extract_sift(img_gray, sift):
    kp, des = sift.detectAndCompute(img_gray, None)
    return kp, des


def flann_match(des1, des2):
    """FLANN matcher — faster and more accurate than BF for SIFT."""
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return []
    index_params  = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    try:
        raw = flann.knnMatch(des1, des2, k=2)
    except cv2.error:
        return []
    return [m for m, n in raw if m.distance < RATIO_TEST * n.distance]


def ransac_inliers(kp1, kp2, good):
    if len(good) < 8:
        return good, None
    pts1    = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2    = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    if mask is None:
        return good, None
    return [m for m, mk in zip(good, mask.ravel()) if mk], mask


# ── Preview ───────────────────────────────────────────────────────────────────

def draw_preview(master_gray, kp1, test_gray, kp2,
                 inliers, match_ratio, avg_dist, result, filename,
                 ratio_ok, dist_ok):
    color = (0, 220, 0) if result == "GOOD" else (0, 0, 220)

    # Draw all keypoints on both images
    master_img = cv2.cvtColor(master_gray, cv2.COLOR_GRAY2BGR)
    test_img   = cv2.cvtColor(test_gray,   cv2.COLOR_GRAY2BGR)
    cv2.drawKeypoints(master_gray, kp1, master_img,
                      color=(0, 180, 255),
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.drawKeypoints(test_gray, kp2, test_img,
                      color=(0, 180, 255),
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    canvas   = np.hstack([master_img, test_img])
    w_master = master_gray.shape[1]

    # Color match lines by distance quality
    for m in inliers[:MAX_DRAW]:
        pt1      = tuple(np.int32(kp1[m.queryIdx].pt))
        pt2      = tuple(np.int32(kp2[m.trainIdx].pt) + np.array([w_master, 0]))
        # Green = close distance (good quality), Yellow = farther distance
        line_col = (0, 220, 0) if m.distance < DIST_THRESHOLD else (0, 200, 220)
        cv2.line(canvas, pt1, pt2, line_col, 1)
        cv2.circle(canvas, pt1, 3, line_col, -1)
        cv2.circle(canvas, pt2, 3, line_col, -1)

    cv2.line(canvas, (w_master, 0), (w_master, canvas.shape[0]), (255, 255, 0), 2)
    cv2.putText(canvas, "MASTER", (10, canvas.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    cv2.putText(canvas, "TEST",   (w_master + 10, canvas.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    target_h = 480
    scale    = target_h / canvas.shape[0]
    canvas   = cv2.resize(canvas, (int(canvas.shape[1] * scale), target_h))

    # Header bar — separate from image, no overlap
    bar_h = 90
    bar   = np.zeros((bar_h, canvas.shape[1], 3), dtype=np.uint8)
    bar[:] = (25, 25, 25)

    cv2.putText(bar, f"File: {filename}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)

    ratio_col = (0, 220, 0) if ratio_ok else (0, 0, 220)
    dist_col  = (0, 220, 0) if dist_ok  else (0, 0, 220)
    cv2.putText(bar, f"Match: {match_ratio:.1f}%",
                (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, ratio_col, 2)
    cv2.putText(bar, f"AvgDist: {avg_dist:.1f}",
                (220, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, dist_col, 2)
    cv2.putText(bar, f"->  {result}",
                (430, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    reason = []
    if not ratio_ok: reason.append(f"match {match_ratio:.1f}% < {RATIO_THRESHOLD}%")
    if not dist_ok:  reason.append(f"dist {avg_dist:.1f} > {DIST_THRESHOLD}")
    hint = "  [" + "  &  ".join(reason) + "]" if reason else ""
    cv2.putText(bar, hint, (10, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 120, 120), 1)

    return np.vstack([bar, canvas])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref",    default="reference/good2.jpeg")
    parser.add_argument("--labels", default="labels")
    parser.add_argument("--ratio",  type=float, default=RATIO_THRESHOLD,
                        help=f"Min match ratio %% (default {RATIO_THRESHOLD})")
    parser.add_argument("--dist",   type=float, default=DIST_THRESHOLD,
                        help=f"Max avg descriptor distance (default {DIST_THRESHOLD})")
    args = parser.parse_args()

    master_bgr = cv2.imread(args.ref)
    if master_bgr is None:
        print(f"ERROR: reference not found: {args.ref}")
        return

    master_gray = cv2.cvtColor(master_bgr, cv2.COLOR_BGR2GRAY)
    master_gray = cv2.resize(master_gray, IMG_SIZE)

    sift      = cv2.SIFT_create(nfeatures=5000)
    kp1, des1 = extract_sift(master_gray, sift)

    print(f"\nReference  : {args.ref}  ({len(kp1)} keypoints)")
    print(f"Threshold  : match_ratio >= {args.ratio:.1f}%  AND  avg_dist <= {args.dist:.1f}  → GOOD")
    print(f"             GREEN line = strong match  |  CYAN line = weak match")
    print(f"\nAny key = next    ESC = quit\n")

    files = sorted(f for f in os.listdir(args.labels)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png')))

    pass_count = fail_count = 0
    results = []

    for file in files:
        path     = os.path.join(args.labels, file)
        test_bgr = cv2.imread(path)
        if test_bgr is None:
            continue

        test_gray = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)
        test_gray = cv2.resize(test_gray, IMG_SIZE)

        kp2, des2  = extract_sift(test_gray, sift)
        good       = flann_match(des1, des2)
        inliers, _ = ransac_inliers(kp1, kp2, good)

        match_ratio = len(inliers) / max(len(kp1), 1) * 100
        avg_dist    = np.mean([m.distance for m in inliers]) if inliers else 999.0

        ratio_ok = match_ratio >= args.ratio
        dist_ok  = avg_dist   <= args.dist
        result   = "GOOD" if (ratio_ok and dist_ok) else "BAD"

        reason = []
        if not ratio_ok: reason.append(f"match {match_ratio:.1f}%<{args.ratio}%")
        if not dist_ok:  reason.append(f"dist {avg_dist:.1f}>{args.dist}")
        reason_str = "  [" + " & ".join(reason) + "]" if reason else ""

        print(f"  {file}")
        print(f"    kp_test     : {len(kp2)}")
        print(f"    inliers     : {len(inliers)}")
        print(f"    match_ratio : {match_ratio:.2f}%  ({'ok' if ratio_ok else 'FAIL'})")
        print(f"    avg_distance: {avg_dist:.1f}  ({'ok' if dist_ok else 'FAIL'})")
        print(f"    Result      : {'✅' if result == 'GOOD' else '❌'} {result}{reason_str}\n")

        results.append((file, len(inliers), match_ratio, avg_dist, result))
        if result == "GOOD": pass_count += 1
        else:                 fail_count += 1

        preview = draw_preview(master_gray, kp1, test_gray, kp2,
                               inliers, match_ratio, avg_dist, result, file,
                               ratio_ok, dist_ok)
        cv2.imshow("SIFT Match  (Master | Test)", preview)

        if cv2.waitKey(0) == 27:
            break

    total = pass_count + fail_count
    print(f"\n{'='*62}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*62}")
    print(f"  GOOD : {pass_count}/{total}")
    print(f"  BAD  : {fail_count}/{total}")
    print(f"\n  {'File':<24} {'Inliers':>7} {'Match%':>7} {'AvgDist':>9}  Result")
    print(f"  {'-'*58}")
    for name, inl, mr, ad, res in results:
        icon = "✅" if res == "GOOD" else "❌"
        print(f"  {icon} {name:<22} {inl:7d} {mr:6.2f}% {ad:9.1f}  {res}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

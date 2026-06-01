# sift_align_diff.py — How it Works

---

## What is SIFT?

SIFT = Scale-Invariant Feature Transform

It finds special points (corners, edges, blobs) on an image
and describes each point as a unique number fingerprint.

The key word is SCALE-INVARIANT:
- It finds the SAME points regardless of zoom level
- It finds the SAME points regardless of rotation
- It finds the SAME points regardless of lighting

---

## What is ORB?

ORB = Oriented FAST and Rotated BRIEF

Also finds special points — but it is NOT scale-invariant.
It uses a fixed pyramid (fixed zoom steps).

---

## Why NOT ORB?

Your photos are taken by hand.
Camera distance changes every shot.

Example:
  Master photo  → label fills 80% of frame
  Test photo    → label fills 50% of frame (shot from further)

  ORB sees them as DIFFERENT SIZE → cannot match → wrong result

  good12.jpg with ORB → 7.62% match → BAD  ← WRONG
  good12.jpg with SIFT → 28.79% match → GOOD ← CORRECT

ORB works like stairs — fixed steps, cannot handle in-between zoom.
SIFT works like a ramp — continuous, handles any zoom.

  ORB:                        SIFT:
  Step 1: 100%                Any zoom level
  Step 2:  75%                works automatically
  Step 3:  50%
  (nothing between steps)

---

## Why SIFT?

  Camera too close   → SIFT handles it
  Camera too far     → SIFT handles it
  Label slightly tilted → SIFT handles it
  Different lighting → SIFT handles it

ORB fails all of the above when difference is too large.

---

## Full Pipeline

  Step 1: SIFT finds matching points on master + test
          (works at any zoom/angle)
                |
  Step 2: Compute homography (warp formula)
          Warp test image → align exactly to master
                |
  Step 3: Compare every pixel — SSIM per-pixel score
          Count pixels where score < 0.5 → diff_area %
                |
  Step 4: Decision
          SSIM >= 0.73  AND  diff_area < 10% → GOOD
          otherwise → BAD

---

## Why two checks?

  SSIM score    → overall label structure match
  diff_area %   → catches stamps, handwriting, marks added on label

  A bad label with a stamp:
    SSIM = 0.82  (still looks similar overall)
    diff_area = 13%  (but 13% of pixels changed) → BAD ✅

---

## Result

  Good images : 14/14  (100%)
  Bad images  : 26/33  ( 79%)
  Overall     : 40/47  ( 85%)
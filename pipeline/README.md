# Label Inspector Pipeline

## Folder structure
```
pipeline/
  inspector.py           ← SIFT+SSIM inspector (main)
  anomalib_inspector.py  ← PatchCore deep-learning inspector
  region.json            ← saved label region from setup
  mask.json              ← saved hologram mask (relative offsets)
  reference/
    master.jpg           ← master reference (auto-saved on setup)
  results/
    good/                ← auto-saved GOOD label crops + heatmaps
    bad/                 ← auto-saved BAD label crops + heatmaps
```

---

## inspector.py — SIFT + SSIM

### Workflow

```
START
  │
  ├─ master.jpg exists? ──YES──► Skip to INSPECTION
  │
  NO
  │
  ▼
SETUP (press M)
  1. Draw box around a GOOD label         → region.json  (stores master_area)
  2. Draw box over hologram zone (opt.)   → mask.json    (relative offsets)
  3. Crop saved as reference/master.jpg

  │
  ▼
INSPECTION (automatic)
  1. V-threshold detects label bboxes in every frame
  2. LabelTracker watches center zone:
       WAITING → label enters zone → IN_ZONE
       IN_ZONE → label stable (IoU check) → CHECKED  ← check fires here
       CHECKED → new label arrives (low IoU) → IN_ZONE
  3. On CHECKED trigger (or C key if AUTO_CHECK=False):
       a. Area check     — detected w×h vs master_area  (±AREA_TOL_PCT %)
       b. Crop bbox      — extract label from frame
       c. SIFT align     — find keypoints, warp test onto master
       d. Two-scale SSIM — fine (blur=3) + coarse (blur=31) diff
       e. Apply mask     — hologram zone zeroed out before compare
       f. Verdict        — ALL checks pass → GOOD, any fail → BAD
  4. Crop saved to results/good/ or results/bad/
  5. 4-panel result window: Master | Aligned | Diff | Heatmap
```

### How the comparison works

```
SIFT matching (RATIO_TEST=0.75)
  → homography needs MIN_INLIERS=80 agreeing points
    → warp test label onto master
      → SSIM score      ≥ SSIM_PASS (0.73)?       ✓/✗
      → fine diff %     < FINE_FAIL_PCT (2%)?      ✓/✗   ← catches small print errors
      → coarse diff %   < COARSE_FAIL_PCT (2%)?    ✓/✗   ← catches big missing zones
        → ALL pass → GOOD   |   any fail → BAD
```

### AUTO_CHECK flag
```python
# top of inspector.py
AUTO_CHECK = True   # True  = auto-check when tracker fires
                    # False = only check on C key press
```

### Run
```bash
python3 pipeline/inspector.py                        # TCP stream (default)
python3 pipeline/inspector.py --camera 0             # webcam
python3 pipeline/inspector.py --ref reference/master.jpg
```

### Controls
| Key | Action |
|-----|--------|
| M   | Setup — draw region + mask, capture master |
| C   | Manually check current center label |
| R   | Reset tracker + last result |
| Q   | Quit |

### Tuning (top of inspector.py)
| Variable | Default | Meaning |
|----------|---------|---------|
| `AUTO_CHECK` | `True` | Enable/disable automatic checking |
| `V_LOW` | 130 | Min brightness to detect label |
| `CENTER_ZONE` | 0.35 | How centred label must be (0=exact, 1=anywhere) |
| `AREA_TOL_PCT` | 20 % | Allowed area size difference vs master |
| `CHECK_COOLDOWN` | 2.0 s | Min seconds between auto-checks |
| `SSIM_PASS` | 0.73 | SSIM score minimum to pass |
| `FINE_FAIL_PCT` | 2.0 % | Max fine-detail diff % allowed |
| `COARSE_FAIL_PCT` | 2.0 % | Max coarse diff % allowed |
| `MIN_INLIERS` | 80 | Min SIFT keypoints needed for alignment |

---

## anomalib_inspector.py — PatchCore (deep learning)

### Workflow

```
START
  │
  ▼
TRAINING (press T)
  1. Hold a GOOD label in the center zone
  2. Press T — auto-collects --samples crops (default 15)
     one crop every 0.4 s while label is stable in zone
  3. PatchCore trains on those crops (~30–60 s on CPU)
  4. Model ready for the session

  │
  ▼
INSPECTION (automatic)
  1. V-threshold detects label bboxes (same as inspector.py)
  2. LabelTracker watches center zone (WAITING → IN_ZONE → CHECKED)
  3. On CHECKED trigger (or C key):
       a. Crop the detected bbox
       b. Score with PatchCore → raw anomaly score + spatial heatmap
       c. Normalise score across all scores seen this session
       d. norm score ≥ thresh → BAD   |   norm score < thresh → GOOD
  4. Crop + heatmap overlay saved to results/good/ or results/bad/
  5. Result panel: original crop | jet heatmap overlay + score bar
```

### Run
```bash
python3 pipeline/anomalib_inspector.py                   # TCP stream
python3 pipeline/anomalib_inspector.py --camera 0        # webcam
python3 pipeline/anomalib_inspector.py --thresh 0.4 --samples 25
```

### Controls
| Key | Action |
|-----|--------|
| T   | Collect training crops and train PatchCore |
| C   | Manually check current center label |
| R   | Reset tracker + last result |
| Q   | Quit |

### Tuning (top of anomalib_inspector.py)
| Variable | Default | Meaning |
|----------|---------|---------|
| `V_LOW` | 130 | Min brightness to detect label |
| `CENTER_ZONE` | 0.35 | How centred label must be |
| `TRAIN_SAMPLES` | 15 | Number of crops to collect for training |
| `SAMPLE_INTERVAL` | 0.4 s | Seconds between training crop captures |
| `CHECK_COOLDOWN` | 2.0 s | Min seconds between auto-checks |
| `IMG_SIZE` | 256 | Image size fed to PatchCore |

---

## Which inspector to use?

| | `inspector.py` | `anomalib_inspector.py` |
|---|---|---|
| Method | SIFT keypoint align + SSIM | PatchCore deep feature matching |
| Setup | Draw box + capture master | Collect 15 good-label crops |
| Speed | Fast (CPU, ~50 ms) | Slower first run (training), then ~200 ms |
| Best for | Flat labels, clear print | Complex textures, holograms, gradients |
| Explainability | Diff map + SSIM score | Spatial heatmap only |

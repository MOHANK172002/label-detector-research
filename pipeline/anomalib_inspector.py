"""
Label Defect Inspector — PatchCore Edition
==========================================
Pipeline:

  TRAINING (press T):
    1. Bring a GOOD label into the center zone.
    2. Press T — the system auto-collects TRAIN_SAMPLES crops of that label
       as it sits still (one crop per stable frame, ~0.5 s apart).
    3. PatchCore is trained on those crops.
    4. Model is saved to pipeline/model/ for reuse next run.

  INSPECTION (automatic after training):
    1. V-threshold detects individual label bboxes (same as inspector.py).
    2. LabelTracker watches the center zone (WAITING → IN_ZONE → CHECKED).
    3. When a NEW label stabilises in the zone:
         a. Crop the detected bbox.
         b. Score with PatchCore → anomaly_score + heatmap.
         c. GOOD / BAD decision → save to pipeline/results/.

Controls:
  T       = start training capture (collect good-label crops)
  C       = manual check of current center label
  R       = reset last result / tracker
  Q/ESC   = quit

Usage:
  python3 pipeline/anomalib_inspector.py
  python3 pipeline/anomalib_inspector.py --camera 0
  python3 pipeline/anomalib_inspector.py --thresh 0.5 --samples 20
"""

import argparse
import os
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "model")
GOOD_DIR    = os.path.join(BASE_DIR, "results", "good")
BAD_DIR     = os.path.join(BASE_DIR, "results", "bad")

DEFAULT_URL = (
    "tcp://192.168.1.11:8888"
    "?fflags=nobuffer&flags=low_delay&framedrop=1"
)

# ── detection constants (mirrors inspector.py) ─────────────────────────────────
V_LOW       = 130
V_HIGH      = 255
MORPH_K     = 0
MIN_AREA_P  = 9        # % of frame area

CENTER_ZONE = 0.35     # fraction of half-frame

# ── PatchCore / scoring ────────────────────────────────────────────────────────
IMG_SIZE       = 256   # square resize fed to PatchCore
TRAIN_SAMPLES  = 15    # crops to collect during training phase
SAMPLE_INTERVAL = 0.4  # seconds between training crops
BACKBONE       = "wide_resnet50_2"

# ── tracker ────────────────────────────────────────────────────────────────────
IOU_SAME       = 0.35
IOU_NEW        = 0.10
CHECK_COOLDOWN = 2.0


# ══════════════════════════════════════════════════════════════════════════════
# LABEL DETECTION  (from inspector.py)
# ══════════════════════════════════════════════════════════════════════════════

def detect_labels(frame):
    fh, fw = frame.shape[:2]
    v_ch   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]
    mask   = cv2.inRange(v_ch, V_LOW, V_HIGH)
    if MORPH_K > 0:
        k    = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_K, MORPH_K))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    flood_m = np.zeros((fh + 2, fw + 2), dtype=np.uint8)
    flooded = mask.copy()
    cv2.floodFill(flooded, flood_m, (0, 0), 255)
    mask = mask | cv2.bitwise_not(flooded)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = (MIN_AREA_P / 100.0) * fh * fw
    boxes = []
    for cnt in cnts:
        if cv2.contourArea(cnt) < min_area:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bw > fw * 0.95 or bh > fh * 0.95:
            continue
        if bw < bh * 0.5:
            continue
        boxes.append((bx, by, bw, bh))
    boxes.sort(key=lambda b: b[1])
    return boxes


def pick_center(boxes, fw, fh):
    cx_f, cy_f = fw // 2, fh // 2
    zx = fw * CENTER_ZONE
    zy = fh * CENTER_ZONE
    best_i, best_d = None, float("inf")
    for i, (x, y, w, h) in enumerate(boxes):
        if x < 4 or y < 4 or x + w > fw - 4 or y + h > fh - 4:
            continue
        cx, cy = x + w // 2, y + h // 2
        if abs(cx - cx_f) > zx or abs(cy - cy_f) > zy:
            continue
        d = ((cx - cx_f) ** 2 + (cy - cy_f) ** 2) ** 0.5
        if d < best_d:
            best_d, best_i = d, i
    return best_i


# ══════════════════════════════════════════════════════════════════════════════
# TRACKER  (from inspector.py)
# ══════════════════════════════════════════════════════════════════════════════

class LabelTracker:
    """WAITING → IN_ZONE → CHECKED state machine using IoU."""

    def __init__(self):
        self.state      = "WAITING"
        self.last_box   = None
        self.last_check = 0.0

    def _iou(self, a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
        iy = max(0, min(ay + ah, by + bh) - max(ay, by))
        inter = ix * iy
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    def update(self, center_box):
        """Returns True when a new, stable label should be checked."""
        now = time.time()

        if center_box is None:
            if self.state in ("IN_ZONE", "CHECKED"):
                self.state    = "WAITING"
                self.last_box = None
            return False

        if self.state == "WAITING":
            self.state    = "IN_ZONE"
            self.last_box = center_box
            return False

        if self.state == "IN_ZONE":
            iou = self._iou(center_box, self.last_box)
            if iou > IOU_SAME:
                self.last_box = center_box
                if (now - self.last_check) > CHECK_COOLDOWN:
                    self.state      = "CHECKED"
                    self.last_check = now
                    return True
            else:
                self.last_box = center_box
            return False

        if self.state == "CHECKED":
            iou = self._iou(center_box, self.last_box)
            if iou < IOU_NEW:
                self.state    = "IN_ZONE"
                self.last_box = center_box
            else:
                self.last_box = center_box
            return False

        return False

    def get_state(self):
        return self.state


# ══════════════════════════════════════════════════════════════════════════════
# PATCHCORE TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_patchcore(crop_paths: list, work_dir: Path):
    """Train PatchCore on a list of good-label crop paths. Returns (engine, model)."""
    import torch
    from anomalib.data import Folder
    from anomalib.models import Patchcore
    from anomalib.engine import Engine
    from torchvision.transforms import v2 as T

    normal_dir = work_dir / "normal"
    normal_dir.mkdir(parents=True, exist_ok=True)
    for p in crop_paths:
        shutil.copy2(p, normal_dir / Path(p).name)

    aug = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ])

    datamodule = Folder(
        name="label",
        normal_dir=str(normal_dir),
        root=str(work_dir),
        augmentations=aug,
        train_batch_size=16,
        eval_batch_size=1,
        num_workers=0,
        test_split_mode="none",
        val_split_mode="from_train",
        val_split_ratio=0.2,
    )

    model = Patchcore(
        backbone=BACKBONE,
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
    )

    engine = Engine(
        default_root_dir=str(work_dir / "lightning"),
        max_epochs=1,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_progress_bar=True,
    )

    print(f"[PatchCore] Training on {len(crop_paths)} label crops...")
    t0 = time.time()
    engine.fit(model=model, datamodule=datamodule)
    print(f"[PatchCore] Done — {time.time() - t0:.1f}s")
    return engine, model


# ══════════════════════════════════════════════════════════════════════════════
# SCORING
# ══════════════════════════════════════════════════════════════════════════════

_tmp_score_dir = None

def _get_tmp_score_dir():
    global _tmp_score_dir
    if _tmp_score_dir is None:
        _tmp_score_dir = tempfile.mkdtemp(prefix="anomalib_score_")
    return _tmp_score_dir


def score_crop(engine, model, crop_bgr: np.ndarray):
    """
    Score a BGR crop in memory.
    Returns (score: float, heatmap: np.ndarray H×W float32, infer_ms: float).
    score: mean anomaly_map value — higher = more anomalous.
    """
    tmp_dir  = _get_tmp_score_dir()
    tmp_path = os.path.join(tmp_dir, f"_score_{time.time_ns()}.jpg")
    cv2.imwrite(tmp_path, crop_bgr)
    try:
        results = engine.predict(model=model, data_path=tmp_path, return_predictions=True)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    if not results:
        return 0.0, None, 0.0

    batch   = results[0]
    amap    = batch.anomaly_map
    if amap is None:
        return 0.0, None, 0.0

    amap_np = amap.squeeze().cpu().numpy().astype(np.float32)
    return float(amap_np.mean()), amap_np, 0.0


def _heatmap_overlay(bgr: np.ndarray, amap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    if amap is None:
        return bgr
    h, w = bgr.shape[:2]
    amap_r = cv2.resize(amap, (w, h))
    mn, mx = amap_r.min(), amap_r.max()
    if mx > mn:
        amap_u8 = ((amap_r - mn) / (mx - mn) * 255).astype(np.uint8)
    else:
        amap_u8 = np.zeros((h, w), dtype=np.uint8)
    heat = cv2.applyColorMap(amap_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr, 1 - alpha, heat, alpha, 0)


# ══════════════════════════════════════════════════════════════════════════════
# RESULT PANEL
# ══════════════════════════════════════════════════════════════════════════════

def build_result_panel(crop_bgr, amap, score, norm_score, verdict, thresh):
    """Two-panel: original crop | heatmap overlay, with score bar below."""
    color = (0, 220, 0) if verdict == "GOOD" else (0, 0, 220)
    ph    = 300
    def rs(img):
        s = ph / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * s), ph))

    p1 = rs(crop_bgr)
    p2 = rs(_heatmap_overlay(crop_bgr, amap))

    cv2.putText(p1, "CROP",    (8, ph - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    cv2.putText(p2, "HEATMAP", (8, ph - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    grid = np.hstack([p1, p2])
    bar  = np.full((90, grid.shape[1], 3), 25, dtype=np.uint8)

    cv2.putText(bar, f"Raw score: {score:.4f}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.putText(bar, f"Norm score: {norm_score:.3f}  (thresh {thresh:.2f})",
                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.putText(bar, f">> {verdict}",
                (int(grid.shape[1] * 0.6), 56), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(bar, "T=train  C=check  R=reset  Q=quit",
                (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    return np.vstack([bar, grid])


# ══════════════════════════════════════════════════════════════════════════════
# LIVE FEED DRAWING
# ══════════════════════════════════════════════════════════════════════════════

TRACKER_COLOR = {
    "WAITING": (180, 180, 180),
    "IN_ZONE": (0, 220, 255),
    "CHECKED": (0, 200, 0),
}


def draw_live(frame, boxes, center_i, tracker_state,
              last_verdict, model_ready, collecting, collected, total_samples):
    out = frame.copy()
    fh, fw = frame.shape[:2]

    for i, (x, y, w, h) in enumerate(boxes):
        is_c  = (i == center_i)
        fully = x > 4 and y > 4 and x + w < fw - 4 and y + h < fh - 4

        if is_c:
            color = TRACKER_COLOR.get(tracker_state, (0, 255, 0))
            thick = 3
            tag   = f"{tracker_state}  {w}x{h}px"
        elif fully:
            color, thick, tag = (255, 255, 255), 1, f"{w}x{h}"
        else:
            color, thick, tag = (0, 140, 255), 1, "PARTIAL"

        cv2.rectangle(out, (x, y), (x + w, y + h), color, thick)
        t = 18 if is_c else 10
        for px, py, dx, dy in [(x,y,1,1),(x+w,y,-1,1),(x+w,y+h,-1,-1),(x,y+h,1,-1)]:
            cv2.line(out, (px, py), (px + dx * t, py), color, thick)
            cv2.line(out, (px, py), (px, py + dy * t), color, thick)
        cv2.putText(out, tag, (x + 4, max(y - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 if is_c else 0.4, color, 1)

    # Center zone
    zx = int(fw * CENTER_ZONE)
    zy = int(fh * CENTER_ZONE)
    cx_f, cy_f = fw // 2, fh // 2
    cv2.rectangle(out, (cx_f - zx, cy_f - zy), (cx_f + zx, cy_f + zy), (0, 200, 255), 1)
    cv2.line(out, (cx_f - 15, cy_f), (cx_f + 15, cy_f), (0, 200, 255), 2)
    cv2.line(out, (cx_f, cy_f - 15), (cx_f, cy_f + 15), (0, 200, 255), 2)

    # Status bar
    if collecting:
        pct = collected / total_samples
        bar_w = int((fw - 20) * pct)
        cv2.rectangle(out, (0, fh - 55), (fw, fh), (20, 60, 20), -1)
        cv2.rectangle(out, (10, fh - 40), (10 + bar_w, fh - 15), (0, 200, 0), -1)
        cv2.putText(out, f"COLLECTING TRAINING CROPS  {collected}/{total_samples}",
                    (10, fh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    elif last_verdict:
        bc = (0, 60, 0) if last_verdict == "GOOD" else (60, 0, 0)
        cv2.rectangle(out, (0, fh - 55), (fw, fh), bc, -1)
        cv2.putText(out, f"LAST: {last_verdict}   T=train  C=check  R=reset  Q=quit",
                    (10, fh - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    else:
        if not model_ready:
            msg = "NO MODEL — press T with a GOOD label in view to train"
            bc  = (60, 20, 20)
        else:
            msg = "Ready — bring label to CENTER zone   T=retrain  C=check  Q=quit"
            bc  = (25, 25, 25)
        cv2.rectangle(out, (0, fh - 38), (fw, fh), bc, -1)
        cv2.putText(out, msg, (10, fh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════

def save_result(crop_bgr, amap, verdict):
    folder = GOOD_DIR if verdict == "GOOD" else BAD_DIR
    os.makedirs(folder, exist_ok=True)
    ts  = time.strftime("%Y%m%d_%H%M%S")
    ms  = int((time.time() % 1) * 1000)
    base = f"{verdict}_{ts}_{ms:03d}"
    cv2.imwrite(os.path.join(folder, f"{base}.jpg"), crop_bgr)
    if amap is not None:
        h, w = crop_bgr.shape[:2]
        amap_r = cv2.resize(amap, (w, h))
        mn, mx = amap_r.min(), amap_r.max()
        if mx > mn:
            u8 = ((amap_r - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            u8 = np.zeros((h, w), dtype=np.uint8)
        heat = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(crop_bgr, 0.5, heat, 0.5, 0)
        cv2.imwrite(os.path.join(folder, f"{base}_heat.jpg"), overlay)
    return os.path.join(folder, f"{base}.jpg")


# ══════════════════════════════════════════════════════════════════════════════
# NORMALISER  (keeps a running min/max to normalise scores session-wide)
# ══════════════════════════════════════════════════════════════════════════════

class ScoreNormaliser:
    def __init__(self):
        self._min = None
        self._max = None

    def update(self, score):
        if self._min is None or score < self._min:
            self._min = score
        if self._max is None or score > self._max:
            self._max = score

    def normalise(self, score):
        if self._min is None or self._max == self._min:
            return 0.5
        return (score - self._min) / (self._max - self._min)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING CAPTURE LOOP
# ══════════════════════════════════════════════════════════════════════════════

def collect_training_crops(cap, n_samples, sample_interval):
    """
    Collect n_samples crops of the center label while the user holds it still.
    Shows progress overlay. Returns list of saved file paths (in a temp dir),
    or [] on cancel.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="anomalib_train_"))
    crops   = []
    last_t  = 0.0
    tracker = LabelTracker()
    print(f"[Training] Hold a GOOD label in the center zone. "
          f"Collecting {n_samples} crops...")

    while len(crops) < n_samples:
        ret, frame = cap.read()
        if not ret:
            continue

        fh, fw = frame.shape[:2]
        boxes    = detect_labels(frame)
        center_i = pick_center(boxes, fw, fh)
        center_box = boxes[center_i] if center_i is not None else None

        # Use tracker to know when label is stable
        stable = tracker.update(center_box)
        now    = time.time()

        disp = frame.copy()

        if center_box is not None:
            bx, by, bw, bh = center_box
            state = tracker.get_state()
            color = TRACKER_COLOR.get(state, (0, 200, 255))
            cv2.rectangle(disp, (bx, by), (bx + bw, by + bh), color, 3)

        # Collect a crop when label is stable and interval has passed
        if center_box is not None and tracker.get_state() in ("IN_ZONE", "CHECKED"):
            if (now - last_t) >= sample_interval:
                bx, by, bw, bh = center_box
                crop = frame[by:by + bh, bx:bx + bw]
                path = str(tmp_dir / f"train_{len(crops):04d}.jpg")
                cv2.imwrite(path, crop)
                crops.append(path)
                last_t = now
                print(f"  Crop {len(crops)}/{n_samples}")

        # Progress overlay
        pct   = len(crops) / n_samples
        bar_w = int((fw - 20) * pct)
        cv2.rectangle(disp, (0, fh - 55), (fw, fh), (20, 60, 20), -1)
        cv2.rectangle(disp, (10, fh - 40), (10 + bar_w, fh - 15), (0, 200, 0), -1)
        cv2.putText(disp,
                    f"COLLECTING  {len(crops)}/{n_samples}   ESC=cancel",
                    (10, fh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Zone crosshair
        zx = int(fw * CENTER_ZONE)
        zy = int(fh * CENTER_ZONE)
        cx_f, cy_f = fw // 2, fh // 2
        cv2.rectangle(disp, (cx_f - zx, cy_f - zy), (cx_f + zx, cy_f + zy), (0, 200, 255), 1)

        cv2.imshow("Label Inspector — Anomalib", disp)
        if cv2.waitKey(1) & 0xFF == 27:
            print("[Training] Cancelled.")
            return []

    print(f"[Training] Collected {len(crops)} crops → {tmp_dir}")
    return crops, tmp_dir


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera",  type=int,   default=None,    help="Webcam index (default: TCP stream)")
    parser.add_argument("--thresh",  type=float, default=0.5,     help="Normalised anomaly threshold 0-1 (default 0.5)")
    parser.add_argument("--samples", type=int,   default=TRAIN_SAMPLES, help="Training crops to collect (default 15)")
    args = parser.parse_args()

    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
    else:
        cap = cv2.VideoCapture(DEFAULT_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("ERROR: cannot open stream/camera")
        return

    WIN_LIVE   = "Label Inspector — Anomalib"
    WIN_RESULT = "Result"
    cv2.namedWindow(WIN_LIVE,   cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_RESULT, cv2.WINDOW_NORMAL)

    # State
    engine       = None
    model        = None
    model_ready  = False
    normaliser   = ScoreNormaliser()
    tracker      = LabelTracker()
    last_verdict = None
    last_score   = None
    last_norm    = None
    result_panel = None
    _train_tmp   = None   # keep tmp dir alive between train & main loop

    print("T=train  C=check  R=reset  Q=quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        fh, fw = frame.shape[:2]
        boxes    = detect_labels(frame)
        center_i = pick_center(boxes, fw, fh)
        center_box = boxes[center_i] if center_i is not None else None

        # Tracker → auto-check when new label stabilises
        should_check = tracker.update(center_box)

        if should_check and model_ready and center_box is not None:
            bx, by, bw, bh = center_box
            crop     = frame[by:by + bh, bx:bx + bw]
            t0       = time.time()
            score, amap, _ = score_crop(engine, model, crop)
            ms       = (time.time() - t0) * 1000
            normaliser.update(score)
            norm     = normaliser.normalise(score)
            verdict  = "GOOD" if norm < args.thresh else "BAD"

            last_verdict = verdict
            last_score   = score
            last_norm    = norm
            fname        = save_result(crop, amap, verdict)

            print(f"  [{verdict}]  raw={score:.4f}  norm={norm:.3f}  "
                  f"thresh={args.thresh}  ({ms:.0f}ms)  → {fname}")

            result_panel = build_result_panel(
                crop, amap, score, norm, verdict, args.thresh)

        # Draw live
        display = draw_live(
            frame, boxes, center_i, tracker.get_state(),
            last_verdict, model_ready,
            collecting=False, collected=0, total_samples=args.samples)
        cv2.imshow(WIN_LIVE, display)

        if result_panel is not None:
            cv2.imshow(WIN_RESULT, result_panel)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break

        elif key in (ord('r'), ord('R')):
            tracker      = LabelTracker()
            last_verdict = None
            last_score   = None
            last_norm    = None
            result_panel = None
            print("Reset.")

        elif key in (ord('t'), ord('T')):
            # ── Training capture ──────────────────────────────────────────
            result = collect_training_crops(cap, args.samples, SAMPLE_INTERVAL)
            if result:
                crop_paths, tmp_dir = result
                _train_tmp = tmp_dir   # keep alive
                work_dir   = tmp_dir / "patchcore_work"
                work_dir.mkdir(parents=True, exist_ok=True)
                try:
                    engine, model = train_patchcore(crop_paths, work_dir)
                    model_ready   = True
                    normaliser    = ScoreNormaliser()
                    tracker       = LabelTracker()
                    last_verdict  = None
                    result_panel  = None
                    print("[Training] Model ready. Bring labels into the zone to inspect.\n")
                except Exception as e:
                    print(f"[Training] ERROR: {e}")

        elif key in (ord('c'), ord('C')):
            # ── Manual check ──────────────────────────────────────────────
            check_box = center_box if center_box is not None else tracker.last_box
            if not model_ready:
                print("No model — press T to train first.")
            elif check_box is None:
                print("No label in view.")
            else:
                bx, by, bw, bh = check_box
                crop    = frame[by:by + bh, bx:bx + bw]
                score, amap, _ = score_crop(engine, model, crop)
                normaliser.update(score)
                norm    = normaliser.normalise(score)
                verdict = "GOOD" if norm < args.thresh else "BAD"
                last_verdict = verdict
                last_score   = score
                last_norm    = norm
                fname        = save_result(crop, amap, verdict)
                print(f"  [C/{verdict}]  raw={score:.4f}  norm={norm:.3f}  → {fname}")
                result_panel = build_result_panel(
                    crop, amap, score, norm, verdict, args.thresh)

    cap.release()
    cv2.destroyAllWindows()

    # Cleanup tmp score dir
    global _tmp_score_dir
    if _tmp_score_dir and os.path.isdir(_tmp_score_dir):
        shutil.rmtree(_tmp_score_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

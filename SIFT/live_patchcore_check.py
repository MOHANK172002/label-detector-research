"""
PatchCore Live Label Defect Checker
=====================================
Auto-detects the label (HSV sat-suppress), collects reference crops,
trains PatchCore on them, then scores every new label crop.

Workflow
--------
  TRAIN MODE  (no trained model present or press T):
    - Camera runs; label is auto-detected every frame
    - Each stable detection is saved as a training crop (target = TRAIN_SAMPLES)
    - Once enough crops are collected PatchCore trains (takes ~30-60s on CPU)
    - Model saved to SIFT/patchcore_model/  and inference starts immediately

  INFER MODE  (after training or model already exists):
    - Every stable label crop is scored against the model
    - Anomaly score 0→1;  score >= THRESH → BAD
    - Result panel shows the anomaly heatmap (bright = defect region)
    - Crops saved to SIFT/results/good/ or SIFT/results/bad/

Controls
--------
  T       = re-train  (clears old crops, collects new ones)
  S       = force-score the current frame right now
  R       = reset tracker / last result
  Q / ESC = quit

Usage
-----
  python3 SIFT/live_patchcore_check.py
  python3 SIFT/live_patchcore_check.py --camera 0
  python3 SIFT/live_patchcore_check.py --thresh 0.5 --samples 20
"""

import cv2
import numpy as np
import os
import time
import argparse
import shutil
import tempfile
import threading
from pathlib import Path

# ── Stream ──────────────────────────────────────────────────────────────────────
DEFAULT_URL = (
    "tcp://192.168.1.11:8888"
    "?fflags=nobuffer&flags=low_delay&framedrop=1"
)

# ── Paths ────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "patchcore_model")
TRAIN_DIR   = os.path.join(BASE_DIR, "train_samples")   # saved crops you can inspect
GOOD_DIR    = os.path.join(BASE_DIR, "results", "good")
BAD_DIR     = os.path.join(BASE_DIR, "results", "bad")

# ── PatchCore ────────────────────────────────────────────────────────────────────
IMG_SIZE      = 256       # resize label crop for PatchCore (square)
BACKBONE      = "wide_resnet50_2"
TRAIN_SAMPLES = 70        # how many reference crops to collect before training
ANOMALY_THRESH = 0.5      # normalised score >= this → BAD  (re-calibrated after train)

# ── Label detector defaults (from crop_tool.py) ─────────────────────────────────
AD_S_MAX        = 255
AD_V_MIN        = 176
AD_MORPH_K      = 7
AD_MIN_AREA_PCT = 4
AD_PADDING      = 0

# ── Tracker ──────────────────────────────────────────────────────────────────────
CHECK_COOLDOWN = 2.0   # seconds between auto-scores in infer mode


# ══════════════════════════════════════════════════════════════════════════════
# LABEL DETECTOR  (identical to crop_tool.py / live_damage_check.py)
# ══════════════════════════════════════════════════════════════════════════════

def detect_label(frame):
    fh, fw = frame.shape[:2]
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]

    low_sat    = cv2.inRange(s_ch, 0,        AD_S_MAX)
    bright     = cv2.inRange(v_ch, AD_V_MIN, 255)
    label_mask = cv2.bitwise_and(low_sat, bright)
    label_mask = cv2.bitwise_not(label_mask)

    k_size  = max(AD_MORPH_K, 1)
    k_small = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    closed  = cv2.morphologyEx(label_mask, cv2.MORPH_CLOSE, k_small)

    big = max(k_size * 3 + 1, 11)
    if big % 2 == 0:
        big += 1
    k_big = cv2.getStructuringElement(cv2.MORPH_RECT, (big, big))
    solid = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_big)

    cnts, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = (AD_MIN_AREA_PCT / 100.0) * fh * fw
    cx_f, cy_f = fw // 2, fh // 2
    best_box  = None
    best_dist = float("inf")

    for cnt in cnts:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w > fw * 0.95 or h > fh * 0.95:
            continue
        if w < h * 0.3:
            continue
        dist = ((x + w // 2 - cx_f) ** 2 + (y + h // 2 - cy_f) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_box  = (x, y, w, h)

    if best_box is None:
        return None

    x, y, w, h = best_box
    x = max(0, x - AD_PADDING)
    y = max(0, y - AD_PADDING)
    w = min(fw - x, w + AD_PADDING * 2)
    h = min(fh - y, h + AD_PADDING * 2)
    return {"x": x, "y": y, "w": w, "h": h}


def crop_region(frame, region):
    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
    return frame[max(0, y):min(frame.shape[0], y + h),
                 max(0, x):min(frame.shape[1], x + w)]


# ══════════════════════════════════════════════════════════════════════════════
# STABILITY TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class LabelTracker:
    def __init__(self, cooldown=CHECK_COOLDOWN):
        self.cooldown     = cooldown
        self.last_mean    = None
        self.stable_since = 0.0
        self.last_check   = 0.0

    def update(self, crop):
        now  = time.time()
        mean = float(crop.mean())

        if self.last_mean is None:
            self.last_mean    = mean
            self.stable_since = now
            return False

        if abs(mean - self.last_mean) > 2.0:
            self.stable_since = now
            self.last_mean    = mean
            return False

        self.last_mean = mean

        if (now - self.stable_since) < 0.5:
            return False

        if (now - self.last_check) > self.cooldown:
            self.last_check = now
            return True

        return False

    def reset(self):
        self.last_mean    = None
        self.stable_since = 0.0
        self.last_check   = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# PATCHCORE TRAINER  (runs in a background thread)
# ══════════════════════════════════════════════════════════════════════════════

class PatchCoreTrainer:
    def __init__(self):
        self.engine = None
        self.model  = None
        self.score_min = 0.0   # raw score range captured from training images
        self.score_max = 1.0   # used to normalise inference scores
        self.ready  = False
        self.training = False
        self.status = "No model — collecting training crops"
        self._work_dir = None  # kept alive so model weights persist

    def train_async(self, image_paths: list, image_size: int, backbone: str,
                    on_done):
        self.training = True
        self.ready    = False
        self.status   = f"Training PatchCore on {len(image_paths)} crops …"

        def _run():
            try:
                import torch
                from anomalib.data    import Folder
                from anomalib.models  import Patchcore
                from anomalib.engine  import Engine
                from torchvision.transforms import v2 as T

                work_dir = Path(MODEL_DIR)
                work_dir.mkdir(parents=True, exist_ok=True)

                normal_dir = work_dir / "normal"
                if normal_dir.exists():
                    shutil.rmtree(normal_dir)
                normal_dir.mkdir()

                for p in image_paths:
                    shutil.copy2(p, normal_dir / Path(p).name)

                aug = T.Compose([
                    T.Resize((image_size, image_size)),
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
                    backbone=backbone,
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
                    enable_progress_bar=False,
                )

                engine.fit(model=model, datamodule=datamodule)

                # Score ALL training images to get the good-label score distribution.
                # score_max = mean + 3*std of training scores (3-sigma upper bound).
                # Anything above that at inference time is anomalous.
                raw_scores = []
                for p in image_paths:
                    r = engine.predict(model=model, data_path=p,
                                       return_predictions=True)
                    if r:
                        amap = r[0].anomaly_map
                        if amap is not None:
                            raw_scores.append(float(amap.squeeze().cpu().numpy().mean()))

                if raw_scores:
                    arr = np.array(raw_scores)
                    s_min = float(arr.min())
                    s_max = float(arr.mean() + 3.0 * arr.std()) if len(arr) > 1 else float(arr.max() * 2)
                else:
                    s_min, s_max = 0.0, 1.0

                self.engine    = engine
                self.model     = model
                self.score_min = s_min
                self.score_max = s_max
                self.ready     = True
                self.training  = False
                self.status    = "Model ready"
                print(f"[PatchCore] Training done. "
                      f"Train scores: min={s_min:.4f}  mean={float(np.mean(raw_scores)):.4f}  "
                      f"max={float(np.max(raw_scores)):.4f}  3σ-ceil={s_max:.4f}")
                print(f"  → A score of 0.0 = typical good label, 1.0 = very different")
                on_done()
            except Exception as exc:
                self.training = False
                self.status   = f"Training FAILED: {exc}"
                print(f"[PatchCore] ERROR: {exc}")
                import traceback; traceback.print_exc()

        threading.Thread(target=_run, daemon=True).start()

    def score(self, image_path: str):
        """Returns (normalised_score 0-1, anomaly_map HxW float32 or None).
        The amap is returned in raw units — use score_min/score_max to display it."""
        if not self.ready:
            return None, None
        try:
            results = self.engine.predict(
                model=self.model, data_path=image_path, return_predictions=True)
            if not results:
                return None, None
            amap = results[0].anomaly_map
            if amap is None:
                return None, None
            amap_np = amap.squeeze().cpu().numpy().astype(np.float32)
            raw     = float(amap_np.mean())
            lo, hi  = self.score_min, self.score_max
            norm    = float(np.clip((raw - lo) / max(hi - lo, 1e-9), 0.0, 1.0))
            print(f"  [raw={raw:.5f}  norm={norm:.3f}  floor={lo:.5f}  ceil={hi:.5f}]")
            return norm, amap_np
        except Exception as exc:
            print(f"[Score] ERROR: {exc}")
            return None, None


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def heatmap_overlay(bgr: np.ndarray, amap: np.ndarray,
                    amap_floor: float = 0.0, amap_ceil: float = None,
                    alpha: float = 0.5) -> np.ndarray:
    """
    Overlay anomaly heatmap on bgr image.

    amap_floor / amap_ceil: the raw score range that maps to cold→hot.
    These should come from the training score distribution so the heatmap
    is calibrated — a clean label stays cold even if local pixel variance
    causes tiny amap differences.

    If ceil is None we fall back to local min/max (always dramatic — avoid).
    """
    if amap is None:
        return bgr
    h, w   = bgr.shape[:2]
    amap_r = cv2.resize(amap, (w, h))

    ceil = amap_ceil if amap_ceil is not None else amap_r.max()
    span = max(ceil - amap_floor, 1e-9)
    u8   = np.clip((amap_r - amap_floor) / span * 255, 0, 255).astype(np.uint8)

    heat = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr, 1 - alpha, heat, alpha, 0)


def draw_live(frame, region, mode, status, last_verdict, n_collected, target):
    out = frame.copy()
    fh, fw = frame.shape[:2]

    if region is not None:
        rx, ry, rw, rh = region["x"], region["y"], region["w"], region["h"]
        color = (0, 255, 255) if mode == "TRAIN" else (0, 255, 0)
        cv2.rectangle(out, (rx, ry), (rx + rw, ry + rh), color, 2)
        cv2.putText(out, f"LABEL  {rw}x{rh}px",
                    (rx, max(ry - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    else:
        cv2.putText(out, "No label detected",
                    (fw // 2 - 140, fh // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 140, 255), 2)

    # Status bar
    bc = (40, 40, 40)
    if last_verdict == "GOOD":
        bc = (0, 60, 0)
    elif last_verdict == "BAD":
        bc = (0, 0, 80)
    cv2.rectangle(out, (0, fh - 60), (fw, fh), bc, -1)

    if mode == "TRAIN":
        prog = f"Collecting: {n_collected}/{target} crops  |  {status}"
        cv2.putText(out, prog, (10, fh - 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
    else:
        if last_verdict:
            cv2.putText(out, f"LAST: {last_verdict}",
                        (10, fh - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 220, 0) if last_verdict == "GOOD" else (0, 0, 220), 2)
        cv2.putText(out, status, (10, fh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    cv2.putText(out, "T=re-train  S=score  R=reset  Q=quit",
                (fw - 310, fh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)
    return out


def draw_result_panel(crop_bgr, amap, score, verdict, thresh,
                      amap_floor=0.0, amap_ceil=None):
    h = 280
    crop_r = cv2.resize(crop_bgr, (int(crop_bgr.shape[1] * h / crop_bgr.shape[0]), h))
    over_r = heatmap_overlay(
        crop_r,
        cv2.resize(amap, (crop_r.shape[1], h)) if amap is not None else None,
        amap_floor=amap_floor,
        amap_ceil=amap_ceil,
    )

    panel  = np.hstack([crop_r, over_r])
    bar    = np.full((80, panel.shape[1], 3), 25, dtype=np.uint8)

    color  = (0, 220, 0) if verdict == "GOOD" else (0, 0, 220)
    cv2.putText(bar, f"{verdict}",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
    cv2.putText(bar, f"Anomaly score: {score:.3f}  (thresh={thresh:.2f})",
                (160, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(bar, "Left: label crop   Right: anomaly heatmap (bright=defect)",
                (160, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)

    return np.vstack([bar, panel])


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════

def save_result(crop_bgr, verdict):
    folder = GOOD_DIR if verdict == "GOOD" else BAD_DIR
    os.makedirs(folder, exist_ok=True)
    ts  = time.strftime("%Y%m%d_%H%M%S")
    ms  = int((time.time() % 1) * 1000)
    out = os.path.join(folder, f"{verdict}_{ts}_{ms:03d}.jpg")
    cv2.imwrite(out, crop_bgr)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PatchCore live label defect checker")
    parser.add_argument("--camera",  type=int, default=None,
                        help="Webcam index (default: TCP stream)")
    parser.add_argument("--thresh",  type=float, default=ANOMALY_THRESH,
                        help=f"Anomaly score threshold 0-1 (default {ANOMALY_THRESH})")
    parser.add_argument("--samples", type=int, default=TRAIN_SAMPLES,
                        help=f"Training crops to collect (default {TRAIN_SAMPLES})")
    parser.add_argument("--size",    type=int, default=IMG_SIZE,
                        help=f"PatchCore image size (default {IMG_SIZE})")
    args = parser.parse_args()

    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
    else:
        cap = cv2.VideoCapture(DEFAULT_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("ERROR: cannot open stream/camera")
        return

    pc        = PatchCoreTrainer()
    tracker   = LabelTracker()
    train_tracker = LabelTracker(cooldown=0.5)  # faster collection during training

    train_crops  = []   # paths of collected training images
    mode         = "TRAIN"
    last_verdict = None
    result_panel = None

    # Clear any leftover crops from previous run so we start fresh
    if os.path.exists(TRAIN_DIR):
        shutil.rmtree(TRAIN_DIR)
    os.makedirs(TRAIN_DIR, exist_ok=True)

    def on_train_done():
        nonlocal mode
        mode = "INFER"

    WIN_LIVE   = "PatchCore Label Inspector"
    WIN_RESULT = "Result — Crop | Anomaly Heatmap"
    cv2.namedWindow(WIN_LIVE,   cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_RESULT, cv2.WINDOW_NORMAL)

    print(f"Thresh={args.thresh}  Samples={args.samples}  Size={args.size}")
    print("Collecting training crops from auto-detected label …")
    print("T=re-train  S=score  R=reset  Q=quit\n")

    # Temp dir for scoring (PatchCore needs a file path)
    tmp_score_dir = tempfile.mkdtemp(prefix="pc_score_")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        region = detect_label(frame)

        # ── TRAIN mode: collect stable crops ──────────────────────────────────
        if mode == "TRAIN" and not pc.training:
            if region is not None:
                crop = crop_region(frame, region)
                if train_tracker.update(crop) and len(train_crops) < args.samples:
                    idx  = len(train_crops) + 1
                    path = os.path.join(TRAIN_DIR, f"train_{idx:03d}.jpg")
                    cv2.imwrite(path, crop)
                    train_crops.append(path)
                    print(f"  Saved training crop {idx}/{args.samples}: {path}")

                if len(train_crops) >= args.samples:
                    pc.train_async(train_crops, args.size, BACKBONE, on_train_done)

        # ── INFER mode: score stable detections ───────────────────────────────
        elif mode == "INFER" and pc.ready:
            if region is not None:
                crop = crop_region(frame, region)
                if tracker.update(crop):
                    tmp_path = os.path.join(tmp_score_dir, "current.jpg")
                    cv2.imwrite(tmp_path, crop)

                    norm_score, amap = pc.score(tmp_path)
                    if norm_score is not None:
                        verdict = "BAD" if norm_score >= args.thresh else "GOOD"
                        last_verdict = verdict
                        fpath = save_result(crop, verdict)
                        print(f"  [{verdict}]  score={norm_score:.3f}  → {fpath}")
                        result_panel = draw_result_panel(
                            crop, amap, norm_score, verdict, args.thresh,
                            amap_floor=pc.score_min, amap_ceil=pc.score_max)
            else:
                tracker.reset()

        # ── Draw live view ────────────────────────────────────────────────────
        display = draw_live(frame, region, mode, pc.status,
                            last_verdict, len(train_crops), args.samples)
        cv2.imshow(WIN_LIVE, display)

        if result_panel is not None:
            cv2.imshow(WIN_RESULT, result_panel)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break

        elif key in (ord('t'), ord('T')):
            # Clear training data and start over
            print("\nRe-training — clearing old crops …")
            if os.path.exists(TRAIN_DIR):
                shutil.rmtree(TRAIN_DIR)
            os.makedirs(TRAIN_DIR, exist_ok=True)
            train_crops.clear()
            mode         = "TRAIN"
            last_verdict = None
            result_panel = None
            pc.ready     = False
            pc.status    = "Collecting training crops …"
            train_tracker.reset()
            tracker.reset()

        elif key in (ord('r'), ord('R')):
            tracker.reset()
            last_verdict = None
            result_panel = None
            print("Reset.")

        elif key in (ord('s'), ord('S')):
            if not pc.ready:
                print("Model not ready yet.")
            elif region is None:
                print("No label detected.")
            else:
                crop     = crop_region(frame, region)
                tmp_path = os.path.join(tmp_score_dir, "force.jpg")
                cv2.imwrite(tmp_path, crop)
                norm_score, amap = pc.score(tmp_path)
                if norm_score is not None:
                    verdict      = "BAD" if norm_score >= args.thresh else "GOOD"
                    last_verdict = verdict
                    fpath        = save_result(crop, verdict)
                    print(f"  [S-key] [{verdict}]  score={norm_score:.3f}  → {fpath}")
                    result_panel = draw_result_panel(
                        crop, amap, norm_score, verdict, args.thresh,
                        amap_floor=pc.score_min, amap_ceil=pc.score_max)

    cap.release()
    cv2.destroyAllWindows()
    shutil.rmtree(tmp_score_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

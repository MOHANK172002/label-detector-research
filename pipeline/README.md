# Label Inspector Pipeline

## Folder structure
```
pipeline/
  inspector.py      ← main script
  reference/
    master.jpg      ← master reference (auto-saved when you press M)
  results/
    good/           ← auto-saved GOOD label crops
    bad/            ← auto-saved BAD label crops
```

## First run — capture master
```bash
cd pipeline
python3 inspector.py
```
1. A good label will appear in the center zone (cyan box)
2. Press **M** — that label is saved as `reference/master.jpg`
3. Inspection starts automatically from that point

## Subsequent runs
```bash
python3 inspector.py                        # TCP stream (default)
python3 inspector.py --camera 0            # webcam
python3 inspector.py --ref reference/master.jpg
```

## Controls
| Key | Action |
|-----|--------|
| M   | Capture center label as new master |
| R   | Reset last result |
| Q   | Quit |

## Tuning thresholds (top of inspector.py)
| Variable | Default | Meaning |
|----------|---------|---------|
| V_LOW | 143 | Min brightness to detect label (raise if background bleeds in) |
| CENTER_ZONE | 0.35 | How centred label must be before checking (0=exact center, 1=anywhere) |
| CHECK_COOLDOWN | 3.0 s | Minimum time between checks (prevents double-checking same label) |
| SSIM_PASS | 0.73 | SSIM score minimum to pass |
| FINE_FAIL_PCT | 6.0 % | Max fine diff % allowed |
| COARSE_FAIL_PCT | 8.0 % | Max coarse diff % allowed |

#!/usr/bin/env python3
"""
Validate wake word recordings and flag bad clips.

Usage:
    python validate_recordings.py [recordings_dir]

Default dir: ./real_recordings

Checks per clip:
  - Duration      : warn if < 0.5s or > 3.5s
  - RMS level     : warn if too quiet (likely silent/distant mic)
  - Speech onset  : warn if word starts too late (> 0.5s silence at start)
  - Speech offset : warn if word cut off (> 0.3s silence at end)
  - Clipping      : warn if peak >= 0.99 (saturated)

Output: ranked list (worst first) + summary counts.
"""

import sys
import numpy as np
import soundfile as sf
from pathlib import Path

# ── Thresholds ────────────────────────────────────────────────────────────────
MIN_DURATION   = 0.5    # seconds
MAX_DURATION   = 3.5    # seconds
RMS_FLOOR      = 0.005  # normalised amplitude — below = too quiet
ONSET_SILENCE  = 0.5    # seconds of leading silence = bad
OFFSET_SILENCE = 0.3    # seconds of trailing silence = bad
CLIP_THRESHOLD = 0.99   # peak >= this = clipping
ENERGY_THRESH  = 0.01   # frame energy threshold for onset/offset detection
FRAME_LEN      = 0.02   # seconds per energy frame

recordings_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./real_recordings")

wavs = sorted(recordings_dir.glob("*.wav"))
if not wavs:
    print(f"[ERROR] No WAV files found in {recordings_dir}")
    sys.exit(1)

print(f"=== Validating {len(wavs)} recordings in {recordings_dir} ===\n")


def energy_frames(data: np.ndarray, sr: int) -> np.ndarray:
    frame_samples = int(FRAME_LEN * sr)
    n_frames = len(data) // frame_samples
    frames = data[:n_frames * frame_samples].reshape(n_frames, frame_samples)
    return np.sqrt((frames ** 2).mean(axis=1))


results = []

for wav in wavs:
    issues = []
    score  = 0  # higher = worse

    try:
        data, sr = sf.read(str(wav), always_2d=False)
    except Exception as e:
        results.append((999, wav.name, [f"unreadable: {e}"]))
        continue

    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)

    # Normalise to [-1, 1] if int
    if data.max() > 1.0:
        data = data / 32768.0

    duration = len(data) / sr

    # ── Duration ──────────────────────────────────────────────────────────────
    if duration < MIN_DURATION:
        issues.append(f"too short ({duration:.2f}s)")
        score += 40
    elif duration > MAX_DURATION:
        issues.append(f"too long ({duration:.2f}s)")
        score += 10

    # ── RMS ───────────────────────────────────────────────────────────────────
    rms = float(np.sqrt((data ** 2).mean()))
    if rms < RMS_FLOOR:
        issues.append(f"too quiet (rms={rms:.4f})")
        score += 50

    # ── Clipping ──────────────────────────────────────────────────────────────
    if np.abs(data).max() >= CLIP_THRESHOLD:
        issues.append("clipping (peak >= 0.99)")
        score += 20

    # ── Onset / offset silence ────────────────────────────────────────────────
    energies = energy_frames(data, sr)
    speech_frames = np.where(energies > ENERGY_THRESH)[0]

    if len(speech_frames) == 0:
        issues.append("no speech detected (all silence)")
        score += 60
    else:
        onset_s  = speech_frames[0]  * FRAME_LEN
        offset_s = (len(energies) - speech_frames[-1] - 1) * FRAME_LEN

        if onset_s > ONSET_SILENCE:
            issues.append(f"late onset ({onset_s:.2f}s silence at start)")
            score += 30

        if offset_s > OFFSET_SILENCE:
            issues.append(f"early cutoff ({offset_s:.2f}s silence at end)")
            score += 20

    results.append((score, wav.name, issues, duration, rms))

# ── Sort worst first ──────────────────────────────────────────────────────────
results.sort(key=lambda x: -x[0])

bad   = [r for r in results if r[0] > 0]
ok    = [r for r in results if r[0] == 0]

print(f"{'─'*60}")
print(f"  BAD clips ({len(bad)}) — re-record these")
print(f"{'─'*60}")
for r in bad:
    score, name, issues = r[0], r[1], r[2]
    print(f"  [{score:3d}] {name}")
    for issue in issues:
        print(f"         ✗ {issue}")

print(f"\n{'─'*60}")
print(f"  SUMMARY")
print(f"{'─'*60}")
print(f"  Total    : {len(results)}")
print(f"  OK       : {len(ok)}")
print(f"  Bad      : {len(bad)}")

issue_counts = {}
for r in bad:
    for issue in r[2]:
        key = issue.split("(")[0].strip()
        issue_counts[key] = issue_counts.get(key, 0) + 1
for k, v in sorted(issue_counts.items(), key=lambda x: -x[1]):
    print(f"    {v:3d}x  {k}")

print(f"\n  Re-record the {len(bad)} flagged clips above,")
print(f"  then re-run training from split:")
print(f"    echo 'hey murph' | python 02_training.py --from split")

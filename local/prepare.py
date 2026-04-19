#!/usr/bin/env python3
"""
Center speech within each recording and trim excess silence.

Run after a recording session, before transferring clips to the VM.

Usage:
    python local/prepare.py [--rec-dir ./real_recordings]

What it does:
  - Finds the speech region in each clip (energy threshold)
  - Centers the speech in the clip with PAD_S seconds of silence on each side
  - Clips shorter than MIN_DURATION after centering are kept as-is (not over-trimmed)
  - Saves in-place; originals backed up to <rec_dir>_backup/ on first run

Install deps (same as record.py):
    pip install -r local/requirements.txt
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ENERGY_THRESH = 0.01
FRAME_LEN     = 0.02   # seconds per energy frame
PAD_S         = 0.15   # silence padding to keep around speech on each side
MIN_DURATION  = 1.0    # never produce a clip shorter than this (seconds)

parser = argparse.ArgumentParser()
parser.add_argument("--rec-dir", type=Path, default=Path("./real_recordings"))
args = parser.parse_args()

wavs = sorted(args.rec_dir.glob("*.wav"))
if not wavs:
    print(f"[ERROR] No WAV files in {args.rec_dir}")
    sys.exit(1)

backup_dir = Path(str(args.rec_dir) + "_backup")
if not backup_dir.exists():
    print(f"Backing up originals -> {backup_dir} ...")
    shutil.copytree(args.rec_dir, backup_dir)
    print(f"  {len(wavs)} files backed up.\n")
else:
    print(f"[OK] Backup already exists at {backup_dir}\n")

print(f"=== Centering {len(wavs)} recordings ===\n")

centered = skipped = errors = 0

for wav in wavs:
    try:
        data, sr = sf.read(str(wav), always_2d=False)
    except Exception as e:
        print(f"  [ERROR] {wav.name}: {e}")
        errors += 1
        continue

    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if np.abs(data).max() > 1.0:
        data = data / 32768.0

    frame_samples = int(FRAME_LEN * sr)
    pad_samples   = int(PAD_S * sr)
    n_frames      = len(data) // frame_samples

    if n_frames == 0:
        skipped += 1
        continue

    frames   = data[:n_frames * frame_samples].reshape(n_frames, frame_samples)
    energies = np.sqrt((frames ** 2).mean(axis=1))
    speech   = np.where(energies > ENERGY_THRESH)[0]

    if len(speech) == 0:
        print(f"  [SKIP] {wav.name}: no speech detected")
        skipped += 1
        continue

    speech_start = max(0, speech[0]  * frame_samples - pad_samples)
    speech_end   = min(len(data), (speech[-1] + 1) * frame_samples + pad_samples)
    trimmed      = data[speech_start:speech_end]

    # Enforce minimum duration — pad symmetrically with silence
    min_samples = int(MIN_DURATION * sr)
    if len(trimmed) < min_samples:
        pad = (min_samples - len(trimmed)) // 2
        trimmed = np.pad(trimmed, (pad, min_samples - len(trimmed) - pad))

    sf.write(str(wav), (trimmed * 32767).astype(np.int16), sr)
    centered += 1

print(f"Done.  Centered: {centered}  Skipped: {skipped}  Errors: {errors}")
if centered:
    print(f"\nOriginals preserved in {backup_dir}")
    print(f"Next: transfer to VM and train.")

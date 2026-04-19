#!/usr/bin/env python3
"""
Trim leading/trailing silence from wake word recordings.

Usage:
    python trim_recordings.py [recordings_dir]

Default dir: ./real_recordings

Trims in-place. Originals backed up to <dir>_backup/ (first run only).
"""

import sys
import shutil
import numpy as np
import soundfile as sf
from pathlib import Path

ENERGY_THRESH  = 0.01   # frame energy threshold
FRAME_LEN      = 0.02   # seconds per frame
PAD_S          = 0.05   # silence padding to keep around speech (seconds)

recordings_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./real_recordings")
backup_dir     = Path(str(recordings_dir) + "_backup")

wavs = sorted(recordings_dir.glob("*.wav"))
if not wavs:
    print(f"[ERROR] No WAV files in {recordings_dir}")
    sys.exit(1)

# ── Backup (once) ─────────────────────────────────────────────────────────────
if not backup_dir.exists():
    print(f"Backing up originals -> {backup_dir} ...")
    shutil.copytree(recordings_dir, backup_dir)
    print(f"  {len(wavs)} files backed up.\n")
else:
    print(f"[SKIP] Backup already exists at {backup_dir}\n")

print(f"=== Trimming {len(wavs)} recordings ===\n")

trimmed = skipped = 0

for wav in wavs:
    try:
        data, sr = sf.read(str(wav), always_2d=False)
    except Exception as e:
        print(f"  [ERROR] {wav.name}: {e}")
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
        print(f"  [SKIP] {wav.name}: too short")
        skipped += 1
        continue

    frames   = data[:n_frames * frame_samples].reshape(n_frames, frame_samples)
    energies = np.sqrt((frames ** 2).mean(axis=1))
    speech   = np.where(energies > ENERGY_THRESH)[0]

    if len(speech) == 0:
        print(f"  [SKIP] {wav.name}: no speech detected")
        skipped += 1
        continue

    start = max(0, speech[0]  * frame_samples - pad_samples)
    end   = min(len(data), (speech[-1] + 1) * frame_samples + pad_samples)
    trimmed_data = data[start:end]

    # Write back as int16 at original sample rate
    sf.write(str(wav), (trimmed_data * 32767).astype(np.int16), sr)
    trimmed += 1

print(f"Done. Trimmed: {trimmed}  Skipped: {skipped}")
print(f"\nNext: python validate_recordings.py {recordings_dir}")

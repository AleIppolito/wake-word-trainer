#!/usr/bin/env python3
"""
Prepare raw recordings for training: normalize, center, quality-check.

Reads from real_rec_raw/, writes to real_rec_prepared/.
Raw files are never modified.

Usage:
    python local/prepare.py [--rec-dir ./real_rec_raw] [--out-dir ./real_rec_prepared]

What it does per clip:
  1. Peak-normalize to TARGET_PEAK (-2 dBFS)
  2. Detect speech region via energy threshold
  3. Trim leading/trailing silence, keep PAD_S on each side
  4. Enforce MIN_DURATION (pads symmetrically if needed)
  5. Estimate SNR and speech duration → quality flag

Quality flags:
  ✓ GOOD  SNR > 15 dB  and speech > 0.4s
  ⚠ WARN  SNR 8–15 dB  or  speech 0.25–0.4s
  ✗ BAD   SNR < 8 dB   or  speech < 0.25s  or no speech detected

All clips are saved regardless of quality flag. Re-record BAD ones.

Install deps (same as record.py):
    pip install -r local/requirements.txt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ENERGY_THRESH = 0.01   # energy threshold for speech detection
FRAME_LEN     = 0.02   # seconds per energy analysis frame
PAD_S         = 0.15   # silence padding to keep around speech on each side
MIN_DURATION  = 1.0    # minimum clip duration after trimming (seconds)
TARGET_PEAK   = 0.8    # peak normalize to this level (< 1.0 to avoid clipping)

SNR_GOOD      = 15.0   # dB
SNR_WARN      = 8.0    # dB
DUR_GOOD      = 0.40   # seconds of detected speech
DUR_WARN      = 0.25   # seconds of detected speech

parser = argparse.ArgumentParser()
parser.add_argument("--rec-dir", type=Path, default=Path("./real_rec_raw"))
parser.add_argument("--out-dir", type=Path, default=Path("./real_rec_prepared"))
args = parser.parse_args()

wavs = sorted(args.rec_dir.glob("*.wav"))
if not wavs:
    print(f"[ERROR] No WAV files in {args.rec_dir}")
    sys.exit(1)

args.out_dir.mkdir(parents=True, exist_ok=True)

print(f"=== Preparing {len(wavs)} clips ===")
print(f"    raw -> {args.rec_dir}")
print(f"    out -> {args.out_dir}\n")

counts = {"GOOD": 0, "WARN": 0, "BAD": 0, "ERROR": 0}


def analyze(data: np.ndarray, sr: int):
    """Return (speech_duration_s, snr_db, has_speech)."""
    frame_samples = int(FRAME_LEN * sr)
    n_frames = len(data) // frame_samples
    if n_frames == 0:
        return 0.0, 0.0, False

    frames   = data[:n_frames * frame_samples].reshape(n_frames, frame_samples)
    energies = np.sqrt((frames ** 2).mean(axis=1))
    speech   = np.where(energies > ENERGY_THRESH)[0]

    if len(speech) == 0:
        return 0.0, 0.0, False

    speech_dur = len(speech) * FRAME_LEN
    rms_speech = float(np.sqrt((frames[speech] ** 2).mean()))

    noise_mask = energies <= ENERGY_THRESH
    if noise_mask.any():
        rms_noise = float(np.sqrt((frames[noise_mask] ** 2).mean()))
    else:
        rms_noise = 1e-10  # all speech, no noise reference → treat as clean

    snr_db = 20.0 * np.log10(max(rms_speech, 1e-10) / max(rms_noise, 1e-10))
    return speech_dur, snr_db, True


def quality(speech_dur: float, snr_db: float, has_speech: bool) -> tuple[str, list[str]]:
    if not has_speech:
        return "BAD", ["no speech detected"]
    reasons = []
    if snr_db < SNR_WARN:
        reasons.append(f"low SNR ({snr_db:.1f} dB)")
    elif snr_db < SNR_GOOD:
        reasons.append(f"SNR ok ({snr_db:.1f} dB)")
    if speech_dur < DUR_WARN:
        reasons.append(f"short speech ({speech_dur:.2f}s)")
    elif speech_dur < DUR_GOOD:
        reasons.append(f"speech ok ({speech_dur:.2f}s)")

    if snr_db < SNR_WARN or speech_dur < DUR_WARN:
        return "BAD", reasons
    if snr_db < SNR_GOOD or speech_dur < DUR_GOOD:
        return "WARN", reasons
    return "GOOD", []


def center_and_trim(data: np.ndarray, sr: int) -> np.ndarray:
    """Trim silence, add PAD_S on each side, enforce MIN_DURATION."""
    frame_samples = int(FRAME_LEN * sr)
    pad_samples   = int(PAD_S * sr)
    n_frames      = len(data) // frame_samples

    if n_frames == 0:
        return data

    frames   = data[:n_frames * frame_samples].reshape(n_frames, frame_samples)
    energies = np.sqrt((frames ** 2).mean(axis=1))
    speech   = np.where(energies > ENERGY_THRESH)[0]

    if len(speech) == 0:
        return data

    start = max(0, speech[0]  * frame_samples - pad_samples)
    end   = min(len(data), (speech[-1] + 1) * frame_samples + pad_samples)
    out   = data[start:end]

    min_samples = int(MIN_DURATION * sr)
    if len(out) < min_samples:
        deficit = min_samples - len(out)
        out = np.pad(out, (deficit // 2, deficit - deficit // 2))

    return out


ICONS = {"GOOD": "✓", "WARN": "⚠", "BAD": "✗", "ERROR": "!"}

for wav in wavs:
    try:
        data, sr = sf.read(str(wav), always_2d=False)
    except Exception as e:
        print(f"  ! ERROR  {wav.name}  {e}")
        counts["ERROR"] += 1
        continue

    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if np.abs(data).max() > 1.0:
        data = data / 32768.0

    # Analyze before processing (SNR on original signal)
    speech_dur, snr_db, has_speech = analyze(data, sr)
    grade, reasons = quality(speech_dur, snr_db, has_speech)

    # Normalize
    peak = np.abs(data).max()
    if peak > 0:
        data = data * (TARGET_PEAK / peak)

    # Center + trim
    data = center_and_trim(data, sr)

    # Save
    out_path = args.out_dir / wav.name
    sf.write(str(out_path), (data * 32767).astype(np.int16), sr)

    counts[grade] += 1
    icon   = ICONS[grade]
    detail = f"SNR={snr_db:5.1f}dB  speech={speech_dur:.2f}s"
    suffix = f"  ({', '.join(reasons)})" if reasons else ""
    print(f"  {icon} {grade:<4}  {wav.name:<45}  {detail}{suffix}")


total = sum(counts.values())
print(f"\n{'='*60}")
print(f"  Processed : {total}")
print(f"  ✓ GOOD    : {counts['GOOD']}  ({100*counts['GOOD']//max(1,total)}%)")
print(f"  ⚠ WARN    : {counts['WARN']}  ({100*counts['WARN']//max(1,total)}%)")
print(f"  ✗ BAD     : {counts['BAD']}  ({100*counts['BAD']//max(1,total)}%)")
if counts["ERROR"]:
    print(f"  ! ERROR   : {counts['ERROR']}")
print(f"{'='*60}")
print(f"\n  Output: {args.out_dir}/")
if counts["BAD"]:
    print(f"\n  Re-record the {counts['BAD']} BAD clips for best results.")
print(f"\n  Next: transfer to VM")
print(f"    scp -r {args.out_dir}/ root@<SERVER_IP>:/path/to/wake-word-trainer/real_rec_prepared")

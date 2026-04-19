#!/usr/bin/env python3
"""
Evaluate a trained wake word model: recall on real recordings + FP/hour on background audio.

Usage:
    python eval.py <wake_model.onnx> [options]

    --rec-dir    DIR   recordings dir  (default: ./real_rec_raw)
    --fp-dir     DIR   background dir  (default: ./audioset_16k)
    --threshold  F     score threshold (default: 0.3)
    --fp-samples N     audioset clips to sample for FP eval (default: 500)

Model search path for mel/embedding models:
    ./openwakeword/openwakeword/resources/models/
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import librosa
from _log import setup_log

log = setup_log("eval")

CHUNK_SAMPLES = 1280
MEL_FEATURES  = 32
MEL_WINDOW    = 76
EMB_WINDOW    = 16
EMB_DIM       = 96

parser = argparse.ArgumentParser()
parser.add_argument("wake_model", type=Path)
parser.add_argument("--rec-dir",    type=Path, default=Path("./real_rec_prepared"))
parser.add_argument("--fp-dir",     type=Path, default=Path("./audioset_16k"))
parser.add_argument("--threshold",  type=float, default=0.3)
parser.add_argument("--fp-samples", type=int,   default=500)
args = parser.parse_args()

FP_COOLDOWN = EMB_WINDOW

models_dir = Path("./openwakeword/openwakeword/resources/models")
mel_path = models_dir / "melspectrogram.onnx"
emb_path = models_dir / "embedding_model.onnx"

for p in [mel_path, emb_path, args.wake_model]:
    if not p.exists():
        log.error(f"model not found: {p}")
        print(f"[ERROR] Not found: {p}")
        sys.exit(1)

log.info(f"eval start  wake={args.wake_model}  threshold={args.threshold}  rec_dir={args.rec_dir}")
print(f"mel : {mel_path}")
print(f"emb : {emb_path}")
print(f"wake: {args.wake_model}")
print()

_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
mel_sess  = ort.InferenceSession(str(mel_path),        providers=_providers)
emb_sess  = ort.InferenceSession(str(emb_path),        providers=_providers)
wake_sess = ort.InferenceSession(str(args.wake_model), providers=_providers)

mel_in  = mel_sess.get_inputs()[0].name;  mel_out  = mel_sess.get_outputs()[0].name
emb_in  = emb_sess.get_inputs()[0].name;  emb_out  = emb_sess.get_outputs()[0].name
wk_in   = wake_sess.get_inputs()[0].name; wk_out   = wake_sess.get_outputs()[0].name


def run_pipeline(wav_path, count_mode=False):
    data, sr = sf.read(str(wav_path), always_2d=False)
    if sr != 16000:
        data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=16000)
    if data.ndim > 1:
        data = data.mean(axis=1)
    audio = np.clip(data.astype(np.float32) * 32767.0, -32768, 32767)
    duration_sec = len(audio) / 16000

    min_samples = CHUNK_SAMPLES * (MEL_WINDOW // 4 + EMB_WINDOW + 4)
    if len(audio) < min_samples:
        audio = np.pad(audio, (min_samples - len(audio), 0))

    mel_buf, emb_buf, scores = [], [], []
    step = CHUNK_SAMPLES // 2
    n_chunks = max(1, (len(audio) - CHUNK_SAMPLES) // step + 1)
    cooldown = 0
    n_activations = 0

    for i in range(n_chunks):
        chunk = audio[i*step : i*step + CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))

        mel_o = mel_sess.run([mel_out], {mel_in: chunk.reshape(1, -1)})[0]
        mel_o = mel_o / 10 + 2
        n_frames = mel_o.size // MEL_FEATURES
        mel_flat = mel_o.flatten()
        for f in range(n_frames):
            mel_buf.append(mel_flat[f*MEL_FEATURES:(f+1)*MEL_FEATURES].copy())
        while len(mel_buf) > MEL_WINDOW: mel_buf.pop(0)
        if len(mel_buf) < MEL_WINDOW: continue

        flat_mel = np.stack(mel_buf).reshape(1, MEL_WINDOW, MEL_FEATURES, 1).astype(np.float32)
        emb_o = emb_sess.run([emb_out], {emb_in: flat_mel})[0]
        emb_buf.append(emb_o.flatten()[:EMB_DIM].copy())
        while len(emb_buf) > EMB_WINDOW: emb_buf.pop(0)
        if len(emb_buf) < EMB_WINDOW: continue

        flat_emb = np.stack(emb_buf).reshape(1, EMB_WINDOW, EMB_DIM).astype(np.float32)
        score = wake_sess.run([wk_out], {wk_in: flat_emb})[0].flatten()[0]

        if count_mode:
            if cooldown > 0:
                cooldown -= 1
            elif score >= args.threshold:
                n_activations += 1
                cooldown = FP_COOLDOWN
        else:
            scores.append(float(score))

    if count_mode:
        return n_activations, duration_sec
    return max(scores) if scores else 0.0


# ── Recall ────────────────────────────────────────────────────────────────────
wavs = sorted(args.rec_dir.glob("*.wav"))
if not wavs:
    print(f"[ERROR] No WAV files in {args.rec_dir}")
    sys.exit(1)

from collections import Counter
EDGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
print(f"{'─'*52}")
print(f"  RECALL  —  {len(wavs)} recordings  (threshold={args.threshold})")
print(f"{'─'*52}")

peaks, n_detected, buckets = [], 0, Counter()
for wav in wavs:
    try:
        peak = run_pipeline(wav)
    except Exception as e:
        print(f"  ERROR {wav.name}: {e}")
        continue
    peaks.append(peak)
    if peak >= args.threshold:
        n_detected += 1
    for j in range(len(EDGES) - 1):
        if EDGES[j] <= peak < EDGES[j+1]:
            buckets[f"{EDGES[j]:.1f}-{EDGES[j+1]:.1f}"] += 1
            break
    print(f"  {'✓' if peak >= args.threshold else '✗'} {wav.name}  peak={peak:.4f}")

print(f"\n{'='*52}")
print(f"  Files      : {len(peaks)}")
print(f"  Detected   : {n_detected}/{len(peaks)}  ({100*n_detected/max(1,len(peaks)):.1f}%)")
print(f"  Avg peak   : {np.mean(peaks):.4f}")
print(f"  Median     : {np.median(peaks):.4f}")
print(f"  Max        : {np.max(peaks):.4f}")
print()
for label in sorted(buckets):
    print(f"    {label}  {'█'*buckets[label]} ({buckets[label]})")
log.info(f"recall  detected={n_detected}/{len(peaks)}  ({100*n_detected/max(1,len(peaks)):.1f}%)  "
         f"avg={np.mean(peaks):.4f}  median={np.median(peaks):.4f}  max={np.max(peaks):.4f}")
print(f"{'='*52}\n")


# ── False positives ───────────────────────────────────────────────────────────
fp_wavs = sorted(args.fp_dir.glob("*.wav")) if args.fp_dir.exists() else []
if not fp_wavs:
    print(f"[SKIP] FP eval — {args.fp_dir} not found or empty")
    sys.exit(0)

sample = random.sample(fp_wavs, min(args.fp_samples, len(fp_wavs)))
total_activations, total_seconds, fp_errors = 0, 0.0, 0

print(f"{'─'*52}")
print(f"  FALSE POSITIVES  —  {len(sample)} audioset clips sampled")
print(f"{'─'*52}")

for wav in sample:
    try:
        n_act, dur = run_pipeline(wav, count_mode=True)
    except Exception:
        fp_errors += 1
        continue
    total_activations += n_act
    total_seconds += dur
    if n_act > 0:
        print(f"  ! {wav.name}  activations={n_act}  dur={dur:.1f}s")

total_hours = total_seconds / 3600
fp_per_hour = total_activations / total_hours if total_hours > 0 else float("inf")

print(f"\n{'='*52}")
print(f"  Clips sampled  : {len(sample)}")
print(f"  Total audio    : {total_seconds/60:.1f} min  ({total_hours:.3f} hr)")
print(f"  FP activations : {total_activations}")
print(f"  FP / hour      : {fp_per_hour:.2f}")
if fp_errors:
    print(f"  Errors         : {fp_errors}")
print(f"{'='*52}")
log.info(f"fp_per_hour={fp_per_hour:.2f}  activations={total_activations}  "
         f"hours={total_hours:.3f}  errors={fp_errors}")
print(f"\n  Target: FP/hour < 0.5 (good)  < 1.0 (acceptable)")

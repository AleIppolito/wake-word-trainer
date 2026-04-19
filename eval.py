#!/usr/bin/env python3
import sys
import random
import numpy as np
import librosa
import onnxruntime as ort
from pathlib import Path
from collections import Counter

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
REC_DIR       = Path("/workspace/wake-word-trainer/real_recordings")
FP_DIR        = Path("/workspace/wake-word-trainer/audioset_16k")
MEL_PATH      = None   # auto-detected from /workspace if None
EMB_PATH      = None
WAKE_PATH     = Path("/workspace/wake-word-trainer/my_custom_model/hey_murph.onnx")
THRESHOLD     = 0.3
N_FP_SAMPLES  = 500    # audioset clips to sample for FP eval
FP_COOLDOWN   = 25     # frames to ignore after a detection (matches EMB_WINDOW)
# ─────────────────────────────────────────────

CHUNK_SAMPLES = 1280
MEL_FEATURES  = 32
MEL_WINDOW    = 76
EMB_WINDOW    = 25
EMB_DIM       = 96

# ── Model loading ─────────────────────────────────────────────────────────────
if MEL_PATH is None:
    import subprocess
    mel = subprocess.check_output(["find", "/workspace", "-name", "melspectrogram.onnx"]).decode().strip().split("\n")[0]
    emb = subprocess.check_output(["find", "/workspace", "-name", "embedding_model.onnx"]).decode().strip().split("\n")[0]
    MEL_PATH = Path(mel)
    EMB_PATH = Path(emb)

print(f"mel : {MEL_PATH}")
print(f"emb : {EMB_PATH}")
print(f"wake: {WAKE_PATH}")
print()

_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
mel_sess  = ort.InferenceSession(str(MEL_PATH),  providers=_providers)
emb_sess  = ort.InferenceSession(str(EMB_PATH),  providers=_providers)
wake_sess = ort.InferenceSession(str(WAKE_PATH), providers=_providers)

mel_in  = mel_sess.get_inputs()[0].name;  mel_out  = mel_sess.get_outputs()[0].name
emb_in  = emb_sess.get_inputs()[0].name;  emb_out  = emb_sess.get_outputs()[0].name
wk_in   = wake_sess.get_inputs()[0].name; wk_out   = wake_sess.get_outputs()[0].name


def run_pipeline(wav_path, count_mode=False):
    """
    Run the full 3-stage inference pipeline on a WAV file.

    count_mode=False (recall eval): returns max score across all chunks.
    count_mode=True  (FP eval):     returns (n_activations, duration_seconds).
      Activations are threshold crossings with FP_COOLDOWN frame suppression,
      matching the real detection behaviour in murph.
    """
    audio, _ = librosa.load(str(wav_path), sr=16000, mono=True)
    audio = np.clip(audio * 32767.0, -32768, 32767).astype(np.float32)
    duration_sec = len(audio) / 16000

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
            elif score >= THRESHOLD:
                n_activations += 1
                cooldown = FP_COOLDOWN
        else:
            scores.append(float(score))

    if count_mode:
        return n_activations, duration_sec
    return max(scores) if scores else 0.0


# ── 1. Recall eval on real recordings ────────────────────────────────────────
wavs = sorted(REC_DIR.glob("*.wav"))
print(f"{'─'*52}")
print(f"  RECALL  —  {len(wavs)} recordings  (threshold={THRESHOLD})")
print(f"{'─'*52}")

peaks, n_detected = [], 0
EDGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
buckets = Counter()

for wav in wavs:
    try:
        peak = run_pipeline(wav)
    except Exception as e:
        print(f"  ERROR {wav.name}: {e}")
        continue
    peaks.append(peak)
    if peak >= THRESHOLD:
        n_detected += 1
    for j in range(len(EDGES) - 1):
        if EDGES[j] <= peak < EDGES[j+1]:
            buckets[f"{EDGES[j]:.1f}-{EDGES[j+1]:.1f}"] += 1
            break
    print(f"  {'✓' if peak >= THRESHOLD else '✗'} {wav.name}  peak={peak:.4f}")

print(f"\n{'='*52}")
print(f"  Files      : {len(peaks)}")
print(f"  Detected   : {n_detected}/{len(peaks)}  ({100*n_detected/max(1,len(peaks)):.1f}%)")
print(f"  Avg peak   : {np.mean(peaks):.4f}")
print(f"  Median     : {np.median(peaks):.4f}")
print(f"  Max        : {np.max(peaks):.4f}")
print()
for label in sorted(buckets):
    print(f"    {label}  {'█'*buckets[label]} ({buckets[label]})")
print(f"{'='*52}\n")


# ── 2. False positive eval on background audio ───────────────────────────────
fp_wavs = sorted(FP_DIR.glob("*.wav")) if FP_DIR.exists() else []
if not fp_wavs:
    print(f"[SKIP] FP eval — {FP_DIR} not found or empty")
    sys.exit(0)

sample = random.sample(fp_wavs, min(N_FP_SAMPLES, len(fp_wavs)))
total_activations = 0
total_seconds = 0.0
fp_errors = 0

print(f"{'─'*52}")
print(f"  FALSE POSITIVES  —  {len(sample)} audioset clips sampled")
print(f"{'─'*52}")

for wav in sample:
    try:
        n_act, dur = run_pipeline(wav, count_mode=True)
    except Exception as e:
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
print()
print(f"  Target: FP/hour < 0.5 (good)  < 1.0 (acceptable)")

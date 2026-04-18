#!/usr/bin/env python3
import sys
import numpy as np
import librosa
import onnxruntime as ort
from pathlib import Path
from collections import Counter

REC_DIR    = Path("/workspace/wake-word-trainer/real_recordings")
MEL_PATH   = None   # fill after: find /workspace -name melspectrogram.onnx
EMB_PATH   = None   # fill after: find /workspace -name embedding_model.onnx
WAKE_PATH  = Path("/workspace/wake-word-trainer/my_custom_model/hey_murph.onnx")
THRESHOLD  = 0.3

CHUNK_SAMPLES = 1280
MEL_FEATURES  = 32
MEL_WINDOW    = 76
EMB_WINDOW    = 25
EMB_DIM       = 96

# Auto-find mel/emb if not set
if MEL_PATH is None:
    import subprocess
    mel = subprocess.check_output(["find", "/workspace", "-name", "melspectrogram.onnx"]).decode().strip().split("\n")[0]
    emb = subprocess.check_output(["find", "/workspace", "-name", "embedding_model.onnx"]).decode().strip().split("\n")[0]
    MEL_PATH  = Path(mel)
    EMB_PATH  = Path(emb)

print(f"mel : {MEL_PATH}")
print(f"emb : {EMB_PATH}")
print(f"wake: {WAKE_PATH}")
print()

mel_sess  = ort.InferenceSession(str(MEL_PATH),  providers=["CUDAExecutionProvider","CPUExecutionProvider"])
emb_sess  = ort.InferenceSession(str(EMB_PATH),  providers=["CUDAExecutionProvider","CPUExecutionProvider"])
wake_sess = ort.InferenceSession(str(WAKE_PATH), providers=["CUDAExecutionProvider","CPUExecutionProvider"])

mel_in_name  = mel_sess.get_inputs()[0].name
mel_out_name = mel_sess.get_outputs()[0].name
emb_in_name  = emb_sess.get_inputs()[0].name
emb_out_name = emb_sess.get_outputs()[0].name
wk_in_name   = wake_sess.get_inputs()[0].name
wk_out_name  = wake_sess.get_outputs()[0].name

def run_pipeline(wav_path):
    audio, _ = librosa.load(str(wav_path), sr=16000, mono=True)
    audio = np.clip(audio * 32767.0, -32768, 32767).astype(np.float32)
    mel_buf, emb_buf, scores = [], [], []
    step = CHUNK_SAMPLES // 2
    n_chunks = max(1, (len(audio) - CHUNK_SAMPLES) // step + 1)
    for i in range(n_chunks):
        chunk = audio[i*step : i*step + CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
        mel_out  = mel_sess.run([mel_out_name], {mel_in_name: chunk.reshape(1, -1)})[0]
        mel_out  = mel_out / 10 + 2  # match openWakeWord internal transform
        n_frames = mel_out.size // MEL_FEATURES
        mel_flat = mel_out.flatten()
        for f in range(n_frames):
            mel_buf.append(mel_flat[f*MEL_FEATURES:(f+1)*MEL_FEATURES].copy())
        while len(mel_buf) > MEL_WINDOW: mel_buf.pop(0)
        if len(mel_buf) < MEL_WINDOW: continue
        flat_mel = np.stack(mel_buf).reshape(1, MEL_WINDOW, MEL_FEATURES, 1).astype(np.float32)
        emb_out  = emb_sess.run([emb_out_name], {emb_in_name: flat_mel})[0]
        emb_buf.append(emb_out.flatten()[:EMB_DIM].copy())
        while len(emb_buf) > EMB_WINDOW: emb_buf.pop(0)
        if len(emb_buf) < EMB_WINDOW: continue
        flat_emb = np.stack(emb_buf).reshape(1, EMB_WINDOW, EMB_DIM).astype(np.float32)
        score    = wake_sess.run([wk_out_name], {wk_in_name: flat_emb})[0].flatten()[0]
        scores.append(float(score))
    return max(scores) if scores else 0.0

wavs = sorted(REC_DIR.glob("*.wav"))
print(f"Evaluating {len(wavs)} recordings (threshold={THRESHOLD})...\n")

peaks, n_detected = [], 0
EDGES = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.01]
buckets = Counter()

for wav in wavs:
    try:    peak = run_pipeline(wav)
    except Exception as e: print(f"  ERROR {wav.name}: {e}"); continue
    peaks.append(peak)
    if peak >= THRESHOLD: n_detected += 1
    for j in range(len(EDGES)-1):
        if EDGES[j] <= peak < EDGES[j+1]:
            buckets[f"{EDGES[j]:.1f}-{EDGES[j+1]:.1f}"] += 1; break
    print(f"  {'✓' if peak>=THRESHOLD else '✗'} {wav.name}  peak={peak:.4f}")

print(f"\n{'='*52}")
print(f"  Files      : {len(peaks)}")
print(f"  Detected   : {n_detected}/{len(peaks)}  ({100*n_detected/len(peaks):.1f}%)")
print(f"  Avg peak   : {np.mean(peaks):.4f}")
print(f"  Median     : {np.median(peaks):.4f}")
print(f"  Max        : {np.max(peaks):.4f}")
print()
for label in sorted(buckets):
    print(f"    {label}  {'█'*buckets[label]} ({buckets[label]})")
print('='*52)

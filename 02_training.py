#!/usr/bin/env python3
"""
STEP 2 - Train on real voice recordings (no synthetic TTS for positive examples).

Prerequisites:
  - 00_fix_dependencies.py and 01_setup_and_download.py already run
  - WAV recordings in RECORDINGS_SOURCE_DIR (produced by record.py)
    Files at any sample rate are automatically resampled to 16 kHz.

Pipeline:
  1. Prompt for wake word phrase (used for adversarial TTS negative generation)
  2. Validate recordings
  3. Split 80% train / 20% test -> positive_train / positive_test
  4. Augment clips (x AUGMENTATION_ROUNDS to expand the dataset)
  5. Train model
  6. Convert ONNX -> TFLite

Output:
  - ./my_custom_model/<model_name>.onnx
  - ./my_custom_model/<model_name>.tflite
  - ./my_custom_model/<model_name>_float16.tflite
"""

import os
import sys
import subprocess
import resource
import random

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
RECORDINGS_SOURCE_DIR    = "./real_recordings"
TRAIN_SPLIT              = 0.8
AUGMENTATION_ROUNDS      = 50
NUMBER_OF_TRAINING_STEPS = 50000
FALSE_ACTIVATION_PENALTY = 1300
# ─────────────────────────────────────────────

def run(cmd, ignore_error=False):
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0 and not ignore_error:
        print(f"[WARN] exit code {result.returncode}")

# Fix ulimit
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (65536, hard))
print(f"ulimit -n set to 65536 (was {soft})")

import yaml
import numpy as np
import scipy.io.wavfile
from pathlib import Path

# ── 1. Prompt for wake word ───────────────────────────────────────────────────
print("\nEnter the wake word phrase.")
print("This is used to generate adversarial TTS negative samples and to name the output model.")
TARGET_WORD = input("Wake word: ").strip()
if not TARGET_WORD:
    print("[ERROR] Wake word cannot be empty.")
    sys.exit(1)

# ── 2. Validate recordings ────────────────────────────────────────────────────
print(f"\n=== Checking recordings in '{RECORDINGS_SOURCE_DIR}' ===")
if not os.path.exists(RECORDINGS_SOURCE_DIR):
    print(f"[ERROR] Folder '{RECORDINGS_SOURCE_DIR}' not found.")
    print("Transfer your recordings to that folder and try again.")
    sys.exit(1)

all_wavs = sorted(Path(RECORDINGS_SOURCE_DIR).glob("*.wav"))
if len(all_wavs) < 20:
    print(f"[ERROR] Only {len(all_wavs)} WAV files found. At least 20 are required (200+ recommended).")
    sys.exit(1)
print(f"Found {len(all_wavs)} WAV files.")

# ── 3. Create output directories ──────────────────────────────────────────────
model_name = TARGET_WORD.replace(" ", "_")
output_dir = "./my_custom_model"
model_dir  = os.path.join(output_dir, model_name)
pos_train  = os.path.join(model_dir, "positive_train")
pos_test   = os.path.join(model_dir, "positive_test")

for d in [pos_train, pos_test,
          os.path.join(model_dir, "negative_train"),
          os.path.join(model_dir, "negative_test")]:
    os.makedirs(d, exist_ok=True)

# ── 4. Split and copy (with automatic resampling) ─────────────────────────────
print(f"\n=== 80/20 split -> {model_dir} ===")
shuffled = list(all_wavs)
random.shuffle(shuffled)
split_idx  = int(len(shuffled) * TRAIN_SPLIT)
train_wavs = shuffled[:split_idx]
test_wavs  = shuffled[split_idx:]

def copy_and_resample(src_paths, dest_dir):
    from scipy.signal import resample_poly
    from math import gcd
    ok = 0
    for i, src in enumerate(src_paths):
        dst = os.path.join(dest_dir, f"{model_name}_{i:04d}.wav")
        try:
            sr, data = scipy.io.wavfile.read(str(src))
        except Exception as e:
            print(f"  [SKIP] {src.name}: {e}")
            continue
        if data.ndim > 1:                          # stereo -> mono
            data = data.mean(axis=1)
        if sr != 16000:                            # resample if needed
            g = gcd(16000, sr)
            data = resample_poly(data, 16000 // g, sr // g)
        scipy.io.wavfile.write(dst, 16000, data.astype(np.int16))
        ok += 1
    print(f"  {ok} clips -> {dest_dir}")

copy_and_resample(train_wavs, pos_train)
copy_and_resample(test_wavs,  pos_test)

# ── 5. Generate training config YAML ─────────────────────────────────────────
config = yaml.load(
    open("openwakeword/examples/custom_model.yml", "r").read(), yaml.Loader
)

config["target_phrase"]                       = [TARGET_WORD]
config["model_name"]                          = model_name
config["n_samples"]                           = len(train_wavs)
config["n_samples_val"]                       = len(test_wavs)
config["steps"]                               = NUMBER_OF_TRAINING_STEPS
config["target_accuracy"]                     = 0.7
config["target_recall"]                       = 0.5
config["output_dir"]                          = output_dir
config["max_negative_weight"]                 = FALSE_ACTIVATION_PENALTY
config["augmentation_rounds"]                 = AUGMENTATION_ROUNDS
config["background_paths"]                    = ["./audioset_16k", "./fma"]
config["false_positive_validation_data_path"] = "validation_set_features.npy"
config["feature_data_files"]                  = {
    "ACAV100M_sample": "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
}

with open("my_model.yaml", "w") as f:
    yaml.dump(config, f)

eff_train = len(train_wavs) * AUGMENTATION_ROUNDS
print(f"\nWake word:         {TARGET_WORD}")
print(f"Train clips:       {len(train_wavs)}  x{AUGMENTATION_ROUNDS} aug = {eff_train} effective samples")
print(f"Test clips:        {len(test_wavs)}")
print(f"Training steps:    {NUMBER_OF_TRAINING_STEPS}")
print(f"FP penalty:        {FALSE_ACTIVATION_PENALTY}")
print(f"Target accuracy:   0.7  (early stop)")
print(f"Target recall:     0.5  (early stop)")

# ── 6. Generate adversarial negative clips (TTS) ─────────────────────────────
print("\n=== STEP 0/2: Generate adversarial negative clips (TTS) ===")
run(f"{sys.executable} openwakeword/openwakeword/train.py --training_config my_model.yaml --generate_clips")

# ── 7. Augment ───────────────────────────────────────────────────────────────
print("\n=== STEP 1/2: Augment clips ===")
run(f"{sys.executable} openwakeword/openwakeword/train.py --training_config my_model.yaml --augment_clips --overwrite")

# ── 8. Train ──────────────────────────────────────────────────────────────────
print("\n=== STEP 2/2: Train model ===")
run(f"{sys.executable} openwakeword/openwakeword/train.py --training_config my_model.yaml --train_model")

# ── 9. ONNX -> TFLite ────────────────────────────────────────────────────────
print("\n=== Convert ONNX -> TFLite (via onnx2tf) ===")
onnx_path    = f"my_custom_model/{model_name}.onnx"
tflite_tmp   = f"my_custom_model/{model_name}_float32.tflite"
tflite_final = f"my_custom_model/{model_name}.tflite"

if os.path.exists(onnx_path):
    run(f"onnx2tf -i {onnx_path} -o my_custom_model/ -kat onnx____Flatten_0")
    if os.path.exists(tflite_tmp):
        os.rename(tflite_tmp, tflite_final)
    print(f"\n=== DONE ===")
    print(f"  ONNX       -> {onnx_path}")
    print(f"  TFLite f32 -> {tflite_final}")
    print(f"  TFLite f16 -> my_custom_model/{model_name}_float16.tflite")
else:
    print(f"\n[ERROR] {onnx_path} not found — training failed.")
    print("Check the log for errors in STEP 2/2.")
    sys.exit(1)

#!/usr/bin/env python3
"""
STEP 2 - Train on real voice recordings (no synthetic TTS for positive examples).

Prerequisites:
  - 00_fix_dependencies.py and 01_setup_and_download.py already run
  - WAV recordings in RECORDINGS_SOURCE_DIR (produced by record.py)
    Files at any sample rate are automatically resampled to 16 kHz.

Pipeline (each step is skipped if already completed):
  1. Prompt for wake word phrase
  2. Validate recordings
  3. Split 80% train / 20% test  ->  positive_train / positive_test
  4. Generate adversarial TTS negative clips
  5. Augment positive clips
  6. Train model
  7. Convert ONNX -> TFLite

Re-run options:
  --force            Re-run all steps from scratch
  --from <step>      Re-run from a specific step onward
                     Steps: split | generate | augment | train | convert

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
import argparse
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
RECORDINGS_SOURCE_DIR    = "./real_recordings"
TRAIN_SPLIT              = 0.8
AUGMENTATION_ROUNDS      = 50
NUMBER_OF_TRAINING_STEPS = 50000
FALSE_ACTIVATION_PENALTY = 1300
# ─────────────────────────────────────────────

STEPS_ORDER = ["split", "generate", "augment", "train", "convert"]

parser = argparse.ArgumentParser(description="Train a custom wake word model.")
group = parser.add_mutually_exclusive_group()
group.add_argument(
    "--force",
    action="store_true",
    help="Re-run all steps from scratch, ignoring any checkpoints.",
)
group.add_argument(
    "--from",
    dest="from_step",
    choices=STEPS_ORDER,
    metavar="STEP",
    help=f"Re-run from this step onward. One of: {', '.join(STEPS_ORDER)}",
)
args = parser.parse_args()

# Fix ulimit
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (65536, hard))
print(f"ulimit -n set to 65536 (was {soft})")

import yaml
import numpy as np
import scipy.io.wavfile


def run(cmd, ignore_error=False):
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0 and not ignore_error:
        print(f"[WARN] exit code {result.returncode}")
    return result.returncode == 0


def should_run(step: str) -> bool:
    """Return True if this step needs to run given --force / --from flags."""
    if args.force:
        return True
    if args.from_step:
        return STEPS_ORDER.index(step) >= STEPS_ORDER.index(args.from_step)
    return False  # checkpoints decide below


# ── 1. Prompt for wake word ───────────────────────────────────────────────────
print("\nEnter the wake word phrase.")
print("This is used to generate adversarial TTS negative samples and to name the output model.")
TARGET_WORD = input("Wake word: ").strip()
if not TARGET_WORD:
    print("[ERROR] Wake word cannot be empty.")
    sys.exit(1)

model_name = TARGET_WORD.replace(" ", "_")
output_dir = "./my_custom_model"
model_dir  = os.path.join(output_dir, model_name)
pos_train  = os.path.join(model_dir, "positive_train")
pos_test   = os.path.join(model_dir, "positive_test")
onnx_path  = os.path.join(output_dir, f"{model_name}.onnx")
tflite_final = os.path.join(output_dir, f"{model_name}.tflite")
tflite_tmp   = os.path.join(output_dir, f"{model_name}_float32.tflite")

# Sentinel files — created on successful step completion
sentinels = {
    step: Path(model_dir) / f".done_{step}"
    for step in STEPS_ORDER
}

os.makedirs(model_dir, exist_ok=True)

# Clear sentinels for steps that need to re-run
for step in STEPS_ORDER:
    if should_run(step) and sentinels[step].exists():
        sentinels[step].unlink()
        print(f"[RESET] checkpoint cleared for step: {step}")


def step_done(step: str) -> bool:
    return sentinels[step].exists()


def mark_done(step: str):
    sentinels[step].touch()


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
for d in [pos_train, pos_test,
          os.path.join(model_dir, "negative_train"),
          os.path.join(model_dir, "negative_test")]:
    os.makedirs(d, exist_ok=True)

# ── 4. Split and copy ─────────────────────────────────────────────────────────
if step_done("split"):
    print(f"\n[SKIP] split — reading clip counts from existing directories.")
    train_count = len(list(Path(pos_train).glob("*.wav")))
    test_count  = len(list(Path(pos_test).glob("*.wav")))
    print(f"  {train_count} train clips, {test_count} test clips.")
else:
    print(f"\n=== [split] 80/20 split -> {model_dir} ===")
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
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != 16000:
                g = gcd(16000, sr)
                data = resample_poly(data, 16000 // g, sr // g)
            scipy.io.wavfile.write(dst, 16000, data.astype(np.int16))
            ok += 1
        print(f"  {ok} clips -> {dest_dir}")

    copy_and_resample(train_wavs, pos_train)
    copy_and_resample(test_wavs,  pos_test)
    train_count = len(train_wavs)
    test_count  = len(test_wavs)
    mark_done("split")

# ── 5. Write training config YAML ────────────────────────────────────────────
config = yaml.load(
    open("openwakeword/examples/custom_model.yml", "r").read(), yaml.Loader
)
config["target_phrase"]                       = [TARGET_WORD]
config["model_name"]                          = model_name
config["n_samples"]                           = train_count
config["n_samples_val"]                       = test_count
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

eff_train = train_count * AUGMENTATION_ROUNDS
print(f"\nWake word:         {TARGET_WORD}")
print(f"Train clips:       {train_count}  x{AUGMENTATION_ROUNDS} aug = {eff_train} effective samples")
print(f"Test clips:        {test_count}")
print(f"Training steps:    {NUMBER_OF_TRAINING_STEPS}")
print(f"FP penalty:        {FALSE_ACTIVATION_PENALTY}")
print(f"Target accuracy:   0.7  (early stop)")
print(f"Target recall:     0.5  (early stop)")

# ── 6. Generate adversarial negative clips ────────────────────────────────────
if step_done("generate"):
    print("\n[SKIP] generate — adversarial negative clips already generated.")
else:
    print("\n=== [generate] Adversarial negative clips (TTS) ===")
    if run(f"{sys.executable} openwakeword/openwakeword/train.py --training_config my_model.yaml --generate_clips"):
        mark_done("generate")
    else:
        print("[ERROR] generate step failed. Fix the issue and re-run.")
        sys.exit(1)

# ── 7. Augment ────────────────────────────────────────────────────────────────
if step_done("augment"):
    print("\n[SKIP] augment — clips already augmented.")
else:
    print("\n=== [augment] Augment clips ===")
    if run(f"{sys.executable} openwakeword/openwakeword/train.py --training_config my_model.yaml --augment_clips"):
        mark_done("augment")
    else:
        print("[ERROR] augment step failed. Fix the issue and re-run.")
        sys.exit(1)

# ── 8. Train ──────────────────────────────────────────────────────────────────
if step_done("train"):
    print("\n[SKIP] train — model already trained.")
else:
    print("\n=== [train] Train model ===")
    if run(f"{sys.executable} openwakeword/openwakeword/train.py --training_config my_model.yaml --train_model"):
        mark_done("train")
    else:
        print("[ERROR] train step failed. Fix the issue and re-run.")
        sys.exit(1)

# ── 9. ONNX -> TFLite ────────────────────────────────────────────────────────
if step_done("convert"):
    print(f"\n[SKIP] convert — TFLite model already exists.")
else:
    print("\n=== [convert] ONNX -> TFLite (via onnx2tf) ===")
    if not os.path.exists(onnx_path):
        print(f"[ERROR] {onnx_path} not found — training output is missing.")
        sys.exit(1)
    if run(f"onnx2tf -i {onnx_path} -o {output_dir}/ -kat onnx____Flatten_0"):
        if os.path.exists(tflite_tmp):
            os.rename(tflite_tmp, tflite_final)
        mark_done("convert")
        print(f"\n=== DONE ===")
        print(f"  ONNX       -> {onnx_path}")
        print(f"  TFLite f32 -> {tflite_final}")
        print(f"  TFLite f16 -> {output_dir}/{model_name}_float16.tflite")
    else:
        print("[ERROR] convert step failed. Check the onnx2tf output.")
        sys.exit(1)

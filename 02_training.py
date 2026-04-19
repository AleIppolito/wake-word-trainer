#!/usr/bin/env python3
"""
STEP 2 - Train a custom wake word model on real voice recordings.

Prerequisites:
  - setup.sh already run (or 00_download.py + 01_fix_n_patch.py)
  - WAV recordings in --rec-dir (produced by record.py)

Pipeline (each step skipped if already completed):
  split    -> 80/20 train/test split of real recordings
  generate -> populate negative dirs from audioset
  augment  -> augment positive clips via openwakeword
  train    -> train the DNN model
  convert  -> ONNX -> TFLite (skipped if tensorflow-cpu not installed)

Usage:
  python 02_training.py <wake-word-phrase> [options]

  --rec-dir      DIR  recordings folder        (default: ./real_rec_raw)
  --steps        N    training steps            (default: 100000)
  --penalty      N    false activation penalty  (default: 300)
  --aug-rounds   N    augmentation rounds       (default: 150)
  --neg-train    N    audioset neg train clips  (default: 200)
  --neg-test     N    audioset neg test clips   (default: 50)
  --force            re-run all steps
  --from STEP        re-run from step onward
                     steps: split | generate | augment | train | convert
"""

import argparse
import os
import random
import resource
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import yaml
from _log import setup_log

log = setup_log("training")

STEPS_ORDER = ["split", "generate", "augment", "train", "convert"]

parser = argparse.ArgumentParser()
parser.add_argument("wake_word", help="Wake word phrase (e.g. 'hey murph')")
parser.add_argument("--rec-dir",    type=Path,  default=Path("./real_rec_prepared"))
parser.add_argument("--steps",      type=int,   default=100000)
parser.add_argument("--penalty",    type=int,   default=300)
parser.add_argument("--aug-rounds", type=int,   default=150)
parser.add_argument("--neg-train",         type=int, default=200)
parser.add_argument("--neg-test",          type=int, default=50)
parser.add_argument("--neg-speech-train",  type=int, default=200,
                    help="LibriSpeech clips added to negative train (0 to skip)")
parser.add_argument("--neg-speech-test",   type=int, default=50,
                    help="LibriSpeech clips added to negative test (0 to skip)")
group = parser.add_mutually_exclusive_group()
group.add_argument("--force", action="store_true")
group.add_argument("--from", dest="from_step", choices=STEPS_ORDER, metavar="STEP")
args = parser.parse_args()

TARGET_WORD  = args.wake_word.strip().lower()
model_name   = TARGET_WORD.replace(" ", "_")
output_dir   = Path("./my_custom_model")
model_dir    = output_dir / model_name
pos_train    = model_dir / "positive_train"
pos_test     = model_dir / "positive_test"
neg_train    = model_dir / "negative_train"
neg_test     = model_dir / "negative_test"
onnx_path    = output_dir / f"{model_name}.onnx"
tflite_final = output_dir / f"{model_name}.tflite"
tflite_tmp   = output_dir / f"{model_name}_float32.tflite"
sentinels    = {step: model_dir / f".done_{step}" for step in STEPS_ORDER}

soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (65536, hard))

model_dir.mkdir(parents=True, exist_ok=True)


def run(cmd, ignore_error=False):
    print(f"\n$ {cmd}")
    log.debug(f"run: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0 and not ignore_error:
        print(f"[WARN] exit code {result.returncode}")
        log.warning(f"exit code {result.returncode}: {cmd}")
    return result.returncode == 0


def should_run(step):
    if args.force:
        return True
    if args.from_step:
        return STEPS_ORDER.index(step) >= STEPS_ORDER.index(args.from_step)
    return False


def step_done(step):
    return sentinels[step].exists()


def mark_done(step):
    sentinels[step].touch()


def clear_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


# Reset sentinels + artifacts for forced steps
for step in STEPS_ORDER:
    if not should_run(step):
        continue
    if sentinels[step].exists():
        sentinels[step].unlink()
    print(f"[RESET] {step}")
    if step == "split":
        clear_dir(pos_train); clear_dir(pos_test)
    elif step == "generate":
        clear_dir(neg_train); clear_dir(neg_test)
    elif step == "augment":
        clear_dir(pos_train); clear_dir(pos_test)
        clear_dir(neg_train); clear_dir(neg_test)
        for s in ["generate", "split"]:
            if sentinels[s].exists(): sentinels[s].unlink()
        for npy in ["positive_features_train.npy", "positive_features_test.npy",
                    "negative_features_train.npy", "negative_features_test.npy"]:
            p = model_dir / npy
            if p.exists(): p.unlink()
    elif step == "train":
        if onnx_path.exists(): onnx_path.unlink()
        if sentinels["augment"].exists(): sentinels["augment"].unlink()
        for npy in ["positive_features_train.npy", "positive_features_test.npy",
                    "negative_features_train.npy", "negative_features_test.npy"]:
            p = model_dir / npy
            if p.exists(): p.unlink()
    elif step == "convert":
        for f in [tflite_final, tflite_tmp, output_dir / f"{model_name}_float16.tflite"]:
            if f.exists(): f.unlink()


# ── Validate recordings ───────────────────────────────────────────────────────
if not args.rec_dir.exists():
    print(f"[ERROR] Recordings folder not found: {args.rec_dir}")
    sys.exit(1)

all_wavs = sorted(args.rec_dir.glob("*.wav"))
if len(all_wavs) < 200:
    print(f"[ERROR] Only {len(all_wavs)} WAV files found. At least 200 required (300+ recommended).")
    sys.exit(1)
print(f"Found {len(all_wavs)} recordings in {args.rec_dir}")

for d in [pos_train, pos_test, neg_train, neg_test]:
    d.mkdir(parents=True, exist_ok=True)


# ── Split ─────────────────────────────────────────────────────────────────────
if step_done("split"):
    train_count = len(list(pos_train.glob("*.wav")))
    test_count  = len(list(pos_test.glob("*.wav")))
    print(f"\n[SKIP] split — {train_count} train / {test_count} test")
else:
    print("\n=== [split] 80/20 split ===")
    shuffled = list(all_wavs)
    random.shuffle(shuffled)
    split_idx  = int(len(shuffled) * 0.8)
    train_wavs = shuffled[:split_idx]
    test_wavs  = shuffled[split_idx:]

    def copy_and_resample(src_paths, dest_dir):
        from math import gcd
        from scipy.signal import resample_poly
        ok = 0
        for i, src in enumerate(src_paths):
            dst = dest_dir / f"{model_name}_{i:04d}.wav"
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
            scipy.io.wavfile.write(str(dst), 16000, data.astype(np.int16))
            ok += 1
        print(f"  {ok} clips -> {dest_dir}")

    copy_and_resample(train_wavs, pos_train)
    copy_and_resample(test_wavs, pos_test)
    train_count = len(train_wavs)
    test_count  = len(test_wavs)
    mark_done("split")


# ── Write training config YAML ────────────────────────────────────────────────
config = yaml.load(
    open("openwakeword/examples/custom_model.yml").read(), yaml.Loader
)
config["target_phrase"]                       = [TARGET_WORD]
config["model_name"]                          = model_name
config["n_samples"]                           = train_count
config["n_samples_val"]                       = test_count
config["steps"]                               = args.steps
config["target_accuracy"]                     = 0.85
config["target_recall"]                       = 0.7
config["target_false_positives_per_hour"]     = 50.0
config["output_dir"]                          = str(output_dir)
config["max_negative_weight"]                 = args.penalty
config["augmentation_rounds"]                 = args.aug_rounds
bg_paths = ["./audioset_16k", "./fma"]
if Path("./librispeech_16k").exists():
    bg_paths.append("./librispeech_16k")
config["background_paths"]                    = bg_paths
config["false_positive_validation_data_path"] = "validation_set_features.npy"
config["feature_data_files"]                  = {
    "ACAV100M_sample": "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
}
config_path = Path(f"{model_name}.yaml")
with open(config_path, "w") as f:
    yaml.dump(config, f)

log.info(f"wake_word={TARGET_WORD} steps={args.steps} penalty={args.penalty} "
         f"aug_rounds={args.aug_rounds} train={train_count} test={test_count} "
         f"neg_audioset={args.neg_train}/{args.neg_test} "
         f"neg_speech={args.neg_speech_train}/{args.neg_speech_test}")
print(f"\nWake word    : {TARGET_WORD}")
print(f"Train clips  : {train_count} x{args.aug_rounds} aug = {train_count * args.aug_rounds} effective")
print(f"Test clips   : {test_count}")
print(f"Steps        : {args.steps}")
print(f"FP penalty   : {args.penalty}")
print(f"Audioset negs: {args.neg_train} train / {args.neg_test} test")
print(f"Speech negs  : {args.neg_speech_train} train / {args.neg_speech_test} test")


# ── Generate (populate negative dirs from audioset) ───────────────────────────
if step_done("generate"):
    print("\n[SKIP] generate")
else:
    print("\n=== [generate] Populating negatives from audioset ===")
    audioset_wavs = sorted(Path("./audioset_16k").glob("*.wav"))
    if not audioset_wavs:
        print("[ERROR] audioset_16k/ empty — run 00_download.py first.")
        sys.exit(1)
    random.shuffle(audioset_wavs)
    for i, src in enumerate(audioset_wavs[:args.neg_train]):
        shutil.copy(src, neg_train / f"neg_train_{i:04d}.wav")
    for i, src in enumerate(audioset_wavs[args.neg_train:args.neg_train + args.neg_test]):
        shutil.copy(src, neg_test / f"neg_test_{i:04d}.wav")
    print(f"  {args.neg_train} audioset clips -> {neg_train}")
    print(f"  {args.neg_test} audioset clips  -> {neg_test}")

    ls_wavs = sorted(Path("./librispeech_16k").glob("*.wav")) if Path("./librispeech_16k").exists() else []
    if ls_wavs and (args.neg_speech_train > 0 or args.neg_speech_test > 0):
        random.shuffle(ls_wavs)
        for i, src in enumerate(ls_wavs[:args.neg_speech_train]):
            shutil.copy(src, neg_train / f"neg_speech_train_{i:04d}.wav")
        for i, src in enumerate(ls_wavs[args.neg_speech_train:args.neg_speech_train + args.neg_speech_test]):
            shutil.copy(src, neg_test / f"neg_speech_test_{i:04d}.wav")
        print(f"  {args.neg_speech_train} librispeech clips -> {neg_train}")
        print(f"  {args.neg_speech_test} librispeech clips  -> {neg_test}")
    elif args.neg_speech_train > 0:
        print("  [WARN] librispeech_16k/ not found — run 00_download.py to add speech negatives")

    mark_done("generate")


# ── Augment ───────────────────────────────────────────────────────────────────
if step_done("augment"):
    log.info("step augment: skipped (already done)")
    print("\n[SKIP] augment")
else:
    log.info("step augment: starting")
    print("\n=== [augment] ===")
    if run(f"{sys.executable} openwakeword/openwakeword/train.py --training_config {config_path} --augment_clips"):
        mark_done("augment")
        log.info("step augment: done")
    else:
        log.error("step augment: failed")
        print("[ERROR] augment failed.")
        sys.exit(1)


# ── Train ─────────────────────────────────────────────────────────────────────
if step_done("train"):
    log.info("step train: skipped (already done)")
    print("\n[SKIP] train")
else:
    log.info("step train: starting")
    print("\n=== [train] ===")
    if run(f"{sys.executable} openwakeword/openwakeword/train.py --training_config {config_path} --train_model"):
        mark_done("train")
        log.info("step train: done")
    else:
        log.error("step train: failed")
        print("[ERROR] train failed.")
        sys.exit(1)


# ── Convert ONNX -> TFLite ────────────────────────────────────────────────────
if step_done("convert"):
    print("\n[SKIP] convert")
else:
    print("\n=== [convert] ONNX -> TFLite ===")
    if not onnx_path.exists():
        print(f"[ERROR] {onnx_path} not found.")
        sys.exit(1)
    if run(f"onnx2tf -i {onnx_path} -o {output_dir}/ -kat onnx____Flatten_0"):
        if tflite_tmp.exists():
            tflite_tmp.rename(tflite_final)
        mark_done("convert")
        print(f"  ONNX       -> {onnx_path}")
        print(f"  TFLite f32 -> {tflite_final}")
        print(f"  TFLite f16 -> {output_dir}/{model_name}_float16.tflite")
    else:
        print("[WARN] TFLite conversion failed (tensorflow-cpu not installed?). ONNX model ready.")
        mark_done("convert")


# ── Download hint ─────────────────────────────────────────────────────────────
if onnx_path.exists():
    port = "22"
    try:
        for line in Path("/etc/ssh/sshd_config").read_text().splitlines():
            if line.strip().startswith("Port "):
                port = line.split()[1]; break
    except Exception:
        pass
    ip = "<server-ip>"
    try:
        result = subprocess.run("curl -s --max-time 3 ifconfig.me", shell=True,
                                capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            ip = result.stdout.strip()
    except Exception:
        pass

    cwd = output_dir.resolve()
    print(f"\n{'═'*60}")
    print(f"  Models ready. To download:")
    print(f"  scp -P {port} root@{ip}:{cwd}/{model_name}.onnx .")
    print(f"  scp -P {port} root@{ip}:{cwd}/{model_name}.tflite .")
    print(f"{'═'*60}")

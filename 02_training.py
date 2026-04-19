#!/usr/bin/env python3
"""
STEP 2 - Train on real voice recordings (no synthetic TTS for positive examples).

Prerequisites:
  - 00_download.py and 01_fix_n_patch.py already run (via setup.sh)
  - WAV recordings in RECORDINGS_SOURCE_DIR (produced by record.py)
    Files at any sample rate are automatically resampled to 16 kHz.

Pipeline (each step is skipped if already completed):
  1. Prompt for wake word phrase
  2. Validate recordings
  3. Split 80% train / 20% test  ->  positive_train / positive_test
  4. Generate adversarial TTS negative clips (Italian piper voice)
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
import shutil
import tempfile
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
RECORDINGS_SOURCE_DIR    = "./real_recordings"
TRAIN_SPLIT              = 0.8
AUGMENTATION_ROUNDS      = 150
NUMBER_OF_TRAINING_STEPS = 100000
FALSE_ACTIVATION_PENALTY = 100

# Italian Piper TTS voices for synthetic positive generation.
# Both downloaded by 00_download.py to ./models/
PIPER_MODELS = [
    "./models/it_IT-riccardo-x_low.onnx",  # male Italian — closest to user voice
]

# Number of synthetic TTS positive clips to generate per voice.
N_TTS_POSITIVE_TRAIN     = 200
N_TTS_POSITIVE_TEST      = 50
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
print("This is used to generate TTS positive samples and to name the output model.")
TARGET_WORD = input("Wake word: ").strip().lower()
if not TARGET_WORD:
    print("[ERROR] Wake word cannot be empty.")
    sys.exit(1)

model_name = TARGET_WORD.replace(" ", "_")
output_dir = "./my_custom_model"
model_dir  = os.path.join(output_dir, model_name)
pos_train  = os.path.join(model_dir, "positive_train")
pos_test   = os.path.join(model_dir, "positive_test")
neg_train  = os.path.join(model_dir, "negative_train")
neg_test   = os.path.join(model_dir, "negative_test")
onnx_path  = os.path.join(output_dir, f"{model_name}.onnx")
tflite_final = os.path.join(output_dir, f"{model_name}.tflite")
tflite_tmp   = os.path.join(output_dir, f"{model_name}_float32.tflite")

# Sentinel files - created on successful step completion
sentinels = {
    step: Path(model_dir) / f".done_{step}"
    for step in STEPS_ORDER
}

os.makedirs(model_dir, exist_ok=True)

def _clear_dir(path):
    """Remove and recreate a directory."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


# Clear sentinels AND output artifacts for steps that need to re-run
for step in STEPS_ORDER:
    if not should_run(step):
        continue
    if sentinels[step].exists():
        sentinels[step].unlink()
    print(f"[RESET] clearing step: {step}")
    if step == "split":
        _clear_dir(pos_train)
        _clear_dir(pos_test)
    elif step == "generate":
        _clear_dir(neg_train)
        _clear_dir(neg_test)
    elif step == "augment":
        # augmented clips live alongside originals; wipe and re-split from source
        _clear_dir(pos_train)
        _clear_dir(pos_test)
        # clear negative dirs and force generate to re-run so they get refilled
        _clear_dir(neg_train)
        _clear_dir(neg_test)
        if sentinels["generate"].exists():
            sentinels["generate"].unlink()
        # clear the split sentinel too so files are re-copied
        if sentinels["split"].exists():
            sentinels["split"].unlink()
        # delete feature .npy files — train.py skips augmentation if they exist
        for npy in ["positive_features_train.npy", "positive_features_test.npy",
                    "negative_features_train.npy", "negative_features_test.npy"]:
            p = os.path.join(model_dir, npy)
            if os.path.exists(p):
                os.remove(p)
    elif step == "train":
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
        # Feature .npy files are produced by the augment step and consumed by
        # train.  If we're re-running train we must regenerate them, so drop
        # the augment sentinel and the stale feature files.
        if sentinels["augment"].exists():
            sentinels["augment"].unlink()
        for npy in ["positive_features_train.npy", "positive_features_test.npy",
                    "negative_features_train.npy", "negative_features_test.npy"]:
            p = os.path.join(model_dir, npy)
            if os.path.exists(p):
                os.remove(p)
    elif step == "convert":
        for f in [tflite_final, tflite_tmp,
                  os.path.join(output_dir, f"{model_name}_float16.tflite")]:
            if os.path.exists(f):
                os.remove(f)


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
if len(all_wavs) < 100:
    print(f"[ERROR] Only {len(all_wavs)} WAV files found. At least 100 are required.")
    sys.exit(1)
print(f"Found {len(all_wavs)} WAV files.")

# ── 3. Create output directories ──────────────────────────────────────────────
for d in [pos_train, pos_test, neg_train, neg_test]:
    os.makedirs(d, exist_ok=True)

# ── 4. Split and copy ─────────────────────────────────────────────────────────
if step_done("split"):
    print(f"\n[SKIP] split - reading clip counts from existing directories.")
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
config["target_accuracy"]                     = 0.85
config["target_recall"]                       = 0.7
config["target_false_positives_per_hour"]     = 50.0
config["output_dir"]                          = output_dir
config["max_negative_weight"]                 = FALSE_ACTIVATION_PENALTY
config["augmentation_rounds"]                 = AUGMENTATION_ROUNDS
config["background_paths"]                    = ["./audioset_16k", "./fma"]
config["false_positive_validation_data_path"] = "validation_set_features.npy"
config["feature_data_files"]                  = {
    "ACAV100M_sample": "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
}
with open(f"{model_name}.yaml", "w") as f:
    yaml.dump(config, f)

tts_train_total = N_TTS_POSITIVE_TRAIN * len(PIPER_MODELS)
tts_test_total  = N_TTS_POSITIVE_TEST  * len(PIPER_MODELS)
eff_train = (train_count + tts_train_total) * AUGMENTATION_ROUNDS
print(f"\nWake word:         {TARGET_WORD}")
print(f"Real train clips:  {train_count}")
print(f"TTS positives:     +{tts_train_total} train / +{tts_test_total} test ({len(PIPER_MODELS)} voices)")
print(f"Total train:       {train_count + tts_train_total}  x{AUGMENTATION_ROUNDS} aug = {eff_train} effective samples")
print(f"Test clips:        {test_count + tts_test_total}")
print(f"Training steps:    {NUMBER_OF_TRAINING_STEPS}")
print(f"FP penalty:        {FALSE_ACTIVATION_PENALTY}")
print(f"Target accuracy:   0.7  (early stop)")
print(f"Target recall:     {config['target_recall']}  (early stop)")


# ── 6. Generate synthetic positive clips (Italian piper TTS) ─────────────────
def generate_tts_positives(dest_dir: str, n_clips: int, phrase: str, prefix: str) -> int:
    """
    Generate synthetic positive clips using all configured piper voices.
    Each voice contributes n_clips at 5 length scales.
    Clips land in dest_dir named <prefix>_<voice>_<n>.wav at 16kHz.
    Returns total clips written.
    """
    length_scales = [0.75, 0.85, 1.0, 1.15, 1.25]
    per_scale = max(1, n_clips // len(length_scales))
    remainder = n_clips - per_scale * len(length_scales)
    total = 0

    for model_path in PIPER_MODELS:
        if not os.path.exists(model_path):
            print(f"  [WARN] piper model not found: {model_path} — skipping")
            continue
        voice_tag = Path(model_path).stem  # e.g. it_IT-paola-medium

        for i, scale in enumerate(length_scales):
            count = per_scale + (1 if i < remainder else 0)
            text_input = (phrase + "\n") * count

            with tempfile.TemporaryDirectory() as tmpdir:
                result = subprocess.run(
                    f"piper --model {model_path} --length-scale {scale} --output_dir {tmpdir}/",
                    input=text_input.encode(),
                    shell=True,
                    capture_output=True,
                )
                if result.returncode != 0:
                    print(f"  [WARN] piper {voice_tag} scale={scale}: {result.stderr.decode()[:200]}")
                    continue

                for wav in sorted(Path(tmpdir).glob("*.wav")):
                    dst = os.path.join(dest_dir, f"{prefix}_{voice_tag}_{total:04d}.wav")
                    _sr, _data = scipy.io.wavfile.read(str(wav))
                    if _sr != 16000:
                        from math import gcd as _gcd
                        from scipy.signal import resample_poly as _rp
                        _g = _gcd(16000, _sr)
                        _data = _rp(_data.astype(np.float32), 16000 // _g, _sr // _g)
                    scipy.io.wavfile.write(dst, 16000, _data.astype(np.int16))
                    total += 1

    print(f"  {total} Italian TTS positives -> {dest_dir}")
    return total


if step_done("generate"):
    print("\n[SKIP] generate - TTS positive clips already generated.")
else:
    print("\n=== [generate] Synthetic positive clips (Italian piper TTS) ===")
    n_train = generate_tts_positives(pos_train, N_TTS_POSITIVE_TRAIN, TARGET_WORD, "tts_train")
    n_test  = generate_tts_positives(pos_test,  N_TTS_POSITIVE_TEST,  TARGET_WORD, "tts_test")
    if n_train > 0 and n_test > 0:
        mark_done("generate")
    else:
        print("[ERROR] generate step failed. Check piper installation and model paths.")
        sys.exit(1)

    # Populate neg dirs with audioset clips (train.py requires non-empty negative dirs)
    print("\n=== [generate] Populating negative dirs from audioset ===")
    audioset_wavs = sorted(Path("./audioset_16k").glob("*.wav"))
    if not audioset_wavs:
        print("[ERROR] audioset_16k/ empty — run 00_download.py first.")
        sys.exit(1)
    random.shuffle(audioset_wavs)
    for i, src in enumerate(audioset_wavs[:200]):
        shutil.copy(src, os.path.join(neg_train, f"neg_train_{i:04d}.wav"))
    for i, src in enumerate(audioset_wavs[200:250]):
        shutil.copy(src, os.path.join(neg_test, f"neg_test_{i:04d}.wav"))
    print(f"  200 audioset clips -> {neg_train}")
    print(f"  50  audioset clips -> {neg_test}")


# ── 7. Augment ────────────────────────────────────────────────────────────────
if step_done("augment"):
    print("\n[SKIP] augment - clips already augmented.")
else:
    print("\n=== [augment] Augment clips ===")
    if run(f"{sys.executable} openwakeword/openwakeword/train.py --training_config {model_name}.yaml --augment_clips"):
        mark_done("augment")
    else:
        print("[ERROR] augment step failed. Fix the issue and re-run.")
        sys.exit(1)

# ── 8. Train ──────────────────────────────────────────────────────────────────
if step_done("train"):
    print("\n[SKIP] train - model already trained.")
else:
    print("\n=== [train] Train model ===")
    if run(f"{sys.executable} openwakeword/openwakeword/train.py --training_config {model_name}.yaml --train_model"):
        mark_done("train")
    else:
        print("[ERROR] train step failed. Fix the issue and re-run.")
        sys.exit(1)

# ── 9. ONNX -> TFLite ────────────────────────────────────────────────────────
if step_done("convert"):
    print(f"\n[SKIP] convert - TFLite model already exists.")
else:
    print("\n=== [convert] ONNX -> TFLite (via onnx2tf) ===")
    if not os.path.exists(onnx_path):
        print(f"[ERROR] {onnx_path} not found - training output is missing.")
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
        print("[WARN] TFLite conversion failed (tensorflow not installed?). ONNX model is ready.")
        print(f"  ONNX -> {onnx_path}")
        mark_done("convert")


# ── Download instructions ─────────────────────────────────────────────────────
def _get_ssh_port() -> str:
    try:
        for line in Path("/etc/ssh/sshd_config").read_text().splitlines():
            line = line.strip()
            if line.startswith("Port "):
                return line.split()[1]
    except Exception:
        pass
    return "22"


def _get_ip() -> str:
    try:
        result = subprocess.run(
            "curl -s --max-time 3 ifconfig.me",
            shell=True, capture_output=True, text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "<server-ip>"


if os.path.exists(onnx_path):
    _port = _get_ssh_port()
    _ip   = _get_ip()
    _cwd  = os.path.abspath(output_dir)
    print(f"\n{'═' * 60}")
    print(f"  Models ready. To download from your local machine:")
    print(f"")
    print(f"  scp -P {_port} root@{_ip}:{_cwd}/{model_name}.onnx .")
    print(f"  scp -P {_port} root@{_ip}:{_cwd}/{model_name}.tflite .")
    print(f"  scp -P {_port} root@{_ip}:{_cwd}/{model_name}_float16.tflite .")
    print(f"{'═' * 60}")

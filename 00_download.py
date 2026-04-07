#!/usr/bin/env python3
"""
STEP 0 - Set up piper-sample-generator and download datasets.
Run ONCE. Skips anything already downloaded.

Prerequisites:
  pip install -r requirements.txt

Output:
  - ./piper-sample-generator/                           (TTS engine)
  - ./openwakeword/                                     (training framework)
  - ./mit_rirs/                                         (Room Impulse Responses)
  - ./audioset_16k/                                     (background noise)
  - ./fma/                                              (background music)
  - ./openwakeword_features_ACAV100M_2000_hrs_16bit.npy (~17 GB, training features)
  - ./validation_set_features.npy                       (~180 MB, validation features)
"""

import os
import sys
import subprocess
import locale


def run(cmd, ignore_error=False):
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0 and not ignore_error:
        print(f"[WARN] exit code {result.returncode}")


def fix_locale():
    def getpreferredencoding(do_setlocale=True):
        return "UTF-8"
    locale.getpreferredencoding = getpreferredencoding


fix_locale()

# ── piper-sample-generator ────────────────────────────────────────────────────
print("=== Setup piper-sample-generator ===")
if not os.path.exists("./piper-sample-generator"):
    run("git clone https://github.com/rhasspy/piper-sample-generator")
    run("git -C piper-sample-generator checkout 213d4d5")
    run("wget -O piper-sample-generator/models/en_US-libritts_r-medium.pt "
        "'https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt'")
else:
    print("piper-sample-generator already present, skipping.")

if "piper-sample-generator/" not in sys.path:
    sys.path.append("piper-sample-generator/")

# ── openwakeword ──────────────────────────────────────────────────────────────
print("\n=== Setup openwakeword ===")
if not os.path.exists("./openwakeword"):
    run("git clone https://github.com/dscripka/openwakeword")
    # --no-deps: all dependencies are already in requirements.txt
    run("pip install -e ./openwakeword --no-deps")
else:
    print("openwakeword already present, skipping.")

# ── openWakeWord models ───────────────────────────────────────────────────────
print("\n=== Download openWakeWord models ===")
models_dir = "./openwakeword/openwakeword/resources/models"
os.makedirs(models_dir, exist_ok=True)
base_url = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"
for fname in ["embedding_model.onnx", "embedding_model.tflite",
              "melspectrogram.onnx", "melspectrogram.tflite"]:
    out = f"{models_dir}/{fname}"
    if not os.path.exists(out):
        run(f"wget {base_url}/{fname} -O {out}")
    else:
        print(f"  {fname} already present, skipping.")

# ── datasets ──────────────────────────────────────────────────────────────────
import numpy as np
from pathlib import Path
import datasets as hf_datasets
import scipy.io.wavfile
from tqdm import tqdm

# MIT RIR
print("\n=== Download MIT RIR ===")
if not os.path.exists("./mit_rirs"):
    os.mkdir("./mit_rirs")
    run("git lfs install")
    run("git clone https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses")
    rir_dataset = hf_datasets.Dataset.from_dict({
        "audio": [str(i) for i in Path("./MIT_environmental_impulse_responses/16khz").glob("*.wav")]
    }).cast_column("audio", hf_datasets.Audio())
    for row in tqdm(rir_dataset, desc="MIT RIR"):
        name = row["audio"]["path"].split("/")[-1]
        scipy.io.wavfile.write(f"./mit_rirs/{name}", 16000,
                               (row["audio"]["array"] * 32767).astype(np.int16))
else:
    print("MIT RIR already present, skipping.")

# AudioSet
print("\n=== Download AudioSet ===")
if not os.path.exists("audioset"):
    os.mkdir("audioset")
    fname = "bal_train09.tar"
    run(f"wget -O audioset/{fname} https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/{fname}")
    run("cd audioset && tar -xvf bal_train09.tar")
    os.makedirs("./audioset_16k", exist_ok=True)
    audioset_dataset = hf_datasets.Dataset.from_dict({
        "audio": [str(i) for i in Path("audioset/audio").glob("**/*.flac")]
    }).cast_column("audio", hf_datasets.Audio(sampling_rate=16000))
    for row in tqdm(audioset_dataset, desc="AudioSet"):
        name = row["audio"]["path"].split("/")[-1].replace(".flac", ".wav")
        scipy.io.wavfile.write(f"./audioset_16k/{name}", 16000,
                               (row["audio"]["array"] * 32767).astype(np.int16))
else:
    print("AudioSet already present, skipping.")

# FMA
print("\n=== Download FMA ===")
if not os.path.exists("./fma"):
    os.mkdir("./fma")
    fma_dataset = hf_datasets.load_dataset("rudraml/fma", name="small", split="train", streaming=True)
    fma_dataset = iter(fma_dataset.cast_column("audio", hf_datasets.Audio(sampling_rate=16000)))
    n_hours = 1
    for i in tqdm(range(n_hours * 3600 // 30), desc="FMA"):
        row = next(fma_dataset)
        name = row["audio"]["path"].split("/")[-1].replace(".mp3", ".wav")
        scipy.io.wavfile.write(f"./fma/{name}", 16000,
                               (row["audio"]["array"] * 32767).astype(np.int16))
else:
    print("FMA already present, skipping.")

# Pre-computed features
print("\n=== Download ACAV100M features (~17 GB) ===")
if not os.path.exists("./openwakeword_features_ACAV100M_2000_hrs_16bit.npy"):
    run("wget https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy")
else:
    print("ACAV100M features already present, skipping.")

print("\n=== Download validation set ===")
if not os.path.exists("./validation_set_features.npy"):
    run("wget https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy")
else:
    print("Validation set already present, skipping.")

print("\n=== Download complete ===")

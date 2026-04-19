#!/usr/bin/env python3
"""
STEP 0 - Set up openwakeword and download training datasets.
Run ONCE. Skips anything already downloaded.

Prerequisites:
  pip install -r requirements.txt

Output:
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
import io
import numpy as np
import pandas as pd
from pathlib import Path
import librosa
import datasets as hf_datasets
import scipy.io.wavfile
import shutil
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

# MIT RIR
print("\n=== Download MIT RIR ===")
if not os.path.exists("./mit_rirs"):
    os.mkdir("./mit_rirs")
    run("git lfs install")
    run("git clone https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses")
    for wav_path in tqdm(sorted(Path("./MIT_environmental_impulse_responses/16khz").glob("*.wav")), desc="MIT RIR"):
        data, _ = librosa.load(wav_path, sr=16000, mono=True)
        scipy.io.wavfile.write(f"./mit_rirs/{wav_path.name}", 16000,
                               (data * 32767).astype(np.int16))
else:
    print("MIT RIR already present, skipping.")

# AudioSet
print("\n=== Download AudioSet ===")
if not os.path.exists("./audioset_16k"):
    os.makedirs("./audioset_16k", exist_ok=True)
    # Dataset migrated from .tar to parquet; bal_train has shards 00–37
    parquet_shards = [f"data/bal_train/{i:02d}.parquet" for i in range(38)]
    for shard in tqdm(parquet_shards, desc="AudioSet shards"):
        cached = hf_hub_download(
            repo_id="agkphysics/AudioSet",
            filename=shard,
            repo_type="dataset",
        )
        df = pd.read_parquet(cached)
        for _, row in df.iterrows():
            audio_bytes = row["audio"]["bytes"]
            data, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
            name = f"{row['video_id']}.wav"
            scipy.io.wavfile.write(f"./audioset_16k/{name}", 16000,
                                   (data * 32767).astype(np.int16))
else:
    print("AudioSet already present, skipping.")

# FMA (using lewtun/music_genres — parquet shards with HF Audio bytes format)
print("\n=== Download FMA ===")
if not os.path.exists("./fma"):
    os.mkdir("./fma")
    parquet_shards = sorted(
        f for f in list_repo_files("lewtun/music_genres", repo_type="dataset")
        if f.startswith("data/train-") and f.endswith(".parquet")
    )
    i = 0
    for shard in tqdm(parquet_shards, desc="FMA shards"):
        cached = hf_hub_download(
            repo_id="lewtun/music_genres",
            filename=shard,
            repo_type="dataset",
        )
        df = pd.read_parquet(cached)
        for _, row in df.iterrows():
            audio_bytes = row["audio"]["bytes"]
            data, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
            scipy.io.wavfile.write(f"./fma/track_{i:04d}.wav", 16000,
                                   (data * 32767).astype(np.int16))
            i += 1
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

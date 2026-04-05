#!/usr/bin/env python3
"""
STEP 1 - Setup piper-sample-generator + download dataset
Da eseguire UNA VOLTA SOLA. Se i dati sono già scaricati salta tutto.

Prerequisiti:
  pip install -r requirements.txt
  python 00_fix_dependencies.py

Output:
  - ./piper-sample-generator/     (TTS engine)
  - ./openwakeword/               (framework training)
  - ./mit_rirs/                   (Room Impulse Responses)
  - ./audioset_16k/               (background noise)
  - ./fma/                        (musica background)
  - ./openwakeword_features_ACAV100M_2000_hrs_16bit.npy  (~17GB, training features)
  - ./validation_set_features.npy (~180MB, validation features)
"""

import os
import sys
import subprocess
import locale


def run(cmd, ignore_error=False):
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0 and not ignore_error:
        print(f"[WARN] codice di uscita {result.returncode}")


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
    # piper-tts e piper-phonemize-cross NON sono in requirements.txt perché
    # piper_phonemize viene installato come pacchetto locale da piper-phonemize-cross
    # e il path cambia ad ogni installazione.
    run("pip install piper-tts piper-phonemize-cross webrtcvad-wheels")
else:
    print("piper-sample-generator già presente, skip.")

if "piper-sample-generator/" not in sys.path:
    sys.path.append("piper-sample-generator/")

# ── openwakeword ──────────────────────────────────────────────────────────────
print("\n=== Setup openwakeword ===")
if not os.path.exists("./openwakeword"):
    run("git clone https://github.com/dscripka/openwakeword")
    # --no-deps: tutte le dipendenze sono già in requirements.txt
    run("pip install -e ./openwakeword --no-deps")
    # datasets e deep-phonemizer servono solo per il download; non pinniamo
    # la versione per evitare conflitti con pyarrow già installato.
    run("pip install 'datasets>=2.14' deep-phonemizer==0.0.19")
else:
    print("openwakeword già presente, skip.")

# ── modelli openWakeWord ──────────────────────────────────────────────────────
print("\n=== Download modelli openWakeWord ===")
models_dir = "./openwakeword/openwakeword/resources/models"
os.makedirs(models_dir, exist_ok=True)
base_url = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"
for fname in ["embedding_model.onnx", "embedding_model.tflite",
              "melspectrogram.onnx", "melspectrogram.tflite"]:
    out = f"{models_dir}/{fname}"
    if not os.path.exists(out):
        run(f"wget {base_url}/{fname} -O {out}")
    else:
        print(f"  {fname} già presente, skip.")

# ── dataset ───────────────────────────────────────────────────────────────────
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
    print("MIT RIR già presente, skip.")

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
    print("AudioSet già presente, skip.")

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
    print("FMA già presente, skip.")

# Feature pre-calcolate
print("\n=== Download feature ACAV100M (~17GB) ===")
if not os.path.exists("./openwakeword_features_ACAV100M_2000_hrs_16bit.npy"):
    run("wget https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy")
else:
    print("Feature ACAV100M già presenti, skip.")

print("\n=== Download validation set ===")
if not os.path.exists("./validation_set_features.npy"):
    run("wget https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy")
else:
    print("Validation set già presente, skip.")

print("\n=== Setup e download completati ===")
print("Procedi con: python 02_training.py")

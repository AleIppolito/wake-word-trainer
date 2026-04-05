# Wake Word Trainer — openWakeWord

Train a custom wake word model using your own voice recordings and the [openWakeWord](https://github.com/dscripka/openwakeword) framework. The pipeline uses real voice samples (no synthetic TTS for positive examples) and produces both ONNX and TFLite models ready for deployment.

---

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.12 |
| CUDA | 13.0 |
| Disk space | ~20 GB (dataset + venv) |
| RAM | 16 GB+ recommended |

---

## Pipeline overview

```
[any machine]                     [Linux server / VM]
record.py  ──────── scp ────────► real_recordings/
                                  │
                                  ├─ 00_fix_dependencies.py   (patch libraries)
                                  ├─ 01_setup_and_download.py (clone repos + download ~17 GB)
                                  └─ 02_training.py           (train + export)
```

---

## Step-by-step usage

### 1. Record your voice

Run on any machine with a microphone:

```bash
pip install sounddevice soundfile numpy
python record.py
```

Press **Enter** to record a 2-second clip, say your wake word when prompted, then repeat.
Press `q` to quit. Aim for **at least 200 clips** (300+ recommended).

Tips for a robust model:
- Vary your distance from the mic (20 cm / 50 cm / 1 m)
- Vary pitch, speed, and volume
- Record in different rooms if possible

Clips are saved in `real_recordings/`. Transfer them to the server:

```bash
scp -r real_recordings/ root@<SERVER_IP>:/root/wake-word-trainer/real_recordings
```

---

### 2. Full setup (one-time, on the server)

The easiest path is the all-in-one script:

```bash
bash setup.sh
```

This will:
1. Create a Python 3.12 venv at `./venv`
2. Install all dependencies from `requirements.txt`
3. Apply compatibility patches (`00_fix_dependencies.py`)
4. Clone `piper-sample-generator` and `openwakeword`
5. Download all training data (~17 GB total)

Alternatively, run each step manually:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python 00_fix_dependencies.py
python 01_setup_and_download.py
```

**Downloaded assets:**

| Asset | Size | Purpose |
|---|---|---|
| `openwakeword_features_ACAV100M_2000_hrs_16bit.npy` | ~17 GB | Pre-computed background features |
| `validation_set_features.npy` | ~180 MB | Validation set |
| `mit_rirs/` | small | Room impulse responses for augmentation |
| `audioset_16k/` | ~1 GB | Background noise |
| `fma/` | ~1 GB | Background music (1 hour) |

---

### 3. Train the model

```bash
source venv/bin/activate
python 02_training.py
```

You will be prompted to type your wake word phrase. This is used to generate adversarial TTS negative samples (a synthesized voice saying the phrase, so the model learns to reject near-matches) and to name the output files.

**What it does:**

1. Prompts for the wake word phrase
2. Validates that `./real_recordings/` contains at least 20 WAV files
3. Splits them 80% train / 20% test, resampling to 16 kHz if needed
4. Generates adversarial negative clips via TTS (Piper)
5. Augments positive clips (default: ×50 rounds → 10 000+ examples)
6. Trains the DNN model for up to 50 000 steps
7. Converts the output ONNX model to TFLite (float32 + float16)

**Output files:**

```
my_custom_model/
├── <model_name>.onnx
├── <model_name>.tflite        (float32)
└── <model_name>_float16.tflite
```

where `<model_name>` is the wake word phrase with spaces replaced by underscores.

---

## Configuration reference

Key parameters at the top of `02_training.py`:

| Parameter | Default | Description |
|---|---|---|
| `RECORDINGS_SOURCE_DIR` | `./real_recordings` | Folder with your WAV recordings |
| `TRAIN_SPLIT` | `0.8` | Fraction used for training (rest = test) |
| `AUGMENTATION_ROUNDS` | `50` | How many times to augment each clip |
| `NUMBER_OF_TRAINING_STEPS` | `50000` | Max training steps |
| `FALSE_ACTIVATION_PENALTY` | `1300` | Weight penalty for false positives |

---

## Known issues and fixes

### `ModuleNotFoundError: No module named 'pkg_resources'`

**Cause:** `pronouncing` uses `pkg_resources`, which was removed from the standard library in Python 3.12.

**Fix:** Handled automatically by `00_fix_dependencies.py`. It patches `pronouncing/__init__.py` to use `importlib.resources` instead.

---

### `ImportError: cannot import name 'sph_harm' from 'scipy.special'`

**Cause:** `acoustics` imports `sph_harm`, which was renamed to `sph_harm_y` in SciPy ≥ 1.15.

**Fix:** Handled automatically by `00_fix_dependencies.py`. It patches `acoustics/directivity.py` with an alias import.

---

### Wake word not activating reliably

- Lower `FALSE_ACTIVATION_PENALTY` (try 1000–1500)
- Add more recording variety (different environments, distances)
- Increase `AUGMENTATION_ROUNDS`

---

### Too many false positives

- Raise `FALSE_ACTIVATION_PENALTY` (try 2500–3000)
- Re-run `02_training.py` with the updated value

---

### Training crashes at augmentation step (too many open files)

`02_training.py` automatically raises the file descriptor limit to 65536 via `ulimit`. If you still hit the limit, raise it manually before running:

```bash
ulimit -n 65536
```

---

### `onnx2tf` TFLite conversion fails

The conversion uses a fixed key axis tag (`-kat onnx____Flatten_0`). If the ONNX graph changes due to a different openwakeword version, this tag may be wrong. Check the onnx2tf output for the correct tensor name and update the flag in `02_training.py`:

```python
run(f"onnx2tf -i {onnx_path} -o my_custom_model/ -kat <correct_tensor_name>")
```

---

## Project structure

```
wake-word-trainer/
├── setup.sh                    # One-command full setup
├── requirements.txt            # Pinned dependencies (Python 3.12 + CUDA 13.0)
├── 00_fix_dependencies.py      # Post-install patches for pronouncing & acoustics
├── 01_setup_and_download.py    # Clone repos + download datasets
├── 02_training.py              # Main training pipeline
└── record.py                   # Voice recorder
```

---

## License

This project is a training harness that wraps [openWakeWord](https://github.com/dscripka/openwakeword) and [piper-sample-generator](https://github.com/rhasspy/piper-sample-generator). Refer to their respective licenses for redistribution terms.

# Wake Word Trainer - openWakeWord

Train a custom wake word model using real voice recordings and the [openWakeWord](https://github.com/dscripka/openwakeword) framework. Produces ONNX and TFLite models.

---

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.12 |
| CUDA | 13.0 |
| Disk space | ~200 GB (datasets + venv + training) |
| RAM | 16 GB+ recommended |

---

## Pipeline

```
[any machine]                     [Linux VM]
record.py  ──────── scp ────────► real_recordings/
                                  │
                                  ├─ bash setup.sh          (one-time)
                                  └─ python 02_training.py  (train + export)
```

---

## Step-by-step

### 1. Record your voice (local machine)

```bash
pip install sounddevice soundfile numpy
python record.py
```

Press **Enter** to record a 2-second clip, `q` to quit. Aim for **300+ clips**.

Tips:
- Vary distance: 20 cm / 50 cm / 1 m
- Vary pitch, speed, volume
- Record in different rooms

Transfer recordings to the VM:

```bash
scp -r real_recordings/ root@<SERVER_IP>:/path/to/wake-word-trainer/real_recordings
```

---

### 2. Setup (one-time, on the VM)

```bash
git clone <repo>
cd wake-word-trainer
bash setup.sh
```

This will:
1. Create a Python 3.12 venv at `./.venv`
2. Install all dependencies from `requirements.txt`
3. Clone `openwakeword`, download all training data (~17 GB)
4. Apply compatibility patches (`01_fix_n_patch.py`)

**Downloaded assets:**

| Asset | Size | Purpose |
|---|---|---|
| `openwakeword_features_ACAV100M_2000_hrs_16bit.npy` | ~17 GB | Pre-computed background features |
| `validation_set_features.npy` | ~180 MB | Validation set |
| `mit_rirs/` | small | Room impulse responses |
| `audioset_16k/` | ~1 GB | Background noise (negatives) |
| `fma/` | ~1 GB | Background music (negatives) |

---

### 3. Train

```bash
source .venv/bin/activate
python 02_training.py "hey murph"
```

**Key options:**

| Flag | Default | Description |
|---|---|---|
| `--steps` | 100000 | Training steps |
| `--penalty` | 300 | False activation weight penalty |
| `--aug-rounds` | 150 | Augmentation rounds per clip |
| `--neg-train` | 200 | Audioset clips for negative train |
| `--neg-test` | 50 | Audioset clips for negative test |
| `--force` | — | Re-run all steps from scratch |
| `--from STEP` | — | Re-run from step: `split\|generate\|augment\|train\|convert` |

**Output:**

```
my_custom_model/
├── hey_murph.onnx
├── hey_murph.tflite        (float32, requires tensorflow-cpu)
└── hey_murph_float16.tflite
```

---

### 4. Evaluate

```bash
python eval.py my_custom_model/hey_murph.onnx
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--rec-dir` | `./real_recordings` | Recordings for recall eval |
| `--fp-dir` | `./audioset_16k` | Background clips for FP eval |
| `--threshold` | 0.3 | Score threshold |
| `--fp-samples` | 500 | Background clips to sample |

**Targets:**

| Metric | Good | Acceptable |
|---|---|---|
| Recall | > 80% | > 70% |
| FP/hour | < 0.5 | < 1.0 |

---

### 5. Validate recordings (optional)

```bash
python validate_recordings.py [recordings_dir]
```

Flags bad clips by duration, RMS, onset/offset silence, and clipping.

---

## Architecture

openWakeWord 3-stage ONNX pipeline:
```
WAV → melspectrogram.onnx → embedding_model.onnx → hey_murph.onnx → score
```
- mel: 76-frame window, 32 features, 80ms chunks (1280 samples @ 16 kHz)
- emb: 96-dim embedding per window
- wake: takes **16** embedding frames → scalar score (0–1)

**Critical:** `EMB_WINDOW` must be **16**. Model input shape is `(1, 16, 96)`.

---

## VM setup notes (CUDA situation)

The VM has both CUDA 13.0 and CUDA 12.9. PyTorch uses 13.0; ORT needs CUDA 12.9 libs. `setup.sh` handles this automatically.

If ORT falls back to CPU, verify:
```bash
source .venv/bin/activate
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# should include CUDAExecutionProvider
```

---

## Known issues and fixes (handled by `01_fix_n_patch.py`)

| Fix | Cause |
|---|---|
| `acoustics/directivity.py`: `sph_harm` → `sph_harm_y` | scipy ≥ 1.15 renamed it |
| `torchaudio.info()` stub | removed in torchaudio 2.9+ |
| `torchaudio.load()` librosa fallback | torchcodec removed in 2.9+ |
| `train.py` `--convert_to_tflite` default bug | string `"False"` is always truthy |
| `train.py` ONNX export: add `.eval()` + opset 13→18 | LayerNorm requires opset ≥ 17 |
| `train.py` suppress onnxscript/onnx_ir DEBUG spam | hundreds of irrelevant log lines |
| `torch_audiomentations` FutureWarning | noisy deprecation warnings |
| `train.py` lazy `generate_samples` import | piper-sample-generator not required |

---

## TFLite conversion

TFLite (for RPi deployment) requires `tensorflow-cpu` (~500 MB). It is included in `requirements.txt`. If you don't need TFLite, the ONNX model is ready after the `train` step.

If `onnx2tf` conversion fails, check the key axis tag:
```python
# In 02_training.py convert step:
run(f"onnx2tf -i {onnx_path} -o {output_dir}/ -kat onnx____Flatten_0")
# Update -kat value if the ONNX graph changes
```

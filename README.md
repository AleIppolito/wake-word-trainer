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
[local machine]                           [Linux VM]
record.py  → real_rec_raw/
prepare.py → real_rec_prepared/ ─── scp ──► real_rec_prepared/
                                             │
                                             ├─ bash setup.sh          (one-time)
                                             └─ python 02_training.py
```

---

## Step-by-step

### 1. Record your voice (local machine)

```bash
bash local/setup.sh
source local/.venv/bin/activate
python local/record.py --mode close --room cucina
```

Press **Enter** to record a 2-second clip, `q` to quit.

Each clip is validated immediately after recording (RMS, onset/offset silence, clipping). Bad clips are rejected on the spot and not saved.

Clips are saved as `recording_NNNN_ROOM_MODE.wav`. A variety matrix tracks counts per room×distance combination and warns when a cell hits its per-cell cap (`target // (n_rooms × n_modes)`).

```
  Variety matrix  (per-cell target: 55)
  ──────────────────────────────────────
               close   mid    far
  cucina        48     41     38  ◄
  sala           3      0      0
  camera         0      0      0
  ──────────────────────────────────────
  Total: 130 / 500
```

Run sessions across different rooms and distances to fill the matrix evenly:

```bash
python local/record.py --mode far --room sala
python local/record.py --mode mid --room camera
```

Validate existing clips without recording:

```bash
python local/record.py --validate
```

Run prepare to normalize, trim, center, and quality-check before transferring:

```bash
python local/prepare.py
```

Output goes to `real_rec_prepared/`. Raw files in `real_rec_raw/` are never modified. Re-record any clips flagged as BAD.

Transfer prepared recordings to the VM:

```bash
scp -r real_rec_prepared/ root@<SERVER_IP>:/path/to/wake-word-trainer/real_rec_prepared
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
| `librispeech_16k/` | ~500 MB | Real speech negatives (FP suppression for speech) |

---

### 3. Train

```bash
source .venv/bin/activate
python 02_training.py "hey murph"
```

**Key options:**

| Flag | Default | Description |
|---|---|---|
| `--steps` | 200000 | Training steps |
| `--penalty` | 500 | False activation weight penalty |
| `--aug-rounds` | 150 | Augmentation rounds per clip |
| `--neg-train` | 2000 | Audioset clips for negative train |
| `--neg-test` | 200 | Audioset clips for negative test |
| `--neg-speech-train` | 2000 | LibriSpeech clips for negative train |
| `--neg-speech-test` | 200 | LibriSpeech clips for negative test |
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
| `--rec-dir` | `./real_rec_raw` | Recordings for recall eval |
| `--fp-dir` | `./audioset_16k` | Background clips for FP eval |
| `--threshold` | 0.3 | Score threshold |
| `--fp-samples` | 500 | Background clips to sample |

**Targets:**

| Metric | Good | Acceptable |
|---|---|---|
| Recall | > 80% | > 70% |
| FP/hour | < 0.5 | < 1.0 |

---

---

## Project structure

```
wake-word-trainer/
├── setup.sh               One-command VM setup
├── 00_download.py         Clone openwakeword + download datasets
├── 01_fix_n_patch.py      Post-install patches for library incompatibilities
├── 02_training.py         Main training pipeline (split → augment → train → convert)
├── eval.py                Recall + FP/hour evaluation
├── requirements.txt       Pinned deps (Python 3.12 + CUDA 13.0)
├── _log.py                Shared logging helper (writes to log/)
└── local/
    ├── setup.sh           Creates local venv + installs deps
    ├── record.py          Voice recorder — runs on local machine, not the VM
    ├── prepare.py         Center speech + trim silence before sending to VM
    └── requirements.txt   Local deps only (sounddevice, soundfile, numpy)
```

`local/` contains tools that run on your local machine. Everything else runs on the VM.

Logs are written to `log/` (gitignored, auto-created):

| File | Written by |
|---|---|
| `log/setup.log` | `setup.sh` |
| `log/download.log` | `00_download.py` |
| `log/patch.log` | `01_fix_n_patch.py` |
| `log/training.log` | `02_training.py` |
| `log/eval.log` | `eval.py` |
| `log/record.log` | `local/record.py` |

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

TFLite conversion runs automatically after training via `onnx2tf`.

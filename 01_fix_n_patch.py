#!/usr/bin/env python3
"""
STEP 1 - Apply patches to installed libraries.
Run ONCE after: python 00_download.py

Actions:
  2. acoustics:   sph_harm renamed in scipy >= 1.15 -> sph_harm_y
  3. piper:       torch.load weights_only=False (torch >= 2.6)  [no-op if piper-sample-generator absent]
  4. torchaudio:  load() librosa fallback when torchcodec absent (removed in 2.9+)
  5. torchaudio:  info() soundfile stub (removed in 2.9+)
  6. train.py:    --convert_to_tflite default='False' bug (string always truthy)
  7. train.py:    ONNX export — add .eval() + opset 13->18
  8. train.py:    suppress onnxscript/onnx_ir DEBUG spam
  9. torch_audiomentations: silence output_type FutureWarning/DeprecationWarning
 10. train.py:    make generate_samples import lazy (piper-sample-generator not required)
"""

import subprocess
import sys
import site
from pathlib import Path
from _log import setup_log

log = setup_log("patch")
log.info("=== 01_fix_n_patch.py start ===")


def run(cmd, ignore_error=False):
    print(f"\n$ {cmd}")
    log.debug(f"run: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0 and not ignore_error:
        print(f"[WARN] exit code {result.returncode}")
        log.warning(f"exit code {result.returncode}: {cmd}")


def find_site_packages():
    """Returns the site-packages path for the current venv."""
    for p in site.getsitepackages():
        if "site-packages" in p:
            return Path(p)
    return Path(site.getsitepackages()[0])


sp = find_site_packages()
log.info(f"site-packages: {sp}")
print(f"site-packages: {sp}")


def _patch(label: str, applied: bool, skipped: bool = False, missing: bool = False):
    if missing:
        log.info(f"patch [{label}]: file not found, skipped")
    elif skipped:
        log.info(f"patch [{label}]: already applied")
    elif applied:
        log.info(f"patch [{label}]: applied")
    else:
        log.warning(f"patch [{label}]: pattern not found — may need update")

# ── 2. acoustics ──────────────────────────────────────────────────────────────
# ImportError: cannot import name 'sph_harm' from 'scipy.special'
# File: acoustics/directivity.py line 20: from scipy.special import sph_harm
print("\n=== Fix 2: acoustics (sph_harm -> sph_harm_y for scipy >= 1.15) ===")
target = sp / "acoustics" / "directivity.py"
if target.exists():
    if "sph_harm_y" in target.read_text():
        print("[SKIP] already patched")
        _patch("acoustics", applied=False, skipped=True)
    else:
        run(
            f"sed -i 's/from scipy.special import sph_harm/"
            f"from scipy.special import sph_harm_y as sph_harm/' {target}"
        )
        result = subprocess.run(
            [sys.executable, "-c", "import acoustics; print('acoustics OK')"],
            capture_output=True, text=True,
        )
        print(result.stdout.strip() or result.stderr.strip())
        _patch("acoustics", applied=True)
else:
    print(f"[SKIP] {target} not found")
    _patch("acoustics", applied=False, missing=True)

# ── 5. torchaudio.info() missing in 2.9+ ─────────────────────────────────────
# torchaudio 2.9+ removed torchaudio.info() — torch_audiomentations calls it
# to read audio metadata. Inject a soundfile-backed stub into the namespace.
print("\n=== Fix 5: torchaudio.info() stub (soundfile backend, removed in 2.9+) ===")
target = sp / "torchaudio" / "__init__.py"
if target.exists():
    content = target.read_text()
    if "def info(" not in content:
        stub = (
            "\n\n"
            "# Patch: torchaudio.info() removed in 2.9+ — inject soundfile stub\n"
            "def info(uri, format=None, buffer_size=4096):\n"
            "    import soundfile as _sf\n"
            "    from types import SimpleNamespace\n"
            "    _i = _sf.info(str(uri))\n"
            "    return SimpleNamespace(num_frames=_i.frames, sample_rate=_i.samplerate,\n"
            "                           num_channels=_i.channels)\n"
        )
        target.write_text(content + stub)
        result = subprocess.run(
            [sys.executable, "-c", "import torchaudio; torchaudio.info; print('torchaudio.info patch OK')"],
            capture_output=True, text=True,
        )
        print(result.stdout.strip() or result.stderr.strip())
    else:
        print("[SKIP] torchaudio.info already defined")
else:
    print(f"[SKIP] {target} not found")

# ── train.py patches ──────────────────────────────────────────────────────────
# openwakeword/openwakeword/train.py has several issues fixed below.
train_py = Path("./openwakeword/openwakeword/train.py")

# ── 6. convert_to_tflite default="False" bug ─────────────────────────────────
# action="store_true" with default="False" (a string) makes the flag always
# truthy, so TFLite conversion (which needs onnx_tf/TensorFlow) always runs.
# 02_training.py handles TFLite conversion separately via onnx2tf.
print("\n=== Fix 6: train.py --convert_to_tflite default='False' bug ===")
if train_py.exists():
    content = train_py.read_text()
    old = (
        '    parser.add_argument(\n'
        '        "--convert_to_tflite",\n'
        '        help="Convert the trained ONNX model to TFLite format",\n'
        '        action="store_true",\n'
        '        default="False",\n'
        '        required=False\n'
        '    )'
    )
    new = (
        '    parser.add_argument(\n'
        '        "--convert_to_tflite",\n'
        '        help="Convert the trained ONNX model to TFLite format",\n'
        '        action="store_true",\n'
        '        required=False\n'
        '    )'
    )
    if old in content:
        train_py.write_text(content.replace(old, new))
        print("  applied: removed default='False' from --convert_to_tflite")
    else:
        print("  [SKIP] already patched or pattern not found")
else:
    print(f"  [SKIP] {train_py} not found")

# ── 7. ONNX export: model not in eval mode ───────────────────────────────────
# torch.onnx.export is called while model is in training mode, which can affect
# dropout/batchnorm behaviour at inference time.
# Also bumps opset from 13 to 18: LayerNormalization requires opset >= 17,
# so the downgrade always fails and staying at 18 eliminates the warning noise.
print("\n=== Fix 7: train.py ONNX export — eval() + opset 13->18 ===")
if train_py.exists():
    content = train_py.read_text()
    old = (
        '        model_to_save = copy.deepcopy(model)\n'
        '        torch.onnx.export(model_to_save.to("cpu"), torch.rand(self.input_shape)[None, ],\n'
        '                          os.path.join(output_dir, model_name + ".onnx"), opset_version=13)'
    )
    new = (
        '        model_to_save = copy.deepcopy(model)\n'
        '        model_to_save.eval()\n'
        '        torch.onnx.export(model_to_save.to("cpu"), torch.rand(self.input_shape)[None, ],\n'
        '                          os.path.join(output_dir, model_name + ".onnx"), opset_version=18)'
    )
    if old in content:
        train_py.write_text(content.replace(old, new))
        print("  applied: model_to_save.eval() added, opset_version 13->18")
    else:
        print("  [SKIP] already patched or pattern not found")
else:
    print(f"  [SKIP] {train_py} not found")

# ── 8. Suppress onnxscript/onnx_ir DEBUG spam ────────────────────────────────
# The ONNX exporter emits hundreds of DEBUG lines ("An OpSchema was not provided
# for Op ...") that bury real warnings. Suppress at WARNING level.
print("\n=== Fix 8: train.py — suppress onnxscript/onnx_ir DEBUG logging ===")
if train_py.exists():
    content = train_py.read_text()
    old = "import logging\n"
    new = (
        "import logging\n"
        "logging.getLogger('onnxscript').setLevel(logging.WARNING)\n"
        "logging.getLogger('onnx_ir').setLevel(logging.WARNING)\n"
    )
    if old in content and "onnxscript').setLevel" not in content:
        train_py.write_text(content.replace(old, new, 1))
        print("  applied: onnxscript/onnx_ir loggers set to WARNING")
    else:
        print("  [SKIP] already patched or pattern not found")
else:
    print(f"  [SKIP] {train_py} not found")

# ── 4. torchaudio torchcodec fallback ────────────────────────────────────────
# torchaudio 2.9+ routes torchaudio.load() through torchcodec which may not be
# installed (breaks other deps). Patch load() to fall back to librosa when
# torchcodec is unavailable.
print("\n=== Fix 4: torchaudio.load() librosa fallback (no torchcodec) ===")
target = sp / "torchaudio" / "__init__.py"
if target.exists():
    old = "    return load_with_torchcodec("
    new = (
        "    try:\n"
        "        return load_with_torchcodec("
    )
    content = target.read_text()
    if old in content and "librosa fallback" not in content:
        patched = content.replace(
            "    return load_with_torchcodec(\n"
            "        uri,\n"
            "        frame_offset=frame_offset,\n"
            "        num_frames=num_frames,\n"
            "        normalize=normalize,\n"
            "        channels_first=channels_first,\n"
            "        format=format,\n"
            "        buffer_size=buffer_size,\n"
            "        backend=backend,\n"
            "    )",
            "    try:  # librosa fallback\n"
            "        return load_with_torchcodec(\n"
            "            uri,\n"
            "            frame_offset=frame_offset,\n"
            "            num_frames=num_frames,\n"
            "            normalize=normalize,\n"
            "            channels_first=channels_first,\n"
            "            format=format,\n"
            "            buffer_size=buffer_size,\n"
            "            backend=backend,\n"
            "        )\n"
            "    except ImportError:\n"
            "        import numpy as np\n"
            "        import librosa\n"
            "        data, sr = librosa.load(uri, sr=None, mono=False)\n"
            "        if data.ndim == 1:\n"
            "            data = data[np.newaxis, :]  # [1, time]\n"
            "        if frame_offset > 0:\n"
            "            data = data[:, frame_offset:]\n"
            "        if num_frames > 0:\n"
            "            data = data[:, :num_frames]\n"
            "        if not channels_first:\n"
            "            data = data.T\n"
            "        return torch.from_numpy(np.ascontiguousarray(data)), sr",
        )
        target.write_text(patched)
        result = subprocess.run(
            [sys.executable, "-c", "import torchaudio; print('torchaudio load patch OK')"],
            capture_output=True, text=True,
        )
        print(result.stdout.strip() or result.stderr.strip())
    else:
        print("[SKIP] patch already applied or pattern not found")
else:
    print(f"[SKIP] {target} not found")

# ── 3. piper-sample-generator ────────────────────────────────────────────────
# PyTorch 2.6: torch.load default weights_only=True -> breaks complete models
# No-op if piper-sample-generator was not cloned (Italian branch uses piper-tts instead).
print("\n=== Fix 3: piper-sample-generator (torch.load weights_only=False) ===")
target = Path("./piper-sample-generator/generate_samples.py")
if target.exists():
    run(
        f"sed -i 's/torch.load(model_path)/torch.load(model_path, weights_only=False)/' {target}"
    )
    result = subprocess.run(
        [sys.executable, "-c", f"import sys; sys.path.append('./piper-sample-generator'); import generate_samples; print('piper patch OK')"],
        capture_output=True, text=True,
    )
    print(result.stdout.strip() or result.stderr.strip())
else:
    print(f"[SKIP] {target} not found — piper-sample-generator not cloned (expected on italian-tts branch)")

# ── 9. torch_audiomentations FutureWarning ───────────────────────────────────
# output_type=None triggers a FutureWarning; output_type="tensor" a
# DeprecationWarning. Both branches only warn, no other logic. Replace both
# with a silent default so behaviour is unchanged but the noise is gone.
# No-op if upstream (>= 0.12.0) already removed the warning block.
print("\n=== Fix 9: torch_audiomentations — silence output_type warnings ===")
_warn_block = (
    '        if output_type is None:\n'
    '            warnings.warn(\n'
    '                f"Transforms now expect an `output_type` argument that currently defaults to \'tensor\', "\n'
    '                f"will default to \'dict\' in v0.12, and will be removed in v0.13. Make sure to update "\n'
    '                f"your code to something like:\\n"\n'
    '                f"  >>> augment = {self.__class__.__name__}(..., output_type=\'dict\')\\n"\n'
    '                f"  >>> augmented_samples = augment(samples).samples",\n'
    '                FutureWarning,\n'
    '            )\n'
    '            output_type = "tensor"\n'
    '\n'
    '        elif output_type == "tensor":\n'
    '            warnings.warn(\n'
    '                f"`output_type` argument will default to \'dict\' in v0.12, and will be removed in v0.13. "\n'
    '                f"Make sure to update your code to something like:\\n"\n'
    '                f"your code to something like:\\n"\n'
    '                f"  >>> augment = {self.__class__.__name__}(..., output_type=\'dict\')\\n"\n'
    '                f"  >>> augmented_samples = augment(samples).samples",\n'
    '                DeprecationWarning,\n'
    '            )\n'
)
_warn_replacement = '        if output_type is None:\n            output_type = "tensor"\n'

for _ta_file in [
    sp / "torch_audiomentations" / "core" / "transforms_interface.py",
    sp / "torch_audiomentations" / "core" / "composition.py",
]:
    print(f"  patching {_ta_file.name} ...", end=" ")
    if _ta_file.exists():
        _content = _ta_file.read_text()
        if _warn_block in _content:
            _ta_file.write_text(_content.replace(_warn_block, _warn_replacement))
            print("applied")
        else:
            print("[SKIP] already patched or pattern not found")
    else:
        print("[SKIP] not found")

# ── 10. train.py: lazy generate_samples import ───────────────────────────────
# train.py imports generate_samples from piper-sample-generator inside a function
# body (indented). str.replace on just the inner text loses the indentation context.
# Use re.sub with a capture group to preserve whatever leading whitespace exists.
import re as _re
print("\n=== Fix 10: train.py — lazy generate_samples import ===")
if train_py.exists():
    content = train_py.read_text()
    _pattern = r'( *)from generate_samples import generate_samples'
    # Repair previously mis-patched version (wrong indentation from str.replace)
    _bad = '    try:\n    from generate_samples import generate_samples\nexcept ImportError:\n    generate_samples = None'
    _good = '    try:\n        from generate_samples import generate_samples\n    except ImportError:\n        generate_samples = None'
    if _bad in content:
        train_py.write_text(content.replace(_bad, _good))
        content = train_py.read_text()
        print("  repaired: fixed bad indentation from previous patch run")

    _match = _re.search(_pattern, content)
    if _match and "except ImportError" not in content:
        ind = _match.group(1)
        new = (
            f"{ind}try:\n"
            f"{ind}    from generate_samples import generate_samples\n"
            f"{ind}except ImportError:\n"
            f"{ind}    generate_samples = None"
        )
        train_py.write_text(_re.sub(_pattern, new, content, count=1))
        print(f"  applied: generate_samples import is now lazy (indent={repr(ind)})")
    else:
        print("  [SKIP] already patched or pattern not found")
else:
    print(f"  [SKIP] {train_py} not found")

log.info("=== 01_fix_n_patch.py complete ===")
print("\n=== Patches applied ===")
print("Next: python 02_training.py")

#!/usr/bin/env python3
"""
STEP 1 - Remove temporary dependencies and apply patches to installed libraries.
Run ONCE after: python 00_download.py

Actions:
  1. pronouncing: uses pkg_resources (removed in 3.12) -> replaced with importlib.resources
  2. acoustics:   sph_harm renamed in scipy >= 1.15 -> sph_harm_y
"""

import subprocess
import sys
import site
from pathlib import Path


def run(cmd, ignore_error=False):
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0 and not ignore_error:
        print(f"[WARN] exit code {result.returncode}")


def find_site_packages():
    """Returns the site-packages path for the current venv."""
    for p in site.getsitepackages():
        if "site-packages" in p:
            return Path(p)
    return Path(site.getsitepackages()[0])


sp = find_site_packages()
print(f"site-packages: {sp}")

# ── 1. pronouncing ────────────────────────────────────────────────────────────
# ModuleNotFoundError: No module named 'pkg_resources'
# File: pronouncing/__init__.py line 3: from pkg_resources import resource_stream
print("\n=== Fix 1: pronouncing (pkg_resources -> importlib.resources) ===")
target = sp / "pronouncing" / "__init__.py"
if target.exists():
    run(
        f"sed -i 's/from pkg_resources import resource_stream/"
        f"from importlib.resources import open_binary as resource_stream/' {target}"
    )
    run(f"find {sp} -name '*.pyc' -path '*/pronouncing*' -delete")
    result = subprocess.run(
        [sys.executable, "-c", "import pronouncing; print('pronouncing OK')"],
        capture_output=True, text=True,
    )
    print(result.stdout.strip() or result.stderr.strip())
else:
    print(f"[SKIP] {target} not found")

# ── 2. acoustics ──────────────────────────────────────────────────────────────
# ImportError: cannot import name 'sph_harm' from 'scipy.special'
# File: acoustics/directivity.py line 20: from scipy.special import sph_harm
print("\n=== Fix 2: acoustics (sph_harm -> sph_harm_y for scipy >= 1.15) ===")
target = sp / "acoustics" / "directivity.py"
if target.exists():
    run(
        f"sed -i 's/from scipy.special import sph_harm/"
        f"from scipy.special import sph_harm_y as sph_harm/' {target}"
    )
    result = subprocess.run(
        [sys.executable, "-c", "import acoustics; print('acoustics OK')"],
        capture_output=True, text=True,
    )
    print(result.stdout.strip() or result.stderr.strip())
else:
    print(f"[SKIP] {target} not found")

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

print("\n=== Patches applied ===")
print("Next: python 02_training.py")

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
    print(f"[SKIP] {target} not found")

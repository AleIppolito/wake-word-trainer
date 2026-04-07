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

print("\n=== Patches applied ===")
print("Next: python 02_training.py")

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

#!/usr/bin/env python3
"""
Applica patch a librerie installate che hanno bug su Python 3.12.
Da eseguire UNA VOLTA dopo: pip install -r requirements.txt

Patch applicate:
  1. pronouncing: usa pkg_resources (rimosso in 3.12) -> sostituito con importlib.resources
  2. acoustics:   sph_harm rinominata in scipy >= 1.15 -> sph_harm_y
"""

import subprocess
import sys
import site
from pathlib import Path


def run(cmd, ignore_error=False):
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0 and not ignore_error:
        print(f"[WARN] codice di uscita {result.returncode}")


def find_site_packages():
    """Trova il percorso site-packages del venv corrente."""
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
    print(f"[SKIP] {target} non trovato")

# ── 2. acoustics ──────────────────────────────────────────────────────────────
# ImportError: cannot import name 'sph_harm' from 'scipy.special'
# File: acoustics/directivity.py line 20: from scipy.special import sph_harm
print("\n=== Fix 2: acoustics (sph_harm -> sph_harm_y per scipy >= 1.15) ===")
target = sp / "acoustics" / "directivity.py"
if target.exists():
    run(
        f"sed -i 's/from scipy.special import sph_harm$/"
        f"from scipy.special import sph_harm_y as sph_harm/' {target}"
    )
    result = subprocess.run(
        [sys.executable, "-c", "import acoustics; print('acoustics OK')"],
        capture_output=True, text=True,
    )
    print(result.stdout.strip() or result.stderr.strip())
else:
    print(f"[SKIP] {target} non trovato")

print("\n=== Patch completate ===")
print("Procedi con: python 01_setup_and_download.py")

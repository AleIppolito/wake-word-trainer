#!/usr/bin/env python3
"""
Custom Wake Word - Recorder

Install dependencies (once):
    pip install sounddevice soundfile numpy

Usage:
    python record.py

Each clip is 2 seconds long. Press Enter, wait for the cue, then say your wake word.
Vary distance, speed, pitch, and volume across recordings.
Aim for at least 200 clips (300+ recommended).
"""

import os
import sys
import time
import numpy as np

try:
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    print("[ERROR] Missing dependencies. Run:")
    print("    pip install sounddevice soundfile numpy")
    sys.exit(1)

SAMPLE_RATE = 16000
DURATION    = 2.0
OUTPUT_DIR  = "real_recordings"

os.makedirs(OUTPUT_DIR, exist_ok=True)

existing = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".wav")])
count = existing

print("=" * 52)
print("  Custom Wake Word - Recorder")
print("=" * 52)
print(f"  Output:    {OUTPUT_DIR}/")
print(f"  Format:    WAV 16 kHz mono 16-bit")
print(f"  Duration:  {DURATION}s per clip")
print(f"  Existing:  {existing}")
print()
print("  Tips for a robust model:")
print("  - Distance: 20 cm / 50 cm / 1 m from the mic")
print("  - Tone: normal / slightly higher / slightly lower")
print("  - Speed: normal / a bit slower / a bit faster")
print("  - Room: record in different rooms if possible")
print()
print("  Enter = record    q + Enter = quit")
print()

try:
    while True:
        cmd = input(f"  [{count:3d} clips]  > ").strip().lower()
        if cmd == "q":
            break

        print("  ", end="", flush=True)
        for i in range(3, 0, -1):
            print(f"{i}.. ", end="", flush=True)
            time.sleep(0.35)
        print(">>> SPEAK NOW <<<")

        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
        )
        sd.wait()

        rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
        if rms < 80:
            print("  [!] Too quiet (RMS={:.0f}) - move closer to the mic and try again.".format(rms))
            continue

        filename = os.path.join(OUTPUT_DIR, f"recording_{count:04d}.wav")
        sf.write(filename, audio, SAMPLE_RATE, subtype="PCM_16")
        print(f"  OK  {filename}  (volume={rms:.0f})")
        count += 1

except KeyboardInterrupt:
    print()

total = count - existing
print()
print(f"  Session ended: +{total} new clips, {count} total.")
print()
print("  Next - transfer the folder to the training server:")
print(f"    scp -r {OUTPUT_DIR} root@<SERVER_IP>:/root/wake-word-trainer/real_recordings")
print()
if count < 100:
    print(f"  [!] You have {count} clips. At least 200 are recommended.")
elif count < 200:
    print(f"  [OK] You have {count} clips. 200+ is better, but you can proceed.")
else:
    print(f"  [OK] You have {count} clips. Great, proceed with training.")

#!/usr/bin/env python3
"""
Hey Murph - Recorder (esegui su Windows)

Installazione (una volta sola):
    pip install sounddevice soundfile numpy

Uso:
    python record_windows.py

Ogni clip dura 2 secondi. Premi INVIO, aspetta il segnale, di' "hey murph".
Cerca di variare: distanza dal microfono, velocita', tono, volume.
Obiettivo: almeno 200 registrazioni, meglio 300.
"""

import os
import sys
import time
import numpy as np

try:
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    print("[ERRORE] Dipendenze mancanti. Esegui:")
    print("    pip install sounddevice soundfile numpy")
    sys.exit(1)

SAMPLE_RATE = 16000
DURATION    = 2.0          # secondi per clip
OUTPUT_DIR  = "hey_murph_recordings"

os.makedirs(OUTPUT_DIR, exist_ok=True)

existing = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".wav")])
count = existing

print("=" * 52)
print("  Hey Murph - Recorder")
print("=" * 52)
print(f"  Output:    {OUTPUT_DIR}/")
print(f"  Formato:   WAV 16kHz mono 16-bit")
print(f"  Durata:    {DURATION}s per clip")
print(f"  Esistenti: {existing}")
print()
print("  Consigli per variare le registrazioni:")
print("  - distanza: 20cm / 50cm / 1m dal microfono")
print("  - tono: normale / leggermente piu' alto / piu' basso")
print("  - velocita': normale / un po' piu' lenta / piu' veloce")
print("  - ambiente: se possibile, registra in stanze diverse")
print()
print("  INVIO = registra    q + INVIO = esci")
print()

try:
    while True:
        cmd = input(f"  [{count:3d} clip]  > ").strip().lower()
        if cmd == "q":
            break

        print("  ", end="", flush=True)
        for i in range(3, 0, -1):
            print(f"{i}.. ", end="", flush=True)
            time.sleep(0.35)
        print(">>> HEY MURPH <<<")

        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
        )
        sd.wait()

        rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
        if rms < 80:
            print("  [!] Troppo silenziosa (RMS={:.0f}) - riprova, avvicinati al microfono".format(rms))
            continue

        filename = os.path.join(OUTPUT_DIR, f"hey_murph_{count:04d}.wav")
        sf.write(filename, audio, SAMPLE_RATE, subtype="PCM_16")
        print(f"  OK  {filename}  (volume={rms:.0f})")
        count += 1

except KeyboardInterrupt:
    print()

total = count - existing
print()
print(f"  Sessione terminata: +{total} nuove clip, {count} totali.")
print()
print("  Prossimo passo - trasferisci la cartella sul server:")
print(f"    scp -r {OUTPUT_DIR} root@<IP_SERVER>:/root/real_recordings")
print()
if count < 100:
    print(f"  [!] Hai {count} clip. Consigliati almeno 200 per un modello decente.")
elif count < 200:
    print(f"  [OK] Hai {count} clip. 200+ e' meglio, ma puoi procedere.")
else:
    print(f"  [OK] Hai {count} clip. Ottimo, puoi procedere con il training.")

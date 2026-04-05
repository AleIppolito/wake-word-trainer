#!/usr/bin/env python3
"""
STEP 3 - Solo training (skip generate e augment se già fatti)
Utile per:
  - ritentare il training con parametri diversi
  - cambiare FALSE_ACTIVATION_PENALTY o NUMBER_OF_TRAINING_STEPS
  - ripartire dopo un crash nello step 3/3

Assume che generate_clips e augment_clips siano già stati completati da 02_training.py

Suggerimenti per migliorare il modello:
  - Se non si attiva abbastanza: abbassa FALSE_ACTIVATION_PENALTY (es. 1000-1500)
  - Se troppi falsi positivi: alza FALSE_ACTIVATION_PENALTY (es. 2500-3000)
  - Se pronuncia italiana "ei marf": prova TARGET_WORD = "ei maarf" o "ay maarf"
    (il DeepPhonemizer parte dall'inglese, spellings fonetiche aiutano)
  - Per aggiungere la tua voce reale: usa wakeword-data-collector e includi
    i sample in background_paths
"""

import os
import sys
import subprocess
import resource

# ─────────────────────────────────────────────
# CONFIGURAZIONE - modifica questi valori
# ─────────────────────────────────────────────
TARGET_WORD              = "hey murph"
NUMBER_OF_EXAMPLES       = 50000          # non usato in retrain, lasciato per compatibilita'
NUMBER_OF_TRAINING_STEPS = 50000
FALSE_ACTIVATION_PENALTY = 2000
# ─────────────────────────────────────────────

def run(cmd, ignore_error=False):
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0 and not ignore_error:
        print(f"[WARN] codice di uscita {result.returncode}")

# Fix ulimit
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (65536, hard))
print(f"ulimit -n impostato a 65536 (era {soft})")

import yaml

config = yaml.load(open("openwakeword/examples/custom_model.yml", "r").read(), yaml.Loader)

config["target_phrase"]                       = [TARGET_WORD]
config["model_name"]                          = TARGET_WORD.replace(" ", "_")
config["n_samples"]                           = NUMBER_OF_EXAMPLES
config["n_samples_val"]                       = max(500, NUMBER_OF_EXAMPLES // 10)
config["steps"]                               = NUMBER_OF_TRAINING_STEPS
config["target_accuracy"]                     = 0.7
config["target_recall"]                       = 0.5
config["output_dir"]                          = "./my_custom_model"
config["max_negative_weight"]                 = FALSE_ACTIVATION_PENALTY
config["background_paths"]                    = ["./audioset_16k", "./fma"]
config["false_positive_validation_data_path"] = "validation_set_features.npy"
config["feature_data_files"]                  = {"ACAV100M_sample": "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"}

with open("my_model.yaml", "w") as f:
    yaml.dump(config, f)

print(f"\nTarget word:     {TARGET_WORD}")
print(f"Training steps:  {NUMBER_OF_TRAINING_STEPS}")
print(f"Penalty FP:      {FALSE_ACTIVATION_PENALTY}")

print("\n=== Train model ===")
run(f"{sys.executable} openwakeword/openwakeword/train.py --training_config my_model.yaml --train_model")

print("\n=== Conversione ONNX -> TFLite ===")

model_name   = config["model_name"]
onnx_path    = f"my_custom_model/{model_name}.onnx"
tflite_tmp   = f"my_custom_model/{model_name}_float32.tflite"
tflite_final = f"my_custom_model/{model_name}.tflite"

if os.path.exists(onnx_path):
    run(f"onnx2tf -i {onnx_path} -o my_custom_model/ -kat onnx____Flatten_0")
    if os.path.exists(tflite_tmp):
        os.rename(tflite_tmp, tflite_final)
    print(f"\n=== DONE ===")
    print(f"  ONNX      -> {onnx_path}")
    print(f"  TFLite f32 -> {tflite_final}")
    print(f"  TFLite f16 -> my_custom_model/{model_name}_float16.tflite")
else:
    print(f"\n[ERRORE] {onnx_path} non trovato — il training e' fallito.")
    sys.exit(1)

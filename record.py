#!/usr/bin/env python3
"""
Record and validate wake word clips.

Recording mode (manual):
    python record.py --mode close --room cucina
    Enter = record next clip    q + Enter = quit

Recording mode (auto-loop):
    python record.py --mode far --room sala --auto [--pause 2.0]
    Loops continuously. Ctrl+C to stop.
    Audio cues: high beep = start speaking, low beep = stop.

    --mode   close | mid | far | other   (microphone distance)
    --room   any name (cucina, sala, camera, ...)
    --target total clip target across all cells (default: 500)
    --auto   loop automatically without keypresses
    --pause  seconds between clips in auto mode (default: 2.0)

Validate existing clips without recording:
    python record.py --validate [--rec-dir ./real_recordings]

Clips saved as: recording_NNNN_ROOM_MODE.wav
Per-cell cap = target // (n_rooms × n_modes). Warns when a cell is full.

Install deps (once):
    pip install sounddevice soundfile numpy
"""

import argparse
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    print("[ERROR] pip install sounddevice soundfile numpy")
    sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE    = 16000
DURATION       = 2.0
RMS_FLOOR      = 0.005
CLIP_THRESHOLD = 0.99
ENERGY_THRESH  = 0.01
FRAME_LEN      = 0.02
ONSET_MAX_S    = 0.5
OFFSET_MAX_S   = 0.3

MODES  = ("close", "mid", "far", "other")
TAG_RE = re.compile(r'^recording_\d{4}_(.+)_(close|mid|far|other)\.wav$')


# ── Audio cues ────────────────────────────────────────────────────────────────
def _beep(freq: float, duration_s: float = 0.12, volume: float = 0.4):
    t     = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    tone  = (np.sin(2 * np.pi * freq * t) * volume * 32767).astype(np.int16)
    # fade out last 20% to avoid click
    fade  = int(len(tone) * 0.2)
    tone[-fade:] = (tone[-fade:] * np.linspace(1, 0, fade)).astype(np.int16)
    sd.play(tone, samplerate=SAMPLE_RATE)
    sd.wait()

def beep_start(): _beep(880)        # high — speak now
def beep_stop():  _beep(440, 0.08)  # low  — window closed


# ── Quality check ─────────────────────────────────────────────────────────────
def validate_clip(data: np.ndarray, sr: int) -> list[str]:
    issues = []
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if np.abs(data).max() > 1.0:
        data = data / 32768.0

    rms = float(np.sqrt((data ** 2).mean()))
    if rms < RMS_FLOOR:
        issues.append(f"too quiet (rms={rms:.4f})")

    if np.abs(data).max() >= CLIP_THRESHOLD:
        issues.append("clipping")

    frame_samples = int(FRAME_LEN * sr)
    n_frames = len(data) // frame_samples
    if n_frames == 0:
        issues.append("too short")
        return issues

    frames   = data[:n_frames * frame_samples].reshape(n_frames, frame_samples)
    energies = np.sqrt((frames ** 2).mean(axis=1))
    speech   = np.where(energies > ENERGY_THRESH)[0]

    if len(speech) == 0:
        issues.append("no speech detected")
    else:
        onset_s  = speech[0]  * FRAME_LEN
        offset_s = (n_frames - speech[-1] - 1) * FRAME_LEN
        if onset_s > ONSET_MAX_S:
            issues.append(f"late onset ({onset_s:.2f}s — speak sooner after beep)")
        if offset_s > OFFSET_MAX_S:
            issues.append(f"early cutoff ({offset_s:.2f}s — word may be cut off)")

    return issues


# ── Matrix helpers ─────────────────────────────────────────────────────────────
def scan_clips(rec_dir: Path) -> dict:
    matrix = defaultdict(lambda: defaultdict(int))
    for wav in sorted(rec_dir.glob("*.wav")):
        m = TAG_RE.match(wav.name)
        if m:
            matrix[m.group(1)][m.group(2)] += 1
    return matrix


def all_rooms_modes(matrix: dict, cur_room: str = None, cur_mode: str = None):
    rooms = sorted(set(list(matrix.keys()) + ([cur_room] if cur_room else [])))
    modes = sorted(set(
        [m for r in matrix for m in matrix[r]] + ([cur_mode] if cur_mode else [])
    ))
    if not modes:
        modes = list(MODES)
    return rooms, modes


def print_matrix(matrix: dict, target: int, cur_room: str = None, cur_mode: str = None,
                 per_cell: int = None):
    rooms, modes = all_rooms_modes(matrix, cur_room, cur_mode)
    if per_cell is None:
        n_cells  = max(1, len(rooms) * len(modes))
        per_cell = max(1, target // n_cells)
    total = sum(matrix[r][m] for r in rooms for m in modes)

    col_w = max(7, max((len(m) for m in modes), default=5) + 2)
    row_w = max(10, max((len(r) for r in rooms), default=6) + 2)
    sep   = "─" * (row_w + col_w * len(modes) + 2)

    print(f"\n  Variety matrix  (per-cell target: {per_cell})")
    print(f"  {sep}")
    header = " " * row_w + "".join(m.center(col_w) for m in modes)
    print(f"  {header}")
    print(f"  {sep}")
    for room in rooms:
        marker = " ◄" if room == cur_room else ""
        row = room.ljust(row_w)
        for mode in modes:
            count = matrix[room][mode]
            cell  = f"{count}/{per_cell}"
            flag  = "!" if count >= per_cell and room == cur_room and mode == cur_mode else ""
            row  += (cell + flag).center(col_w)
        print(f"  {row}{marker}")
    print(f"  {sep}")
    print(f"  Total: {total} / {target}")

    if cur_room and cur_mode and matrix[cur_room][cur_mode] >= per_cell:
        print(f"\n  [!] Cell {cur_room}×{cur_mode} full ({matrix[cur_room][cur_mode]}/{per_cell})")
        print(f"      Consider switching room or distance.")

    return per_cell


def next_clip_number(rec_dir: Path) -> int:
    nums = [
        int(m.group(1))
        for wav in rec_dir.glob("*.wav")
        if (m := re.match(r'^recording_(\d{4})', wav.name))
    ]
    return max(nums) + 1 if nums else 0


# ── Record one clip (shared by manual + auto) ─────────────────────────────────
def record_one(args, matrix, count, per_cell, session_good, session_bad):
    """Record, validate, and save one clip. Returns updated (count, session_good, session_bad)."""
    print(f"  >>> SPEAK NOW <<<", flush=True)
    beep_start()

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )
    sd.wait()
    beep_stop()

    flat   = audio.flatten().astype(np.float32) / 32768.0
    issues = validate_clip(flat, SAMPLE_RATE)

    if issues:
        session_bad += 1
        print(f"  ✗  REJECTED — {' | '.join(issues)}")
        return count, session_good, session_bad

    filename = args.rec_dir / f"recording_{count:04d}_{args.room}_{args.mode}.wav"
    sf.write(str(filename), audio, SAMPLE_RATE, subtype="PCM_16")
    matrix[args.room][args.mode] += 1
    session_good += 1
    count += 1
    print(f"  ✓  {filename.name}")

    if matrix[args.room][args.mode] == per_cell:
        print(f"\n  [!] Cell {args.room}×{args.mode} reached target ({per_cell}).")
        print(f"      Switch room or distance for better variety.\n")

    return count, session_good, session_bad


# ── Validate mode ─────────────────────────────────────────────────────────────
def run_validate(rec_dir: Path):
    wavs = sorted(rec_dir.glob("*.wav"))
    if not wavs:
        print(f"[ERROR] No WAV files in {rec_dir}")
        sys.exit(1)

    matrix = scan_clips(rec_dir)
    rooms, modes = all_rooms_modes(matrix)
    total    = len(wavs)
    untagged = total - sum(matrix[r][m] for r in rooms for m in modes)

    print(f"\n{'='*52}")
    print(f"  Validating {total} clips in {rec_dir}")
    print(f"{'='*52}")

    bad, ok = [], []
    for wav in wavs:
        try:
            data, sr = sf.read(str(wav), always_2d=False)
        except Exception as e:
            bad.append((wav.name, [f"unreadable: {e}"]))
            continue
        issues = validate_clip(data, sr)
        if issues:
            bad.append((wav.name, issues))
        else:
            ok.append(wav.name)

    if bad:
        print(f"\n  BAD clips ({len(bad)}) — re-record these:")
        for name, issues in bad:
            print(f"    {name}")
            for issue in issues:
                print(f"      ✗ {issue}")

    print(f"\n  OK:      {len(ok)}")
    print(f"  Bad:     {len(bad)}")
    if untagged:
        print(f"  Untagged (legacy): {untagged}")

    if any(matrix[r][m] for r in rooms for m in modes):
        print_matrix(matrix, target=500, per_cell=500 // 9)


# ── Recording mode ────────────────────────────────────────────────────────────
def run_record(args):
    args.rec_dir.mkdir(parents=True, exist_ok=True)
    matrix = scan_clips(args.rec_dir)
    matrix[args.room][args.mode]

    hints = {"close": "(~20 cm)", "mid": "(~50 cm)", "far": "(~1 m)", "other": ""}
    print("=" * 52)
    print("  Wake Word Recorder")
    print("=" * 52)
    print(f"  Room:   {args.room}")
    print(f"  Mode:   {args.mode}  {hints.get(args.mode, '')}")
    print(f"  Output: {args.rec_dir}/")
    if args.auto:
        print(f"  Mode:   AUTO  (pause={args.pause}s between clips)  Ctrl+C to stop")
        print(f"  Cues:   high beep = speak  |  low beep = stop")
    else:
        print(f"  Enter = record    q + Enter = quit")
    print(f"\n  Tips: vary pitch, speed, energy across clips.")

    per_cell = args.max if args.max else args.target // 9
    print_matrix(matrix, args.target, args.room, args.mode, per_cell=per_cell)
    print()

    count        = next_clip_number(args.rec_dir)
    session_good = 0
    session_bad  = 0

    try:
        if args.auto:
            while True:
                cur = matrix[args.room][args.mode]
                print(f"\n  [{count:4d} total | {args.room}×{args.mode}: {cur}/{per_cell}]")
                count, session_good, session_bad = record_one(
                    args, matrix, count, per_cell, session_good, session_bad
                )
                time.sleep(args.pause)
        else:
            while True:
                cur         = matrix[args.room][args.mode]
                full_marker = " [FULL]" if cur >= per_cell else ""
                prompt      = f"  [{count:4d} total | {args.room}×{args.mode}: {cur}/{per_cell}{full_marker}]  > "
                cmd         = input(prompt).strip().lower()
                if cmd == "q":
                    break
                print("  ", end="", flush=True)
                for i in range(3, 0, -1):
                    print(f"{i}.. ", end="", flush=True)
                    time.sleep(0.3)
                count, session_good, session_bad = record_one(
                    args, matrix, count, per_cell, session_good, session_bad
                )

    except KeyboardInterrupt:
        print()

    print()
    print(f"  Session: +{session_good} saved, {session_bad} rejected.")
    print_matrix(matrix, args.target, args.room, args.mode, per_cell=per_cell)
    print()


# ── Entry point ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Record and validate wake word clips.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("--mode",     choices=MODES)
parser.add_argument("--room",     help="Room name (e.g. cucina, sala, camera)")
parser.add_argument("--target",   type=int,   default=500)
parser.add_argument("--max",      type=int,   default=None,
                    help="Max clips per room×mode cell. Default: target // (3×3) = 55")
parser.add_argument("--rec-dir",  type=Path,  default=Path("./real_recordings"))
parser.add_argument("--auto",     action="store_true", help="Loop automatically")
parser.add_argument("--pause",    type=float, default=2.0,
                    help="Seconds between clips in auto mode (default: 2.0)")
parser.add_argument("--validate", action="store_true",
                    help="Validate existing clips without recording")
args = parser.parse_args()

if args.validate:
    run_validate(args.rec_dir)
else:
    if not args.mode or not args.room:
        parser.error("--mode and --room are required for recording")
    args.room = args.room.strip().lower().replace(" ", "_")
    run_record(args)

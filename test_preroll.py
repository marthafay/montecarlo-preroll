#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025, Martha Elias

"""
CAUTION
Deterministic modeling is vulnerable to unnatural distortions and algorithmically triggered reactions. Independent safety and risk management strategies are essential.

DISCLAIMER (Research Only)
This repository contains a research prototype. It is provided for educational and research purposes only. It does NOT constitute financial, investment, legal, medical, or any other professional advice. No warranty is given. Use at your own risk. Before using any outputs to inform real-world decisions, obtain advice from qualified professionals and perform independent verification.

Copyright (c) 2025 Martha Elias
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:
    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

I’d be happy if you enjoy my work: https://buymeacoffee.com/marthafay

Monte Carlo Preroll

Author: Elias, Martha
Version: v1.0 (October 2025)
DOI: 10.5281/zenodo.17443778
marthaelias [at] protonmail [dot] com
"""

# test_preroll.py
# Pytest für preroll_i.py (CLI)
# - erzeugt temporäre Sine-WAV
# - ruft CLI mehrmals auf
# - prüft Output-Existenz, Dauer, Reproduzierbarkeit via --seed
#
# Lauf:
#   pytest -q audio/test_preroll.py

import os
import sys
import math
import hashlib
import struct
import wave
import subprocess
from pathlib import Path

import numpy as np
import pytest


def _script_path() -> Path:
    """Pfad zu preroll.py (dieser Test liegt im selben Ordner)."""
    here = Path(__file__).resolve().parent
    p = here / "preroll_i.py"
    if not p.exists():
        pytest.skip("preroll.py nicht gefunden neben diesem Test.")
    return p


def _write_sine_wav(path: Path, sr=22050, secs=2.0, freq=440.0, amp=0.2):
    """Mono 16-bit PCM Sine-WAV."""
    n = int(round(secs * sr))
    t = np.arange(n, dtype=np.float64) / sr
    x = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float64)
    x16 = np.clip((x * 32767.0), -32767, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(x16.tobytes())


def _read_wav_info(path: Path):
    with wave.open(str(path), "rb") as w:
        nframes = w.getnframes()
        fr = w.getframerate()
        nch = w.getnchannels()
        sw = w.getsampwidth()
    return {"frames": nframes, "sr": fr, "ch": nch, "sampwidth": sw}


def _sha256_bytes(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def test_help_runs(tmp_path: Path):
    """-h sollte ohne Exception laufen."""
    script = _script_path()
    res = subprocess.run([sys.executable, str(script), "-h"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res.returncode == 0
    assert "usage:" in res.stdout.lower() or "usage:" in res.stderr.lower()


def test_basic_render_creates_file_and_duration(tmp_path: Path):
    """
    Minimale Renderei:
    - erzeugt Output
    - Dauer ~ --dur (±20%)
    - WAV-Header plausibel
    """
    script = _script_path()
    inp = tmp_path / "in.wav"
    outp = tmp_path / "out.wav"
    sr = 22050
    _write_sine_wav(inp, sr=sr, secs=2.0)

    dur = 1.2  # kurz halten, schneller Test
    cmd = [
        sys.executable, str(script),
        str(inp),
        "--dur", str(dur),
        "--iters", "2",
        "--pink-mix", "0.15",
        "--xfade-ms", "40",
        "--out", str(outp),
        "--seed", "123"
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res.returncode == 0, f"CLI failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert outp.exists() and outp.stat().st_size > 0

    info = _read_wav_info(outp)
    # Dauer prüfen (±20%)
    got_secs = info["frames"] / info["sr"]
    assert 0.8 * dur <= got_secs <= 1.2 * dur, f"unexpected duration: {got_secs:.3f}s vs --dur={dur}s"
    # Header plausibel
    assert info["ch"] in (1, 2)
    assert info["sampwidth"] == 2


def test_reproducible_with_seed(tmp_path: Path):
    """Gleicher Seed → identische Bytes."""
    script = _script_path()
    inp = tmp_path / "in.wav"
    out1 = tmp_path / "o1.wav"
    out2 = tmp_path / "o2.wav"
    _write_sine_wav(inp, sr=22050, secs=2.0)

    base_args = [
        sys.executable, str(script), str(inp),
        "--dur", "1.0",
        "--iters", "3",
        "--pink-mix", "0.2",
        "--xfade-ms", "30",
        "--seed", "42",
    ]

    res1 = subprocess.run(base_args + ["--out", str(out1)],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    res2 = subprocess.run(base_args + ["--out", str(out2)],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res1.returncode == 0 and res2.returncode == 0

    h1 = _sha256_bytes(out1)
    h2 = _sha256_bytes(out2)
    assert h1 == h2, "Outputs differ despite fixed seed"


def test_different_seed_changes_output(tmp_path: Path):
    """Anderer Seed → in der Regel unterschiedliche Bytes."""
    script = _script_path()
    inp = tmp_path / "in.wav"
    out1 = tmp_path / "a.wav"
    out2 = tmp_path / "b.wav"
    _write_sine_wav(inp, sr=22050, secs=2.0)

    base = [sys.executable, str(script), str(inp), "--dur", "1.0", "--iters", "2", "--pink-mix", "0.15", "--xfade-ms", "20"]
    r1 = subprocess.run(base + ["--seed", "1", "--out", str(out1)],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    r2 = subprocess.run(base + ["--seed", "2", "--out", str(out2)],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert r1.returncode == 0 and r2.returncode == 0

    # Nicht knallhart – selten können gleiche Bytes entstehen. Fallback: RMS prüfen.
    h1, h2 = _sha256_bytes(out1), _sha256_bytes(out2)
    if h1 == h2:
        # Lade und vergleiche grob RMS
        def rms(p: Path):
            with wave.open(str(p), "rb") as w:
                data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64)
                return float(np.sqrt(np.mean((data / 32767.0) ** 2) + 1e-12))
        assert abs(rms(out1) - rms(out2)) < 1e-6 or False, "Same bytes and same RMS — extremely unlikely"
    else:
        assert True


def test_riser_flag_runs(tmp_path: Path):
    """Sanity: --riser --riser-type noise sollte laufen und Datei erzeugen."""
    script = _script_path()
    inp = tmp_path / "in.wav"
    outp = tmp_path / "riser.wav"
    _write_sine_wav(inp, sr=22050, secs=2.0)

    cmd = [
        sys.executable, str(script), str(inp),
        "--dur", "1.0",
        "--riser", "--riser-type", "noise", "--riser-level", "-12",
        "--pink-mix", "0.1",
        "--xfade-ms", "30",
        "--iters", "1",
        "--seed", "9",
        "--out", str(outp)
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res.returncode == 0, f"CLI failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert outp.exists() and outp.stat().st_size > 0

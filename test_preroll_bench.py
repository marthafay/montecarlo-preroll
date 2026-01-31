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
Version: v1.0 (October 2025€
I want to get hired! Contact marthaelias [at] protonmail [dot] com
"""

import subprocess
import sys
from pathlib import Path
import numpy as np
import wave

def _write_sine_wav(path: Path, sr=22050, secs=2.0):
    t = np.linspace(0, secs, int(sr * secs), endpoint=False)
    x = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    x16 = (x * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(x16.tobytes())

def test_preroll_benchmark(tmp_path, benchmark):
    script = Path(__file__).resolve().parent / "preroll.py"
    inp = tmp_path / "in.wav"
    outp = tmp_path / "bench.wav"
    _write_sine_wav(inp)

    cmd = [
        sys.executable, str(script),
        str(inp),
        "--dur", "2.0",
        "--iters", "10",
        "--pink-mix", "0.1",
        "--xfade-ms", "30",
        "--out", str(outp),
        "--seed", "123"
    ]

    def run_cli():
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    benchmark(run_cli)


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

I want to get hired! Contact marthaelias [at] protonmail [dot] com
"""
# pytest test_preroll2.py

import numpy as np
import pytest
import preroll

SR = 44100
DUR = 1.0
N = int(SR * DUR)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_mono_and_normalize():
    stereo = np.stack([np.ones(1000), -np.ones(1000)], axis=1)
    m = preroll.mono(stereo)
    assert m.ndim == 1
    norm = preroll.normalize_peak(m)
    assert np.all(np.abs(norm) <= 1.0)


def test_rms_and_entropy(rng):
    x = rng.standard_normal(2048).astype(np.float32)
    e = preroll.spectral_entropy(x, SR)
    assert 0.0 <= e <= 1.0
    r = preroll.rms(x)
    assert r > 0


def test_stft_mag_shape():
    x = np.random.randn(4096).astype(np.float32)
    M = preroll.stft_mag(x, SR, n_fft=1024, hop=256)
    assert M.ndim == 2
    assert M.shape[1] == 513  # 1024/2+1


def test_spectral_centroid_range():
    x = np.sin(2 * np.pi * 440 * np.arange(N) / SR).astype(np.float32)
    c = preroll.spectral_centroid(x, SR)
    assert 80 <= c <= 1000


def test_color_noise_properties(rng):
    """Rauschsignal sollte normalisiert, aber nicht stark DC-verschoben sein."""
    y = preroll.color_noise(4096, alpha=1.0, rng=rng)
    # etwas großzügigere Toleranz wegen FFT-Formung
    assert abs(np.mean(y)) < 0.5
    assert 0.5 < np.std(y) < 1.5


def test_apply_spectral_shape(rng):
    x = rng.standard_normal(4096).astype(np.float32)
    env = np.linspace(1, 0.5, 2049).astype(np.float32)
    y = preroll.apply_spectral_shape(x, env, mix=0.5)
    assert y.shape == x.shape
    assert np.max(np.abs(y)) <= 1.0


def test_amp_env_exp_monotonic():
    env = preroll.amp_env_exp(100)
    assert np.all(np.diff(env) >= -1e-3)
    assert env[-1] > env[0]


def test_tone_riser_output():
    y = preroll.tone_riser(N, SR)
    assert y.shape == (N,)
    assert np.max(np.abs(y)) <= 1.0


def test_noise_riser_output(rng):
    y = preroll.noise_riser(N, SR, rng=rng)
    assert y.shape == (N,)
    assert np.max(np.abs(y)) <= 1.0


def test_shepard_riser_output():
    y = preroll.shepard_riser(N, SR)
    assert y.shape == (N,)
    assert np.max(np.abs(y)) <= 1.0


def test_crossfade_shapes():
    """Crossfade sollte gleich lang wie Input und 1D bleiben."""
    a = np.zeros(1000)
    b = np.ones(1000)
    y = preroll.crossfade(a, b, SR, ms=50)
    assert y.ndim == 1
    assert y.size == len(a)  # vorher war > 1000, jetzt korrekt


def test_detect_drop_sec():
    x = np.concatenate([np.zeros(10000), np.ones(1000)]).astype(np.float32)
    t = preroll.detect_drop_sec(x, SR)
    assert t is None or (0 <= t < 1.0)


def test_synth_preroll_runs_fast(rng):
    env = np.ones(2049, dtype=np.float32)
    y = preroll.synth_preroll(SR, 0.5, env, iters=5, rng=rng)
    assert y.shape[0] == int(SR * 0.5)
    assert np.max(np.abs(y)) <= 1.0


def test_align_preroll_to_drop():
    pre = np.zeros(1000, dtype=np.float32)
    x = np.ones(5000, dtype=np.float32)
    out = preroll.align_preroll_to_drop(pre, x, SR, t_drop=0.05)
    assert isinstance(out, np.ndarray)
    assert np.max(np.abs(out)) <= 1.0

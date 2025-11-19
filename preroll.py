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

"""
preroll.py — Monte-Carlo + Spektral-Entropie + Drop-Riser (Tone/Noise/Shepard)
"""

import argparse
import numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import get_window

EPS = 1e-12

# ------------------------- Core Utils -------------------------

def mono(x):
    if x.ndim == 1: return x.astype(np.float32)
    return np.mean(x.astype(np.float32), axis=1)

def normalize_peak(x, peak=0.999):
    m = np.max(np.abs(x)) or 1.0
    return np.clip(x / m * peak, -peak, peak)

def rms(x):
    x = np.asarray(x, float); return float(np.sqrt(np.mean(x*x) + EPS))

def stft_mag(x, sr, n_fft=2048, hop=512, win="hann"):
    w = get_window(win, n_fft, fftbins=True).astype(np.float32)
    n = len(x); out = []
    for i in range(0, n - n_fft + 1, hop):
        frame = x[i:i+n_fft] * w
        spec = np.fft.rfft(frame)
        out.append(np.abs(spec))
    return np.array(out, dtype=np.float32)

def spectral_entropy(x, sr, n_fft=2048, hop=512):
    M = stft_mag(x, sr, n_fft=n_fft, hop=hop)
    if M.size == 0: return 0.0
    P = M**2
    P = P / (np.sum(P, axis=1, keepdims=True) + EPS)
    H = -np.sum(P * np.log(P + EPS), axis=1)
    Hmax = np.log(P.shape[1] + EPS)
    return float(np.clip(np.mean(H / (Hmax + EPS)), 0.0, 1.0))

def spectral_envelope(x, sr, n_fft=4096, hop=1024, smooth_bins=5):
    M = stft_mag(x, sr, n_fft=n_fft, hop=hop)
    if M.size == 0: return np.ones(n_fft//2 + 1, dtype=np.float32)
    env = np.mean(M, axis=0)
    if smooth_bins > 1:
        k = int(smooth_bins); env = np.convolve(env, np.ones(k)/k, mode="same")
    return np.maximum(env, EPS).astype(np.float32)

def spectral_centroid(x, sr, n_fft=4096, hop=1024):
    x = x[: min(len(x), 10*sr)]
    M = stft_mag(x, sr, n_fft=n_fft, hop=hop)
    if M.size == 0: return 440.0
    freqs = np.fft.rfftfreq(n_fft, 1.0/sr)
    num = (M * freqs).sum(axis=1)
    den = (M + EPS).sum(axis=1)
    return float(np.clip(np.median(num/den), 80.0, sr/4))

# ------------------------- Drop Detection -------------------------

def spectral_flux(x, sr, n_fft=2048, hop=256, smooth=5):
    M = stft_mag(x, sr, n_fft=n_fft, hop=hop)
    if len(M) < 2: return np.zeros(1), hop/sr
    diff = np.maximum(M[1:] - M[:-1], 0.0)
    flux = diff.sum(axis=1)
    if smooth > 1:
        k = int(smooth); ker = np.ones(k)/k
        flux = np.convolve(flux, ker, mode="same")
    t_step = hop / float(sr)
    return flux.astype(np.float32), t_step

def detect_drop_sec(x, sr, search_from_s=2.0, search_to_s=30.0):
    flux, dt = spectral_flux(x, sr, n_fft=2048, hop=256, smooth=7)
    times = np.arange(len(flux)) * dt
    lo = int(np.searchsorted(times, float(search_from_s)))
    hi = int(np.searchsorted(times, float(search_to_s)))
    lo = max(0, min(lo, len(flux)-1))
    hi = max(lo+1, min(hi, len(flux)))
    if hi - lo < 5: return None
    idx_rel = int(np.argmax(flux[lo:hi]))
    t_drop = float(times[lo + idx_rel])
    return t_drop

# ------------------------- Noise & shaping -------------------------

def color_noise(n, alpha, rng):
    x = rng.standard_normal(n).astype(np.float32)
    X = np.fft.rfft(x)
    freqs = np.linspace(1.0, 1.0 + len(X) - 1, len(X))
    shape = 1.0 / (freqs ** (alpha/2.0))
    y = np.fft.irfft(X * shape, n=n).astype(np.float32)
    y = y / (np.std(y) + EPS)
    return y

def apply_spectral_shape(x, target_env, mix):
    X = np.fft.rfft(x)
    mag = np.abs(X) + EPS
    phase = np.angle(X)
    if len(target_env) < len(mag):
        env = np.interp(np.linspace(0, 1, len(mag)),
                        np.linspace(0, 1, len(target_env)),
                        target_env)
    else:
        env = target_env[:len(mag)]
    shaped_mag = (1.0 - mix) * mag + mix * env * (np.mean(mag) / (np.mean(env) + EPS))
    y = np.fft.irfft(shaped_mag * np.exp(1j * phase), n=len(x)).astype(np.float32)
    return normalize_peak(y)

def amp_env_exp(n, start=0.05, end=1.0, curve=4.0):
    t = np.linspace(0, 1, n)
    y = start * (end/start) ** (t**(1.0/curve))
    return y.astype(np.float32)

# ------------------------- Riser Generators -------------------------

def tone_riser(n, sr, f_start=120.0, f_end=1000.0, amp_curve=4.0, level=0.25):
    t = np.arange(n) / float(sr)
    f_t = f_start * ((f_end / max(f_start, 1.0)) ** t)
    phase = 2*np.pi * np.cumsum(f_t) / sr
    sig = np.sin(phase).astype(np.float32)
    env = amp_env_exp(n, start=0.03, end=1.0, curve=amp_curve)
    return normalize_peak(sig * env) * float(level)

def noise_riser(n, sr, alpha=1.0, f_lo=200.0, f_hi=None, amp_curve=4.0, level=0.25, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    base = color_noise(n, alpha=alpha, rng=rng)
    if f_hi is None: f_hi = sr/2.2
    freqs = np.fft.rfftfreq(n, 1.0/sr)
    X = np.fft.rfft(base)
    blocks = 16
    blk = n // blocks
    out = np.zeros(n, dtype=np.float32)
    win = np.hanning(2*blk).astype(np.float32)
    for b in range(blocks):
        i0 = b*blk; i1 = min((b+2)*blk, n)
        if i1 - i0 <= 0: continue
        t = b / (blocks-1 + EPS)
        fc = f_lo * ((f_hi / f_lo) ** (t))
        mask = 1.0 / (1.0 + (freqs / max(fc, 40.0))**4)
        Y = X * mask
        yb = np.fft.irfft(Y, n=n).astype(np.float32)[i0:i1]
        w = win[:len(yb)]
        out[i0:i1] += yb * w
    out = normalize_peak(out)
    env = amp_env_exp(n, start=0.05, end=1.0, curve=amp_curve)
    return out * env * float(level)

def shepard_riser(n, sr, f_base=110.0, octaves=5, rate=2.0, amp_curve=4.0, level=0.22):
    t = np.arange(n) / float(sr)
    sig = np.zeros(n, dtype=np.float32)
    for k in range(-octaves//2, octaves//2+1):
        f0 = f_base * (2.0 ** k)
        f_t = f0 * (2.0 ** (rate * t))
        phase = 2*np.pi * np.cumsum(f_t)/sr
        layer = np.sin(phase)
        logf = np.log2(f_t / (f_base * (2.0 ** (k))))
        g = np.exp(-0.5 * (logf / 0.75)**2)
        sig += (layer * g).astype(np.float32)
    sig = normalize_peak(sig)
    env = amp_env_exp(n, start=0.03, end=1.0, curve=amp_curve)
    return sig * env * float(level)

# ------------------------- MC Preroll -------------------------

def slow_amp_mod(n, sr, rng):
    t = np.arange(n) / sr
    f1, f2, f3 = rng.uniform(0.05, 0.25), rng.uniform(0.25, 0.6), rng.uniform(0.6, 1.0)
    a1, a2, a3 = rng.uniform(0.2, 0.6), rng.uniform(0.1, 0.4), rng.uniform(0.05, 0.2)
    base = 0.7 + 0.3 * np.sin(2*np.pi*f1*t)
    mod  = 1.0 + a1*np.sin(2*np.pi*f1*t) + a2*np.sin(2*np.pi*f2*t + rng.uniform(0, 2*np.pi)) \
                 + a3*np.sin(2*np.pi*f3*t + rng.uniform(0, 2*np.pi))
    attack = int(0.15 * sr); release = int(0.25 * sr)
    env = np.ones(n, dtype=np.float32)
    if attack > 0: env[:attack] = np.linspace(0.1, 1.0, attack)
    if release > 0: env[-release:] = np.linspace(1.0, 0.2, release)
    return (base * mod * env).astype(np.float32)

def hard_limit(x, ceiling=0.98): return np.clip(x, -ceiling, ceiling)

def crossfade(a, b, sr, ms=120):
    n = int(sr * ms / 1000.0)
    n = max(1, min(n, len(a), len(b)))
    fo = np.linspace(1.0, 0.0, n); fi = np.linspace(0.0, 1.0, n)
    return np.concatenate([a[:-n], a[-n:]*fo + b[:n]*fi, b[n:]])

def synth_preroll(sr, dur_s, env, target_entropy=0.85, iters=200, seed=None,
                  base_loudness=0.5, rng=None, pink_mix=0.0):
    rng = np.random.default_rng(seed) if rng is None else rng
    n = int(sr * dur_s)
    best, best_err = None, 1e9
    for _ in range(iters):
        alpha = rng.uniform(0.0, 2.0)
        mix   = rng.uniform(0.2, 0.9)
        y = color_noise(n, alpha, rng)
        y = apply_spectral_shape(y, env, mix)
        y *= slow_amp_mod(n, sr, rng)
        y = normalize_peak(y)
        if pink_mix > 0.0:
            y += normalize_peak(color_noise(n, 1.0, rng)) * float(np.clip(pink_mix, 0.0, 1.0))
            y = normalize_peak(y)
        y *= float(base_loudness)
        H = spectral_entropy(y, sr)
        err = abs(H - target_entropy)
        if err < best_err:
            best, best_err = y, err
            if err < 0.01: break
    return hard_limit(best if best is not None else np.zeros(n, dtype=np.float32))

# ------------------------- Align to Drop -------------------------

def align_preroll_to_drop(preroll, x, sr, t_drop, xfade_ms=120):
    pre = np.array(preroll, dtype=np.float32)
    start_idx = max(0, int((t_drop - len(pre)/sr) * sr))
    x_cut = x[start_idx:].astype(np.float32)
    joined = crossfade(pre, x_cut, sr, ms=xfade_ms)
    return normalize_peak(joined)

# ------------------------- Main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", help="Pfad zum Eingangs-WAV")
    ap.add_argument("--dur", type=float, default=6.0)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--target", type=float, default=0.85)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--pink-mix", type=float, default=0.0)
    ap.add_argument("--xfade-ms", type=int, default=120)
    ap.add_argument("--riser", action="store_true")
    ap.add_argument("--riser-type", choices=["tone","noise","shepard"], default="tone")
    ap.add_argument("--riser-level", type=float, default=0.25)
    ap.add_argument("--riser-curve", type=float, default=4.0)
    ap.add_argument("--riser-fstart", type=float, default=120.0)
    ap.add_argument("--riser-fend", type=float, default=1200.0)
    ap.add_argument("--align-drop", action="store_true")
    ap.add_argument("--search-from", type=float, default=2.0)
    ap.add_argument("--search-to", type=float, default=30.0)
    args = ap.parse_args()

    sr, xraw = wavfile.read(args.wav)
    x = mono(xraw.astype(np.float32))
    if x.dtype.kind in "iu":
        maxv = np.max(np.abs(x)) or 1.0
        x = x / maxv
    x = normalize_peak(x)

    env = spectral_envelope(x[:min(len(x), sr*20)], sr, n_fft=4096, hop=1024, smooth_bins=7)
    base_loud = 0.6 * rms(x[:sr*5])
    f0 = spectral_centroid(x, sr)

    rng = np.random.default_rng(args.seed)
    preroll = synth_preroll(sr, args.dur, env, target_entropy=args.target,
                            iters=args.iters, seed=args.seed,
                            base_loudness=base_loud, rng=rng, pink_mix=args.pink_mix)

    if args.riser:
        n = len(preroll)
        if args.riser_type == "tone":
            riser = tone_riser(n, sr, args.riser_fstart, args.riser_fend,
                               args.riser_curve, args.riser_level)
        elif args.riser_type == "noise":
            riser = noise_riser(n, sr, 1.0, 200.0, sr/2.2,
                                args.riser_curve, args.riser_level, rng)
        else:
            riser = shepard_riser(n, sr, max(80.0, f0/4), 5,
                                  1.5, args.riser_curve, args.riser_level)
        preroll = normalize_peak(preroll + riser)

    joined = crossfade(preroll, x, sr, ms=args.xfade_ms)

    joined_drop = None
    if args.align_drop:
        t_drop = detect_drop_sec(x, sr, args.search_from, args.search_to)
        if t_drop is not None and t_drop > 0:
            joined_drop = align_preroll_to_drop(preroll, x, sr, t_drop, args.xfade_ms)

    # ✅ FIXED OUTPUT LOGIC
    if args.out:
        out_path = Path(args.out)
        wavfile.write(out_path, sr, (preroll * 32767).astype(np.int16))
    else:
        prefix = args.wav.rsplit(".", 1)[0]
        wavfile.write(f"{prefix}_preroll.wav", sr, (preroll * 32767).astype(np.int16))

    prefix = args.out or args.wav.rsplit(".", 1)[0]
    wavfile.write(f"{prefix}_joined.wav", sr, (joined * 32767).astype(np.int16))
    if joined_drop is not None:
        wavfile.write(f"{prefix}_joined_to_drop.wav", sr, (joined_drop * 32767).astype(np.int16))

    print(f"[ok] preroll entropy≈{spectral_entropy(preroll, sr):.3f} (target {args.target})")
    print(f"[ok] riser={'on' if args.riser else 'off'} type={args.riser_type} f0≈{int(f0)} Hz pink={args.pink_mix}")
    if joined_drop is not None:
        print(f"[ok] drop aligned join written: {prefix}_joined_to_drop.wav")

if __name__ == "__main__":
    main()

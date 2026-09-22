# -*- coding: utf-8 -*-
"""Shared detector and assembly parameters (one set for BOTH methods: muq and band).
Sign convention: right/lagging = +, left/leading = -. 1 frame = 41.708 ms (23.976 fps)."""
import numpy as np

FRAME = 41.708               # ms/frame
STEP = 0.5                   # anchor grid T step (s)
MIN_FR = 3.0                 # cut-size threshold = 120 ms
PEN = 700.0                  # DP segmentation penalty (middle of the zero-false-positive plateau)
QPOW = 3.0                   # anchor weight exponent = quality^qpow
SMAX = 0.3                   # max segment |slope| (drift physics ≤~2%)
MSIZE_S = 20.0               # min segment length (s)
T0 = 2.5                     # grid start (s): the band window minimum (half 2.5s); muq (half 4s) interpolates up to 4s
T_MAX_DEFAULT = 1400         # top of the DEFAULT grid (s) — fallback for a standalone caller or a test without a known duration


def make_T(dur_s):
    """The anchor grid sized to the pair's ACTUAL duration: [T0, dur) with step STEP. The top
    is dur, not dur-T0: vision only needs frames to EXIST (no 5s window), so it covers the
    tail in full; for band/multispec, tail windows that run past the track are clamped by
    the leaf clamp (o=NaN/w=0). Production builds the grid from the actual duration (any
    length, not just up to about 23 min); the global T below is only a default for a
    standalone caller or a test."""
    last = max(T0 + STEP, float(dur_s))
    return np.arange(T0, last, STEP)


T = np.arange(T0, T_MAX_DEFAULT, STEP)   # DEFAULT, for a standalone caller or a test. Production calls make_T(dur_pair).

# -*- coding: utf-8 -*-
"""The audio anchor pipeline.

Two meters on ONE core; they differ only in the map `build_arr(ref_mono, dub_mono, T) -> (o, w)`:
  • band — pure DSP (48 log bands, wmedian+agree), NO model/license;
  • muq  — a musical SSL embedder (OpenMuQ/MuQ-large-msd-iter, licensed, optional).

Shared core: detect (weight = quality^qpow, broken lines by DP, tail_filter, edge_refine); the laid
sound is warped by apply. multispec measures what remains.

Sign: rightward/lagging=+, leftward/leading=−. 1 frame = 41.708 ms (23.976 fps).
"""
from . import params, detect  # noqa: F401

__all__ = ["params", "detect"]

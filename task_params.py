"""Task parameters of conform: what a user chooses about a job. The list is closed.

A task parameter is a choice about the method or the content of the result. It has a default and a
declared domain; a value outside the domain is refused, so every accepted job is a valid one.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import Field

from track_muxer.conform.schema import ApiModel


class AudioMethod(StrEnum):
    """Meter of the audio layer."""

    BAND = "band"       # 48-band DSP, no model
    MUQ = "muq"         # neural embedder, wants a GPU


# Ceiling of the speed at which the laid sound follows a drift, % per second. It has to be positive:
# at zero the curve cannot move at all, below zero it runs away from the reference.
DRIFT_SPEED_MIN, DRIFT_SPEED_DEFAULT, DRIFT_SPEED_MAX = 0.25, 1.25, 10.0
DriftSpeedPct = Annotated[float, Field(ge=DRIFT_SPEED_MIN, le=DRIFT_SPEED_MAX)]


class DriftSpeedLimits(ApiModel):
    min: float = DRIFT_SPEED_MIN
    max: float = DRIFT_SPEED_MAX
    default: float = DRIFT_SPEED_DEFAULT


class AudioMethodLimits(ApiModel):
    values: list[AudioMethod] = list(AudioMethod)
    default: AudioMethod = AudioMethod.BAND


class TaskParamLimits(ApiModel):
    """Domains of the task parameters, for a client to show instead of restating them."""

    drift_speed_pct: DriftSpeedLimits = DriftSpeedLimits()
    audio_method: AudioMethodLimits = AudioMethodLimits()

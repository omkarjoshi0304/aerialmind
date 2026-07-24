"""AerialMind core data types.

All enums, value objects, and data structures shared across modules.
Every module imports from here — this file has ZERO internal dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray

    import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class OperatingMode(Enum):
    """Drone operating mode — determines detection models, ROE policy, and threat weights."""

    MILITARY = auto()
    CIVIL = auto()


class ThreatLevel(Enum):
    """Composite threat score mapped to discrete levels.

    Explicit integer values allow numeric comparison:
        if assessment.level.value >= ThreatLevel.HIGH.value: ...
    """

    NONE = 0
    LOW = 1
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4


class CERState(Enum):
    """Cognitive Edge Resilience state machine.

    Tracks GPS health and link health on two independent axes.
    Only when BOTH are lost (CER_FULL) does the drone enter fully
    autonomous survival behavior.
    """

    NOMINAL = auto()
    GPS_DEGRADED = auto()
    GPS_DENIED = auto()
    LINK_DEGRADED = auto()
    LINK_LOST = auto()
    CER_PARTIAL = auto()
    CER_FULL = auto()
    SAFE_ZONE_RETURN = auto()
    EMERGENCY_LAND = auto()
    LANDED = auto()


class DecisionState(Enum):
    """Decision engine state machine — what the drone is currently doing."""

    IDLE = auto()
    MONITORING = auto()
    TRACKING = auto()
    ALERTING = auto()
    AUTONOMOUS_ACTION = auto()
    RETURNING = auto()
    LANDED = auto()


class ResourcePriority(Enum):
    """QoS priority for compute resource allocation.

    The Orchestrator's QoS Manager uses these to dynamically throttle
    modules when compute is scarce. In CER mode, navigation gets
    CRITICAL priority and vision drops to LOW/BACKGROUND.
    """

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


# ---------------------------------------------------------------------------
# Sensor Data Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimestampedFrame:
    """A camera frame with capture metadata.

    Attributes:
        frame: Raw pixel data as H x W x C numpy array (uint8).
        mono_ts: Monotonic timestamp in seconds — never goes backward,
                 never affected by NTP clock corrections.
        seq_id: Sequential frame counter from the camera driver.
        width: Frame width in pixels.
        height: Frame height in pixels.
    """

    frame: NDArray[np.uint8]
    mono_ts: float
    seq_id: int
    width: int
    height: int


@dataclass(frozen=True)
class TimestampedIMU:
    """Accelerometer + gyroscope reading at a single instant.

    Arrives at 200-400 Hz — the fastest data in the system.
    The EKF uses this for its prediction step between slower updates.

    Attributes:
        accel: Acceleration in m/s^2, body frame (x=forward, y=right, z=down).
        gyro: Angular velocity in rad/s, body frame.
        mono_ts: Monotonic timestamp in seconds.
    """

    accel: tuple[float, float, float]
    gyro: tuple[float, float, float]
    mono_ts: float


@dataclass(frozen=True)
class GPSFix:
    """A GPS position fix with quality indicators.

    Attributes:
        latitude: Degrees, WGS84.
        longitude: Degrees, WGS84.
        altitude_msl: Meters above mean sea level.
        hdop: Horizontal Dilution of Precision — lower is better.
              Under 1.0 = excellent, 1-2 = good, 2-5 = moderate, >5 = poor.
        fix_type: 0 = no fix, 2 = 2D, 3 = 3D, 4 = DGPS, 5 = RTK.
        num_satellites: Visible satellites used in the solution.
        mono_ts: Monotonic timestamp in seconds.
        valid: Set False by GPS integrity checker if spoofing is suspected.
    """

    latitude: float
    longitude: float
    altitude_msl: float
    hdop: float
    fix_type: int
    num_satellites: int
    mono_ts: float
    valid: bool


@dataclass(frozen=True)
class OpticalFlowReading:
    """Raw X/Y velocity from a downward-facing optical flow sensor.

    Works even when ORB-SLAM3 can't find visual features
    (desert, water, night). Almost zero compute cost.

    Attributes:
        vx: Velocity in m/s, body-frame X axis.
        vy: Velocity in m/s, body-frame Y axis.
        quality: Sensor-reported confidence, 0-255.
        ground_distance: Distance to ground in meters (from integrated rangefinder).
        mono_ts: Monotonic timestamp in seconds.
    """

    vx: float
    vy: float
    quality: int
    ground_distance: float
    mono_ts: float
